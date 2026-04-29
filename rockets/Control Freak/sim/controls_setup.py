import numpy as np
from sympy import Matrix

from controls import Controls
from sensors.sensor_model import IMU, make_accel_gyro_sensor_model
from sensors.sensor_fusion import SensorFusion


# Controller limits / activation settings
control_params = {
    "u0": np.array([0.0]),  # rad, initial canard deflection command
    "max_input": np.deg2rad(8),  # rad, actuator saturation limit
    "max_input_rate": np.deg2rad(545),  # rad/s, actuator maximum deflection slew rate
    "roll_damping_lambda": 2.0,  # 1/s, desired exponential roll-rate decay rate
    "min_control_speed": 30.0,  # m/s, avoid huge scheduled gains at low dynamic pressure
    "mach_activation_on": False,  # bool, enables the post-burn Mach gate in the controller
    "mach_activation_threshold": 0.6,  # Mach, controller activates below this value after burnout
}

# Roll-control effectiveness model
canard_model_params = {
    "moment_coeff_per_deg": -4.23e-7,  # N*m/(m^2/s^2*deg), zero-AoA symmetric CFD fit
}

# IMU model inputs
imu_params = {
    "update_rate": 100,  # Hz, IMU sample rate
    "accel_range": 160.0,  # m/s^2, accelerometer full-scale range
    "gyro_range": np.deg2rad(2000),  # rad/s, gyroscope full-scale range
    "accel_noise_density": 0.003,  # accelerometer white-noise sigma used by the IMU model
    "gyro_noise_density": 0.0003,  # gyroscope white-noise sigma used by the IMU model
    "accel_random_walk": 0.0001,  # accelerometer bias random-walk sigma used by the IMU model
    "gyro_random_walk": 0.00001,  # gyroscope bias random-walk sigma used by the IMU model
    "accel_bit_depth": 16,  # bits, accelerometer ADC resolution
    "gyro_bit_depth": 16,  # bits, gyroscope ADC resolution
    "g": 9.81,  # m/s^2, local gravity used by the sensor model
}

# EKF covariance inputs
ekf_params = {
    "initial_covariance_scale": 0.01,  # diagonal scale for the initial EKF covariance matrix
    "process_noise_diag": [
        1e-4, 1e-4, 1e-4,  # angular-rate process noise terms
        1e-2, 1e-2, 1e-2,  # body-velocity process noise terms
        1e-5, 1e-5, 1e-5, 1e-5,  # quaternion process noise terms
    ],
    "measurement_noise_std": {
        "accel_g": imu_params["accel_noise_density"] / imu_params["g"],  # g, match IMU output units
        "accel_model_g": 0.5,  # g, EKF trust limit for accel model mismatch
        "accel_burn_g": 5.0,  # g, downweight accel during motor burn vibration/thrust mismatch
        "gyro_rad_s": imu_params["gyro_noise_density"],  # rad/s, match IMU model noise
    },
}


def build_controls_stack(parameter_bundle):
    """Create controls, IMU, and EKF objects from the internal-dynamics setup."""
    p = parameter_bundle["parameter"]
    motor_burn_time = parameter_bundle["motor_burn_time"]
    drag_func = parameter_bundle["drag_func"]

    controls = Controls(IREC_COMPLIANT=True, rocket_name="my_rocket")
    controls.load_params(p)
    controls.set_controls_params(
        u0=control_params["u0"],
        max_input=control_params["max_input"],
        max_input_rate=control_params["max_input_rate"],
    )

    if control_params["mach_activation_on"]:
        controls.set_mach_activation(control_params["mach_activation_threshold"])

    controls.set_symbols()
    w1, w2, w3, _, _, _, _, _, _, _ = controls.state_vars

    controls.set_drag_func(drag_func)

    moment_coeff_per_deg = canard_model_params["moment_coeff_per_deg"]

    def canard_moment_cfd(v_mag: float, zeta: float) -> float:
        delta_deg = np.rad2deg(zeta)
        return moment_coeff_per_deg * (v_mag ** 2) * delta_deg

    def canard_moment_jacobian(v_mag: float) -> float:
        return moment_coeff_per_deg * (v_mag ** 2) * (180.0 / np.pi)

    controls.add_control_surface_moments(lambda sv, iv: Matrix([0, 0, 0]))
    controls.set_canard_cfd_func(canard_moment_cfd)
    controls.set_canard_jacobian_func(canard_moment_jacobian)

    roll_damping_lambda = control_params["roll_damping_lambda"]
    min_control_speed = control_params["min_control_speed"]

    def K_func(t: float, xhat: np.ndarray) -> np.ndarray:
        K = np.zeros((1, 10))

        v_air_mag = float(np.linalg.norm(xhat[3:6]))
        if v_air_mag < min_control_speed:
            return K

        I3 = float(controls.get_inertia(t)[2])
        dM_dzeta = canard_moment_jacobian(v_air_mag)
        if abs(dM_dzeta) < 1e-9:
            return K

        # Desired roll damping: w3_dot_cmd = -lambda * w3.
        # Since M = dM_dzeta*zeta and w3_dot = M/I3,
        # zeta_cmd = (-I3*lambda/dM_dzeta) * w3.
        K[0, 2] = -I3 * roll_damping_lambda / dM_dzeta
        return K

    controls.setK(K_func)
    controls.setL(np.zeros((10, 3)))
    controls.set_reference(lambda t: np.zeros(10))

    imu = IMU(**imu_params)

    sensor_model = make_accel_gyro_sensor_model(imu, controls)
    sensor_model.accel_model_std_g = ekf_params["measurement_noise_std"]["accel_model_g"]
    sensor_model.accel_burn_std_g = ekf_params["measurement_noise_std"]["accel_burn_g"]

    controls.set_sensor_params(
        sensor_vars=["a1", "a2", "a3", "w1", "w2", "w3"],
        sensor_model=sensor_model,
    )

    P0 = np.eye(10) * ekf_params["initial_covariance_scale"]
    Q = np.diag(ekf_params["process_noise_diag"])
    accel_std = max(
        ekf_params["measurement_noise_std"]["accel_g"],
        ekf_params["measurement_noise_std"]["accel_model_g"],
    )
    gyro_std = ekf_params["measurement_noise_std"]["gyro_rad_s"]
    accel_var = accel_std ** 2
    gyro_var = gyro_std ** 2
    R_ekf = np.diag([accel_var, accel_var, accel_var, gyro_var, gyro_var, gyro_var])

    ekf = SensorFusion(
        initial_state=p.x0,
        initial_covariance=P0,
        Q=Q,
        R=R_ekf,
        controls_model=controls,
    )

    return {
        "controls": controls,
        "imu": imu,
        "ekf": ekf,
        "motor_burn_time": motor_burn_time,
    }
