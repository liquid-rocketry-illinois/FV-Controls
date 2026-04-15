import numpy as np
from sympy import Matrix

from controls import Controls
from sensor_model import IMU, make_accel_gyro_sensor_model
from sensor_fusion import SensorFusion


# Controller limits / activation settings
control_params = {
    "u0": np.array([0.0]),  # rad, initial canard deflection command
    "max_input": np.deg2rad(8),  # rad, actuator saturation limit
    "max_input_rate": np.deg2rad(545),  # rad/s, actuator maximum deflection slew rate
    "roll_rate_gain": -1.0,  # rad/(rad/s), state-feedback gain on w3
    "mach_activation_on": False,  # bool, enables the post-burn Mach gate in the controller
    "mach_activation_threshold": 0.6,  # Mach, controller activates below this value after burnout
}

# Roll-control effectiveness model
canard_model_params = {
    "a_canard": -0.625484,  # N*m/rad, zero-speed roll-moment slope
    "b_canard": 0.011041,  # N*m/(rad*(m/s)), speed-dependent roll-moment slope term
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

    a_canard = canard_model_params["a_canard"]
    b_canard = canard_model_params["b_canard"]

    def canard_moment_cfd(v_mag: float, zeta: float) -> float:
        return (a_canard + b_canard * v_mag) * zeta

    def canard_moment_jacobian(v_mag: float) -> float:
        return a_canard + b_canard * v_mag

    controls.add_control_surface_moments(lambda sv, iv: Matrix([0, 0, 0]))
    controls.set_canard_cfd_func(canard_moment_cfd)
    controls.set_canard_jacobian_func(canard_moment_jacobian)

    roll_rate_gain = control_params["roll_rate_gain"]

    def K_func(t: float, xhat: np.ndarray) -> np.ndarray:
        K = np.zeros((1, 10))
        K[0, 2] = roll_rate_gain
        return K

    controls.setK(K_func)
    controls.setL(np.zeros((10, 3)))
    controls.set_reference(lambda t: np.zeros(10))

    imu = IMU(**imu_params)

    controls.set_sensor_params(
        sensor_vars=["a1", "a2", "a3", "w1", "w2", "w3"],
        sensor_model=make_accel_gyro_sensor_model(imu, controls),
    )

    P0 = np.eye(10) * ekf_params["initial_covariance_scale"]
    Q = np.diag(ekf_params["process_noise_diag"])
    accel_std = ekf_params["measurement_noise_std"]["accel_g"]
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
