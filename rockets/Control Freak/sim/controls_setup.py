import numpy as np
from sympy import Matrix

from controls import Controls
from sensors.sensor_model import IMU, make_accel_gyro_sensor_model
from sensors.sensor_fusion import SensorFusion


# Controller limits / activation settings
control_params = {
    "u0": np.array([0.0]),  # rad, initial canard deflection command
    "max_input": np.deg2rad(9),  # rad, actuator saturation limit
    "max_input_rate": np.deg2rad(428.571428571),  # rad/s, actuator maximum deflection slew rate
    "roll_damping_lambda": 30,  # 1/s, desired exponential roll-rate decay rate
    "roll_effectiveness_auto_flip": True,  # flips K sign if measured roll accel opposes modeled canard effect
    "roll_effectiveness_initial_sign": 1.0,  # set to -1.0 if ground testing already proves reversal
    "roll_effectiveness_post_burn_delay": 0.25,  # s, wait after burnout before checking sign
    "roll_effectiveness_min_command": np.deg2rad(2.0),  # rad, ignore tiny canard commands
    "roll_effectiveness_min_expected_accel": 0.5,  # rad/s^2, modeled canard roll accel threshold
    "roll_effectiveness_min_measured_accel": 0.5,  # rad/s^2, gyro-derived roll accel threshold
    "roll_effectiveness_mismatch_count": 5,  # consecutive opposite-sign samples before flipping
    "min_control_speed": 30.0,  # m/s, avoid huge scheduled gains at low dynamic pressure
    "irec_compliant": True,  # bool, forces zero control during motor burn when enabled
    "mach_activation_on": False,  # bool, enables the post-burn Mach gate in the controller
    "mach_activation_threshold": 0.6,  # Mach, controller activates below this value after burnout
}

# Roll-control effectiveness model
canard_model_params = {
    "moment_coeff_per_deg": -2.23e-6, #-4.23e-7,  # N*m/(m^2/s^2*deg), zero-AoA symmetric CFD fit
}

# HAL1 bench/noise test calibration from data/sensor/HAL1_LOG.TXT.
# Statistics use the initial static noise block: 199999 samples over 4018.02 s.
hal1_noise_calibration = {
    "update_rate": 50,  # Hz, raw HAL1_LOG.TXT cadence is about 20 ms/sample
    "accel_noise_std_g": np.array([0.00294723, 0.01054800, 0.00821427]),
    "gyro_noise_std_rad_s": np.deg2rad(np.array([0.339727, 0.147199, 0.122455])),
    "gyro_static_bias_rad_s": np.deg2rad(np.array([0.0327518, 0.0417113, 0.101654])),
}

hal1_accel_noise_std_g_rms = float(np.sqrt(np.mean(hal1_noise_calibration["accel_noise_std_g"] ** 2)))
hal1_gyro_noise_std_rad_s_rms = float(np.sqrt(np.mean(hal1_noise_calibration["gyro_noise_std_rad_s"] ** 2)))

# EKF trust tuning:
#   larger measurement_std_scale => trust that sensor less
#   larger process_noise_scale   => trust internal dynamics less
ekf_trust_tuning = {
    "angular_rate_process_scale": np.array([2.0, 2.0, 3.0]),  # trust gyro correction, especially roll w3
    "linear_velocity_process_scale": np.array([0.25, 0.25, 0.25]),  # trust internal dynamics more for v1/v2/v3
    "quaternion_process_scale": np.array([1.0, 1.0, 1.0, 1.0]),
    "accel_measurement_std_scale": np.array([2.0, 2.0, 2.0]),  # trust accel less for linear motion
    "gyro_measurement_std_scale": np.array([1.0, 1.0, 0.5]),  # trust roll gyro w3 more
}

# IMU model inputs
imu_params = {
    "update_rate": hal1_noise_calibration["update_rate"],  # Hz, IMU sample rate from HAL1 bench data
    "accel_range": 160.0,  # m/s^2, accelerometer full-scale range
    "gyro_range": np.deg2rad(2000),  # rad/s, gyroscope full-scale range
    "accel_noise_density": hal1_accel_noise_std_g_rms * 9.81,  # m/s^2, HAL1 measured white-noise sigma
    "gyro_noise_density": hal1_gyro_noise_std_rad_s_rms,  # rad/s, HAL1 measured white-noise sigma
    "accel_random_walk": 0.0001,  # accelerometer bias random-walk sigma used by the IMU model
    "gyro_random_walk": 0.00001,  # gyroscope bias random-walk sigma used by the IMU model
    "accel_bit_depth": 16,  # bits, accelerometer ADC resolution
    "gyro_bit_depth": 16,  # bits, gyroscope ADC resolution
    "g": 9.81,  # m/s^2, local gravity used by the sensor model
}

# EKF covariance inputs
base_process_noise_diag = np.array([
    1e-4, 1e-4, 1e-4,  # angular-rate process noise terms
    1e-2, 1e-2, 1e-2,  # body-velocity process noise terms
    1e-5, 1e-5, 1e-5, 1e-5,  # quaternion process noise terms
])

ekf_params = {
    "initial_covariance_scale": 0.01,  # diagonal scale for the initial EKF covariance matrix
    "process_noise_diag": base_process_noise_diag * np.concatenate((
        ekf_trust_tuning["angular_rate_process_scale"],
        ekf_trust_tuning["linear_velocity_process_scale"],
        ekf_trust_tuning["quaternion_process_scale"],
    )),
    "measurement_noise_std": {
        "accel_g": (
            hal1_noise_calibration["accel_noise_std_g"]
            * ekf_trust_tuning["accel_measurement_std_scale"]
        ),  # g, scaled HAL1 measured per-axis noise
        "accel_model_g": 0.5,  # g, EKF trust limit for accel model mismatch
        "accel_burn_g": 5.0,  # g, downweight accel during motor burn vibration/thrust mismatch
        "gyro_rad_s": (
            hal1_noise_calibration["gyro_noise_std_rad_s"]
            * ekf_trust_tuning["gyro_measurement_std_scale"]
        ),  # rad/s, scaled HAL1 measured per-axis noise
    },
}


def build_controls_stack(parameter_bundle):
    """Create controls, IMU, and EKF objects from the internal-dynamics setup."""
    p = parameter_bundle["parameter"]
    motor_burn_time = parameter_bundle["motor_burn_time"]
    drag_func = parameter_bundle["drag_func"]

    controls = Controls(
        IREC_COMPLIANT=control_params["irec_compliant"],
        rocket_name="my_rocket",
    )
    controls.load_params(p)
    controls.set_controls_params(
        u0=control_params["u0"],
        max_input=control_params["max_input"],
        max_input_rate=control_params["max_input_rate"],
    )
    controls.configure_roll_effectiveness_monitor(
        enabled=control_params["roll_effectiveness_auto_flip"],
        initial_sign=control_params["roll_effectiveness_initial_sign"],
        post_burn_delay_s=control_params["roll_effectiveness_post_burn_delay"],
        min_abs_command_rad=control_params["roll_effectiveness_min_command"],
        min_abs_expected_roll_accel_rad_s2=control_params["roll_effectiveness_min_expected_accel"],
        min_abs_measured_roll_accel_rad_s2=control_params["roll_effectiveness_min_measured_accel"],
        required_consecutive_mismatches=control_params["roll_effectiveness_mismatch_count"],
        allow_flip_back=False,
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
        # u = -K*(xhat-r), so K = I3*lambda/dM_dzeta.
        K[0, 2] = controls.roll_effectiveness_sign * I3 * roll_damping_lambda / dM_dzeta
        return K

    controls.setK(K_func)
    controls.setL(np.zeros((10, 3)))
    controls.set_reference(lambda t: np.zeros(10))

    imu = IMU(**imu_params)
    imu.static_bias_gyro = hal1_noise_calibration["gyro_static_bias_rad_s"].copy()

    sensor_model = make_accel_gyro_sensor_model(imu, controls)
    sensor_model.accel_model_std_g = ekf_params["measurement_noise_std"]["accel_model_g"]
    sensor_model.accel_burn_std_g = ekf_params["measurement_noise_std"]["accel_burn_g"]

    controls.set_sensor_params(
        sensor_vars=["a1", "a2", "a3", "w1", "w2", "w3"],
        sensor_model=sensor_model,
    )

    P0 = np.eye(10) * ekf_params["initial_covariance_scale"]
    Q = np.diag(ekf_params["process_noise_diag"])
    accel_std = np.maximum(
        np.asarray(ekf_params["measurement_noise_std"]["accel_g"], dtype=float),
        float(ekf_params["measurement_noise_std"]["accel_model_g"]),
    )
    gyro_std = np.asarray(ekf_params["measurement_noise_std"]["gyro_rad_s"], dtype=float)
    R_ekf = np.diag(np.concatenate((accel_std ** 2, gyro_std ** 2)))

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
