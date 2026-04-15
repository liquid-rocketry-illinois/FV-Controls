import numpy as np

class Sensor:
    """base class for all sensors (IMU, Baromoter, GPS),
    handles sampling and lag"""

    def __init__(self, update_rate, lag=0.0):
        self.dt = 1.0 / update_rate
        self.lag = lag
        self.last_update_time = 0.0
        self.buffer = [] #for simulating lag

    def is_ready(self, t):
        """checks if enough time has passed to trigger a new sample"""
        return t >= (self.last_update_time + self.dt)
    
    
class IMU(Sensor):
    """IMU model (accelerometer + gyroscope)
    True State -> Misalignment/Scale -> Bias/Walk/Noise -> Check Saturation -> Quantization"""

    def __init__(self, update_rate,
                 accel_range, gyro_range,
                 accel_noise_density, gyro_noise_density,
                 accel_random_walk, gyro_random_walk,
                 accel_bit_depth=16, gyro_bit_depth=16,
                 g=9.81,
                 temp_noise_std=0.5, temp_bias=0.0, temp_bit_depth=12, temp_range_K=100.0):
        """CHECK: AI, initializes parameters"""
        super().__init__(update_rate)

        self.g = g

        # --- 1. HARDWARE SPECS ---
        self.accel_max = accel_range # m/s^2
        self.gyro_max = gyro_range   # rad/s

        # Resolution (Lsb) = Full Scale Range / (2^Bits)
        # Note: Range is usually +/- max, so total span is 2*max
        self.accel_lsb = (2 * self.accel_max) / (2**accel_bit_depth)
        self.gyro_lsb = (2 * self.gyro_max) / (2**gyro_bit_depth)

        # --- 2. NOISE PARAMETERS ---
        # White Noise Std Dev = Density * sqrt(sampling_rate)
        # Or provided directly as sigma
        self.accel_std = accel_noise_density
        self.gyro_std = gyro_noise_density

        # --- TEMPERATURE SENSOR (on-chip) ---
        self.temp_noise_std = temp_noise_std                                    # K
        self.temp_bias      = temp_bias                                         # K, static offset
        self.temp_lsb       = temp_range_K / (2**temp_bit_depth) if temp_bit_depth > 0 else 0.0
        
        # Bias Instability (Random Walk) parameters
        self.accel_walk_sigma = accel_random_walk
        self.gyro_walk_sigma = gyro_random_walk

        # --- 3. INTERNAL STATES ---
        # Deterministic parameters (set via setters later)
        self.misalignment = np.eye(3)
        self.scale_factor = np.eye(3)
        self.static_bias_accel = np.zeros(3)
        self.static_bias_gyro = np.zeros(3)
        
        # Stochastic States (The "Walk" accumulator)
        self.bias_walk_accel = np.zeros(3)
        self.bias_walk_gyro = np.zeros(3)

        self.saturation_warning_triggered = False 


    def extract_physics(self, rocket_state, derivatives):
        """gets ideal physical states from simulation
        args:
            rocket_state: [w1, w2, w3, v1, v2, v3, qw, qx, qy, qz]
            derivatives:  [w1dot, w2dot, w3dot, v1dot, v2dot, v3dot, qdot...] 
            
        returns:
            true_accel, true_gyro (arrays)
        """

        #w1, w2, w3
        true_gyro = np.array(rocket_state[0:3])

        #deriving from derivatives (vdot).
        #1dot, v2dot, v3dot (body frame kinematic acceleration)
        kinematic_accel = np.array(derivatives[3:6])

        #have to add gravity
        #IMU measure acceleration = kinematic accel - gravity (body)
        #need to rotate gravity vector [0, 0, -g] into body frame
        
        qw, qx, qy, qz = rocket_state[6:10]

        #normalize quat to prevent drift
        norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        if norm > 0:
            qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm

        # World-to-body rotation matrix R_BW (matches dynamics.py convention)
        xx, yy, zz = qx*qx, qy*qy, qz*qz
        wx, wy, wz = qw*qx, qw*qy, qw*qz
        xy, xz, yz = qx*qy, qx*qz, qy*qz

        R_BW = np.array([
            [1 - 2*(yy + zz),   2*(xy + wz),       2*(xz - wy)],
            [2*(xy - wz),       1 - 2*(xx + zz),   2*(yz + wx)],
            [2*(xz + wy),       2*(yz - wx),       1 - 2*(xx + yy)]
        ])

        #grav vector in world frame
        g_world = np.array([0, 0, -self.g])
        g_body = R_BW @ g_world

        #actual acceleration, what sensor feels
        true_accel = kinematic_accel - g_body

        return true_accel, true_gyro
    
    def apply_deterministic_errors(self, true_vec, scale_matrix, misalign_matrix, static_bias):
        """
        applies scale Factor, misalignment, and static bias
        y = M * S * x + b_0
        """
        scaled = scale_matrix @ true_vec

        misaligned = misalign_matrix @ scaled

        #static bias: offset
        biased = misaligned + static_bias

        return biased
    
    def update_random_walk(self, current_walk, sigma_walk, dt):
        """updates bias instability
            New_Bias = Old_Bias + N(0, sigma * sqrt(dt))
            """
        if sigma_walk > 0: #ensure instability
           step = np.random.normal(0, sigma_walk * np.sqrt(dt), 3) #takes 3 random numbers from gaussian distribution with mean 0
           return current_walk+step
        return current_walk #just return if std dev of instability is 0


    def add_stochastic_noise(self, vec, walk_bias, noise_sigma):
        """adds current random walk bias and new noise"""

        #high freq noise
        noise = np.random.normal(0, noise_sigma, 3)

        return vec+walk_bias+noise


    def check_lims(self, vec, max_val):
        """cuts the signal to the sensors range"""
        if np.any(np.abs(vec) > max_val):
            if not self.saturation_warning_triggered:
                print(f"SENSOR HAS SATURATED, VALUE: {vec} EXCEEDS {max_val}")
                self.saturation_warning_triggered = True
        return np.clip(vec, -max_val, max_val)
    
    def quantize(self, vec, lsb):
        """changes continous value to steps"""
        if lsb <= 0: return vec
        return np.round(vec/lsb) * lsb
    
    def read(self, t, rocket_state, derivatives, true_temperature=288.15):
        """executes all other functions in order and reads final result
        Args:
            t (float): current simulation time
            rocket_state (array): [w1, w2, w3, v1, v2, v3, qw, qx, qy, qz]
            derivatives (array): [w1dot, w2dot, w3dot, v1dot, v2dot, v3dot, qdot...]
            true_temperature (float): true atmospheric temperature in Kelvin (default ISA sea level)

        Returns:
            np.ndarray: [a1, a2, a3, w1, w2, w3, T]  — accel (g), gyro (rad/s), temperature (K)
        """

        #check rate
        self.last_update_time = t

        #physics
        a_true, w_true = self.extract_physics(rocket_state, derivatives)

        #update internal random walk
        self.bias_walk_accel = self.update_random_walk(self.bias_walk_accel, self.accel_walk_sigma, self.dt)
        self.bias_walk_gyro = self.update_random_walk(self.bias_walk_gyro, self.gyro_walk_sigma, self.dt)

        #for accelerometer — pipeline in m/s², convert to g at output
        a_det = self.apply_deterministic_errors(a_true, self.scale_factor, self.misalignment, self.static_bias_accel)
        a_stoch = self.add_stochastic_noise(a_det, self.bias_walk_accel, self.accel_std)
        a_sat = self.check_lims(a_stoch, self.accel_max)   # saturation in m/s²
        a_dig = self.quantize(a_sat, self.accel_lsb) / self.g  # output in g

        #for gyroscope
        w_det = self.apply_deterministic_errors(w_true, self.scale_factor, self.misalignment, self.static_bias_gyro)
        w_stoch = self.add_stochastic_noise(w_det, self.bias_walk_gyro, self.gyro_std)
        w_sat = self.check_lims(w_stoch, self.gyro_max)
        w_dig = self.quantize(w_sat, self.gyro_lsb)

        #for temperature sensor — bias + white noise + quantization
        T_noisy = true_temperature + self.temp_bias + np.random.normal(0.0, self.temp_noise_std)
        T_dig   = self.quantize(np.array([T_noisy]), self.temp_lsb)[0]

        #concat for sensor fusion, make sure order matches for C
        # [0:3] = accel, [3:6] = gyro, [6] = temperature
        return np.concatenate((a_dig, w_dig, [T_dig]))


def make_gyro_only_sensor_model(imu: IMU, state_dim: int = 10):
    """Return a sensor-model callback that exposes only the gyro channels."""
    def sensor_model_func(t: float, x: np.ndarray, u=None) -> np.ndarray:
        return np.asarray(x[0:3], dtype=float)

    sensor_model_func.measurement_type = "gyro_only"
    sensor_model_func.measurement_dim = 3
    sensor_model_func.accel_scale = float(imu.g)
    return sensor_model_func


def make_accel_gyro_sensor_model(imu: IMU, controls) -> callable:
    """Return a noiseless EKF measurement model for accel + gyro channels.

    The output matches IMU.read() ordering for the first six channels:
    [a1, a2, a3, w1, w2, w3], where accelerometer outputs are in g units and
    gyroscope outputs are in rad/s.
    """

    def sensor_model_func(t: float, x: np.ndarray, u=None) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if u is None:
            u = np.zeros_like(controls.u0)
        else:
            u = np.asarray(u, dtype=float)

        xdot = controls.f_numeric(t, x, u)
        omega = x[0:3]
        velocity_body = x[3:6]
        inertial_accel_body = xdot[3:6] + np.cross(omega, velocity_body)

        qw, qx, qy, qz = x[6:10]
        norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        if norm > 0:
            qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

        xx, yy, zz = qx * qx, qy * qy, qz * qz
        wx, wy, wz = qw * qx, qw * qy, qw * qz
        xy, xz, yz = qx * qy, qx * qz, qy * qz
        R_BW = np.array([
            [1 - 2 * (yy + zz), 2 * (xy + wz),     2 * (xz - wy)],
            [2 * (xy - wz),     1 - 2 * (xx + zz), 2 * (yz + wx)],
            [2 * (xz + wy),     2 * (yz - wx),     1 - 2 * (xx + yy)],
        ])

        altitude_asl = float(getattr(controls, "_current_altitude", 0.0) or 0.0)
        if getattr(controls, "_env_gravity_func", None) is not None:
            g_local = float(controls._env_gravity_func(altitude_asl))
        else:
            g_local = float(controls.g)

        g_body = R_BW @ np.array([0.0, 0.0, -g_local])
        specific_force_body = inertial_accel_body - g_body
        accel_g = specific_force_body / float(imu.g)
        gyro = np.asarray(x[0:3], dtype=float)
        return np.concatenate((accel_g, gyro))

    sensor_model_func.measurement_type = "accel_gyro"
    sensor_model_func.measurement_dim = 6
    sensor_model_func.accel_scale = float(imu.g)
    return sensor_model_func

#from silsim.py sensor fusion step is: xdot -= self.controls.L @ (C @ xhat - y)
#measurement vector needs to be concat


#todo
#flag in check lims

#add gravity in extract physics for true values
