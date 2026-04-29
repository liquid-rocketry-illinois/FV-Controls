import numpy as np

class SensorFusion:
    """EKF State estimator, combines noisy vector from sensor class with predictions from dynamics
    to generate state estimate, xhat, using a dynamic Kalman Gain"""

    def __init__(self, initial_state, initial_covariance, Q, R, controls_model):
        #float array so we dont get integer trunc errors
        self.xhat = np.array(initial_state, dtype=float) 
        self.P = np.array(initial_covariance, dtype=float) 
        self.Q = np.array(Q, dtype=float) 
        self.R = np.array(R, dtype=float) 
        self.controls = controls_model 
        self.I = np.eye(len(initial_state)) 

    def _on_launch_rail(self, t: float) -> bool:
        rail_clearance = getattr(self.controls, "t_launch_rail_clearance", None)
        return rail_clearance is not None and float(t) < float(rail_clearance)

    def _measurement_covariance(self, t: float) -> np.ndarray:
        R = self.R.copy()
        measurement_type = getattr(self.controls.sensor_model, "measurement_type", None)
        if measurement_type == "accel_gyro" and self.controls.is_motor_burning(t):
            burn_std = getattr(self.controls.sensor_model, "accel_burn_std_g", None)
            if burn_std is not None:
                R[0:3, 0:3] = np.eye(3) * float(burn_std) ** 2
        return R

    def _limit_measurement_gain(self, t: float, K: np.ndarray) -> np.ndarray:
        """Keep each sensor channel correcting only states it directly observes."""
        measurement_type = getattr(self.controls.sensor_model, "measurement_type", None)
        if measurement_type == "accel_gyro":
            K = K.copy()
            accel_cols = slice(0, 3)
            gyro_cols = slice(3, 6)

            # Accel observes specific force, so do not let it directly jump rates or attitude.
            K[0:3, accel_cols] = 0.0
            K[6:10, accel_cols] = 0.0

            # Gyro observes angular rate, not velocity or attitude.
            K[3:10, gyro_cols] = 0.0

            if self.controls.is_motor_burning(t):
                K[:, accel_cols] = 0.0
        return K

    def _apply_rail_constraint(self):
        """Keep the EKF state consistent with the rail constraint before launch."""
        self.xhat[0] = 0.0
        self.xhat[1] = 0.0
        self.xhat[3] = 0.0
        self.xhat[4] = 0.0
        self.xhat[5] = max(0.0, float(self.xhat[5]))

        q0 = np.asarray(self.controls.x0[6:10], dtype=float).copy()
        norm = np.linalg.norm(q0)
        if norm > 0:
            q0 /= norm
        self.xhat[6:10] = q0

    def _update_on_rail(self, t, dt, y_meas, u):
        """Before rail clearance, only gyro channels are observable/useful."""
        xdot_pred = self.controls.f_numeric(t, self.xhat, u)
        self.xhat = self.xhat + (xdot_pred * dt)
        self._apply_rail_constraint()

        A, _ = self.controls.get_AB(t, self.xhat, u)
        F = self.I + (A * dt)
        self.P = (F @ self.P @ F.T) + self.Q

        C = np.zeros((6, len(self.xhat)), dtype=float)
        C[3:6, 0:3] = np.eye(3)
        y_expected = self.controls.predict_sensor_measurement(t, self.xhat, None)
        residual = y_meas - y_expected

        R = self._measurement_covariance(t)
        S = (C @ self.P @ C.T) + R
        K = self.P @ C.T @ np.linalg.inv(S)

        self.xhat = self.xhat + (K @ residual)

        IKC = self.I - K @ C
        self.P = IKC @ self.P @ IKC.T + K @ R @ K.T

        self._apply_rail_constraint()
        return self.xhat

    def update(self, t, dt, y_meas, u):
        """fusion loop, runs every time step
            t: Current time
            dt : time step since last update
            y_meas (array): noisy measurement vector from IMU 
            u (array): current control inputs (fin angles)
            
            returns xhat: filtered state estimate"""

        if self._on_launch_rail(t):
            return self._update_on_rail(t, dt, y_meas, u)
        
        #prediction step


        """Because controls class has the f_numeric(t, x, u) method, which evaluates the 
        full nonlinear equations of motion, we dont need to piece 
        together gravity and thrust using Ax + B - we can just pass the state 
        directly into the nonlinear physics engine for a more accurate 
        prediction, while reserving the Jacobians (get_AB) just for the 
        covariance (uncertainty) tracking"""


        #predict state using the full nonlinear physics engine
        #replaces old Ax + Bu + accel math
        xdot_pred = self.controls.f_numeric(t, self.xhat, u)
        self.xhat = self.xhat + (xdot_pred * dt)

        #predict covariance: uncertainty grows over time
        #get current linearized physics jacobians to track uncertainty
        A, B = self.controls.get_AB(t, self.xhat, u)
        
        #convert continuous A matrix to discrete F matrix
        F = self.I + (A * dt)
        
        #P = F * P * F^T + Q
        self.P = (F @ self.P @ F.T) + self.Q

        #--correction step--

        #get Jacobian matrix that maps state to sensors
        C = self.controls.get_C(t, self.xhat, u, A=A)
        
        #calc what we expect the imu to read right now
        y_expected = self.controls.predict_sensor_measurement(t, self.xhat, u)
        
        #innovation: reality - expected
        residual = y_meas - y_expected

        #innovation covariance
        R = self._measurement_covariance(t)
        S = (C @ self.P @ C.T) + R
        
        #optimal kalman gain
        K = self.P @ C.T @ np.linalg.inv(S)
        K = self._limit_measurement_gain(t, K)

        #update state using gain and residual
        self.xhat = self.xhat + (K @ residual)

        #update covariance — Joseph form guarantees symmetry and positive semi-definiteness
        IKC = self.I - K @ C
        self.P = IKC @ self.P @ IKC.T + K @ R @ K.T
        
        #quaternions drift over time with matrix math but nrom has to be 1
        quat = self.xhat[6:10]
        norm = np.linalg.norm(quat)
        if norm > 0:
            self.xhat[6:10] = quat / norm

        return self.xhat
