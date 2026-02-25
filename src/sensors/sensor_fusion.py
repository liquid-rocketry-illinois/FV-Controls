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

    def update(self, t, dt, y_meas, u):
        """fusion loop, runs every time step
            t: Current time
            dt : time step since last update
            y_meas (array): noisy measurement vector from IMU 
            u (array): current control inputs (fin angles)
            
            returns xhat: filtered state estimate"""
        
        #prediction step


        """Because controls class has the f_numeric(t, x, u) method, which evaluates the 
        full nonlinear equations of motion, we don't need to piece 
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
        C = self.controls.get_C(self.xhat)
        
        #calc what we expect the imu to read right now
        y_expected = C @ self.xhat
        
        #innovation: reality - expected
        residual = y_meas - y_expected

        #innovation covariance
        S = (C @ self.P @ C.T) + self.R
        
        #optimal kalman gain
        K = self.P @ C.T @ np.linalg.inv(S)

        #update state using gain and residual
        self.xhat = self.xhat + (K @ residual)
        
        #update covariance
        self.P = (self.I - K @ C) @ self.P
        
        #quaternions drift over time with matrix math but nrom has to be 1
        quat = self.xhat[6:10]
        norm = np.linalg.norm(quat)
        if norm > 0:
            self.xhat[6:10] = quat / norm

        return self.xhat