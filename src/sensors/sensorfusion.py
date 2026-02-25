

class SensorFusion:
    """State estimator, combines noisy vector from sensor class with predictions from dynamics
    to generate state estimate, xhat"""


    def __init__(initial_state, controls_model):
        #float array so we dont get integer trunc errors
        xhat = np.array(initial_state, dtype = float) #starting guess of rocket state
        controls = controls_model #controls class instance, holds A,B,C,L matrices

    def update(self, t, dt, y, u):
        """fusion loop, runs every time step
            t: Current time
            dt : time step since last update
            y (array): boisy measurement vector from IMU 
            u (array): current control inputs (fin angles,...)
            
            returns xhat: filtered state estimate"""

        

