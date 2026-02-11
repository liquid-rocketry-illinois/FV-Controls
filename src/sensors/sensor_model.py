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
    """TBI, IMU model"""

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
        g_body = ...
        true_accel = kinematic_accel - g_body

        return true_gyro, true_accel
    
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
           step = np.random.normal(0, sigma_walk, np.sqrt(dt), 3) #takes 3 random numbers from gaussian distribution with mean 0, sqrt dt for random walk algo
           return current_walk+step
        return current_walk #just return if std dev of instability is 0


    def add_stochastic_noise(self, vec, walk_bias, noise_sigma):
        """adds current random walk bias and new noise"""

        #high freq noise
        noise = np.random.normal(0, noise_sigma, 3)

        return vec+walk_bias+noise



#from silsim.py sensor fusion step is: xdot -= self.controls.L @ (C @ xhat - y)
#measurement vector needs to be concat