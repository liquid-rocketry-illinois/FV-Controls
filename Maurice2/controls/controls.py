import numpy as np
from sympy import *
from typing import Callable
from dynamics import Dynamics

class Controls(Dynamics):
    def __init__(self, IREC_COMPLIANT: bool):
        """Initialize the Controls class.

        Args:
            IREC_COMPLIANT (bool): Flag indicating if the control system should comply with IREC requirements (no control during motor burn).
        """
        super().__init__()
        
        self.input_vars : list = [] # List of symbolic input variables
        self.max_input : float = None  # Maximum control input (e.g., max deflection angle)
        self.u0 : np.ndarray = None  # Initial control input vector
        self.sensor_vars : list = []  # List of sensor output variables
        self.M_controls_func : Callable = None  # Moment contributions from control surfaces
        self.IREC_COMPLIANT : bool = IREC_COMPLIANT  # IREC requirement: no control during motor burn

        self.A_sym : Matrix = None  # State matrix
        self.B_sym : Matrix = None  # Input matrix
        self.C_sym : Matrix = None  # Output matrix

        self.A : np.array = None  # Numerical state matrix
        self.B : np.array = None  # Numerical input matrix
        self.C : np.array = None  # Numerical output matrix

        self.K : Callable = None  # User-defined gain-scheduled state feedback matrix
        self.L : np.array = None  # Observer gain matrix

        self.sensor_model : Callable = None  # User-defined sensor output function


    def set_symbols(self):
        """Set the symbolic variables for the control inputs. Supersedes the parent method to include control surface deflection angle.
        If more control surfaces are added in the future, they should be included here. Simply append more symbols to self.input_vars.
        """
        super().set_symbols()
        
        zeta = symbols('zeta', real = True)
        self.input_vars.append(zeta)


    def set_controls_params(self, u0: np.ndarray, max_input: float):
        """Set the symbolic variables for the sensor outputs.

        Args:
            u0 (np.ndarray): Initial control input vector.
            max_input (float): Maximum control input (e.g., max deflection angle). \
            Keep consistent with units used in dynamics (most likely radians).
        """
        self.u0 = u0
        self.max_input = max_input


    def set_sensor_params(self, sensor_vars: list, sensor_model: Callable):
        """Set the sensor output function. User-defined function to simulate sensor measurements.

        Args:
            sensor_vars (list): List of ***symbolic*** variables representing sensor outputs. \
                These should be listed in the same order as returned by the sensor function.
                for example, if your sensor function returns [w1, w2, w3, v1, v2, v3, qw, qx, qy, qz],
                then sensor_vars should be [w1, w2, w3, v1, v2, v3, qw, qx, qy, qz].
                **Must be a subset of self.state_vars (can include all state vars).**
                **Must be defined using self.state_vars symbols.**
            sensor_model (Callable): Function that takes in time and the state vector, and returns the sensor output vector.
        """
        self.sensor_vars = sensor_vars
        self.sensor_model = sensor_model

    
    def checkParamsSet(self):
        """Check if all necessary parameters have been set. Supersedes the parent method to include control-specific parameters.

        Raises:
            ValueError: If any parameter is not set.
        """
        super().checkParamsSet()
        required_controls_params = ['sensor_vars', 'max_input', 'u0', 'K', 'L', 'sensor_model', 'M_controls_func']
        for param in required_controls_params:
            if getattr(self, param) is None:
                raise ValueError(f"Controls parameter '{param}' not set. Please set it before running the simulation.")
    
    
    def get_moments(self, burnout: bool) -> Matrix:
        """Get the total moments acting on the rocket, including contributions from control surfaces.
        Supersedes the parent method to include control surface moments.

        Args:
            burnout (bool): Whether the rocket is in the pre-burnout or post-burnout phase. (True for post-burnout)

        Returns:
            Matrix: Total moments acting on the rocket.
        """
        
        M_dynamics : Matrix = super().get_moments(burnout)
        
        if self.IREC_COMPLIANT and not burnout:
            return M_dynamics
        
        M_controls : Matrix = self.M_controls_func(self.state_vars, self.input_vars)
        self.M = M_dynamics + M_controls
        
        return self.M
    
    
    def add_control_surface_moments(self, M_controls_func: Callable):
        self.M_controls_func = M_controls_func
    
    
    ## Additional implementation of control surface impact on forces on rocket (e.g. drag) possible


    def set_f(self, t: float, xhat: Matrix, u: Matrix):
        """Get the equations of motion evaluated at time t, state xhat, and input u. Supersedes the parent method to include control surface deflection angle.
        Args:
            t (float): The time in seconds.
            xhat (np.array): The state estimation vector as a numpy array.
            u (np.array): The input vector as a numpy array.
        ## Sets:
            self.f_subs_full (Matrix): The substituted equations of motion at a state and input.
        """
        
        super().set_f(t, xhat)
        
        if u is None:
            return
        zeta = self.input_vars[0]
        n_e = {
            zeta: u[0]
        }
        self.f_subs_full = self.f_subs_full.subs(n_e)
    
    
    def get_AB(self, t: float, xhat: Matrix, u: Matrix) -> tuple:
        """Compute the A and B matrices for linearized state-space representation.
        supersedes the parent method to include control surface deflection angle.
        Args:
            xhat (Matrix): State vector at which to linearize.
            u (Matrix): Input vector at which to linearize.
        Returns:
            tuple: A and B matrices.
        """
        self.set_f(t=t, xhat=xhat, u=u)
        w1, w2, w3, v1, v2, v3, qw, qx, qy, qz = self.state_vars
        zeta = self.input_vars[0]
        m = Matrix(self.state_vars)
        n = Matrix(self.input_vars)
        m_e = {
            w1: xhat[0],
            w2: xhat[1],
            w3: xhat[2],
            v1: xhat[3],
            v2: xhat[4],
            v3: xhat[5],
            qw: xhat[6],
            qx: xhat[7],
            qy: xhat[8],
            qz: xhat[9],
        }
        n_e = {
            zeta: u[0]
        }
        A : Matrix = self.f_subs_params.jacobian(m).subs(m_e).subs(n_e).n()
        B : Matrix = self.f_subs_params.jacobian(n).subs(m_e).subs(n_e).n()

        self.A_sym = A
        self.B_sym = B
        
        A_num = np.array(A).astype(np.float64)
        B_num = np.array(B).astype(np.float64)
        self.A = A_num
        self.B = B_num

        return A_num, B_num


    def get_thrust_accel(self, t: float):
        """Get the thrust acceleration at time t.

        Args:
            t (float): The time in seconds.

        Returns:
            np.array: The thrust acceleration vector as a numpy array.
        """
        thrust = self.get_thrust(t)
        m = self.get_mass(t)
        a_thrust = np.zeros(10)
        a_thrust[3] = thrust[0] / m
        a_thrust[4] = thrust[1] / m
        a_thrust[5] = thrust[2] / m
        return a_thrust


    def get_gravity_accel(self, xhat: np.array):
        """Get the gravity acceleration in body frame at time t.

        Args:
            xhat (np.array): The current state estimate as a numpy array.

        Returns:
            np.array: The gravity acceleration vector as a numpy array.
        """
        g = np.array([0.0, 0.0, -self.g])
        qw, qx, qy, qz = xhat[6], xhat[7], xhat[8], xhat[9]
        R_world_to_body = np.array(self.R_BW_from_q(qw, qx, qy, qz)).astype(np.float64)
        g_body = R_world_to_body @ g
        a_gravity = np.zeros(10)
        a_gravity[3:6] = g_body
        return a_gravity


    def get_C(self, xhat: np.ndarray):
        """Compute the control input based on the current state, estimated state, and gain matrix.

        Args:
            xhat (np.ndarray): The estimated state vector.
        
        Returns:
            np.ndarray: The computed control input vector.
        """
        if len(self.sensor_vars) == 0:
            raise ValueError("Sensor variables not set. Please use set_sensor_vars() to define sensor output variables.")
        
        w1, w2, w3, v1, v2, v3, qw, qx, qy, qz = self.state_vars
        m : Matrix = Matrix(self.state_vars)
        g : Matrix = Matrix(self.sensor_vars)
        
        # g = Matrix([
        #     [w1],
        #     [w2],
        #     [w3],
        #     # [v1],
        #     # [v2],
        #     # [v3],
        #     [qw],
        #     [qx],
        #     [qy],
        #     [qz]
        # ])

        m_e = {
            w1: xhat[0],
            w2: xhat[1],
            w3: xhat[2],
            v1: xhat[3],
            v2: xhat[4],
            v3: xhat[5],
            qw: xhat[6],
            qx: xhat[7],
            qy: xhat[8],
            qz: xhat[9],
        }

        C : Matrix = g.jacobian(m).subs(m_e).n()
        
        self.C_sym = C
        C_num = np.array(C).astype(np.float64)
        self.C = C_num
        return C_num


    def setL(self, L: np.ndarray):
        """Set the observer gain matrix L.

        Args:
            L (np.ndarray): The observer gain matrix.
        """
        self.L = L
        

    def setK(self, K: Callable):
        """Set the state feedback gain matrix K. User-defined control law as a function of time and state.

        Args:
            K (Callable): Function that takes in time and state, and returns the gain matrix.
        """
        self.K = K
    
    
    