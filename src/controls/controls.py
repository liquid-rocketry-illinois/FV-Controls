import numpy as np
from sympy import *
from typing import Callable, Optional
from dynamics.dynamics import Dynamics
from scipy.integrate import solve_ivp
import scipy.signal


class Controls(Dynamics):
    def __init__(self, IREC_COMPLIANT: bool, rocket_name: Optional[str] = None, dynamics: Dynamics = None):
        """Initialize the Controls class.

        Args:
            IREC_COMPLIANT (bool): Flag indicating if the control system should comply with IREC requirements (no control during motor burn).
            rocket_name (str, optional): Name of the rocket. Provide this or a `dynamics` object.
            dynamics (Dynamics, optional): Existing dynamics object to inherit `rocket_name` from.
        """
        if dynamics is not None:
            rocket_name = dynamics.rocket_name
        if rocket_name is None:
            raise ValueError("Controls requires a rocket_name. Provide rocket_name directly or pass an existing Dynamics object.")
        super().__init__(rocket_name=rocket_name)

        self.input_vars : list = []  # List of symbolic input variables
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
        self.r : Callable = None  # Reference trajectory function

        # Cached numeric helpers
        self._A_numeric = None
        self._B_numeric = None

    def set_symbols(self):
        """Set the symbolic variables for the control inputs. 
        Supersedes the parent method to include control surface deflection angle. If more control surfaces are added in the future, they should be included here. Simply append more symbols to self.input_vars."""

        super().set_symbols()

        zeta = symbols('zeta', real=True)
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
                These should be listed in the same order as returned by the sensor function
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
        required_controls_params = ['sensor_vars', 'max_input', 'u0', 'K', 'L', 'sensor_model', 'M_controls_func', 'r']
        for param in required_controls_params:
            if getattr(self, param) is None:
                raise ValueError(f"Controls parameter '{param}' not set. Please set it before running the simulation.")


    def get_moments(self) -> Matrix:
        """Get the total moments acting on the rocket, including contributions from control surfaces.
        Supersedes the parent method to include control surface moments.

        Returns:
            Matrix: Total moments acting on the rocket.
        """
        M_dynamics : Matrix = super().get_moments()
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
        n_e = {zeta: u[0]}
        self.f_subs_full = self.f_subs_full.subs(n_e)


    def _compile_linearization_funcs(self):
        """Lazily lambdify Jacobians for A and B matrices."""
        if self._A_numeric is not None and self._B_numeric is not None:
            return
        if self.f is None or self.state_vars is None:
            self.define_eom()

        w1, w2, w3, v1, v2, v3, qw, qx, qy, qz = self.state_vars
        eps = Float(1e-9)
        vxy = sqrt(v1**2 + v2**2 + eps**2)
        repl = {
            sqrt(v1**2 + v2**2): vxy,
            (v1**2 + v2**2)**(Float(1)/2): vxy,
        }

        def _prep(expr: Matrix):
            return expr.xreplace(repl)

        m = Matrix(self.state_vars)
        n = Matrix(self.input_vars)
        arg_syms = self.state_vars + self.input_vars + self.params + [self.t_sym]

        expr = _prep(self.f)

        self._A_numeric = lambdify(arg_syms, expr.jacobian(m), modules="numpy")
        self._B_numeric = lambdify(arg_syms, expr.jacobian(n), modules="numpy")


    def _compile_numeric_funcs(self):
        """Lazily lambdify EOM including control inputs."""
        if self._f_numeric is not None:
            return
        if self.f is None or self.state_vars is None:
            self.define_eom()

        w1, w2, w3, v1, v2, v3, qw, qx, qy, qz = self.state_vars
        eps = Float(1e-9)
        vxy = sqrt(v1**2 + v2**2 + eps**2)
        repl = {
            sqrt(v1**2 + v2**2): vxy,
            (v1**2 + v2**2)**(Float(1)/2): vxy,
        }

        def _prep(expr: Matrix):
            return expr.xreplace(repl)

        arg_syms = self.state_vars + self.input_vars + self.params + [self.t_sym]
        self._f_numeric = lambdify(arg_syms, _prep(self.f), modules="numpy")


    def f_numeric(self, t: float, x: np.ndarray, u: np.ndarray = None) -> np.ndarray:
        """Fast numeric evaluation of EOM including control inputs."""
        self.checkParamsSet()
        self._compile_numeric_funcs()

        state_vals = np.asarray(x, dtype=float).tolist()
        if u is None:
            input_vals = [0.0] * len(self.input_vars)
        else:
            input_vals = np.asarray(u, dtype=float).tolist()

        param_vals = self._gather_param_values(t)
        result = self._f_numeric(*(state_vals + input_vals + param_vals + [float(t)]))
        return np.array(result, dtype=float).reshape(-1)


    def get_AB(self, t: float, xhat: Matrix, u: Matrix) -> tuple:
        """Compute the A and B matrices for linearized state-space representation using cached lambdified Jacobians."""
        self.checkParamsSet()
        self._compile_linearization_funcs()

        param_vals = self._gather_param_values(t)
        args = (
            np.asarray(xhat, dtype=float).tolist()
            + np.asarray(u, dtype=float).tolist()
            + param_vals
            + [float(t)]
        )

        A = np.array(self._A_numeric(*args), dtype=np.float64)
        B = np.array(self._B_numeric(*args), dtype=np.float64)

        self.A = A
        self.B = B

        return A, B


    def get_C(self, xhat: np.ndarray):
        """Compute the control input based on the current state, estimated state, and gain matrix.

        This function input parameter
            xhat (np.ndarray): The estimated state vector.

        It returns
            np.ndarray: The computed control input vector.
        """
        if len(self.sensor_vars) == 0:
            raise ValueError("Sensor variables not set. Please use set_sensor_vars() to define sensor output variables.")

        w1, w2, w3, v1, v2, v3, qw, qx, qy, qz = self.state_vars
        m : Matrix = Matrix(self.state_vars)
        g : Matrix = Matrix(self.sensor_vars)

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
            L: The observer gain matrix.
        """
        self.L = L


    def setK(self, K: Callable):
        """Set the state feedback gain matrix K. User-defined control law as a function of time and state.
        """
        self.K = K


    def set_reference(self, r: Callable):
        """Set the reference trajectory for the control system to track.
        """
        self.r = r


    def compute_control(self, t: float, xhat: np.ndarray) -> np.ndarray:
        """Compute the control input using state feedback control law with saturation.
        u = -K(t, xhat) @ (r(t) - xhat)
        with saturation to respect actuator limits.

        This function takes these two parameters:
            t: Current time in seconds.
            xhat: Current estimated state vector.

        It returns:
            np.ndarray: Control input vector.
        """
        r_t = self.r(t)

        # e = r - x̂
        error = r_t - xhat

        K_t = self.K(t, xhat)

        # u = -K @ e
        u = -K_t @ error

        # Apply saturation with conards limit
        u = np. clip (u, -self.max_input, self.max_input)
                # disable control during motor burn
        if self.IREC_COMPLIANT and self.is_motor_burning(t):
            u = np.zeros_like(u)

        return u


    def is_motor_burning(self, t: float) -> bool:
        """Check if the motor is currently burning at time t.

        It takes
            t: Current time in seconds.
        It return
            True if motor is burning, False otherwise.
        Todo: Change the function to is_contorl_starting because control only happens after motor burns out + Mock Point 7.
        """
        if self.t_motor_burnout is None:
            # No burnout time set, assume motor not burning
            return False

        return t < self.t_motor_burnout


    def observer_dynamics(self, t: float, xhat: np.ndarray, u: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute the observer state derivative (Luenberger observer).

        x̂ = A*x̂ + B*u + L*(y - ŷ)

        it takes:
            t: Current time in seconds.
            xhat: Current estimated state vector.
            u: Current control input vector.
            y: Current sensor measurement vector.

        It returns:
            np.ndarray: Time derivative of estimated state.
        """
        A, B = self.get_AB(t, xhat, u)
        C = self.get_C(xhat)

        y_hat = C @ xhat
        innovation = y - y_hat
        #L is trust on sensor
        # X= A*x + Вжи + L*(y - ý)
        xhat_dot = A @ xhat + B @ u + self.L @ innovation
        return xhat_dot