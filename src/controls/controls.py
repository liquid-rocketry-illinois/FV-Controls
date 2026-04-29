import numpy as np
from sympy import *
from typing import Callable, Optional
from dynamics import Dynamics


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
        self.max_input_rate : float = None  # Maximum actuator slew rate (rad/s)
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
        self._canard_cfd_func      = None
        self._canard_jacobian_func = None

        # Mach-based activation
        self._mach_activation_threshold = None   # None = start control at burnout (default)
        self._current_temperature = 288.15        # K, updated each step from IMU temperature channel

    def set_symbols(self):
        """Set the symbolic variables for the control inputs. 
        Supersedes the parent method to include control surface deflection angle. If more control surfaces are added in the future, they should be included here. Simply append more symbols to self.input_vars."""

        super().set_symbols()

        zeta = symbols('zeta', real=True)
        self.input_vars.append(zeta)


    def set_controls_params(self, u0: np.ndarray, max_input: float, max_input_rate: Optional[float] = None):
        """Set the symbolic variables for the sensor outputs.

        Args:
            u0 (np.ndarray): Initial control input vector.
            max_input (float): Maximum control input (e.g., max deflection angle). \
                Keep consistent with units used in dynamics (most likely radians).
            max_input_rate (float, optional): Maximum command slew rate in rad/s.
                If None, no rate limiting is applied.
        """
        self.u0 = np.asarray(u0, dtype=float)
        self.max_input = max_input
        self.max_input_rate = max_input_rate


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
        required_dynamics_params = [
            'I_0', 'I_f', 'I_3', 'I_3_f',
            'x_CG_0', 'x_CG_f',
            'm_0', 'm_f', 'm_p',
            'd', 'L_ne',
            't_launch_rail_clearance', 't_motor_burnout', 't_estimated_apogee',
            'thrust_times', 'thrust_forces',
            'v_wind', 'rho', 'g',
            'N', 'Cr', 'Ct', 's', 'delta',
        ]
        for param in required_dynamics_params:
            if not hasattr(self, param) or getattr(self, param) is None:
                raise ValueError(f"Dynamics parameter '{param}' not set. Did you call controls.load_params(p)?")

        required_controls_params = ['max_input', 'u0', 'K', 'L', 'sensor_model', 'M_controls_func', 'r']
        for param in required_controls_params:
            if getattr(self, param) is None:
                raise ValueError(f"Controls parameter '{param}' not set. Please set it before running the simulation.")
        if not self.sensor_vars:
            raise ValueError("Controls parameter 'sensor_vars' not set. Please call set_sensor_params() before running.")


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


    def set_canard_cfd_func(self, cfd_func: Callable):

        self._canard_cfd_func = cfd_func


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

        param_vals = self._gather_param_values(t, x, getattr(self, '_current_altitude', None))
        result = np.array(
            self._f_numeric(*(state_vals + input_vals + param_vals + [float(t)])),
            dtype=float
        ).reshape(-1)


        if self._canard_cfd_func is not None:
            v_wind_x = param_vals[19]  # index 19: v_wind1
            v_wind_y = param_vals[20]  # index 20: v_wind2
            va       = self._compute_body_airspeed(x, v_wind_x, v_wind_y)
            v_air_mag = float(np.linalg.norm(va))
            zeta  = float(input_vals[0])
            I3    = float(self.get_inertia(t)[2])
            M_cfd = self._canard_cfd_func(v_air_mag, zeta)
            result[2] += M_cfd / I3   # w3dot is index 2

        return result


    def set_canard_jacobian_func(self, func):
        """Register a function that returns ∂M_canard/∂ζ (scalar, N·m/rad).
        Used to inject the canard's contribution into the B matrix in get_AB,
        since canard_moment_cfd is numeric-only and not in the symbolic EOM.

        Args:
            func: callable(v_mag: float) -> float
                  Returns the partial derivative of canard roll moment w.r.t. zeta.
                  For M = (a + b*v)*zeta, this is simply (a + b*v).
        """
        self._canard_jacobian_func = func

    def get_AB(self, t: float, xhat: Matrix, u: Matrix) -> tuple:
        """Compute the A and B matrices for linearized state-space representation using cached lambdified Jacobians."""
        self.checkParamsSet()
        self._compile_linearization_funcs()

        param_vals = self._gather_param_values(t, xhat, getattr(self, '_current_altitude', None))
        args = (
            np.asarray(xhat, dtype=float).tolist()
            + np.asarray(u, dtype=float).tolist()
            + param_vals
            + [float(t)]
        )

        A = np.array(self._A_numeric(*args), dtype=np.float64)
        B = np.array(self._B_numeric(*args), dtype=np.float64)

        # Inject canard moment Jacobian into B[2] (w3dot row)
        # since canard_moment_cfd is not in the symbolic EOM
        if self._canard_jacobian_func is not None:
            v_wind_x = param_vals[19]
            v_wind_y = param_vals[20]
            va       = self._compute_body_airspeed(xhat, v_wind_x, v_wind_y)
            v_air_mag = float(np.linalg.norm(va))
            I3    = float(self.get_inertia(t)[2])
            dM_dzeta = float(self._canard_jacobian_func(v_air_mag))
            B[2, 0] += dM_dzeta / I3   # ∂w3dot/∂zeta

        self.A = A
        self.B = B

        return A, B


    def predict_sensor_measurement(self, t: float, xhat: np.ndarray, u: np.ndarray = None) -> np.ndarray:
        """Evaluate the configured noiseless measurement model."""
        if self.sensor_model is None:
            raise ValueError("Sensor model not set. Please call set_sensor_params() before running.")
        measurement = self.sensor_model(t, np.asarray(xhat, dtype=float), u)
        return np.asarray(measurement, dtype=float).reshape(-1)

    def get_C(self, t: float, xhat: np.ndarray, u: np.ndarray = None, A: np.ndarray = None):
        """Return the EKF measurement Jacobian for the configured sensor model."""
        if len(self.sensor_vars) == 0:
            raise ValueError("Sensor variables not set. Please use set_sensor_params() to define sensor output variables.")

        measurement_type = getattr(self.sensor_model, "measurement_type", None)

        if measurement_type == "gyro_only":
            C_num = np.zeros((3, 10), dtype=np.float64)
            C_num[:, 0:3] = np.eye(3)
            self.C = C_num
            self.C_sym = None
            return C_num

        if measurement_type == "accel_gyro":
            xhat = np.asarray(xhat, dtype=float).reshape(-1)
            if (
                self.t_launch_rail_clearance is not None
                and float(t) < float(self.t_launch_rail_clearance)
            ) or self.is_motor_burning(t):
                C_num = np.zeros((6, 10), dtype=np.float64)
                C_num[3:6, 0:3] = np.eye(3)
                self.C = C_num
                self.C_sym = None
                return C_num

            if A is None:
                A, _ = self.get_AB(t, xhat, u)

            w1, w2, w3 = xhat[0:3]
            v1, v2, v3 = xhat[3:6]
            qw, qx, qy, qz = xhat[6:10]

            accel_scale = float(getattr(self.sensor_model, "accel_scale", 9.81))
            altitude_asl = float(getattr(self, "_current_altitude", 0.0) or 0.0)
            if self._env_gravity_func is not None:
                g_local = float(self._env_gravity_func(altitude_asl))
            else:
                g_local = float(self.g)

            jac_cross = np.zeros((3, 10), dtype=np.float64)
            jac_cross[0, 1] = v3
            jac_cross[0, 2] = -v2
            jac_cross[0, 4] = -w3
            jac_cross[0, 5] = w2

            jac_cross[1, 0] = -v3
            jac_cross[1, 2] = v1
            jac_cross[1, 3] = w3
            jac_cross[1, 5] = -w1

            jac_cross[2, 0] = v2
            jac_cross[2, 1] = -v1
            jac_cross[2, 3] = -w2
            jac_cross[2, 4] = w1

            jac_minus_g_body = np.zeros((3, 10), dtype=np.float64)
            jac_minus_g_body[0, 6] = -2.0 * g_local * qy
            jac_minus_g_body[0, 7] = 2.0 * g_local * qz
            jac_minus_g_body[0, 8] = -2.0 * g_local * qw
            jac_minus_g_body[0, 9] = 2.0 * g_local * qx

            jac_minus_g_body[1, 6] = 2.0 * g_local * qx
            jac_minus_g_body[1, 7] = 2.0 * g_local * qw
            jac_minus_g_body[1, 8] = 2.0 * g_local * qz
            jac_minus_g_body[1, 9] = 2.0 * g_local * qy

            jac_minus_g_body[2, 7] = -4.0 * g_local * qx
            jac_minus_g_body[2, 8] = -4.0 * g_local * qy

            C_num = np.zeros((6, 10), dtype=np.float64)
            C_num[0:3, :] = (A[3:6, :] + jac_cross + jac_minus_g_body) / accel_scale
            C_num[0:3, 0:3] = 0.0
            C_num[3:6, 0:3] = np.eye(3)

            self.C = C_num
            self.C_sym = None
            return C_num

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


    @staticmethod
    def accel_g_to_ms2(accel_g: np.ndarray, g: float = 9.81) -> np.ndarray:
        """Convert accelerometer reading from g units to m/s².
        Use this when you need the raw IMU accel output (imu_output[0:3]) in SI units.

        Args:
            accel_g: accelerometer reading in g (as returned by IMU.read()[0:3])
            g:       gravitational acceleration in m/s² (default 9.81)

        Returns:
            np.ndarray: acceleration in m/s²
        """
        return np.asarray(accel_g, dtype=float) * g

    def set_mach_activation(self, mach_threshold: float):
        """Enable Mach-based activation: canard control only starts after burnout
        AND once the rocket slows below mach_threshold.
        Pass None to disable (control starts immediately at burnout).

        Args:
            mach_threshold: Mach number below which active control begins.
        """
        self._mach_activation_threshold = mach_threshold

    def set_current_temperature(self, temperature_K: float):
        """Update the atmospheric temperature used for speed-of-sound / Mach calculation.
        Called every step from the IMU temperature channel reading.

        Args:
            temperature_K: measured temperature in Kelvin.
        """
        self._current_temperature = float(temperature_K)

    def _speed_of_sound(self) -> float:
        """Speed of sound from current IMU temperature. c = sqrt(gamma * R_specific * T).
        gamma = 1.4, R_specific = 8314 / 28.97 = 287.05 J/(kg·K) for dry air."""
        return float(np.sqrt(1.4 * 287.05 * self._current_temperature))

    def _current_mach(self, xhat: np.ndarray) -> float:
        """Mach number from EKF velocity estimate and IMU temperature channel."""
        v_mag = float(np.sqrt(xhat[3]**2 + xhat[4]**2 + xhat[5]**2))
        return v_mag / self._speed_of_sound()

    def compute_control(self, t: float, xhat: np.ndarray) -> np.ndarray:
        """Compute the control input using state feedback control law with saturation.
        u = -K(t, xhat) @ (r(t) - xhat)
        with saturation to respect actuator limits.

        Control is inhibited:
          1. During motor burn (when IREC_COMPLIANT=True).
          2. After burnout, while Mach > mach_activation_threshold
             (only when set_mach_activation() has been called).

        Args:
            t: Current time in seconds.
            xhat: Current estimated state vector.

        Returns:
            np.ndarray: Control input vector.
        """
        r_t = self.r(t)
        error = r_t - xhat
        K_t = self.K(t, xhat)
        u_cmd = np.asarray(-K_t @ error, dtype=float).reshape(-1)
        u_prev = np.asarray(getattr(self, "_last_u_cmd", self.u0), dtype=float).reshape(-1)

        if self.max_input_rate is not None:
            last_t = getattr(self, "_last_u_cmd_time", None)
            dt_cmd = 0.0 if last_t is None else max(float(t - last_t), 0.0)
            max_delta = float(self.max_input_rate) * dt_cmd
            if dt_cmd > 0.0:
                u_cmd = np.clip(u_cmd, u_prev - max_delta, u_prev + max_delta)

        u = np.clip(u_cmd, -self.max_input, self.max_input)

        # Inhibit during motor burn
        if self.IREC_COMPLIANT and self.is_motor_burning(t):
            u = np.zeros_like(u)

        # Inhibit post-burnout until Mach drops below threshold.
        # is_motor_burning() check is explicit here so this gate always
        # requires burnout regardless of IREC_COMPLIANT.
        if self._mach_activation_threshold is not None:
            if self.is_motor_burning(t) or self._current_mach(xhat) > self._mach_activation_threshold:
                u = np.zeros_like(u)

        self._last_u_cmd = u.copy()
        self._last_u_cmd_time = float(t)
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
    
    def set_current_altitude(self, altitude: float):
        """Update the current altitude used by _gather_param_values.
        Called by the simulation loop at each timestep."""
        self._current_altitude = altitude

    def load_params(self, p):
        """Copy all parameters from a Parameter object into this Controls instance."""
        attrs = [
            # Inertia & CG
            'I_0', 'I_f', 'I_3', 'I_3_f',
            'x_CG_0', 'x_CG_f',
            # Mass
            'm_0', 'm_f', 'm_p',
            # Body geometry
            'd', 'L_ne',
            # Timing
            't_launch_rail_clearance', 't_motor_burnout', 't_estimated_apogee',
            # Thrust
            'thrust_times', 'thrust_forces',
            # Environment (fallback constants)
            'v_wind', 'rho', 'g',
            # Main fin set (roll EOM + Barrowman main fins)
            'N', 'Cr', 'Ct', 's', 'delta',
            # Legacy CN_alpha / CP — overridden by Barrowman values if compute_cnalpha_barrowman() was called
            'CP_func', 'C_d', 'Cnalpha_rocket', 'Cnalpha_fin',
            # Sim setup
            'dt', 'x0',
            # ---- Barrowman per-component results ----
            'CN_alpha_nose', 'CN_alpha_canards', 'CN_alpha_fins', 'CN_alpha_tail',
            'CP_nose', 'CP_canards', 'CP_fins', 'CP_tail',
            # ---- Barrowman geometry (nose) ----
            'L_nose', 'R_nose', 'nose_shape',
            # ---- Barrowman geometry (canards) ----
            'N_canards', 'Cr_canards', 'Ct_canards', 's_canards',
            'x_canard_LE', 'R_body_at_canard', 'x_sweep_canards', 'canard_plane_angle_deg',
            # ---- Barrowman geometry (main fins) ----
            'x_fin_LE', 'R_body_at_fin', 'x_sweep_fin',
            # ---- Barrowman geometry (tail/boattail) ----
            'tail_type', 'R_boattail_fore', 'R_boattail_aft', 'L_boattail', 'x_boattail',
        ]
        for attr in attrs:
            if hasattr(p, attr):
                setattr(self, attr, getattr(p, attr))
