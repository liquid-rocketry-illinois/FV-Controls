from sympy import *
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from typing import Callable

from dynamics.momentsforces import MomentsForces


def build_power_state_drag_model(
    power_on_csv: str,
    power_off_csv: str,
    burnout_time: float,
):
    """Build a drag callback that matches RocketPy's power-on/power-off CSV usage."""
    power_on_drag_df = pd.read_csv(power_on_csv)
    power_off_drag_df = pd.read_csv(power_off_csv)

    power_on_drag_interp = interp1d(
        power_on_drag_df.iloc[:, 0].to_numpy(dtype=float),
        power_on_drag_df.iloc[:, 1].to_numpy(dtype=float),
        kind="linear",
        bounds_error=False,
        fill_value=(
            float(power_on_drag_df.iloc[0, 1]),
            float(power_on_drag_df.iloc[-1, 1]),
        ),
    )

    power_off_drag_interp = interp1d(
        power_off_drag_df.iloc[:, 0].to_numpy(dtype=float),
        power_off_drag_df.iloc[:, 1].to_numpy(dtype=float),
        kind="linear",
        bounds_error=False,
        fill_value=(
            float(power_off_drag_df.iloc[0, 1]),
            float(power_off_drag_df.iloc[-1, 1]),
        ),
    )

    def drag_func(mach: float, t: float) -> float:
        if t < burnout_time:
            return float(power_on_drag_interp(mach))
        return float(power_off_drag_interp(mach))

    return drag_func


class Dynamics(MomentsForces):
    def __init__(self, rocket_name : str):
        """Initialize the Dynamics class. Rocket body axis is aligned with z-axis.

        Args:
            t_estimated_apogee (float): Estimated time until apogee in seconds.
            dt (float): Time step for simulation in seconds.
            x0 (np.ndarray): Initial state vector.
            [Refer to super for other - Jed]
        """

        super().__init__()

        
        self.f_subs_params : Matrix = None
        self.f_subs_full : Matrix = None
        self.dt : float = None
        self.x0 : np.ndarray = None
        self.t_sym : Symbol = None

        ## Uninitialized parameters ##
        
        # Rocket parameters
        self.I_0 : float = None # Initial moment of inertia in kg·m²
        self.I_f : float = None # Final moment of inertia in kg·m²
        self.I_3 : float = None # Rotational moment of inertia about z-axis at launch in kg·m²
        self.I_3_f : float = None # Rotational moment of inertia about z-axis at burnout in kg·m²
        self.x_CG_0 : float = None # Initial center of gravity location in meters
        self.x_CG_f : float = None # Final center of gravity location in meters


        self.m_0 : float = None # Initial rocket mass in kg
        self.m_f : float = None # Final rocket mass in kg
        self.m_p : float = None # Propellant mass in kg
        self.d : float = None # Rocket body diameter in meters
        self.L_ne : float = None # Length from nose to nozzle in meters
        self.C_d : float = None # Drag coefficient
        self.Cnalpha_rocket : float = None # Total rocket CN_alpha (sum of all components)
        self.t_motor_burnout : float = None # Time to motor burnout in seconds
        self.t_launch_rail_clearance : float = None # Time to launch rail clearance in seconds
        self.t_estimated_apogee : float = None # Time to apogee in esconds
        self._cp_2d_func = None   # CP as function of (AoA_deg, mach)

        # Per-component Barrowman CN_alpha (set by Parameter.compute_cnalpha_barrowman)
        self.CN_alpha_nose     : float = None  # Nose cone contribution (≈ 2.0 for full-diameter nose)
        self.CN_alpha_canards  : float = None  # Canard fin set contribution
        self.CN_alpha_fins     : float = None  # Main fin set contribution (all fins combined)
        self.CN_alpha_tail     : float = None  # Boattail contribution (typically negative)
        self.CP_nose           : float = None  # Nose CP from nose tip (m)
        self.CP_canards        : float = None  # Canard fin set CP from nose tip (m)
        self.CP_fins           : float = None  # Main fin set CP from nose tip (m)
        self.CP_tail           : float = None  # Boattail CP from nose tip (m)

        # Nose geometry (used by Barrowman computation in Parameter)
        self.L_nose     : float = None  # Nose cone length (m)
        self.R_nose     : float = None  # Nose cone base radius (m); None → uses d/2
        self.nose_shape : str   = None  # Shape string: 'conical', 'ogive', 'von_karman', etc.

        # Canard geometry
        self.N_canards        : int   = None
        self.Cr_canards       : float = None
        self.Ct_canards       : float = None
        self.s_canards        : float = None
        self.x_canard_LE      : float = None  # Root LE axial position from nose tip (m)
        self.R_body_at_canard : float = None  # Body radius at canard (m)
        self.x_sweep_canards  : float = 0.0   # Canard LE sweep offset (m)
        self.canard_plane_angle_deg : float = 0.0  # Canard-plane azimuth about body axis (deg)

        # Main fin geometry (supplements setFinParams N/Cr/Ct/s/delta)
        self.x_fin_LE      : float = None  # Root LE axial position from nose tip (m)
        self.R_body_at_fin : float = None  # Body radius at fin (m); None → uses d/2
        self.x_sweep_fin   : float = 0.0   # Main fin LE sweep offset (m)

        # Tail / boattail geometry
        self.tail_type       : str   = None
        self.R_boattail_fore : float = None
        self.R_boattail_aft  : float = None
        self.L_boattail      : float = None
        self.x_boattail      : float = None

        # Fin parameters (main fin set — also used in roll moment EOM)
        self.N : float = None # Number of fins
        self.Cr : float = None # Root chord in meters
        self.Ct : float = None # Tip chord in meters
        self.s : float = None # Span in meters
        self.delta : float = None # Fin cant angle in degrees
        self.Cnalpha_fin : float = None # Per-fin CN_alpha used in roll EOM (1/rad, normalised to A_ref)
    

        
        # Rocket name (used for saving simulation results from Simluation() object to designated path)
        
        
        # State space linearization matrices
        self.A_sym : Matrix = None # Symbolic state matrix
        self.A : np.ndarray = None # State matrix
        
        ## Helpers [Refer to super for other]##
        self._f_numeric = None  # Cached lambdified EOM
        self._A_numeric = None  # Cached lambdified Jacobian of f wrt state

        self._drag_func = None # C_d as function of Mach number
        self._cnalpha_rocket_func = None  # optional: Cnalpha_rocket as function of Mach
        self._cnalpha_fin_func    = None  # optional: Cnalpha_fin as function of Mach

        #environment - call from rocektpy
        self._env_density_func     = None  # rho(altitude) from RocketPy env
        self._env_gravity_func     = None  # g(altitude) from RocketPy env
        self._env_wind_x_func      = None  # v_wind_x(altitude) from RocketPy env
        self._env_wind_y_func      = None  # v_wind_y(altitude) from RocketPy env
        self._env_temperature_func = None  # T(altitude) from RocketPy env, Kelvin


        self.rocket_name = rocket_name

    def get_inertia(self, t: float) -> list:
        """Get the moment of inertia of the rocket at time t.

        Args:
            t (float): The time in seconds.

        Returns:
            float: The moment of inertia of the rocket at time t in kg·m².
        """
        if t <= self.t_motor_burnout:
            I_long = self.I_0 - (self.I_0 - self.I_f) / self.t_motor_burnout * t
            I_roll = self.I_3 - (self.I_3 - self.I_3_f) / self.t_motor_burnout * t
        else:
            I_long = self.I_f
            I_roll = self.I_3_f
        I = [I_long, I_long, I_roll]

        return I

    def get_inertia_dot(self, t: float) -> list:
        """Get the time derivative of the rocket inertia components at time t.

        During motor burn, pitch/yaw and roll inertia are modeled as changing linearly.
        """
        if t <= self.t_motor_burnout:
            I_long_dot = - (self.I_0 - self.I_f) / self.t_motor_burnout
            I_roll_dot = - (self.I_3 - self.I_3_f) / self.t_motor_burnout
        else:
            I_long_dot = 0.0
            I_roll_dot = 0.0
        return [I_long_dot, I_long_dot, I_roll_dot]

    
    def get_CG(self, t: float) -> float:
        """Get the center of gravity location of the rocket at time t.

        Args:
            t (float): The time in seconds.

        Returns:
            float: The center of gravity location of the rocket at time t in meters.
        """
        x_CG = self.x_CG_0 - (self.x_CG_0 - self.x_CG_f) / self.t_motor_burnout * t if t <= self.t_motor_burnout else self.x_CG_f
        return x_CG

    ## BUGGED ##
    def get_AoA(self, v_wind: list, state: list):
        w1, w2, w3, v1, v2, v3, qw, qx, qy, qz = state
        v_wind1, v_wind2 = v_wind
        t_sym = self.t_sym
        H = Heaviside(t_sym - Float(self.t_launch_rail_clearance), 0)  # 0 if t < t_launch_rail_clearance, 1 if t >= t_launch_rail_clearance

        v_wind3 = Float(0)

        # Velocity of rocket relative to air (points where rocket is moving through air)
        va1 = v1 - v_wind1
        va2 = v2 - v_wind2
        va3 = v3 - v_wind3

        eps = Float("1e-6")

        # Body +z axis expressed in world frame (3rd column of DCM)
        b1 = 2 * (qx*qz + qw*qy)
        b2 = 2 * (qy*qz - qw*qx)
        b3 = 1 - 2 * (qx*qx + qy*qy)

        # Parallel component and perpendicular magnitude (stable)
        Vpar = b1*va1 + b2*va2 + b3*va3
        V2   = va1**2 + va2**2 + va3**2
        Vperp2 = Max(Float(0), V2 - Vpar**2)
        Vperp  = sqrt(Vperp2)

        # AoA magnitude (0 when aligned, safe at small speeds)
        AoA = H * atan2(Vperp, Vpar + eps)
        return AoA
        

    def quat_to_euler_xyz(self, q: np.ndarray, degrees=False, eps=1e-9) -> tuple:
        """
        Convert quaternion [w, x, y, z] to Euler angles (theta, phi, psi)
        using the intrinsic XYZ convention:
            theta: rotation about x (pitch)
            phi:   rotation about y (yaw)
            psi:   rotation about z (roll)
        Such that: R = Rz(psi) @ Ry(phi) @ Rx(theta) [INTRISIC]

        Args:
            q (array-like): Quaternion [w, x, y, z].
            degrees (bool): If True, return angles in degrees. (default: radians)
            eps (float):    Small epsilon to handle numerical edge cases.

        Returns:
            (theta, phi, psi): tuple of floats
        """
        # normalize to be safe
        n = np.linalg.norm(q)
        if n < eps:
            raise ValueError("Zero-norm quaternion")
        w = q[0] / n
        x = q[1] / n
        y = q[2] / n
        z = q[3] / n

        # Rotation matrix from quaternion (world<-body)
        # R[i,j] = row i, column j
        xx, yy, zz = x*x, y*y, z*z
        wx, wy, wz = w*x, w*y, w*z
        xy, xz, yz = x*y, x*z, y*z

        R = np.array([
            [1 - 2*(yy + zz),   2*(xy - wz),       2*(xz + wy)],
            [2*(xy + wz),       1 - 2*(xx + zz),   2*(yz - wx)],
            [2*(xz - wy),       2*(yz + wx),       1 - 2*(xx + yy)]
        ])

        # Extract for intrinsic XYZ (q = qz(psi) ⊗ qy(phi) ⊗ qx(theta))
        # From R = Rz(psi) Ry(phi) Rx(theta):
        #   phi   = asin(-R[2,0])
        #   theta = atan2(R[2,1], R[2,2])
        #   psi   = atan2(R[1,0], R[0,0])
        #
        # Handle numerical drift by clamping asin argument.
        s = -R[2, 0]
        s = np.clip(s, -1.0, 1.0)
        phi   = np.arcsin(s)
        theta = np.arctan2(R[2, 1], R[2, 2])

        # If cos(phi) ~ 0 (gimbal lock), fall back to a stable computation for psi
        if abs(np.cos(phi)) < eps:
            # At gimbal lock, theta and psi are coupled; choose a consistent psi:
            # Use elements that remain well-defined:
            # when cos(phi) ~ 0, use psi from atan2(-R[0,1], R[1,1])
            psi = np.arctan2(-R[0, 1], R[1, 1])
        else:
            psi = np.arctan2(R[1, 0], R[0, 0])

        if degrees:
            return np.degrees(theta), np.degrees(phi), np.degrees(psi)
        return theta, phi, psi


    def euler_to_quat_xyz(self, theta, phi, psi, degrees=False) -> np.ndarray:
        """
        Convert Euler angles to a quaternion using intrinsic XYZ:
            - theta: rotation about x (pitch)
            - phi:   rotation about y (yaw)
            - psi:   rotation about z (roll)
        Convention: R = Rz(psi) @ Ry(phi) @ Rx(theta)
        Quaternion is returned as [w, x, y, z].

        Args:
            theta, phi, psi : floats (radians by default; set degrees=True if in deg)
            degrees         : if True, inputs are in degrees

        Returns:
            np.ndarray shape (4,) -> [w, x, y, z]
        """
        if degrees:
            theta, phi, psi = np.radians([theta, phi, psi])

        # half-angles
        cth, sth = np.cos(theta/2.0), np.sin(theta/2.0)
        cph, sph = np.cos(phi/2.0),   np.sin(phi/2.0)
        cps, sps = np.cos(psi/2.0),   np.sin(psi/2.0)

        # intrinsic XYZ closed form (q = qz * qy * qx), scalar-first
        qw =  cph*cps*cth + sph*sps*sth
        qx = -sph*sps*cth + sth*cph*cps
        qy =  sph*cps*cth + sps*sth*cph
        qz = -sph*sth*cps + sps*cph*cth

        q = np.array([qw, qx, qy, qz], dtype=float)
        # normalize to guard against numerical drift
        q /= np.linalg.norm(q)
        return q


    def R_BW_from_q(self, qw, qx, qy, qz) -> Matrix:
        """Convert a quaternion to a rotation matrix. World to body frame.

        Args:
            qw (float): The scalar component of the quaternion.
            qx (float): The x component of the quaternion.
            qy (float): The y component of the quaternion.
            qz (float): The z component of the quaternion.

        Returns:
            Matrix: The rotation matrix from world to body frame.
        """
        s = (qw**2 + qx**2 + qy**2 + qz**2)**-Rational(1,2) # Normalizing factor
        qw, qx, qy, qz = qw*s, qx*s, qy*s, qz*s # Normalized quaternion components. Since quaternions are unit vectors

        xx,yy,zz = qx*qx, qy*qy, qz*qz
        wx,wy,wz = qw*qx, qw*qy, qw*qz
        xy,xz,yz = qx*qy, qx*qz, qy*qz
        return Matrix([
            [1-2*(yy+zz),   2*(xy+wz),   2*(xz-wy)],
            [2*(xy-wz),     1-2*(xx+zz), 2*(yz+wx)],
            [2*(xz+wy),     2*(yz-wx),   1-2*(xx+yy)]
        ])

    def define_eom(self):
        """Get the equations of motion for the rocket. Sets self.f.

        ## Assumptions:
        - Rocket body axis is aligned with z-axis
        - No centrifugal forces are considered to simplify AoA and beta calculations
        - Coefficient of lift is approximated as 2*pi*AoA (thin airfoil theory)
        - Thrust acts only in the z direction of the body frame
        - No wind or atmospheric disturbances are considered
        - Density of air is constant at 1.225 kg/m^3

        ## Notes:
        - The state vector is [w1, w2, w3, v1, v2, v3, qw, qx, qy, qz] where w is angular velocity, v is linear velocity, and q is the quaternion.
        - The input vector is [delta1] where delta1 is the aileron angle
        - Thrust, mass, and inertia are time-varying based on the motor burn state
        - Drag force Fd is modeled as a quadratic function of velocity magnitude
        - Lift force Fl is modeled using thin airfoil theory, proportional to angle of attack (AoA)
        - Corrective moment coefficient C is modeled as a function of velocity magnitude, normal force coefficient Cn, stability margin SM, and rocket diameter
        - Normal force coefficient derivative Cnalpha is modeled as Cn * (AoA / (AoA^2 + aoa_eps^2)) to ensure smoothness at AoA = 0
        - Stability margin SM is modeled as a polynomial function of AoA
        - Small terms are added to avoid division by zero in velocity magnitude and AoA calculations (denoted as eps and aoa_eps)
        - All polynomial equations are determined from experimental OpenRocket data and curve fitting using Google Sheets
        - Piecewise functions are used to bound certain variables (e.g., AoA, Cnalpha, C) to ensure numerical stability and physical realism

        """
        if self.t_sym is None or self.state_vars is None or self.params is None:
            self.set_symbols()
        w1, w2, w3, v1, v2, v3, qw, qx, qy, qz = self.state_vars
        I1, I2, I3, T1, T2, T3, mass, rho, d, g, CG, delta, C_d, Cnalpha_fin, Cnalpha_rocket, Cr, Ct, s, N_fins, v_wind1, v_wind2, CP = self.params
        # I1, I2, I3, T1, T2, T3, mass, rho, d, g, CG, delta, C_d, Cnalpha_fin, Cnalpha_rocket, Cr, Ct, s, N = self.params
        
        v = Matrix([v1, v2, v3]) # Velocity vector
        
        ## Quaternion kinematics ##
        S = Matrix([[0, -w3, w2],
                    [w3, 0, -w1],
                    [-w2, w1, 0]])
        q_vec = Matrix([qw, qx, qy, qz])
        Omega = Matrix([
            [0, -w1, -w2, -w3],
            [w1, 0, w3, -w2],
            [w2, -w3, 0, w1],
            [w3, w2, -w1, 0]
        ])
        
        # -------------------------------------------- #

        F = self.get_forces()
        M = self.get_moments()
        M1, M2, M3 = M[0], M[1], M[2]
        
        ## Equations of motion ##
        I1dot = Piecewise(
            (Float(-(self.I_0 - self.I_f) / self.t_motor_burnout), self.t_sym <= Float(self.t_motor_burnout)),
            (Float(0.0), True),
        )
        I2dot = I1dot
        I3dot = Piecewise(
            (Float(-(self.I_3 - self.I_3_f) / self.t_motor_burnout), self.t_sym <= Float(self.t_motor_burnout)),
            (Float(0.0), True),
        )

        # Euler rigid-body equations with time-varying principal inertias:
        # I * wdot + I_dot * w + w x (I w) = M
        w1dot = ((I2 - I3) * w2 * w3 + M1 - I1dot * w1) / I1
        w2dot = ((I3 - I1) * w3 * w1 + M2 - I2dot * w2) / I2
        w3dot = ((I1 - I2) * w1 * w2 + M3 - I3dot * w3) / I3
        vdot = F/mass - S * v
        qdot = (Omega * q_vec) * Float(1/2)

        f = Matrix([
            [w1dot],
            [w2dot],
            [w3dot],
            [vdot[0]],
            [vdot[1]],
            [vdot[2]],
            [qdot[0]],
            [qdot[1]],
            [qdot[2]],
            [qdot[3]]
        ])

        self.f = f



    def set_f(self, t: float, xhat: np.ndarray):
        """Substitute a numeric state and time into the symbolic EOM.
        Stores the result in self.f_subs_full. Called by Controls.set_f().

        Args:
            t (float): Current time in seconds.
            xhat (np.ndarray): Current state vector [w1,w2,w3,v1,v2,v3,qw,qx,qy,qz].
        """
        if self.f is None:
            self.define_eom()

        state_subs = {sym: float(val) for sym, val in zip(self.state_vars, xhat)}

        param_vals = self._gather_param_values(t, xhat)
        param_subs = {sym: float(val) for sym, val in zip(self.params, param_vals)
                      if sym is not None}

        time_subs = {self.t_sym: float(t)}

        self.f_subs_full = self.f.subs({**state_subs, **param_subs, **time_subs})


    @staticmethod
    def _compute_body_airspeed(x: np.ndarray, v_wind_x: float, v_wind_y: float) -> np.ndarray:
        """Return air-relative velocity [va1, va2, va3] in the body frame.

        Rotates the horizontal wind vector (world frame) into the body frame using
        the quaternion from state x, then subtracts from body-frame inertial velocity.

        Args:
            x: state vector [w1,w2,w3,v1,v2,v3,qw,qx,qy,qz]
            v_wind_x: world-frame wind x-component (m/s)
            v_wind_y: world-frame wind y-component (m/s)

        Returns:
            np.ndarray shape (3,): [va1, va2, va3]
        """
        qw_n = float(x[6]); qx_n = float(x[7])
        qy_n = float(x[8]); qz_n = float(x[9])
        q_norm = np.sqrt(qw_n**2 + qx_n**2 + qy_n**2 + qz_n**2)
        if q_norm > 1e-9:
            qw_n /= q_norm; qx_n /= q_norm
            qy_n /= q_norm; qz_n /= q_norm
        xx, yy, zz = qx_n*qx_n, qy_n*qy_n, qz_n*qz_n
        wx, wy, wz = qw_n*qx_n, qw_n*qy_n, qw_n*qz_n
        xy, xz, yz = qx_n*qy_n, qx_n*qz_n, qy_n*qz_n
        R_BW = np.array([
            [1-2*(yy+zz), 2*(xy+wz),  2*(xz-wy)],
            [2*(xy-wz),   1-2*(xx+zz), 2*(yz+wx)],
            [2*(xz+wy),   2*(yz-wx),  1-2*(xx+yy)]
        ])
        v_wind_body = R_BW @ np.array([v_wind_x, v_wind_y, 0.0])
        return np.array([
            float(x[3]) - v_wind_body[0],
            float(x[4]) - v_wind_body[1],
            float(x[5]) - v_wind_body[2],
        ])

    def fin_roll_moment_numeric(self, t: float, x: np.ndarray,
                                altitude: float = None) -> float:
        """Return the main-fin roll moment used by the dynamics EOM.

        This mirrors the roll part of MomentsForces.set_moments():
        M3_fin = H_rail * (M_f_roll - M_d_roll * w3), where M_f_roll is
        the fin-cant forcing moment and M_d_roll is the fin roll damping
        coefficient.
        """
        if t < self.t_launch_rail_clearance:
            return 0.0

        param_vals = self._gather_param_values(t, x, altitude)
        (
            _I1, _I2, _I3,
            _T1, _T2, _T3,
            _mass, rho, d, _g, _CG,
            delta, _C_d, Cnalpha_fin, _Cnalpha_rocket,
            Cr, Ct, s, N_fins,
            v_wind_x, v_wind_y,
            _CP,
        ) = param_vals

        va = self._compute_body_airspeed(x, v_wind_x, v_wind_y)
        v_air_mag = float(np.sqrt(np.dot(va, va) + 1e-18))
        w3 = float(x[2])

        gamma = Ct / Cr
        r_t = d / 2.0
        tau = (s + r_t) / r_t
        area = np.pi * (d / 2.0) ** 2

        y_ma = (s / 3.0) * (1.0 + 2.0 * gamma) / (1.0 + gamma)
        asin_arg = (tau**2 - 1.0) / (tau**2 + 1.0)
        asin_term = np.arcsin(asin_arg)
        K_f = (1.0 / np.pi**2) * (
            (np.pi**2 / 4.0) * ((tau + 1.0) ** 2 / tau**2)
            + (np.pi * (tau**2 + 1.0) ** 2 / (tau**2 * (tau - 1.0) ** 2)) * asin_term
            - (2.0 * np.pi * (tau + 1.0)) / (tau * (tau - 1.0))
            + ((tau**2 + 1.0) ** 2 / (tau**2 * (tau - 1.0) ** 2)) * asin_term**2
            - (4.0 * (tau + 1.0) / (tau * (tau - 1.0))) * asin_term
            + (8.0 / (tau - 1.0) ** 2) * np.log((tau**2 + 1.0) / (2.0 * tau))
        )
        M_f_roll = K_f * (0.5 * rho * v_air_mag**2) * (
            N_fins * (y_ma + r_t) * Cnalpha_fin * delta * area
        )

        trap_integral = s / 12.0 * (
            (Cr + 3.0 * Ct) * s**2
            + 4.0 * (Cr + 2.0 * Ct) * s * r_t
            + 6.0 * (Cr + Ct) * r_t**2
        )
        C_ldw = 2.0 * N_fins * Cnalpha_fin / (area * d**2) * np.cos(delta) * trap_integral
        K_d = 1.0 + (
            (tau - gamma) / tau - (1.0 - gamma) / (tau - 1.0) * np.log(tau)
        ) / (
            (tau + 1.0) * (tau - gamma) / 2.0
            - (1.0 - gamma) * (tau**3 - 1.0) / (3.0 * (tau - 1.0))
        )
        M_d_roll = K_d * (0.5 * rho * v_air_mag**2) * area * d * C_ldw * (
            d / (2.0 * v_air_mag)
        )

        return float(M_f_roll - M_d_roll * w3)

    def _gather_param_values(self, t: float, x: np.ndarray = None,
                          altitude: float = None) -> list:
        thrust      = self.get_thrust(t)
        mass_rocket = self.get_mass(t)
        inertia     = self.get_inertia(t)
        x_CG        = self.get_CG(t)

        # Atmospheric values — use RocketPy env if available
        if self._env_density_func is not None and altitude is not None:
            rho_val        = float(self._env_density_func(altitude))
            g_val          = float(self._env_gravity_func(altitude))
            v_wind_x       = float(self._env_wind_x_func(altitude))
            v_wind_y       = float(self._env_wind_y_func(altitude))
            # Use altitude-dependent temperature for speed of sound
            if self._env_temperature_func is not None:
                T = float(self._env_temperature_func(altitude))  # Kelvin
                speed_of_sound = float(np.sqrt(1.4 * 287.05 * T))
            else:
                speed_of_sound = 343.0
        else:
            rho_val        = float(self.rho)
            g_val          = float(self.g)
            v_wind_x       = float(self.v_wind[0])
            v_wind_y       = float(self.v_wind[1])
            speed_of_sound = 343.0

        # Air-relative velocity in body frame — used for all aero lookups
        if x is not None:
            va = self._compute_body_airspeed(x, v_wind_x, v_wind_y)
            v_air_mag = float(np.linalg.norm(va))
            AoA_num   = float(np.degrees(
                np.arctan2(np.sqrt(va[0]**2 + va[1]**2), va[2] + 1e-9)
            ))
            AoA_num = np.clip(AoA_num, -15.0, 15.0)  # match Piecewise bound in set_moments
        else:
            v_air_mag = 0.0
            AoA_num   = 0.0

        mach = v_air_mag / speed_of_sound

        # Drag coefficient
        if self._drag_func is not None and x is not None:
            try:
                C_d_val = float(self._drag_func(mach, t))
            except TypeError:
                C_d_val = float(self._drag_func(mach))
        elif self.C_d is not None:
            C_d_val = float(self.C_d)
        else:
            raise RuntimeError(
                "No drag model set. Call controls.set_drag_func() or provide C_d in setRocketParams()."
            )

        # Cnalpha_fin
        if self._cnalpha_fin_func is not None and x is not None:
            Cnalpha_fin_val = float(self._cnalpha_fin_func(mach))
        else:
            Cnalpha_fin_val = float(self.Cnalpha_fin)

        # Cnalpha_rocket
        if self._cnalpha_rocket_func is not None and x is not None:
            Cnalpha_rocket_val = float(self._cnalpha_rocket_func(mach))
        else:
            Cnalpha_rocket_val = float(self.Cnalpha_rocket)

        # CP location — 2D (AoA + Mach) if func registered, else AoA-only fallback
        if self._cp_2d_func is not None:
            CP_val = float(self._cp_2d_func(AoA_num, mach))
        elif self.CP_func is not None:
            CP_val = float(self.CP_func(AoA_num))
        else:
            raise RuntimeError("No CP function set. Call controls.set_cp_func() before running.")

        return [
            float(inertia[0]),      # I1
            float(inertia[1]),      # I2
            float(inertia[2]),      # I3
            float(thrust[0]),
            float(thrust[1]),
            float(thrust[2]),
            float(mass_rocket),
            rho_val,                # altitude-dependent if env set
            float(self.d),
            g_val,                  # altitude-dependent if env set
            float(x_CG),
            float(np.deg2rad(self.delta)),
            C_d_val,                # Mach-dependent if drag_func set
            Cnalpha_fin_val,        # Mach-dependent if cnalpha_fin_func set
            Cnalpha_rocket_val,     # Mach-dependent if cnalpha_rocket_func set
            float(self.Cr),
            float(self.Ct),
            float(self.s),
            float(self.N),
            v_wind_x,               # altitude-dependent if env set
            v_wind_y,               # altitude-dependent if env set
            CP_val                  # AoA+Mach-dependent if cp_2d_func set
        ]

    def set_drag_func(self, drag_func: Callable):
        """Register a velocity-dependent drag function to replace constant C_d.
        
        Args:
            drag_func (Callable): Function with signature (mach: float) -> float
                                or (mach: float, t: float) -> float returning the
                                drag coefficient at a given Mach number. The 2-arg
                                version lets callers switch power-on/power-off drag
                                based on time without changing the rest of the model.
        """
        self._drag_func = drag_func

    def set_cp_func(self, cp_func):
        """Register a 2D CP function for numeric propagation (Track 2).
        Replaces the AoA-only CP_func at runtime with one that also
        accounts for Mach number compressibility effects.

        Args:
            cp_func (Callable): Function with signature
                                (AoA_deg: float, mach: float) -> float
                                returning CP location in meters from nose.
        """
        self._cp_2d_func = cp_func

    def set_cnalpha_rocket_func(self, func: Callable):
        """Register a Mach-dependent Cnalpha function for the rocket body.

            Args:
            func (Callable): Function with signature (mach: float) -> float
                            returning Cnalpha_rocket at a given Mach number.
        """
        self._cnalpha_rocket_func = func

    def set_cnalpha_fin_func(self, func: Callable):
        """Register a Mach-dependent Cnalpha function for the fins.

        Args:
            func (Callable): Function with signature (mach: float) -> float
                            returning Cnalpha_fin at a given Mach number.
        """
        self._cnalpha_fin_func = func

    def set_env_from_rocketpy(self, env):
        """Register altitude-dependent atmospheric functions from a
        RocketPy Environment object. Once set, rho, g, and v_wind are
        computed from current altitude at every timestep instead of
        using the constants from setEnvParams().

        Args:
            env: A RocketPy Environment object that has already been
                configured with an atmospheric model.
        """
        self._env_density_func     = env.density            # callable: altitude (m) -> kg/m³
        self._env_gravity_func     = env.gravity            # callable: altitude (m) -> m/s²
        self._env_wind_x_func      = env.wind_velocity_x   # callable: altitude (m) -> m/s
        self._env_wind_y_func      = env.wind_velocity_y   # callable: altitude (m) -> m/s
        self._env_temperature_func = env.temperature        # callable: altitude (m) -> K
      
# For testing
def main():
    print("Hello there")

if __name__ == "__main__":
    main()
