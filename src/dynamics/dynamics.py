from sympy import *
import numpy as np
import pandas as pd
from typing import Callable

from momentsforces import MomentsForces
from thrust import Thrust

class Dynamics(MomentsForces, Thrust):
    def __init__(self, rocket_name : str):
        """Initialize the Dynamics class. Rocket body axis is aligned with z-axis.

        Args:
            t_estimated_apogee (float): Estimated time until apogee in seconds.
            dt (float): Time step for simulation in seconds.
            x0 (np.ndarray): Initial state vector.
            [Refer to super for other - Jed]
        """

        super.__init__()

        self.f : Matrix = None
        self.state_vars : list = None
        self.params : list = None
        self.f_subs_params : Matrix = None
        self.f_subs_full : Matrix = None
        self.dt : float = None
        self.x0 : np.ndarray = None
        self.t0 : float = 0.0
        self.t_sym : Symbol = None

        ## Uninitialized parameters ##
        
        # Rocket parameters
        self.I_0 : float = None # Initial moment of inertia in kg·m²
        self.I_f : float = None # Final moment of inertia in kg·m²
        self.I_3 : float = None # Rotational moment of inertia about z-axis in kg·m²
        self.x_CG_0 : float = None # Initial center of gravity location in meters
        self.x_CG_f : float = None # Final center of gravity location in meters
        self.m_0 : float = None # Initial rocket mass in kg
        self.m_f : float = None # Final rocket mass in kg
        self.m_p : float = None # Propellant mass in kg
        self.d : float = None # Rocket body diameter in meters
        self.L_ne : float = None # Length from nose to nozzle in meters
        self.C_d : float = None # Drag coefficient
        self.Cnalpha_rocket : float = None # Rocket normal force coefficient derivative
        self.t_motor_burnout : float = None # Time to motor burnout in seconds
        self.t_launch_rail_clearance : float = None # Time to launch rail clearance in seconds
        self.t_estimated_apogee : float = None # Time to apogee in esconds
        self.CP_func : Callable[[Expr], Expr] = None # Center of pressure location as a function of angle of attack in degrees
        
        # Fin parameters
        self.N : float = None # Number of fins
        self.Cr : float = None # Root chord in meters
        self.Ct : float = None # Tip chord in meters
        self.s : float = None # Span in meters
        self.delta : float = None # Fin cant angle in degrees
        
        # Rocket name (used for saving simulation results from Simluation() object to designated path)
        self.rocket_name = rocket_name
        
        # State space linearization matrices
        self.A_sym : Matrix = None # Symbolic state matrix
        self.A : np.ndarray = None # State matrix
        
        ## Helpers [Refer to super for other]##
        self._f_numeric = None  # Cached lambdified EOM
        self._A_numeric = None  # Cached lambdified Jacobian of f wrt state


    def set_symbols(self):
        """Set the symbolic variables for the dynamics equations.
        """
        w1, w2, w3, v1, v2 = symbols('w_1 w_2 w_3 v_1 v_2', real = True) # Angular and linear velocities
        v3 = symbols('v_3', real = True, positive = True) # Longitudinal velocity, assumed positive during flight
        qw, qx, qy, qz = symbols('q_w q_x q_y q_z', real = True) # Quaternion components
        I1, I2, I3 = symbols('I_1 I_2 I_3', real = True, positive = True) # Moments of inertia
        T1, T2, T3 = symbols('T_1 T_2 T_3', real = True, positive = True) # Thrusts
        mass, rho, d, g, CG = symbols('m rho d g CG', real = True, positive = True) # Mass, air density, diameter, gravity, center of gravity
        delta = symbols('delta', real = True) # Fin cant angle
        C_d = symbols('C_d', real = True, positive = True) # Drag coefficient
        Cnalpha_fin, Cnalpha_rocket = symbols('C_n_alpha_fin C_n_alpha_rocket', real = True, positive = True) # Fin and rocket normal force coefficient derivatives
        Cr, Ct, s = symbols('Cr Ct s', real = True, positive = True) # Fin root chord, tip chord, span
        N = symbols('N', real = True, positive = True) # Number of fins
        t_sym = symbols('t', real = True, positive = True) # Time symbol for Heaviside function
        v_wind1, v_wind2 = symbols('v_wind_1 v_wind_2', real = True) # Wind velocity components

        self.state_vars = [w1, w2, w3, v1, v2, v3, qw, qx, qy, qz]
        self.params = [I1, I2, I3, T1, T2, T3, mass, rho, d, g, CG, delta, C_d, Cnalpha_fin, Cnalpha_rocket, Cr, Ct, s, N, v_wind1, v_wind2]
        # self.params = [I1, I2, I3, T1, T2, T3, mass, rho, d, g, CG, delta, C_d, Cnalpha_fin, Cnalpha_rocket, Cr, Ct, s, N]
        self.t_sym = t_sym # Time when rocket leaves the launch rail



        

    def setRocketParams(self, I_0: float, I_f: float, I_3: float,
                        x_CG_0: float, x_CG_f: float,
                        m_0: float, m_f: float, m_p: float,
                        d: float, L_ne: float, C_d: float, Cnalpha_rocket: float,
                        t_launch_rail_clearance: float, t_motor_burnout: float, t_estimated_apogee: float,
                        CP_func: Callable[[Expr], Expr]):
        """Set the rocket parameters.

        Args:
            I_0 (float): Initial moment of inertia in kg·m².
            I_f (float): Final moment of inertia in kg·m².
            I_3 (float): Moment of inertia about the z-axis in kg·m².
            x_CG_0 (float): Initial center of gravity location in meters.
            x_CG_f (float): Final center of gravity location in meters.
            m_0 (float): Initial mass in kg.
            m_f (float): Final mass in kg.
            m_p (float): Propellant mass in kg.
            d (float): Rocket diameter in meters.
            L_ne (float): Length from nose to engine exit in meters.
            Cnalpha_rocket (float): Rocket normal force coefficient derivative.
            t_motor_burnout (float): Time until motor burnout in seconds.
            t_estimated_apogee (float, optional): Estimated time until apogee in seconds.
            t_launch_rail_clearance (float): Time until launch rail clearance in seconds.
            CP_func (Callable[[Expr], Expr]): User defined function of center of pressure location as a function of angle of attack (deg).\
                Takes Expr 'AoA_deg' as parameter. Returns function fit equation of center of pressure location vs AoA (e.g. using Google Sheets).
        """
        self.I_0 = I_0
        self.I_f = I_f
        self.I_3 = I_3
        self.x_CG_0 = x_CG_0
        self.x_CG_f = x_CG_f
        self.m_0 = m_0
        self.m_f = m_f
        self.m_p = m_p
        self.d = d
        self.L_ne = L_ne
        self.C_d = C_d
        self.Cnalpha_rocket = Cnalpha_rocket
        self.t_launch_rail_clearance = t_launch_rail_clearance
        self.t_motor_burnout = t_motor_burnout
        self.t_estimated_apogee = t_estimated_apogee
        self.CP_func = CP_func


    def setFinParams(self, N: int, Cr: float, Ct: float, s: float, Cnalpha_fin: float, delta: float):
        """Set the fin parameters.

        Args:
            N (int): Number of fins.
            Cr (float): Fin root chord in meters.
            Ct (float): Fin tip chord in meters.
            s (float): Fin span in meters.
            Cnalpha_fin (float): Fin normal force coefficient derivative.
            delta (float): Fin cant angle in degrees.
        """
        self.N = N
        self.Cr = Cr
        self.Ct = Ct
        self.s = s
        self.Cnalpha_fin = Cnalpha_fin
        self.delta = delta

    
    
    def setEnvParams(self, v_wind: list, rho: float, g: float):
        """Set the environmental parameters.

        Args:
            v_wind (list): Wind velocity vector [x, y] in m/s.
            rho (float): Air density in kg/m^3.
            g (float): Gravitational acceleration in m/s^2.
        """
        self.v_wind = v_wind
        self.rho = rho
        self.g = g
        
        
    def setSimParams(self, dt: float, x0: np.ndarray):
        """Set the simulation parameters. Appends initial state to states list and initial time to ts list.

        Args:
            dt (float): Time step for simulation in seconds.
            x0 (np.ndarray): Initial state vector.
        """
        self.dt = dt
        self.x0 = np.array(x0, dtype=float)
        
        
    def checkParamsSet(self):
        """Check if all necessary parameters have been set.

        Raises:
            ValueError: If any parameter is not set.
        """
        required_params = [
            'I_0', 'I_f', 'I_3',
            'x_CG_0', 'x_CG_f',
            'm_0', 'm_f', 'm_p',
            'd', 'L_ne',
            't_launch_rail_clearance', 't_motor_burnout', 't_estimated_apogee',
            'thrust_times', 'thrust_forces',
            'v_wind', 'rho', 'g',
            'N', 'Cr', 'Ct', 's', 'Cnalpha_fin',
            'CP_func'
        ]
        for param in required_params:
            if not hasattr(self, param):
                raise ValueError(f"Parameter '{param}' is not set. Please set all necessary parameters before proceeding.")
    
    
    def get_mass(self, t: float) -> float:
        """Get the mass of the rocket at time t.

        Args:
            t (float): The time in seconds.

        Returns:
            float: The mass of the rocket at time t in kg.
        """
        mass_rocket = self.m_0 - self.m_p / self.t_motor_burnout * t if t <= self.t_motor_burnout else self.m_f
        return mass_rocket


    def get_inertia(self, t: float) -> list:
        """Get the moment of inertia of the rocket at time t.

        Args:
            t (float): The time in seconds.

        Returns:
            float: The moment of inertia of the rocket at time t in kg·m².
        """
        I_long = self.I_0 - (self.I_0 - self.I_f) / self.t_motor_burnout * t if t <= self.t_motor_burnout else self.I_f
        I = [I_long, I_long, self.I_3]

        return I

    
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
        Such that: R = Rz(psi) @ Ry(phi) @ Rx(theta)

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


    def euler_to_quat_xyz(self, theta, phi, psi, degrees=False):
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
        qw, qx, qy, qz = qw*s, qx*s, qy*s, qz*s # Normalized quaternion components

        xx,yy,zz = qx*qx, qy*qy, qz*qz
        wx,wy,wz = qw*qx, qw*qy, qw*qz
        xy,xz,yz = qx*qy, qx*qz, qy*qz
        return Matrix([
            [1-2*(yy+zz),   2*(xy+wz),   2*(xz-wy)],
            [2*(xy-wz),     1-2*(xx+zz), 2*(yz+wx)],
            [2*(xz+wy),     2*(yz-wx),   1-2*(xx+yy)]
        ])

    
    def set_forces(self) -> Matrix:
        """Get the forces for the rocket. Sets self.F.
        """
        w1, w2, w3, v1, v2, v3, qw, qx, qy, qz = self.state_vars
        I1, I2, I3, T1, T2, T3, mass, rho, d, g, CG, delta, C_d, Cnalpha_fin, Cnalpha_rocket, Cr, Ct, s, N, v_wind1, v_wind2 = self.params
        # I1, I2, I3, T1, T2, T3, mass, rho, d, g, CG, delta, C_d, Cnalpha_fin, Cnalpha_rocket, Cr, Ct, s, N = self.params        
        t_sym = self.t_sym
    
        H = Heaviside(t_sym - Float(self.t_launch_rail_clearance), 0)  # 0 if t < t_launch_rail_clearance, 1 if t >= t_launch_rail_clearance

        epsAoA = Float(1e-9)  # Small term to avoid division by zero in AoA calculation
        AoA = atan2(sqrt(v1**2 + v2**2), v3 + epsAoA) # Angle of attack
        AoA = Piecewise(
            (0,   Abs(AoA) <= epsAoA),                # inside deadband
            (Min(Abs(AoA), 15 * pi / 180) * (AoA/Abs(AoA)), True)  # ±15°
        )
        # v_wind = (v_wind1, v_wind2)
        # AoA = self.get_AoA(v_wind, self.state_vars)

        eps = Float(1e-9)  # Small term to avoid division by zero
        v = Matrix([v1, v2, v3]) # Velocity vector
        v_mag = sqrt(v1**2 + v2**2 + v3**2 + eps**2) # Magnitude of velocity with small term to avoid division by zero
        vhat = v / v_mag  # Unit vector in direction of velocity

        ## Rocket reference area ##
        A = pi * (d/2)**2 # m^2

        ## Thrust ##
        Ft : Matrix = Matrix([T1, T2, T3])  # Thrust vector, T1 and T2 are assumed 0

        ## Gravity ##
        Fg_world = Matrix([0.0, 0.0, -mass * g])
        R_world_to_body = self.R_BW_from_q(qw, qx, qy, qz)  # Rotation matrix from world to body frame
        Fg : Matrix = R_world_to_body * Fg_world  # Transform gravitational force to body frame

        ## Drag Force ##
        D = C_d * 1/2 * rho * v_mag**2 * A # Drag force using constant drag coefficient
        Fd : Matrix = -D * vhat # Drag force vector

        ## Lift Force ## beta gives direction of the angle of attack to componentize lift, AoA is the direct angle of attack
        eps_beta = Float(1e-9)
        nan_guard = sqrt(v1**2 + v2**2 + eps_beta**2)
        beta = 2 * atan2(v2, nan_guard + v1) # Equivalent to atan2(v2, v1) but avoids NaN at (0,0)
        L = H * 1/2 * rho * v_mag**2 * (2 * pi * AoA) * A # Lift force approximation
        nL = Matrix([
            -cos(AoA) * cos(beta),
            -cos(AoA) * sin(beta),
            sin(AoA)
        ]) # Lift direction unit vector
        Fl : Matrix = L * nL # Lift force vector

        ## Total Forces ##
        F = Ft + Fd + Fl + Fg # Thrust + Drag + Lift + Gravity
        
        self.F = F
    
    
    def get_forces(self):
        """Get the forces for the rocket.
        Returns:
            Matrix: The forces vector.
        """
        self.set_forces()
        return self.F


    def set_moments(self) -> Matrix:
        """Get the moments for the rocket.
        Returns:
            Matrix: The moments vector.
        """
        w1, w2, w3, v1, v2, v3, qw, qx, qy, qz = self.state_vars
        I1, I2, I3, T1, T2, T3, mass, rho, d, g, CG, delta, C_d, Cnalpha_fin, Cnalpha_rocket, Cr, Ct, s, N, v_wind1, v_wind2 = self.params
        # I1, I2, I3, T1, T2, T3, mass, rho, d, g, CG, delta, C_d, Cnalpha_fin, Cnalpha_rocket, Cr, Ct, s, N = self.params
        t_sym = self.t_sym
    
        H = Heaviside(t_sym - Float(self.t_launch_rail_clearance), 0)  # 0 if t < t_launch_rail_clearance, 1 if t >= t_launch_rail_clearance

        epsAoA = Float(1e-9)  # Small term to avoid division by zero in AoA calculation
        AoA = atan2(sqrt(v1**2 + v2**2), v3 + epsAoA) # Angle of attack
        AoA = Piecewise(
            (0,   Abs(AoA) <= epsAoA),                # inside deadband
            (Min(Abs(AoA), 15 * pi / 180) * (AoA/Abs(AoA)), True)  # ±15°
        )
        # v_wind = (v_wind1, v_wind2)
        # AoA = self.get_AoA(v_wind, self.state_vars)

        eps = Float(1e-9)  # Small term to avoid division by zero
        v_mag = sqrt(v1**2 + v2**2 + v3**2 + eps**2) # Magnitude of velocity with small term to avoid division by zero

        ## Rocket reference area ##
        A = pi * (d/2)**2 # m^2
        
        ## Stability Margin ##
        AoA_deg = AoA * 180 / pi # Convert AoA to degrees for polynomial fit

        ## Corrective moment coefficient ##
        # CG is where rotation is about and CP is where force is applied
            # SM = (CP - CG) / d
        # Equations from ApogeeRockets
        C_raw = v_mag**2 * A * Cnalpha_rocket * AoA * (self.CP_func(AoA_deg) - CG) * rho / 2
        
        eps_beta = Float(1e-9)
        nan_guard = sqrt(v1**2 + v2**2 + eps_beta**2)
        beta = 2 * atan2(v2, nan_guard + v1) # Equivalent to atan2(v2, v1) but avoids NaN at (0,0)
        
        M_f_pitch = C_raw * sin(beta) # Pitch forcing moment
        M_f_yaw = -C_raw * cos(beta) # Yaw forcing moment

        ## Propulsive Damping Moment Coefficient (Cdp) ##
        mdot = self.m_p / self.t_motor_burnout # kg/s, average mass flow rate during motor burn
        Cdp = mdot * (self.L_ne - CG)**2 # kg*m^2/s

        ## Aerodynamic Damping Moment Coefficient (Cda) ##
        Cda = (rho * v_mag * A / 2) * (Cnalpha_rocket * AoA * (self.CP_func(AoA_deg) - CG)**2)

        ## Damping Moment Coefficient (Cdm) ##
        M_d_pitch = M_d_yaw = Cdp + Cda

        ## Moment due to fin cant angle ##
        gamma = Ct/Cr
        r_t = d/2
        tau = (s + r_t) / r_t
        
        # Roll forcing moment
        # Equations from RocketPy documentation, Barrowman
        Y_MA = (s/3) * (1 + 2*gamma)/(1+gamma) # Spanwise location of fin aerodynamic center
        K_f = (1/pi**2) * \
            ((pi**2/4)*((tau+1)**2/tau**2) \
            + (pi*(tau**2+1)**2/(tau**2*(tau-1)**2))*asin((tau**2-1)/(tau**2+1)) \
            - (2*pi*(tau+1))/(tau*(tau-1)) \
            + ((tau**2+1)**2/(tau**2*(tau-1)**2))*asin((tau**2-1)/(tau**2+1))**2 \
            - (4*(tau+1)/(tau*(tau-1)))*asin((tau**2-1)/(tau**2+1)) \
            + (8/(tau-1)**2)*log((tau**2+1)/(2*tau)))
        M_f_roll = K_f * (1/2 * rho * v_mag**2) * \
            (N * (Y_MA + r_t) * Cnalpha_fin * delta * A) # Forcing roll moment due to fin cant angle delta

        # Roll damping moment
        trap_integral = s/12 * ((Cr + 3*Ct)*s**2 + 4*(Cr+2*Ct)*s*r_t + 6*(Cr + Ct)*r_t**2)
        C_ldw = 2 * N * Cnalpha_fin / (A * d**2) * cos(delta) * trap_integral
        K_d = 1 + ((tau-gamma)/tau - (1-gamma)/(tau-1)*ln(tau))/ \
            ((tau+1)*(tau-gamma)/2 - (1-gamma)*(tau**3-1)/(3*(tau-1))) # Correction factor for conical fins
        M_d_roll = K_d * (1/2 * rho * v_mag**2) * A * d * C_ldw * (d / (2 * v_mag)) # Damping roll moment
        
        M_f = Matrix([M_f_pitch, M_f_yaw, M_f_roll])  # Corrective moment vector
        M_d = Matrix([M_d_pitch, M_d_yaw, M_d_roll])  # Damping moment vector
        
        M1 = M_f[0] - M_d[0] * w1
        M2 = M_f[1] - M_d[1] * w2
        M3 = M_f[2] - M_d[2] * w3
        
        M = H * Matrix([M1, M2, M3])
        
        self.M = M


    def get_moments(self) -> Matrix:
        """Get the moments for the rocket.
        Returns:
            Matrix: The moments vector.
        """
        self.set_moments()
        return self.M


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
        I1, I2, I3, T1, T2, T3, mass, rho, d, g, CG, delta, C_d, Cnalpha_fin, Cnalpha_rocket, Cr, Ct, s, N, v_wind1, v_wind2 = self.params
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
        w1dot = ((I2 - I3) * w2 * w3 + M1) / I1
        w2dot = ((I3 - I1) * w3 * w1 + M2) / I2
        w3dot = ((I1 - I2) * w1 * w2 + M3) / I3
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


    def set_f(self, t: float, xhat: np.array):
        """Get the equations of motion evaluated at time t and state xhat.
        Args:
            t (float): The time in seconds.
            xhat (np.array): The state estimation vector as a numpy array.
        ## Sets:
            self.f_params (Matrix): The parameterized equations of motion with time-variant constants.
            self.f_subs (Matrix): The substituted equations of motion at a state.
        """
        self.checkParamsSet()
        if self.f is None or self.state_vars is None:
            self.setup_eom()

        w1, w2, w3, v1, v2, v3, qw, qx, qy, qz = self.state_vars
        I1, I2, I3, T1, T2, T3, mass, rho, d, g, CG, delta, C_d, Cnalpha_fin, Cnalpha_rocket, Cr, Ct, s, N, v_wind1, v_wind2 = self.params
        # I1, I2, I3, T1, T2, T3, mass, rho, d, g, CG, delta, C_d, Cnalpha_fin, Cnalpha_rocket, Cr, Ct, s, N = self.params

        ## Get time varying constants ##
        thrust = self.get_thrust(t)
        mass_rocket = self.get_mass(t)
        inertia = self.get_inertia(t)
        x_CG = self.get_CG(t)
        
        params = {
            I1: Float(inertia[0]), # Ixx
            I2: Float(inertia[1]), # Iyy
            I3: Float(inertia[2]), # Izz
            T1: thrust[0],
            T2: thrust[1],
            T3: thrust[2],
            mass: Float(mass_rocket),
            CG: Float(x_CG), # m center of gravity
            v_wind1: Float(self.v_wind[0]), # m/s wind velocity in x direction
            v_wind2: Float(self.v_wind[1]), # m/s wind velocity in y direction
            rho: self.rho, # kg/m^3
            g: self.g, # m/s^2
            N: self.N, # number of fins
            d: self.d, # m rocket diameter
            Cr: self.Cr, # m fin root chord
            Ct: self.Ct, # m fin tip chord
            s: self.s, # m fin span
            C_d: self.C_d, # Drag coefficient
            Cnalpha_fin: self.Cnalpha_fin, # fin normal force coefficient derivative
            Cnalpha_rocket: self.Cnalpha_rocket, # rocket normal force coefficient derivative
            delta: rad(self.delta), # rad fin cant angle
            self.t_sym: Float(t)
        }

        f_params : Matrix = self.f.subs(params)

        ## Replace sqrt(v1^2 + v2^2) with a non-zero term to avoid NaNs in A matrix ##
        eps = Float(1e-9)  # Small term to avoid division by zero
        vxy = sqrt(v1**2 + v2**2 + eps**2)
        repl = {
            sqrt(v1**2 + v2**2): vxy,
            (v1**2 + v2**2)**(Float(1)/2): vxy
        }
        f_params = f_params.xreplace(repl)

        ## Substitute state variables ##
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

        f_subs_full = f_params.subs(m_e)

        self.f_subs_params = f_params
        self.f_subs_full = f_subs_full
    

    def _gather_param_values(self, t: float) -> list:
        """Collect numeric parameter values (same order as self.params) for lambdified EOM."""
        thrust = self.get_thrust(t)
        mass_rocket = self.get_mass(t)
        inertia = self.get_inertia(t)
        x_CG = self.get_CG(t)
        return [
            float(inertia[0]),  # I1
            float(inertia[1]),  # I2
            float(inertia[2]),  # I3
            float(thrust[0]),
            float(thrust[1]),
            float(thrust[2]),
            float(mass_rocket),
            float(self.rho),
            float(self.d),
            float(self.g),
            float(x_CG),
            float(np.deg2rad(self.delta)),
            float(self.C_d),
            float(self.Cnalpha_fin),
            float(self.Cnalpha_rocket),
            float(self.Cr),
            float(self.Ct),
            float(self.s),
            float(self.N),
            float(self.v_wind[0]),
            float(self.v_wind[1]),
        ]


    def _compile_numeric_funcs(self):
        """Lazily lambdify EOM for fast numeric evaluation."""
        if self._f_numeric is not None:
            return
        if self.f is None or self.state_vars is None:
            self.define_eom()

        # Replace sqrt(v1^2 + v2^2) with guarded version to avoid NaNs.
        w1, w2, w3, v1, v2, v3, qw, qx, qy, qz = self.state_vars
        eps = Float(1e-9)
        vxy = sqrt(v1**2 + v2**2 + eps**2)
        repl = {
            sqrt(v1**2 + v2**2): vxy,
            (v1**2 + v2**2)**(Float(1)/2): vxy,
        }

        def _prep(expr: Matrix):
            return expr.xreplace(repl)

        arg_syms = self.state_vars + self.params + [self.t_sym]

        self._f_numeric = lambdify(arg_syms, _prep(self.f), modules="numpy")


    def _compile_A_funcs(self):
        """Lazily lambdify Jacobian of the EOM w.r.t. state for fast A-matrix evaluation."""
        if self._A_numeric is not None:
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
        arg_syms = self.state_vars + self.params + [self.t_sym]

        self._A_numeric = lambdify(arg_syms, _prep(self.f).jacobian(m), modules="numpy")


    def f_numeric(self, t: float, x: np.ndarray, u: np.ndarray = None) -> np.ndarray:
        """Fast numeric evaluation of EOM using cached lambdified functions. Control inputs are ignored."""
        self.checkParamsSet()
        self._compile_numeric_funcs()

        state_vals = np.asarray(x, dtype=float).tolist()

        param_vals = self._gather_param_values(t)
        result = self._f_numeric(*(state_vals + param_vals + [float(t)]))
        return np.array(result, dtype=float).reshape(-1)
        
    
    def getA(self, t: float, xhat: np.array) -> np.ndarray:
        """Get the state matrix A evaluated at time t and state xhat.

        Args:
            t (float): The time in seconds.
            xhat (np.array): The state estimation vector as a numpy array.
        Returns:
            np.ndarray: The state matrix A as a numpy array.
        """
        self.checkParamsSet()
        self._compile_A_funcs()

        param_vals = self._gather_param_values(t)
        args = np.asarray(xhat, dtype=float).tolist() + param_vals + [float(t)]

        A_num = np.array(self._A_numeric(*args), dtype=np.float64)
        self.A = A_num

        return A_num
    
    
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

      
# For testing
def main():
    print("Hello there")

if __name__ == "__main__":
    main()
