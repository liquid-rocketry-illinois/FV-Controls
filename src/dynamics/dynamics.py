from sympy import *
import numpy as np
import pandas as pd
from typing import Callable

class Dynamics:
    def __init__(self, rocket_name : str):
        """Initialize the Dynamics class. Rocket body axis is aligned with z-axis.

        Args:
            t_estimated_apogee (float): Estimated time until apogee in seconds.
            dt (float): Time step for simulation in seconds.
            x0 (np.ndarray): Initial state vector.
        """

        self.f_preburnout : Matrix = None
        self.f_postburnout : Matrix = None
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
        self.SM_func : Callable[[bool, Expr], Expr] = None # Stability margin as a function of angle of attack in degrees
        
        # Fin parameters
        self.N : float = None # Number of fins
        self.Cr : float = None # Root chord in meters
        self.Ct : float = None # Tip chord in meters
        self.s : float = None # Span in meters
        self.Cnalpha_fin : float = None # Normal force coefficient normalized by angle of attack for 1 fin
        self.delta : float = None # Fin cant angle in degrees
        
        # Thrust curve data
        self.thrust_times : np.ndarray = None
        self.thrust_forces : np.ndarray = None

        # Environmental parameters
        self.rho : float = 1.225 # Air density kg/m^3
        self.g : float = 9.81 # Gravitational acceleration m/s^2
        
        # Rocket name (used for saving simulation results from Simluation() object to designated path)
        self.rocket_name = rocket_name
        
        # State space linearization matrices
        self.A_sym : Matrix = None # Symbolic state matrix
        self.A : np.ndarray = None # State matrix
        
        ## Helpers ##
        self.F : Matrix = None # Forces matrix
        self.M : Matrix = None # Moments matrix


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

        self.state_vars = [w1, w2, w3, v1, v2, v3, qw, qx, qy, qz]
        self.params = [I1, I2, I3, T1, T2, T3, mass, rho, d, g, CG, delta, C_d, Cnalpha_fin, Cnalpha_rocket, Cr, Ct, s, N]
        self.t_sym = t_sym


    ## Helper function to print thrust curve ##
    def printThrustCurve(self, thrust_file: str):
        """Print the thrust curve data from a .csv or .eng file. Copy cell output to code block to set thrust curve parameters.
        Replace 'your_object_name' with whatever you name your Dynamics object as (e.g. dynamics = Dynamics(), your object name
        would be 'dynamics').

        Args:
            thrust_file (str): Path to the .csv or .eng file containing thrust curve data. Can be from OpenRocket or thrustcurve.org.
        """
        df = None
        if thrust_file.endswith('.csv'):
            df = pd.read_csv(thrust_file)
        elif thrust_file.endswith('.eng'):
            rows = []
            with open(thrust_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    # skip empty lines and comments
                    if not line or line.startswith(';'):
                        continue

                    parts = line.split()

                    # Data lines in .eng files are usually: "<time> <thrust>"
                    # Header/metadata has more columns, so we ignore those.
                    if len(parts) == 2:
                        try:
                            t = float(parts[0])
                            F = float(parts[1])
                            rows.append((t, F))
                        except ValueError:
                            # In case something weird slips through, just skip the line
                            continue

            df = pd.DataFrame(rows, columns=["# Time (s)", "Thrust (N)"])
        else:
            raise ValueError("Unsupported file format. Please provide a .csv or .eng file.")

        times = df["# Time (s)"]
        thrust = df["Thrust (N)"]
        stop_index = np.argmax(thrust[1:] == 0.0)
        times = times[:stop_index + 2]
        thrust = thrust[:stop_index + 2]
        
        print(f"thrust_times = np.array({times.tolist()})")
        print(f"thrust_forces = np.array({thrust.tolist()})")
        print("your_object_name.setThrustCurve(thrust_times=thrust_times, thrust_forces=thrust_forces)")
        

    def setRocketParams(self, I_0: float, I_f: float, I_3: float,
                        x_CG_0: float, x_CG_f: float,
                        m_0: float, m_f: float, m_p: float,
                        d: float, L_ne: float, C_d: float, Cnalpha_rocket: float,
                        t_launch_rail_clearance: float, t_motor_burnout: float, t_estimated_apogee: float,
                        SM_func: Callable[[bool, Expr], Expr]):
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
            SM_func (Callable[[bool, Expr], Expr]): User defined function of stability margin as a function of angle of attack (deg).\
                Takes bool 'burnout' and Expr 'AoA_deg' as parameters. Separate equations to define burnout=True and burnout=False.\
                    Returns function fit equation of stability margin vs AoA (e.g. using Google Sheets).
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
        self.SM_func = SM_func


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


    def setThrustCurve(self, thrust_times: np.ndarray, thrust_forces: np.ndarray):
        """Set the thrust curve data.

        Args:
            thrust_times (np.ndarray): Array of time points in seconds.
            thrust_force (np.ndarray): Array of thrust values in Newtons corresponding to the time points.
        """
        self.thrust_times = thrust_times
        self.thrust_forces = thrust_forces
    
    
    def setEnvParams(self, rho: float, g: float):
        """Set the environmental parameters.

        Args:
            rho (float): Air density in kg/m^3.
            g (float): Gravitational acceleration in m/s^2.
        """
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
            'rho', 'g',
            'N', 'Cr', 'Ct', 's', 'Cnalpha_fin',
            'SM_func'
        ]
        for param in required_params:
            if not hasattr(self, param):
                raise ValueError(f"Parameter '{param}' is not set. Please set all necessary parameters before proceeding.")


    def get_thrust(self, t: float) -> Matrix:
        """Get the thrust for the rocket at time t.

        Args:
            t (float): The time in seconds.

        Returns:
            dict: A dictionary containing inertia, mass, CG, and thrust at time t.
        """

        T = Matrix([0., 0., 0.])  # N
        motor_burnout = t > self.t_motor_burnout
        if not motor_burnout:
            T[2] = np.interp(t, self.thrust_times, self.thrust_forces) # Thrust acting in z direction
            
        return T
    
    
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
    
    
    def define_forces(self) -> Matrix:
        """Get the forces for the rocket. Sets self.F.
        """
        w1, w2, w3, v1, v2, v3, qw, qx, qy, qz = self.state_vars
        I1, I2, I3, T1, T2, T3, mass, rho, d, g, CG, delta, C_d, Cnalpha_fin, Cnalpha_rocket, Cr, Ct, s, N = self.params
        t_sym = self.t_sym
    
        H = Heaviside(t_sym - Float(self.t_launch_rail_clearance), 0)  # 0 if t < t_launch_rail_clearance, 1 if t >= t_launch_rail_clearance

        epsAoA = Float(1e-9)  # Small term to avoid division by zero in AoA calculation
        AoA = atan2(sqrt(v1**2 + v2**2), v3 + epsAoA) # Angle of attack
        AoA_eff = Piecewise(
            (0,   Abs(AoA) <= epsAoA),                # inside deadband
            (Min(Abs(AoA), 15 * pi / 180) * (AoA/Abs(AoA)), True)  # ±15°
        )

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

        ## Lift Force ##
        eps_beta = Float(1e-9)
        nan_guard = sqrt(v1**2 + v2**2 + eps_beta**2)
        beta = 2 * atan2(v2, nan_guard + v1) # Equivalent to atan2(v2, v1) but avoids NaN at (0,0)
        L = H * 1/2 * rho * v_mag**2 * (2 * pi * AoA_eff) * A # Lift force approximation
        nL = Matrix([
            -cos(AoA_eff) * cos(beta),
            -cos(AoA_eff) * sin(beta),
            sin(AoA_eff)
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
        self.define_forces()
        return self.F


    def define_moments(self, burnout: bool) -> Matrix:
        """Get the moments for the rocket.

        Args:
            burnout (bool): Whether the rocket is in the pre-burnout or post-burnout phase. (True for post-burnout)
        Returns:
            Matrix: The moments vector.
        """
        w1, w2, w3, v1, v2, v3, qw, qx, qy, qz = self.state_vars
        I1, I2, I3, T1, T2, T3, mass, rho, d, g, CG, delta, C_d, Cnalpha_fin, Cnalpha_rocket, Cr, Ct, s, N = self.params
        t_sym = self.t_sym
    
        H = Heaviside(t_sym - Float(self.t_launch_rail_clearance), 0)  # 0 if t < t_launch_rail_clearance, 1 if t >= t_launch_rail_clearance

        epsAoA = Float(1e-9)  # Small term to avoid division by zero in AoA calculation
        AoA = atan2(sqrt(v1**2 + v2**2), v3 + epsAoA) # Angle of attack
        AoA_eff = Piecewise(
            (0,   Abs(AoA) <= epsAoA),                # inside deadband
            (Min(Abs(AoA), 15 * pi / 180) * (AoA/Abs(AoA)), True)  # ±15°
        )

        eps = Float(1e-9)  # Small term to avoid division by zero
        v_mag = sqrt(v1**2 + v2**2 + v3**2 + eps**2) # Magnitude of velocity with small term to avoid division by zero

        ## Rocket reference area ##
        A = pi * (d/2)**2 # m^2
        
        ## Stability Margin ##
        AoA_deg = AoA_eff * 180 / pi # Convert AoA to degrees for polynomial fit
        SM = self.SM_func(burnout, AoA_deg)

        ## Corrective moment coefficient ##
        # Multiplying by stability because CG is where rotation is about and CP is where force is applied
            # SM = (CP - CG) / d
        # Equations from ApogeeRockets
        C_raw = H * v_mag**2 * A * Cnalpha_rocket * AoA_eff * (SM * d) * rho / 2
        
        eps_beta = Float(1e-9)
        nan_guard = sqrt(v1**2 + v2**2 + eps_beta**2)
        beta = 2 * atan2(v2, nan_guard + v1) # Equivalent to atan2(v2, v1) but avoids NaN at (0,0)
        
        M_f_pitch = C_raw * sin(beta) # Pitch forcing moment
        M_f_yaw = -C_raw * cos(beta) # Yaw forcing moment

        ## Propulsive Damping Moment Coefficient (Cdp) ##
        mdot = self.m_p / self.t_motor_burnout # kg/s, average mass flow rate during motor burn
        Cdp = mdot * (self.L_ne - CG)**2 # kg*m^2/s

        ## Aerodynamic Damping Moment Coefficient (Cda) ##
        Cda = H * (rho * v_mag * A / 2) * (Cnalpha_rocket * AoA_eff * (SM * d)**2)

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


    def get_moments(self, burnout: bool) -> Matrix:
        """Get the moments for the rocket.
        Args:
            burnout (bool): Whether the rocket is in the pre-burnout or post-burnout phase. (True for post-burnout)
        Returns:
            Matrix: The moments vector.
        """
        self.define_moments(burnout)
        return self.M


    def define_eom(self, burnout: bool):
        """Get the equations of motion for the rocket. Sets self.f_preburnout or self.f_postburnout.

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
        - Normal force coefficient Cn is modeled as a polynomial function of velocity, with different coefficients pre- and post-motor burnout
        - Drag force Fd is modeled as a quadratic function of velocity magnitude
        - Lift force Fl is modeled using thin airfoil theory, proportional to angle of attack (AoA)
        - Corrective moment coefficient C is modeled as a function of velocity magnitude, normal force coefficient Cn, stability margin SM, and rocket diameter
        - Normal force coefficient derivative Cnalpha is modeled as Cn * (AoA / (AoA^2 + aoa_eps^2)) to ensure smoothness at AoA = 0
        - Stability margin SM is modeled as a polynomial function of AoA
        - Small terms are added to avoid division by zero in velocity magnitude and AoA calculations (denoted as eps and aoa_eps)
        - All polynomial equations are determined from experimental OpenRocket data and curve fitting using Google Sheets
        - Piecewise functions are used to bound certain variables (e.g., AoA, Cnalpha, C) to ensure numerical stability and physical realism

        ## Usage:
        - To derive the full set of equations of motion, call setup_eom() which derives both pre- and post-burnout EOMs.

        Args:
            burnout (bool): Whether the rocket is in the pre-burnout or post-burnout phase. (True for post-burnout)
        """
        if self.t_sym is None or self.state_vars is None or self.params is None:
            self.set_symbols()
        w1, w2, w3, v1, v2, v3, qw, qx, qy, qz = self.state_vars
        I1, I2, I3, T1, T2, T3, mass, rho, d, g, CG, delta, C_d, Cnalpha_fin, Cnalpha_rocket, Cr, Ct, s, N = self.params
        t_sym = self.t_sym
        
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
        M = self.get_moments(burnout)
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

        if (not burnout):
            self.f_preburnout = f
        else:
            self.f_postburnout = f


    def setup_eom(self):
        """Setup the equations of motion by deriving pre- and post-burnout EOMs.
        """
        self.define_eom(burnout=False)
        self.define_eom(burnout=True)


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
        if self.f_preburnout is None or self.f_postburnout is None or self.state_vars is None:
            self.setup_eom()

        w1, w2, w3, v1, v2, v3, qw, qx, qy, qz = self.state_vars
        I1, I2, I3, T1, T2, T3, mass, rho, d, g, CG, delta, C_d, Cnalpha_fin, Cnalpha_rocket, Cr, Ct, s, N = self.params

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

        ## Select pre- or post-burnout equations ##
        preburnout = (t <= self.t_motor_burnout)
        postburnout = (t > self.t_motor_burnout)
        f_params : Matrix = None
        if preburnout:
            f_params = self.f_preburnout.subs(params)
        elif postburnout:
            f_params = self.f_postburnout.subs(params)

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
        
    
    def getA(self, t: float, xhat: np.array) -> np.ndarray:
        """Get the state matrix A evaluated at time t and state xhat.

        Args:
            t (float): The time in seconds.
            xhat (np.array): The state estimation vector as a numpy array.
        Returns:
            np.ndarray: The state matrix A as a numpy array.
        """
        
        self.set_f(t=t, xhat=xhat)
        w1, w2, w3, v1, v2, v3, qw, qx, qy, qz = self.state_vars
        m = Matrix(self.state_vars)
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
        A : Matrix = self.f_subs_params.jacobian(m).subs(m_e).n()

        self.A_sym = A
        
        A_num = np.array(A).astype(np.float64)
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