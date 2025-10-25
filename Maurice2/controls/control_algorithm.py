from sympy import *
import numpy as np
import pandas as pd
import os

class Controls:
    def __init__(
            self,
            t_motor_burnout: float = 1.971,
            t_estimated_apogee: float = 13.571,
            t_launch_rail_clearance: float = 0.164,
            prop_mass: float = 0.355, # kg
            L_ne: float = 1.17, # m
            dt: float = 0.01,
            x0: np.ndarray = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), # Initial state
            u0: np.ndarray = np.array([0.0]), # Initial input
            Ks: np.ndarray = None,
            L: np.ndarray = None,
        ):
        """Initialize the Controls class. Rocket body axis is aligned with y-axis.

        Args:
            t_motor_burnout (float, optional): Time until motor burnout in seconds. Defaults to 1.971.
            t_estimated_apogee (float, optional): Estimated time until apogee in seconds. Defaults to 13.571.
            t_launch_rail_clearance (float, optional): Time until launch rail clearance in seconds. Defaults to 0.164.
            prop_mass (float, optional): Propellant mass in kg. Defaults to 0.355.
            L_ne (float, optional): Distance the nozzle is from the tip of the nose cone in meters. Defaults to 1.17.
            dt (float, required): Time step for simulation in seconds. Defaults to 0.01.
            K (np.ndarray, required): Gain matrix. Defaults to None.
        """
        self.t_motor_burnout = t_motor_burnout # seconds
        self.t_estimated_apogee = t_estimated_apogee # seconds
        self.t_launch_rail_clearance = t_launch_rail_clearance # seconds
        self.prop_mass = prop_mass # kg
        self.L_ne = L_ne # m
        self.csv_path = "Maurice2/data/openrocket_data.csv"
        # self.csv_path = "/Users/dsong/Library/CloudStorage/OneDrive-UniversityofIllinois-Urbana/Club Stuff/LRI/FV-Controls/Control/openrocket_data.csv"
        self.A : Matrix = None
        self.B : Matrix = None
        self.C : Matrix = None
        self.f_preburnout : Matrix = None
        self.f_postburnout : Matrix = None
        self.Ks : np.ndarray = Ks
        self.L : np.ndarray = L
        self.vars : list = None
        self.f_params : Matrix = None
        self.f_subs : Matrix = None
        self.dt = dt
        self.x0 = np.array(x0, dtype=float) if x0 is not None else None
        self.u0 = np.array(u0, dtype=float) if u0 is not None else None
        self.t_sym : Symbol = None
        self.T = None
        self.g = None

        # Logging
        self.states = [self.x0]
        self.inputs = [self.u0]
        self.As = []
        self.Bs = []


    def setRocketParams(self, t_motor_burnout: float, t_estimated_apogee: float, t_launch_rail_clearance: float, prop_mass: float):
        """Set the rocket parameters.

        Args:
            t_motor_burnout (float): Time until motor burnout in seconds.
            t_estimated_apogee (float): Estimated time until apogee in seconds.
            t_launch_rail_clearance (float): Time until launch rail clearance in seconds.
            prop_mass (float): Propellant mass in kg.
        """
        self.t_motor_burnout = t_motor_burnout
        self.t_estimated_apogee = t_estimated_apogee
        self.t_launch_rail_clearance = t_launch_rail_clearance
        self.prop_mass = prop_mass

    
    def getLineOfBestFitTime(self, var: str, n: int = 1):
        """Get the line of best fit for the given data with a polynomial of degree n.

        Args:
            var (str): The variable to fit the line to.
            n (int, optional): The degree of the polynomial to fit. Defaults to 1.

        Returns:
            tuple: A tuple containing the coefficients of the polynomial and its degree.
        """
        # Load the CSV data into a DataFrame
        data = pd.read_csv(self.csv_path)
        t = data["# Time (s)"]
        y = None
        if (var == "mass"):
            y = data["Mass (g)"] / 1000  # Convert to kg
        elif (var == "inertia"):
            y = data["Longitudinal moment of inertia (kg·m²)"]
        elif (var == "CG"):
            y = data["CG location (cm)"] / 100  # Convert to m
        else:
            raise ValueError(
                "Invalid variable. Choose from: " \
                "'mass'," \
                "'inertia'," \
                "'CG'. " \
                )

        # Filter data based on motor burnout
        mask = t <= self.t_motor_burnout
        t = t[mask]
        y = y[mask]
        coeffs = np.polyfit(t, y, n)
        return coeffs, n


    def getLineOfBestFitAoA(self, burnout: str, var: str, n: int = 5):
        """Get the line of best fit for the given data with a polynomial of degree n. Choose from "stability margin" or "normal force coeff".

        Args:
            burnout (str): Choose from "pre burnout" or "post burnout".
            var (str): The variable to fit the line to.
            n (int, optional): The degree of the polynomial to fit. Defaults to 5.

        Returns:
            tuple: A tuple containing the coefficients of the polynomial and its degree.
        """
        # Load the CSV data into a DataFrame
        data = pd.read_csv(self.csv_path)
        x = data["Angle of attack (°)"]
        if (var == "stability margin"):
            y = data["Stability margin calibers (​)"]
        else:
            raise ValueError(
                "Invalid variable. Choose from: " \
                "'stability margin'. "
                )

        launch_to_burnout = (data["# Time (s)"] >= self.t_launch_rail_clearance) & (data["# Time (s)"] < self.t_motor_burnout)
        burnout_to_apogee = (data["# Time (s)"] >= self.t_motor_burnout) & (data["# Time (s)"] <= self.t_estimated_apogee)
        if (burnout == "pre burnout"):
            x = x[launch_to_burnout]
            y = y[launch_to_burnout]
        elif (burnout == "post burnout"):
            x = x[burnout_to_apogee]
            y = y[burnout_to_apogee]
        else:
            raise ValueError(
                "Invalid motor_burnout. Choose from: " \
                "'pre burnout', " \
                "'post burnout'. "
                )

        coeffs = np.polyfit(x, y, n)
        return coeffs, n


    def getLineOfBestFitVel(self, var: str, n: int = 2):
        """Get the line of best fit for the given data with a polynomial of degree n.

        Args:
            var (str): The variable to fit the line to.
            n (int, optional): The degree of the polynomial to fit. Defaults to 2.

        Returns:
            tuple: A tuple containing the coefficients of the polynomial and its degree.
        """
        # Load the CSV data into a DataFrame
        # TODO: set data to __init__ for efficiency
        data = pd.read_csv(self.csv_path)
        x = data["Total velocity (m/s)"]
        if (var == "drag force"):
            y = data["Drag force (N)"]
        else:
            raise ValueError(
                "Invalid variable. Choose from: " \
                "'drag force'. "
                )
        launch_to_apogee = (data["# Time (s)"] >= self.t_launch_rail_clearance) & (data["# Time (s)"] <= self.t_estimated_apogee)
        x = x[launch_to_apogee]
        y = y[launch_to_apogee]

        coeffs = np.polyfit(x, y, n)
        return coeffs, n


    def getTimeConstants(self, t: float):
        """Get the constants for the rocket at time t.

        Args:
            t (float): The time in seconds.

        Returns:
            dict: A dictionary containing inertia, mass, CG, and thrust at time t.
        """

        constants = dict()
        ## Post burnout constants ##
        I = Matrix([0.287, 0.287, 0.0035]) # Post burnout inertia values from OpenRocket, kg*m^2
        m = 2.589  # Post burnout mass from OpenRocket, kg
        CG = 63.5/100  # Post burnout CG from OpenRocket, m
        T = Matrix([0., 0., 0.])  # N

        motor_burnout = t > self.t_motor_burnout

        # TODO: for added efficiency, only call getLineOfBestFitTime once per variable and store the results
        if not motor_burnout:
            coeffs_mass, degree_mass = self.getLineOfBestFitTime("mass")
            m = sum(coeffs_mass[i] * t**(degree_mass - i) for i in range(degree_mass + 1))

            coeffs_inertia, degree_inertia = self.getLineOfBestFitTime("inertia")
            I_long = sum(coeffs_inertia[i] * t**(degree_inertia - i) for i in range(degree_inertia + 1))
            I[0] = I_long # Ixx
            I[1] = I_long # Iyy

            coeffs_CG, degree_CG = self.getLineOfBestFitTime("CG")
            CG = sum(coeffs_CG[i] * t**(degree_CG - i) for i in range(degree_CG + 1))

            times = pd.read_csv(self.csv_path)["# Time (s)"]
            thrust = pd.read_csv(self.csv_path)["Thrust (N)"]
            T[2] = np.interp(t, times, thrust) # Thrust acting in z direction

        constants["inertia"] = I
        constants["mass"] = m
        constants["CG"] = CG
        constants["thrust"] = T
        
        return constants

    
    def quat_to_euler_xyz(self, q: np.ndarray, degrees=False, eps=1e-9):
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
    

    def R_BW_from_q(self, qw, qx, qy, qz):
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

    
    def getAileronMoment(self, delta1: Symbol, v3: Symbol):
        """Get the aileron moment based on the aileron angle.

        Args:
            delta1 (float): The aileron angle in radians.

        Returns:
            Matrix: The symbolic moment vector [Mx, My, Mz] function of the aileron angle and rocket's vertical velocity.
        """
        
        M1, M2 = 0, 0
        # Verify later with data points
        M3 = deg(delta1)/8 * (-2.21e-09*(v3**3) 
                        + 1.58e-06*(v3**2) 
                        + 4.18e-06*v3 
                        ) # v3 = vertical velocity, Mz = roll moment

        return Matrix([M1, M2, M3])
        # -8.26e-05 + 4.18e-06x + 1.58e-06x^2 + -2.21e-09x^3


    def deriveEOM(self, post_burnout: bool):
        """Get the equations of motion for the rocket, derive the A and B matrices at time t.

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
        - To derive the full set of equations of motion, call deriveEOM() twice: once with post_burnout=False and once with post_burnout=True

        Args:
            post_burnout (bool): Whether the rocket is in the post-burnout phase.
        Returns:
            tuple: A tuple containing the A and B Numpy arrays evaluated at the operating state xhat and input u.
        """
        w1, w2, w3, v1, v2 = symbols('w_1 w_2 w_3 v_1 v_2', real = True) # Angular and linear velocities
        v3 = symbols('v_3', real = True, positive = True) # Longitudinal velocity, assumed positive during flight
        qw, qx, qy, qz = symbols('q_w q_x q_y q_z', real = True) # Quaternion components
        I1, I2, I3 = symbols('I_1 I_2 I_3', real = True) # Moments of inertia
        M1, M2, M3 = symbols('M_1 M_2 M_3', real = True) # Moments
        T1, T2, T3 = symbols('T_1 T_2 T_3', real = True) # Thrusts
        mass, rho, A, g, CG = symbols('m rho A g CG', real = True) # Mass, air density, reference area, gravity, center of gravity
        delta1 = symbols('delta_1', real = True) # Aileron angle

        t_sym = symbols('t', real = True) # Time symbol for Heaviside function
        self.t_sym = t_sym
        H = Heaviside(t_sym - Float(self.t_launch_rail_clearance), 0)  # 0 if t < t_launch_rail_clearance, 1 if t >= t_launch_rail_clearance

        epsAoA = Float(1e-3)  # Small term to avoid division by zero in AoA calculation
        AoA = atan2(sqrt(v1**2 + v2**2), v3 + epsAoA) # Angle of attack
        AoA_eff = Piecewise(
            (0,   Abs(AoA) <= epsAoA),                # inside deadband
            (Min(Abs(AoA), 15 * pi / 180) * (AoA/Abs(AoA)), True)  # ±15°
        )

        eps = Float(1e-3)  # Small term to avoid division by zero
        v = Matrix([v1, v2, v3]) # Velocity vector
        v_mag = sqrt(v1**2 + v2**2 + v3**2 + eps**2) # Magnitude of velocity with small term to avoid division by zero
        vhat = v / v_mag  # Unit vector in direction of velocity

        ## Thrust ##
        T : Matrix = Matrix([T1, T2, T3])  # Thrust vector, T1 and T2 are assumed 0

        ## Gravity ##
        Fg_world = Matrix([0.0, 0.0, -mass * g])
        R_world_to_body = self.R_BW_from_q(qw, qx, qy, qz)  # Rotation matrix from world to body frame
        Fg : Matrix = R_world_to_body * Fg_world  # Transform gravitational force to body frame

        ## Drag Force ##
        Fd_mag = -(0.627 + -0.029*v_mag + 1.95e-3*v_mag**2) # Drag force approximation
        Fd : Matrix = Fd_mag * vhat # Drag force vector

        ## Lift Force ##
        eps_beta = Float(1e-3)
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
        # F = T + Fd + Fl + Fg # Total force vector

        F = Fd + Fl # For debugging, add thrust and gravity in loop to avoid differentiating them to 0 in A

        ## Cnalpha ##
        Cnalpha = 0.207  # Linear assumption of Cn vs AoA slope from OpenRocket data (fitted to quadratic, minimal x^2 coefficient)

        ## Stability Margin ##
        AoA_deg = AoA_eff * 180 / pi # Convert AoA to degrees for polynomial fit
        SM = 0
        if not post_burnout:
            SM = 2.8 + -0.48*AoA_deg + 0.163*AoA_deg**2 + -0.0386*AoA_deg**3 + 5.46E-03*AoA_deg**4 + -4.61E-04*AoA_deg**5 + 2.28E-05*AoA_deg**6 + -6.1E-07*AoA_deg**7 + 6.79E-09*AoA_deg**8
        else:
            SM = -0.086*AoA_deg + 2.73

        ## Rocket diameter ##
        d = Float(7.87)/100 # m

        ## FIXME: CCM HAS UNITS N*m, SHOULD USE DAMPING MOMENT COEFFICIENT (Cmq) INSTEAD ##
        ## Corrective moment coefficient ##
        # Multiplying by stability because CG is where rotation is about and CP is where force is applied
            # SM = (CP - CG) / d
        C_raw = H * v_mag**2 * A * Cnalpha * AoA_eff * (SM * d) * rho / 2 # See if it's Cnalpha or Cn, Cn = Cnalpha * AoA_eff
        # Ccm_mag = Piecewise(
        #     (0, C_raw <= 0),
        #     (Min(C_raw, Float(5e-2)), True)                      # cap magnitude (tune)
        # )
        Ccm = Matrix([C_raw * sin(beta), -C_raw * cos(beta), 0])  # Corrective moment vector

        ## Propulsive Damping Moment Coefficient (Cdp) ##
        mdot = self.prop_mass / self.t_motor_burnout # kg/s, average mass flow rate during motor burn
        Cdp = mdot * (self.L_ne - CG)**2 # kg*m^2/s

        ## Aerodynamic Damping Moment Coefficient (Cda) ##
        Cda = H * (rho * v_mag * A / 2) * (Cnalpha * AoA_eff * (SM * d)**2)

        ## Damping Moment Coefficient (Cdm) ##
        Cdm = Cdp + Cda

        ## Moment due to aileron deflection ##
        M_fin = 200 * (Float(1)/2 * rho * v_mag**2) * Matrix([0, 0, 1e-8])  # Fin misalignment, total moment from all four fins, tune later
        M_delta = self.getAileronMoment(delta1, v3)
        M1 = M_fin[0] + M_delta[0] + Ccm[0] - Cdm * w1
        M2 = M_fin[1] + M_delta[1] + Ccm[1] - Cdm * w2
        M3 = M_fin[2] + M_delta[2] + Ccm[2] - Float(0.1) * Cdm * w3

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

        vars = [w1, w2, w3, v1, v2, v3, qw, qx, qy, qz, delta1, I1, I2, I3, T1, T2, T3, mass, rho, A, g, CG]
        self.vars = vars

        if (not post_burnout):
            self.f_preburnout = f
        else:
            self.f_postburnout = f
    

    def computeAB(self, t: float, xhat: np.array, u: np.array):
        """Compute the A and B matrices at time t.
        Args:
            t (float): The time in seconds.
            xhat (np.array): The state estimation vector as a numpy array.
            u (np.array): The input vector as a numpy array.
        ## Sets:
            self.A (Matrix): The A matrix.
            self.B (Matrix): The B matrix.
            self.f_params (Matrix): The parameterized equations of motion with time-variant constants.
            self.f_subs (Matrix): The substituted equations of motion at a state.
        Returns:
            None
        """
        if self.f_preburnout is None or self.f_postburnout is None or self.vars is None:
            print("Equations of motion have not been derived yet. Call deriveEOM() first on pre- and post-burnout.")
            return None, None, None, None
        
        w1, w2, w3, v1, v2, v3, qw, qx, qy, qz, delta1, I1, I2, I3, T1, T2, T3, mass, rho, A, g, CG = self.vars

        m = Matrix([w1, w2, w3, v1, v2, v3, qw, qx, qy, qz]) # State vector
        n = Matrix([delta1]) # Input vector

        ## Get time varying constants ##
        constants = self.getTimeConstants(t)
        mass_rocket = constants["mass"]
        inertia = constants["inertia"]
        CoG = constants["CG"]
        thrust = constants["thrust"]

        params = {
            I1: Float(inertia[0]), # Ixx
            I2: Float(inertia[1]), # Iyy
            I3: Float(inertia[2]), # Izz
            T1: thrust[0],
            T2: thrust[1],
            T3: thrust[2],
            mass: Float(mass_rocket),
            rho: Float(1.225), # kg/m^3 temp constant rho
            A: pi * Float((7.87/100/2)**2), # m^2 reference area
            g: Float(9.81), # m/s^2
            CG: Float(CoG), # m center of gravity
            self.t_sym: Float(t)
        }

        ## Select pre- or post-burnout equations ##
        preburnout = t <= self.t_motor_burnout
        postburnout = t > self.t_motor_burnout
        f_params = None
        if preburnout:
            f_params = self.f_preburnout.subs(params)
        elif postburnout:
            f_params = self.f_postburnout.subs(params)

        ## Replace sqrt(v1^2 + v2^2) with a non-zero term to avoid NaNs in A matrix ##
        eps = Float(1e-3)  # Small term to avoid division by zero
        vxy = sqrt(v1**2 + v2**2 + eps**2)
        repl = {
            sqrt(v1**2 + v2**2): vxy,
            (v1**2 + v2**2)**(Float(1)/2): vxy
        }
        f_params = f_params.xreplace(repl)

        ## NOTE: Not finding equilibrium states, using trajectory/operating-point linearization
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
            delta1: u[0],
        }

        f_subs = f_params.subs(m_e).subs(n_e)

        A = f_params.jacobian(m).subs(m_e).subs(n_e)
        B = f_params.jacobian(n).subs(m_e).subs(n_e)

        self.A = A
        self.B = B
        self.f_params = f_params
        self.f_subs = f_subs

        ## Logging ##
        self.As.append(A)
        self.Bs.append(B)


    def control_law(self, xhat: np.array, t: float):
        """Compute the control input based on the current state estimate and gain matrix.

        Args:
            xhat (np.array): The estimated state vector.
            t (float): The current time in seconds.
        Returns:
            np.ndarray: The computed gain matrix K.
        """
        ## Gain scheduling based on vertical velocity ##
        v3 = xhat[5]

        # Preburnout
        if (t < self.t_motor_burnout):
            Kmax = self.Ks[0]
            Kmin = self.Ks[1]
            v3_mid = 100 # m/s, tune as necessary
            K_val = Kmin + (Kmax - Kmin) / (1 + exp((v3 - v3_mid)/7))

        # Postburnout
        else:
            Kmax = self.Ks[2]
            Kmin = self.Ks[3]
            v3_mid = 80 # m/s, tune as necessary hiii dan :3
            K_val = Kmin + (Kmax - Kmin) / (1 + exp((v3 - v3_mid)/6))
        K = np.zeros((1, 10))
        K[0][2] = K_val

        return K


    def get_thrust_accel(self, t: float):
        """Get the thrust acceleration at time t.

        Args:
            t (float): The time in seconds.

        Returns:
            np.array: The thrust acceleration vector as a numpy array.
        """
        thrust = self.getTimeConstants(t)["thrust"]
        m = self.getTimeConstants(t)["mass"]
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
        g = np.array([0.0, 0.0, -9.81])
        qw, qx, qy, qz = xhat[6], xhat[7], xhat[8], xhat[9]
        R_world_to_body = np.array(self.R_BW_from_q(qw, qx, qy, qz)).astype(np.float64)
        g_body = R_world_to_body @ g
        a_gravity = np.zeros(10)
        a_gravity[3:6] = g_body
        return a_gravity

    ## TODO: Fix this to account for velocity measurements later?? ##
    def computeC(self, xhat: np.ndarray, u: np.ndarray):
        """Compute the control input based on the current state, estimated state, and gain matrix.

        Args:
            xhat (np.ndarray): The estimated state vector.
            u (np.ndarray): The current input vector.
        
        Returns:
            np.ndarray: The computed control input vector.
        """
        w1, w2, w3 = self.vars[0], self.vars[1], self.vars[2]
        v1, v2, v3 = self.vars[3], self.vars[4], self.vars[5]
        qw, qx, qy, qz = self.vars[6], self.vars[7], self.vars[8], self.vars[9]
        delta1 = self.vars[10]

        m = Matrix([w1, w2, w3, v1, v2, v3, qw, qx, qy, qz])
        
        g = Matrix([
            [w1],
            [w2],
            [w3],
            # [v1],
            # [v2],
            # [v3],
            [qw],
            [qx],
            [qy],
            [qz]
        ])

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
            delta1: u[0],
        }

        C = g.jacobian(m).subs(m_e).subs(n_e)
        self.C = C


    def setL(self, L: np.ndarray):
        """Set the observer gain matrix L.

        Args:
            L (np.ndarray): The observer gain matrix.
        """
        self.L = L


    def buildL(self, lw=1.0, lqw=1.0, lqx=1.0, lqy=1.0, lqz=1.0):
        """
        Hand-tuned observer gain L (10x7) for x=[v(3), w(3), q(4)], y=[w(3), q(4)].
        - Only w3 row uses the w3 residual.
        - Quaternion rows use quaternion residuals; roll-weight lqz can be larger.
        - All velocity rows are zero to avoid injecting measurement noise into v.

        Args:
            lw (float): Gain for w residual into w states.
            lqw (float): Gain for qw residual into qw state.
            lqx (float): Gain for qx residual into qx state.
            lqy (float): Gain for qy residual into qy state.
            lqz (float): Gain for qz residual into qz state.

        Returns:
            np.ndarray: The observer gain matrix L.
        """
        L = np.zeros((10, 7))

        L[0, 0] = lw # w1
        L[1, 1] = lw # w2
        L[2, 2] = lw # w3

        # quaternion <- r_quaternion
        L[6, 3] = lqw   # qw residual into qw state
        L[7, 4] = lqx   # qx residual into qx state
        L[8, 5] = lqy   # qy residual into qy state
        L[9, 6] = lqz   # qz residual into qz state (roll-heavy)

        self.L = L


    def deriveSensorModels(self,t: float,
                           w1: float, w2: float, w3: float,
                           theta: float, phi: float, psi: float):
        """Derive the sensor models for the rocket. Assumes small angles for accelerometer bias calculation. Converts Euler angles to quaternion.

        Args:
            t (float): The time in seconds.
            w1 (float): The angular velocity in the x direction in rad/s.
            w2 (float): The angular velocity in the y direction in rad/s.
            w3 (float): The angular velocity in the z direction in rad/s.
            theta (float): The pitch angle in radians.
            phi (float): The yaw angle in radians.
            psi (float): The roll angle in radians.

        Returns:
            np.array: The sensor measurements as a numpy array. [gx, gy, gz, qw, qx, qy, qz]. Gyro in rad/s, quaternion normalized.
        """

        bg1 = np.deg2rad(20.116124137171195 / 3600) # rad/s
        bg2 = np.deg2rad(28.34293605241209 / 3600) # rad/s
        bg3 = np.deg2rad(25.387227350679243 / 3600) # rad/s

        ARW1 = np.deg2rad(6.6977525311882316 / 60) # rad/sqrt(s)
        ARW2 = np.deg2rad(7.733730273458847 / 60) # rad/sqrt(s)
        ARW3 = np.deg2rad(7.020108147288731 / 60) # rad/sqrt(s)

        sigma_w1 = ARW1 / np.sqrt(self.dt)
        sigma_w2 = ARW2 / np.sqrt(self.dt)
        sigma_w3 = ARW3 / np.sqrt(self.dt)

        gw1 = w1 + bg1 + np.random.normal(0.0, sigma_w1)
        gw2 = w2 + bg2 + np.random.normal(0.0, sigma_w2)
        gw3 = w3 + bg3 + np.random.normal(0.0, sigma_w3)

        sigma_theta = np.deg2rad(0.9840317663439204 / (60)) / np.sqrt(self.dt) # rad
        sigma_phi = np.deg2rad(0.872408877638351 / (60)) / np.sqrt(self.dt) # rad

        sigma_Bx = 0.006669083462481435
        sigma_By = 0.003656134366885892
        Bx = 3.500
        By = 28.500
        sigma_psi = np.sqrt((sigma_Bx * By)**2 + (sigma_By * Bx)**2) / (Bx**2 + By**2)

        b_theta = (287.9041401059671 / (9.81 * 3600)) # rad/s (small-angle a_x/g)
        b_phi = (88.37148088944824 / (9.81 * 3600)) # rad/s (small-angle a_y/g)

        bx = 1.5945108844195712
        by = 0.8536236163450057
        b_psi = np.sqrt((bx * By)**2 + (by * Bx)**2) / (Bx**2 + By**2) / 3600 # rad/s?

        theta_m = theta + b_theta * t + np.random.normal(0.0, sigma_theta)
        phi_m = phi + b_phi * t + np.random.normal(0.0, sigma_phi)
        psi_m = psi + b_psi * t + np.random.normal(0.0, sigma_psi)

        q = self.euler_to_quat_xyz(theta_m, phi_m, psi_m)

        g = np.array([gw1, gw2, gw3, q[0], q[1], q[2], q[3]])

        return g
    

    def _f(self, t, x, u):
        # assumes you already called computeAB/computeC to refresh self.f_subs
        f = np.asarray(self.f_subs, float).reshape(-1)
        return f

    def _rk4_step(self, t, x, u):
        dt = self.dt
        k1 = self._f(t,       x,             u)
        k2 = self._f(t+dt/2., x + dt*k1/2.,  u)
        k3 = self._f(t+dt/2., x + dt*k2/2.,  u)
        k4 = self._f(t+dt,    x + dt*k3,     u)
        return x + (dt/6.)*(k1 + 2*k2 + 2*k3 + k4)
    
    def run_rk4(self, t, xhat: np.array, u: np.array):
        """Runge-Kutta 4th order integration of the state estimator recursively until the estimated apogee time is reached.
        Args:
            t (float): The current time in seconds.
            xhat (np.array): The current state estimate as a numpy array.
            u (np.array): The current input as a numpy array.
        Returns:
            np.array: The updated state estimate as a numpy array.
        """
        # loop, not recursion
        while t < self.t_estimated_apogee:
            self.computeAB(t, xhat, u)  # refresh self.f_subs
            # (optionally compute C, y, and use L for a correction later)
            xhat = self._rk4_step(t, xhat, u)

            # quaternion normalize
            qn = np.linalg.norm(xhat[6:10])
            xhat[6:10] = np.array([1.,0.,0.,0.]) if qn < 1e-12 else xhat[6:10]/qn

            # basic control law (keep small for now)
            x_des = self.x0.copy()
            x_des[2] = 0.0
            x_des[5] = float(xhat[5])
            u = np.clip(-self.K @ (xhat - x_des) + self.u0, np.deg2rad(-8), np.deg2rad(8))

            # log + advance time
            self.states.append(xhat.copy())
            self.inputs.append(u.copy())
            t += self.dt
            print(f"t: {t:.3f}")


    def test_eom(self, t: float, xhat: np.array, u: np.array):
        """Test the equations of motion by computing f_subs at the given state and input.

        Args:
            t (float): The current time in seconds.
            xhat (np.array): The current state estimate as a numpy array.
            u (np.array): The current input as a numpy array.
        """
        while t < self.t_estimated_apogee:
            print(f"t: {t:.3f}, xhat: {xhat}, u: {u}")

            self.computeAB(t, xhat, u)
            f_subs = np.array(self.f_subs, dtype=float).reshape(-1)
            xhat = xhat + f_subs * self.dt
            xhat[6:10] /= np.linalg.norm(xhat[6:10])
            u = np.clip(-self.K @ (xhat - self.x0) + self.u0, np.deg2rad(-8), np.deg2rad(8))
            self.states.append(xhat)
            self.inputs.append(u)
            t = t + self.dt
            if f_subs[5] < 0:
                print("Warning: Longitudinal velocity v3 is negative at time t =", t)
                print(f"t: {t:.3f}, xhat: {xhat}, u: {u}")


    def test_AB(self, t: float, xhat: np.array, u: np.array):
        """Test the equations of motion by computing f_subs at the given state and input.

        Args:
            t (float): The current time in seconds.
            xhat (np.array): The current state estimate as a numpy array.
            u (np.array): The current input as a numpy array.
        """
        while t < self.t_estimated_apogee:
            print(f"t: {t:.3f}, xhat: {xhat}, u: {u}")
            self.computeAB(t, xhat, u)
            self.computeC(xhat, u)
            A = np.array(self.A.n()).astype(np.float64)
            B = np.array(self.B.n()).astype(np.float64)
            C = np.array(self.C.n()).astype(np.float64)
            
            ## Control Law ##
            theta, phi, psi = self.quat_to_euler_xyz(xhat[6:10])  # Convert quaternion to Euler angles
            y = self.deriveSensorModels(t, xhat[0], xhat[1], xhat[2],
                                    theta, phi, psi)  # Simulated sensor measurements
            
            ## Add back thrust and gravity terms (differentiated to 0 in computing A) ##
            xdot = A @ xhat + B @ u + self.get_thrust_accel(t) + self.get_gravity_accel(xhat) \
                    # - self.L @ (C @ xhat - y)
            xhat = xhat + xdot * self.dt
            xhat[6:10] /= np.linalg.norm(xhat[6:10])

            # Gain scheduling based on vertical velocity
            K = self.control_law(xhat, t)
            u = np.clip(-K @ (xhat - self.x0) + self.u0, np.deg2rad(-8), np.deg2rad(8))
            self.states.append(xhat)
            self.inputs.append(u)
            t = t + self.dt
            # print(f"K: {K[0][2]}")
        
# For testing
def main():
    ## Define gain matrix ##
    Kmax_preburnout = 100 / 7e1
    Kmin_preburnout = 17.5 / 7e1

    K_max_postburnout = 85 / 6e1
    K_min_postburnout = 17.5 / 6e1

    Ks = np.array([Kmax_preburnout, Kmin_preburnout, K_max_postburnout, K_min_postburnout])  # Gain scheduling based on altitude

    ## Define initial conditions ##
    t0 = 0.0
    xhat0 = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]) # Initial state estimate
    u0 = np.array([0])
    sampling_rate = 20.0  # Hz
    dt = 1.0 / sampling_rate

    controller = Controls(Ks=Ks, dt=dt, x0=xhat0, u0=u0, t_launch_rail_clearance=0.308)
    controller.deriveEOM(post_burnout=False)
    controller.deriveEOM(post_burnout=True)
    controller.buildL(lw=5.0, lqw=1.0, lqx=2.0, lqy=2.0, lqz=2.0)
    controller.test_AB(t0, xhat0, u0)
    print(controller.As)

if __name__ == "__main__":
    main()
