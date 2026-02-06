import sympy as sp
from sympy import *
import numpy as np
import pandas as pd
from typing import Callable
from enum import Enum
import os

# from dynamics.thrust import Thrust
# from dynamics.variables import Variables

class MomentsForces():
    def __init__(self):
        # super().__init__()

        self.M = None

        self.params : list = None

        # Environmental parameters
        self.v_wind : list = [0.0, 0.0]
        self.rho : float = 1.225 # Air density kg/m^3
        self.g : float = 9.81 # Gravitational acceleration m/s^2

        self.F : Matrix = None # Forces matrix
        self.M : Matrix = None # Moments matrix

        self.f : Matrix = None
        self.state_vars : list = None

        # Thrust curve data
        self.thrust_times : np.ndarray = None
        self.thrust_forces : np.ndarray = None

    def set_symbols(self):
        """Set the symbolic variables for the dynamics equations.
        """
        w1, w2, w3, v1, v2 = symbols('w_1 w_2 w_3 v_1 v_2', real = True) # Angular and linear velocities
        v3 = symbols('v_3', real = True, positive = True) # Longitudinal velocity, assumed positive during flight
        qw, qx, qy, qz = symbols('q_w q_x q_y q_z', real = True) # Quaternion components
        # I1, I2, I3 = symbols('I_1 I_2 I_3', real = True, positive = True) # Moments of inertia
        T1, T2, T3 = symbols('T_1 T_2 T_3', real = True, positive = True) # Thrusts
        mass, rho, d, g, CG = symbols('m rho d g CG', real = True, positive = True) # Mass, air density, diameter, gravity, center of gravity
        delta = symbols('delta', real = True) # Fin cant angle
        C_d = symbols('C_d', real = True, positive = True) # Drag coefficient
        # Cnalpha_fin, Cnalpha_rocket = symbols('C_n_alpha_fin C_n_alpha_rocket', real = True, positive = True) # Fin and rocket normal force coefficient derivatives
        # Cr, Ct, s = symbols('Cr Ct s', real = True, positive = True) # Fin root chord, tip chord, span
        # N = symbols('N', real = True, positive = True) # Number of fins
        t_sym = symbols('t', real = True, positive = True) # Time symbol for Heaviside function
        # v_wind1, v_wind2 = symbols('v_wind_1 v_wind_2', real = True) # Wind velocity components

        self.state_vars = [w1, w2, w3, v1, v2, v3, qw, qx, qy, qz]
        # self.params = [I1, I2, I3, T1, T2, T3, mass, rho, d, g, CG, delta, C_d, Cnalpha_fin, Cnalpha_rocket, Cr, Ct, s, N, v_wind1, v_wind2]
        
        self.params = [None, None, None, T1, T2, T3, mass, rho, d, g, CG, delta, C_d, None, None, None, None, None, N, None, None]

        # self.params = [I1, I2, I3, T1, T2, T3, mass, rho, d, g, CG, delta, C_d, Cnalpha_fin, Cnalpha_rocket, Cr, Ct, s, N]
        self.t_sym = t_sym # Time when rocket leaves the launch rail

    def get_thrust_accel(self, t: float) -> np.ndarray:
        """Get the thrust acceleration at time t. Does this by dividing thrusts by m.

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
    
    def setThrustCurve(self, thrust_times: np.ndarray, thrust_forces: np.ndarray):
        """Set the thrust curve data.

        Args:
            thrust_times (np.ndarray): Array of time points in seconds.
            thrust_force (np.ndarray): Array of thrust values in Newtons corresponding to the time points.
        """
        self.thrust_times = thrust_times
        self.thrust_forces = thrust_forces

    def get_thrust(self, t: float) -> Matrix:
        """Get the thrust for the rocket at time t.

        Args:
            t (float): The time in seconds.

        Returns:
            dict: A dictionary containing inertia, mass, CG, and thrust at time t. WRONG. We return T = [0.,0.,thrust]
        """

        T = Matrix([0., 0., 0.])  # N
        motor_burnout = t > self.t_motor_burnout
        if not motor_burnout:
            T[2] = np.interp(t, self.thrust_times, self.thrust_forces) # Thrust acting in z direction
            
        return T

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
    

if __name__ == "__main__":
    x = MomentsForces()
    x.set_moments()
    x.set_forces()
    print(x.get_forces(),x.get_moments())
    print("ssdfds")

