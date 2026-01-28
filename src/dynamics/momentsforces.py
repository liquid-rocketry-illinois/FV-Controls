import sympy as sp
from sympy import *
import numpy as np
import pandas as pd
# from typing import Callable
from enum import Enum
import os

class MomentsForces:
    def __init__(self):
        self.M = None

        # Environmental parameters
        self.v_wind : list = [0.0, 0.0]
        self.rho : float = 1.225 # Air density kg/m^3
        self.g : float = 9.81 # Gravitational acceleration m/s^2


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
