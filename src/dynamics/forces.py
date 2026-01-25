import sympy as sp
from sympy import *
import numpy as np
import pandas as pd
from typing import Callable
from enum import Enum
import os

class Forces():
    def __init__(self):
        """Initialize the Forces class. Rocket body axis is aligned with z-axis.
        """
        self.forcename = ""

        self.g = 9.81
        self.f : Matrix = None
        
        self.f : Matrix = None
        self.state_vars : list = None
        self.params : list = None
        self.f_subs_params : Matrix = None
        self.f_subs_full : Matrix = None
        self.dt : float = None
        self.x0 : np.ndarray = None
        self.t0 : float = 0.0
        self.t_sym : Symbol = None

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

        self.Cnalpha_fin : float = None # Normal force coefficient normalized by angle of attack for 1 fin

        self.F : Matrix = None # Forces matrix

        self.f_sum = sum() # Sum of forces 
 
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
