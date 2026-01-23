from sympy import *
import numpy as np
import pandas as pd
from typing import Callable
from dynamics import *

class EOM:
    def __init__(self):
        """Initialize the Dynamics class. Rocket body axis is aligned with z-axis.

        Args:
            t_estimated_apogee (float): Estimated time until apogee in seconds.
            dt (float): Time step for simulation in seconds.
            x0 (np.ndarray): Initial state vector.
        """

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
        self.Cnalpha_fin : float = None # Normal force coefficient normalized by angle of attack for 1 fin
        self.delta : float = None # Fin cant angle in degrees
        
        # Thrust curve data
        self.thrust_times : np.ndarray = None
        self.thrust_forces : np.ndarray = None

        # Environmental parameters
        self.v_wind : list = [0.0, 0.0]
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
        self._f_numeric = None  # Cached lambdified EOM
        self._A_numeric = None  # Cached lambdified Jacobian of f wrt state
    
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
            dynamics.set_symbols
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
        print(f)

EOM.define_eom