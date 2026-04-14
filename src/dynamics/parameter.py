from sympy import *
import numpy as np
import pandas as pd
from typing import Callable
from dynamics.momentsforces import MomentsForces

class Parameter(MomentsForces):  

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
            
    
