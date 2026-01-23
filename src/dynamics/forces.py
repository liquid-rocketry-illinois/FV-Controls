import sympy as sp
from sympy import *
import numpy as np
import pandas as pd
from typing import Callable
from enum import Enum
import os

class Forces():
    def __init__(self):
        self.forcename = ""

        self.g = 9.81
        self.f : Matrix = None



        self.f = sum() # Sum of forces 

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