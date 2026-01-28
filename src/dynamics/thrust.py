from sympy import *
import numpy as np
import pandas as pd
from typing import Callable

class Thrust:
    def __init__(self):
        
        
        # Thrust curve data
        self.thrust_times : np.ndarray = None
        self.thrust_forces : np.ndarray = None
    
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
            dict: A dictionary containing inertia, mass, CG, and thrust at time t.
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