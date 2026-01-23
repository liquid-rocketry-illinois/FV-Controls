from sympy import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys

from dynamics.dynamics import Dynamics
from controls.controls import Controls

class Simulation():
    """Class to encapsulate the simulation environment, including dynamics and control systems. Sets, runs, and plots the simulation.
    
    ### Usage
        Call set_dynamics() and set_controls() to set the dynamics and control models before running the simulation.
        
    ### Methods
        set_dynamics(dynamics: Dynamics): Set the dynamics model for the simulation.
        set_controls(controls: Controls): Set the control system for the simulation.
    """
    
    def __init__(self, dynamics: Dynamics = None, controls: Controls = None):
        """Initialize the Simulation class.
        
        Attributes:
            dynamics (Dynamics): The dynamics model of the rocket.
            controls (Controls): The control system of the rocket.
            disable_controls (bool): Flag to disable controls during simulation.
        """
        self.dynamics : Dynamics = dynamics # Dynamics model
        self.controls : Controls = controls # Controls model
        self.t0 = 0.0 # Initial time
        self.disable_controls : bool = False # Flag to disable controls during simulation
        self.disable_sensors : bool = False # Flag to disable sensors during simulation
        
        ## Logging ##
        
        # Dynamics logs
        self.dynamics_states = []
        self.dynamics_times = []
        self.dynamics_aoa = []  # Angle of attack log
        
        # Controls logs
        self.controls_states = []
        self.controls_inputs = []
        self.controls_input_moments = []
        self.controls_times = []
        self.controls_aoa = []  # Angle of attack log for controls runs
        
        # CSV logging path
        self.root = Path(__file__).resolve().parents[2]  # …/FV-Controls  
        self.dynamics_path : str = None
        self.controls_path : str = None
        self.dynamics_metadata : dict = {}
        self.controls_metadata : dict = {}


    def set_dynamics(self, dynamics: Dynamics):
        """Set the dynamics model for the simulation.

        Args:
            dynamics (Dynamics): An instance of the Dynamics class.
        """
        dynamics.checkParamsSet()
        self.dynamics = dynamics
        self.dynamics_path : str = self.root / "rockets" / self.dynamics.rocket_name / "data" / "sim_output" / "dynamics"
        
        
    def set_controls(self, controls: Controls):
        """Set the control system for the simulation.

        Args:
            controls (Controls): An instance of the Controls class.
        """
        controls.checkParamsSet()
        self.controls = controls
        self.controls_path : str = self.root / "rockets" / self.controls.rocket_name / "data" / "sim_output" / "controls"
        
    
    def dynamics_step(self, t: float, xhat: np.ndarray, linearized: bool = False) -> np.ndarray:
        """Perform a single dynamics step. Choose between linearized and full nonlinear dynamics.

        Args:
            t (float): Current time.
            xhat (np.ndarray): Current state vector.
            linearized (bool): Whether to use linearized dynamics (xdot = Ax) or nonlinearized (xdot = f(x)). Default is False (nonlinearized).

        Returns:
            np.ndarray: Updated state vector after the dynamics step.
        """
        if linearized:
            A = self.dynamics.getA(t, xhat)
            xdot = A @ xhat + self.dynamics.get_gravity_accel(xhat) + self.dynamics.get_thrust_accel(t)
        else:
            xdot = self.dynamics.f_numeric(t, xhat)
        xhat = xhat + xdot * self.dynamics.dt
        xhat[6:10] /= np.linalg.norm(xhat[6:10])

        self.dynamics_states.append(xhat)
        self.dynamics_times.append(t)
        
        # AoA logging
        try:
            aoa = float(self.dynamics.get_AoA(getattr(self.dynamics, "v_wind", [0.0, 0.0]), xhat))
            self.dynamics_aoa.append(aoa)
        except Exception:
            self.dynamics_aoa.append(None)
        
        # if xhat[5] < 0:
        #     print("Warning: Longitudinal velocity v3 is negative at time t =", t)
        return xhat
    
    
    def dynamics_step_rk4(self, t: float, xhat: np.ndarray) -> np.ndarray:
        """Perform a single dynamics step using the RK4 method.

        Args:
            t (float): Current time.
            xhat (np.ndarray): Current state vector.

        Returns:
            np.ndarray: Updated state vector after the dynamics step.
        """
        dt = self.dynamics.dt
        
        k1 = self.dynamics.f_numeric(t, xhat)
        k2 = self.dynamics.f_numeric(t + dt/2, xhat + k1 * dt/2)
        k3 = self.dynamics.f_numeric(t + dt/2, xhat + k2 * dt/2)
        k4 = self.dynamics.f_numeric(t + dt, xhat + k3 * dt)
        
        xhat = xhat + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        xhat[6:10] /= np.linalg.norm(xhat[6:10])

        self.dynamics_states.append(xhat)
        self.dynamics_times.append(t)
        if xhat[5] < 0:
            print("Warning: Longitudinal velocity v3 is negative at time t =", t)
            
        return xhat


    def controls_step(self, t: float, xhat: np.ndarray, u: np.ndarray, x: np.ndarray = None) -> tuple:
        """Perform a single simulation step, updating the state based on dynamics and control inputs.
        Forward Euler method with ***linearized dynamics*** :math:`(xdot=Ax+Bu-L(Cx-y))`.
        Enabling IREC compliance within the Controls object will disable controls during motor burn and
            override/automate disable_controls flag.

        Args:
            t (float): Current time.
            xhat (np.ndarray): Current state vector.
            u (np.ndarray): Control input vector.
            x (np.ndarray): True state vector for sensor measurements. Not used if sensors are disabled.
        Returns:
            tuple: Updated state and input vectors after the simulation step.
        """
        if self.controls.IREC_COMPLIANT:
            # Disable controls during motor burn for IREC compliance
            if t < self.controls.t_motor_burnout:
                self.disable_controls = True
            else:
                self.disable_controls = False

        if not self.disable_controls:
        # If controls are enabled, compute control input using state feedback
            K = self.controls.K(t, xhat)
            # Subtract self.controls.x0 because it is the desired angular state (No pitch, yaw, roll rates)
            u = np.clip(-K @ (xhat - self.controls.x0) + u, -self.controls.max_input, self.controls.max_input)
        else:
            # If controls are disabled, use zero control input
            u = np.zeros_like(u)
        
        # Get linearized dynamics matrices
        A, B = self.controls.get_AB(t, xhat, u)
        
        ## Add back thrust and gravity terms (differentiated to 0 in computing A) ##
        xdot = A @ xhat + B @ u + self.controls.get_thrust_accel(t) + self.controls.get_gravity_accel(xhat) \
                
        if not self.disable_sensors:
            # Get C matrix and sensor output
            C = self.controls.get_C(xhat)
            y = self.controls.sensor_model(t, x)
            xdot -= self.controls.L @ (C @ xhat - y)
        
        # Forward Euler integration
        xhat = xhat + xdot * self.controls.dt
        
        # Normalize quaternion
        xhat[6:10] /= np.linalg.norm(xhat[6:10])

        self.controls_states.append(xhat)
        self.controls_inputs.append(u)
        self.controls_times.append(t)
        
        # AoA logging
        try:
            aoa = float(self.controls.get_AoA(getattr(self.controls, "v_wind", [0.0, 0.0]), xhat))
            self.controls_aoa.append(aoa)
        except Exception:
            self.controls_aoa.append(None)
        
        return xhat, u
    

    def run_dynamics_simulation(self, rk4 : bool = False, file_name : str = None, show_progress: bool = True, linearized: bool = False):
        """Run the dynamics simulation until t_estimated_apogee. Uses either RK4 or Forward Euler integration. Default is Forward Euler.
        
        Args:
            rk4 (bool): Whether to use RK4 integration method. Default is False (Forward Euler).
            file_name (str): File name to save the dynamics data as CSV after simulation. Don't need to include .csv extension.
            show_progress (bool): Whether to display a progress bar during the run. Default is True.
            linearized (bool): Whether to use linearized dynamics (xdot = Ax) or nonlinearized (xdot = f(x)). Default is False (nonlinearized).
        """
        if rk4 is True and linearized is True:
            raise ValueError("RK4 integration cannot be used with linearized dynamics. Please set linearized to False when using RK4.")
                
        if file_name is None:
            raise ValueError("Please provide a file name to save the dynamics data as CSV after simulation.")

        output_path = self.dynamics_path / f"{file_name}.csv"

        self.reset_dynamics_logs()

        t = self.t0
        xhat = self.dynamics.x0.copy()
        total_time = self.dynamics.t_estimated_apogee

        while t < total_time:
            if (t > self.dynamics.t_launch_rail_clearance and xhat[5] < 0.0):
                break
            if rk4:
                xhat = self.dynamics_step_rk4(t, xhat)
            else:
                xhat = self.dynamics_step(t, xhat, linearized=linearized)
            t += self.dynamics.dt
            if show_progress:
                self._update_progress_bar(t, total_time, prefix="Dynamics ")
            
        self.save_dynamics_to_csv(output_path)
        print(f"\nDynamics data saved to: {output_path}")

    
    def run_controls_simulation(self, file_name : str = None, log_controls_moments: bool = True, show_progress: bool = True):
        """Run the control simulation until t_estimated_apogee.
        Args:
            log_controls_moments (bool): Whether to log control moments at each step. Default is True.
            file_name (str): File name to save the controls data as CSV after simulation. If None, does not save. Don't need to include .csv extension.
            show_progress (bool): Whether to display a progress bar during the run. Default is True.
        """
        if file_name is None:
            raise ValueError("Please provide a file name to save the controls data as CSV after simulation.")

        output_path : Path = self.controls_path / f"{file_name}.csv"
        # If path doesn't exist, create it
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)

        self.reset_controls_logs()

        t = self.t0
        xhat = self.controls.x0.copy()
        u = self.controls.u0.copy()
        total_time = self.controls.t_estimated_apogee
        
        while t < total_time:
            if (t > self.controls.t_launch_rail_clearance and xhat[5] < 0.0):
                break
            xhat, u = self.controls_step(t, xhat, u)
            if log_controls_moments:
                moments = self.controls.M_controls_func(xhat, u)
                self.controls_input_moments.append(np.array(moments, dtype=float).reshape(-1))
            t += self.controls.dt
            if show_progress:
                self._update_progress_bar(t, total_time, prefix="Controls ")

        self.save_controls_to_csv(output_path)
        print(f"\nControls data saved to: {output_path}")


    def save_dynamics_to_csv(self, file_path: str):
        """Save the logged dynamics data to a CSV file.

        Args:
            file_path (str): File path to save the dynamics data as CSV.
        """
        state_cols = self.var_list_to_str(self.dynamics.state_vars)
        dynamics_df = pd.DataFrame(self.dynamics_states, columns=state_cols)
        dynamics_df.insert(0, 'time', self.dynamics_times)
        if len(self.dynamics_aoa) == len(self.dynamics_times):
            dynamics_df['AoA'] = self.dynamics_aoa

        dynamics_df.to_csv(file_path, index=False)
        self._append_metadata_comment(file_path, "motor_burnout_time", getattr(self.dynamics, "t_motor_burnout", None))
        self._append_metadata_comment(file_path, "launch_rail_clearance_time", getattr(self.dynamics, "t_launch_rail_clearance", None))


    def save_controls_to_csv(self, file_path: str):
        """Save the logged controls data to a CSV file.

        Args:
            file_path (str): File path to save the controls data as CSV.
        """
        state_cols = self.var_list_to_str(self.controls.state_vars)
        input_cols = self.var_list_to_str(self.controls.input_vars)
        controls_df = pd.DataFrame(self.controls_states, columns=state_cols)
        controls_df.insert(0, 'time', self.controls_times)
        controls_inputs_df = pd.DataFrame(self.controls_inputs, columns=input_cols)
        controls_df = pd.concat([controls_df, controls_inputs_df], axis=1)
        if len(self.controls_aoa) == len(self.controls_times):
            controls_df['AoA'] = self.controls_aoa
        if len(self.controls_input_moments) > 0:
            moments_arr = np.vstack([np.array(m, dtype=float).reshape(-1) for m in self.controls_input_moments])
            controls_moments_df = pd.DataFrame(moments_arr, columns=['M1', 'M2', 'M3'])
            controls_df = pd.concat([controls_df, controls_moments_df], axis=1)
        controls_df.to_csv(file_path, index=False, mode='w')
        self._append_metadata_comment(file_path, "motor_burnout_time", getattr(self.controls, "t_motor_burnout", None))
        self._append_metadata_comment(file_path, "launch_rail_clearance_time", getattr(self.controls, "t_launch_rail_clearance", None))
        

    def read_dynamics_from_csv(self, file_name: str = None, file_path: str = None):
        """Read dynamics data from a CSV file and populate the logged states and times. If file_path is provided, it overrides file_name.

        Args:
            file_name (str): File name to read the dynamics data from CSV.
            file_path (str): Optional full file path to read the dynamics data from CSV. If provided, overrides file_name.
        Returns:
            states (list): List of state vectors read from the CSV file.
            times (list): List of time values read from the CSV file.
        """
        path = None
        if file_path is not None:
            path = file_path
        elif file_name is not None:
            path = self.dynamics_path / f"{file_name}.csv"
        else:
            raise ValueError("Please provide either file_name or file_path to read the dynamics data from CSV.")
        
        df = pd.read_csv(path, comment='#')
        times = df['time'].to_numpy() if 'time' in df.columns else None
        # Use whatever headers exist in the CSV (except time/AoA) to avoid depending on a Dynamics instance.
        state_cols = [col for col in df.columns if col not in ('time', 'AoA')]
        states = df[state_cols].to_numpy()
        self.dynamics_metadata = self._read_metadata_comments(path)
            
        return times, states
    

    def read_controls_from_csv(self, file_name: str = None, file_path: str = None):
        """Read controls data from a CSV file and populate the logged states, inputs, and times.

        Args:
            file_name (str): File name to read the controls data from CSV.
            file_path (str): Optional full file path to read the controls data from CSV. If provided, overrides file_name.
        Returns:
            states (list): List of state vectors read from the CSV file.
            inputs (list): List of input vectors read from the CSV file.
            times (list): List of time values read from the CSV file.
        """
        path = None
        if file_path is not None:
            path = file_path
        elif file_name is not None:
            path = self.controls_path / f"{file_name}.csv"
        else:
            raise ValueError("Please provide either file_name or file_path to read the controls data from CSV.")

        df = pd.read_csv(path, comment='#')
        times = df['time'].to_numpy() if 'time' in df.columns else None

        moment_cols = [col for col in ['M1', 'M2', 'M3'] if col in df.columns]
        if len(moment_cols) == 3:
            moments = df[moment_cols].to_numpy()
        else:
            moments = None

        # Prefer explicit columns from controls object if available, otherwise infer from headers.
        input_cols = []
        state_cols = []
        if getattr(self, "controls", None) is not None:
            if getattr(self.controls, "input_vars", None):
                input_cols = [c for c in self.var_list_to_str(self.controls.input_vars) if c in df.columns]
            if getattr(self.controls, "state_vars", None):
                state_cols = [c for c in self.var_list_to_str(self.controls.state_vars) if c in df.columns]

        if not input_cols:
            # Heuristic: treat columns commonly used for inputs (u*, zeta, delta) as control inputs.
            candidate_inputs = [c for c in df.columns if c not in ['time'] + moment_cols]
            input_cols = [c for c in candidate_inputs if c.lower().startswith('u') or c.lower() in {'zeta', 'delta'} or c.lower().startswith('zeta') or c.lower().startswith('delta')]

        if not state_cols:
            excluded = set(['time'] + input_cols + moment_cols + ['AoA'])
            state_cols = [c for c in df.columns if c not in excluded]

        states = df[state_cols].to_numpy() if state_cols else None
        inputs = df[input_cols].to_numpy() if input_cols else None
        self.controls_metadata = self._read_metadata_comments(path)
            
        return times, states, inputs, moments


    def plot_dynamics(
        self,
        ang_vel: bool = True,
        lin_vel: bool = True,
        attitude: bool = True,
        quaternion : bool = False,
        file_name : str = None,
        file_path : str = None
    ):
        """Plot the state variables over time. Plots angular velocity, linear velocity, and attitude quaternion on separate subplots.\
            Choose to read from CSV file or locally logged data stored in the Dynamics object.
        
        Args:
            ang_vel (bool): Whether to plot angular velocity.
            lin_vel (bool): Whether to plot linear velocity.
            attitude (bool): Whether to plot attitude quaternion.
            file_name (str): File name to read the CSV file if reading from CSV. Don't need to include .csv extension. Same as used in save_dynamics_to_csv(). If None, uses locally logged data.
            file_path (str): File path to read the CSV file if reading from CSV. If None, uses locally logged data.
        """
        states = None
        times = None
        if file_name is not None or file_path is not None:
            times, states = self.read_dynamics_from_csv(file_name, file_path)
            if len(states) == 0:
                raise ValueError("No dynamics data to plot. Please run the dynamics simulation first using run_dynamics_simulation() and save to CSV using save_dynamics_to_csv().")
        else:
            if len(self.dynamics_states) == 0:
                raise ValueError("No dynamics data to plot. Please run the dynamics simulation first using run_dynamics_simulation().")
            states = np.array(self.dynamics_states)
            times = np.array(self.dynamics_times)
            
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        if ang_vel:
            axs[0].plot(times, states[:, 0], label='ω1')
            axs[0].plot(times, states[:, 1], label='ω2')
            axs[0].plot(times, states[:, 2], label='ω3')
            axs[0].set_title('Angular Velocity vs Time')
            axs[0].set_xlabel('Time (s)')
            axs[0].set_ylabel('Angular Velocity (rad/s)')
            axs[0].legend()
        if lin_vel:
            axs[1].plot(times, states[:, 3], label='v1')
            axs[1].plot(times, states[:, 4], label='v2')
            axs[1].plot(times, states[:, 5], label='v3')
            axs[1].set_title('Linear Velocity vs Time')
            axs[1].set_xlabel('Time (s)')
            axs[1].set_ylabel('Linear Velocity (m/s)')
            axs[1].legend()
        if attitude:
            if quaternion:
                axs[2].plot(times, states[:, 6], label='qw')
                axs[2].plot(times, states[:, 7], label='qx')
                axs[2].plot(times, states[:, 8], label='qy')
                axs[2].plot(times, states[:, 9], label='qz')
                axs[2].set_title('Attitude Quaternion vs Time')
                axs[2].set_ylabel('Quaternion Components')
                axs[2].set_xlabel('Time (s)')
            else:
                euler_angles = np.array([self.quat_to_euler_xyz(q) for q in states[:, 6:10]])
                axs[2].plot(times, np.rad2deg(euler_angles[:, 0]), label='Pitch (deg)')
                axs[2].plot(times, np.rad2deg(euler_angles[:, 1]), label='Yaw (deg)')
                axs[2].plot(times, np.rad2deg(euler_angles[:, 2]), label='Roll (deg)')
                axs[2].set_title('Attitude Euler Angles vs Time')
            axs[2].legend()
            
        # Plot vertical lines using metadata when available
        burnout_time = self.dynamics_metadata.get("motor_burnout_time") if file_name is not None or file_path is not None else getattr(self.dynamics, "t_motor_burnout", None)
        launch_clear_time = self.dynamics_metadata.get("launch_rail_clearance_time") if file_name is not None or file_path is not None else getattr(self.dynamics, "t_launch_rail_clearance", None)
        for ax in axs:
            if launch_clear_time is not None:
                ax.axvline(x=float(launch_clear_time), color='b', linestyle='--', label="Launch Rail Clearance")
            if burnout_time is not None:
                ax.axvline(x=float(burnout_time), color='r', linestyle='--', label='Motor Burnout')
            if launch_clear_time is not None or burnout_time is not None:
                ax.legend()
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        fig.tight_layout()
        return fig, axs


    def plot_controls(
        self,
        ang_vel: bool = True,
        lin_vel: bool = True,
        attitude: bool = True,
        quaternion: bool = False,
        control_inputs: bool = True,
        control_moments: bool = False,
        file_name : str = None,
        file_path : str = None
    ):
        """Plot the state variables and control inputs over time. Plots angular velocity, linear velocity, attitude quaternion, and control inputs on separate subplots.
        Args:
            ang_vel (bool): Whether to plot angular velocity.
            lin_vel (bool): Whether to plot linear velocity.
            attitude (bool): Whether to plot attitude quaternion.
            quaternion (bool): Whether to plot attitude quaternions or Euler angles. Default is False (plots in Euler angles).
            control_inputs (bool): Whether to plot control inputs.
            control_moments (bool): Whether to log control moments at each step. Default is False.
            file_name (str): File name to read the CSV file if reading from CSV. Don't need to include .csv extension. Same as used in save_controls_to_csv(). If None, uses locally logged data.
            file_path (str): File path to read the CSV file if reading from CSV. If None, uses locally logged data. Overrides file_name if both provided.
        """
        states = None
        inputs = None
        moments = None
        times = None
        if file_name is not None or file_path is not None:
            times, states, inputs, moments = self.read_controls_from_csv(file_name, file_path)
            if len(states) == 0:
                raise ValueError("No dynamics data to plot. Please run the controls simulation first using run_controls_simulation() and save to CSV using save_controls_to_csv().")
            print(f"Using controls data from {file_name}.csv for plotting.")
            burnout_time = self.controls_metadata.get("motor_burnout_time")
            launch_clear_time = self.controls_metadata.get("launch_rail_clearance_time")
        else:
            if len(self.controls_states) == 0:
                raise ValueError("No controls data to plot. Please run the controls simulation first using run_controls_simulation().")
            print("Using locally logged controls data for plotting.")
            states = np.array(self.controls_states)
            inputs = np.array(self.controls_inputs)
            moments = np.array(self.controls_input_moments)
            times = np.array(self.controls_times)
            burnout_time = getattr(self.controls, "t_motor_burnout", None)
            launch_clear_time = getattr(self.controls, "t_launch_rail_clearance", None)
            
        fig, axs = plt.subplots(5, 1, figsize=(10, 20))
        if ang_vel:
            axs[0].plot(times, states[:, 0], label='ω1')
            axs[0].plot(times, states[:, 1], label='ω2')
            axs[0].plot(times, states[:, 2], label='ω3')
            axs[0].set_title('Angular Velocity vs Time')
            axs[0].set_xlabel('Time (s)')
            axs[0].set_ylabel('Angular Velocity (rad/s)')
            axs[0].legend()
        if lin_vel:
            axs[1].plot(times, states[:, 3], label='v1')
            axs[1].plot(times, states[:, 4], label='v2')
            axs[1].plot(times, states[:, 5], label='v3')
            axs[1].set_title('Linear Velocity vs Time')
            axs[1].set_xlabel('Time (s)')
            axs[1].set_ylabel('Linear Velocity (m/s)')
            axs[1].legend()
        if attitude:
            if quaternion:
                axs[2].plot(times, states[:, 6], label='qw')
                axs[2].plot(times, states[:, 7], label='qx')
                axs[2].plot(times, states[:, 8], label='qy')
                axs[2].plot(times, states[:, 9], label='qz')
                axs[2].set_title('Attitude Quaternion vs Time')
                axs[2].set_ylabel('Quaternion Components')
            else:
                # Convert quaternions to Euler angles for plotting
                euler_angles = np.array([self.quat_to_euler_xyz(q) for q in states[:, 6:10]])
                axs[2].plot(times, np.rad2deg(euler_angles[:, 0]), label='Pitch (deg)')
                axs[2].plot(times, np.rad2deg(euler_angles[:, 1]), label='Yaw (deg)')
                axs[2].plot(times, np.rad2deg(euler_angles[:, 2]), label='Roll (deg)')
                axs[2].set_title('Attitude Euler Angles vs Time')
                axs[2].set_ylabel('Euler Angles (degrees)')
            axs[2].set_xlabel('Time (s)')
            axs[2].legend()
        if control_inputs:
            axs[3].plot(times, np.rad2deg(inputs[:, 0]), label='u1 (deg)')
            axs[3].set_title('Control Inputs vs Time')
            axs[3].set_xlabel('Time (s)')
            axs[3].set_ylabel('Control Inputs (degrees)')
            axs[3].legend()
        if control_moments:
            axs[4].plot(times, moments[:, 0], label='M1 (Nm)')
            axs[4].plot(times, moments[:, 1], label='M2 (Nm)')
            axs[4].plot(times, moments[:, 2], label='M3 (Nm)')
            axs[4].set_title('Control Moments vs Time')
            axs[4].set_xlabel('Time (s)')
            axs[4].set_ylabel('Control Inputs (degrees) / Moments (Nm)')
            axs[4].legend()

        # Plot vertical lines using metadata when available
        for ax in axs:
            if launch_clear_time is not None:
                ax.axvline(x=float(launch_clear_time), color='b', linestyle='--', label="Launch Rail Clearance")
            if burnout_time is not None:
                ax.axvline(x=float(burnout_time), color='r', linestyle='--', label='Motor Burnout')
            if launch_clear_time is not None or burnout_time is not None:
                ax.legend()
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        fig.tight_layout()
        return fig, axs
        
    
    def compare_controls_dynamics(
        self,
        ang_vel: bool = True,
        lin_vel: bool = True,
        attitude: bool = True,
        quaternion: bool = False,
        control_inputs: bool = True,
        control_moments : bool = True,
        controls_file_name : str = None,
        controls_file_path : str = None,
        dynamics_file_name : str = None,
        dynamics_file_path : str = None
    ):
        fig, axs = self.plot_controls(
            ang_vel=ang_vel,
            lin_vel=lin_vel,
            attitude=attitude,
            quaternion=quaternion,
            control_inputs=control_inputs,
            control_moments=control_moments,
            file_name=controls_file_name,
            file_path=controls_file_path
        )
        print(f"Using dynamics data from {dynamics_file_name}.csv for plotting.")
        states = None
        times = None
        if dynamics_file_name is not None or dynamics_file_path is not None:
            times, states = self.read_dynamics_from_csv(dynamics_file_name, dynamics_file_path)
            if len(states) == 0:
                raise ValueError("No dynamics data to plot. Please run the dynamics simulation first using run_dynamics_simulation() and save to CSV using save_dynamics_to_csv().")
        else:
            if len(self.dynamics_states) == 0:
                raise ValueError("No dynamics data to plot. Please run the dynamics simulation first using run_dynamics_simulation().")
            states = np.array(self.dynamics_states)
            times = np.array(self.dynamics_times)
        
        # Some states are commented out to reduce clutter in the plots
        if ang_vel:
            # axs[0].plot(times, states[:, 0], label='dyn ω1')
            # axs[0].plot(times, states[:, 1], label='dyn ω2')
            axs[0].plot(times, states[:, 2], label='dyn ω3')
            axs[0].legend()
        if lin_vel:
            # axs[1].plot(times, states[:, 3], label='dyn v1')
            # axs[1].plot(times, states[:, 4], label='dyn v2')
            axs[1].plot(times, states[:, 5], label='dyn v3')
            axs[1].legend()
        if attitude:
            if quaternion:
                axs[2].plot(times, states[:, 6], label='qw')
                axs[2].plot(times, states[:, 7], label='qx')
                axs[2].plot(times, states[:, 8], label='qy')
                axs[2].plot(times, states[:, 9], label='qz')
            else:
                euler_angles = np.array([self.controls.quat_to_euler_xyz(q) for q in states[:, 6:10]])
                # axs[2].plot(times, np.rad2deg(euler_angles[:, 0]), label='dyn Pitch (deg)')
                # axs[2].plot(times, np.rad2deg(euler_angles[:, 1]), label='dyn Yaw (deg)')
                axs[2].plot(times, np.rad2deg(euler_angles[:, 2]), label='dyn Roll (deg)')
            axs[2].legend()
            
        plt.tight_layout()
        plt.show()
            
        return fig, axs

        
    
    def compare_dyn_or(self, or_file_path : str, dyn_file_name: str = None, dyn_file_path: str = None):
        """Compare simulation results with OpenRocket data.
        Ensure that the OpenRocket data file contains the required columns:
        Time (s), Vertical Speed (m/s), Total velocity (m/s), Roll rate (°/s).

        Args:
            dyn_file_name (str): File name to read the dynamics CSV file. Don't need to include .csv extension.
            dyn_file_path (str): Path to the dynamics CSV file. If provided, overrides dyn_file_name.
            or_file_path (str): Path to the OpenRocket CSV data file.
        """
        # Load OpenRocket data
        or_data = pd.read_csv(or_file_path)
        or_time = or_data['# Time (s)']
        or_v3 = or_data['Vertical velocity (m/s)']
        or_vmag = or_data['Total velocity (m/s)']
        or_w3 = or_data['Roll rate (°/s)'] * (np.pi / 180.0) # Convert to rad/s
        
        apogee = or_v3 >= 0
        or_time = or_time[apogee]
        or_v3 = or_v3[apogee]
        or_vmag = or_vmag[apogee]
        or_w3 = or_w3[apogee]
        
        # Extract simulation altitude data
        sim_times, sim_states = self.read_dynamics_from_csv(dyn_file_name, dyn_file_path)
        sim_v3 = sim_states[:, 5]
        sim_vmag = np.linalg.norm(sim_states[:, 3:6], axis=1)
        sim_w3 = sim_states[:, 2]
        
        # Plot comparisons on subplots
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        axs[0].plot(or_time, or_v3, label='OpenRocket v3')
        axs[0].plot(sim_times, sim_v3, label='Simulation v3')
        axs[0].set_title('Vertical Velocity Comparison')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Vertical Velocity (m/s)')
        axs[0].legend()

        axs[1].plot(or_time, or_vmag, label='OpenRocket ||v||')
        axs[1].plot(sim_times, sim_vmag, label='Simulation ||v||')
        axs[1].set_title('Total Velocity Comparison')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Total Velocity (m/s)')
        axs[1].legend()

        axs[2].plot(or_time, or_w3, label='OpenRocket w3')
        axs[2].plot(sim_times, sim_w3, label='Simulation w3')
        axs[2].set_title('Roll Rate Comparison')
        axs[2].set_xlabel('Time (s)')
        axs[2].set_ylabel('Roll Rate (rad/s)')
        axs[2].legend()
        
        # Plot vertical lines using metadata when available
        burnout_time = self.dynamics_metadata.get("motor_burnout_time") if dyn_file_name is not None or dyn_file_path is not None else getattr(self.dynamics, "t_motor_burnout", None)
        launch_clear_time = self.dynamics_metadata.get("launch_rail_clearance_time") if dyn_file_name is not None or dyn_file_path is not None else getattr(self.dynamics, "t_launch_rail_clearance", None)
        for ax in axs:
            if launch_clear_time is not None:
                ax.axvline(x=float(launch_clear_time), color='b', linestyle='--', label="Launch Rail Clearance")
            if burnout_time is not None:
                ax.axvline(x=float(burnout_time), color='r', linestyle='--', label='Motor Burnout')
            if launch_clear_time is not None or burnout_time is not None:
                ax.legend()
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        plt.tight_layout()
        plt.show()
        
        print("If results don't match closely, ensure that the rocket configuration and simulation parameters are consistent between OpenRocket and this simulation.")
        
        return fig, axs


    def plot_aoa(self, source: str = "dynamics", file_name: str = None):
        """Plot angle of attack vs time from logged data or a CSV.

        Args:
            source (str): 'dynamics' or 'controls' to select which log to plot.
            file_name (str): Optional CSV name (no extension) to read AoA from; expects an 'AoA' column and 'time'.
                             If None, uses in-memory logs.
        """
        source = source.lower()
        if source not in {"dynamics", "controls"}:
            raise ValueError("source must be 'dynamics' or 'controls'")

        if file_name:
            base_path = self.dynamics_path if source == "dynamics" else self.controls_path
            df = pd.read_csv(base_path / f"{file_name}.csv", comment="#")
            if "AoA" not in df.columns:
                raise ValueError(f"'AoA' column not found in {file_name}.csv")
            times = df["time"].to_numpy() if "time" in df.columns else np.arange(len(df["AoA"]))
            aoa = df["AoA"].to_numpy()
        else:
            if source == "dynamics":
                if not self.dynamics_times or not self.dynamics_aoa:
                    raise ValueError("No dynamics AoA data logged. Run the dynamics simulation first.")
                times = np.array(self.dynamics_times)
                aoa = np.array(self.dynamics_aoa)
            else:
                if not self.controls_times or not self.controls_aoa:
                    raise ValueError("No controls AoA data logged. Run the controls simulation first.")
                times = np.array(self.controls_times)
                aoa = np.array(self.controls_aoa)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(times, aoa, label=f"AoA ({source})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("AoA (rad)")
        ax.set_title("Angle of Attack vs Time")
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        fig.tight_layout()
        return fig, ax


    def reset_logs(self):
        """Reset the logged states and times for dynamics and controls."""
        self.reset_dynamics_logs()
        self.reset_controls_logs()


    def reset_dynamics_logs(self):
        """Reset logged dynamics states and times."""
        self.dynamics_states = []
        self.dynamics_times = []


    def reset_controls_logs(self):
        """Reset logged controls states, inputs, moments, and times."""
        self.controls_states = []
        self.controls_inputs = []
        self.controls_input_moments = []
        self.controls_times = []


    def _update_progress_bar(self, current_time: float, total_time: float, prefix: str = "", bar_len: int = 30):
        """Render an in-place progress bar for long-running loops."""
        frac = min(max(current_time / total_time, 0.0), 1.0) if total_time > 0 else 1.0
        filled = int(round(bar_len * frac))
        bar = '■' * filled + '□' * (bar_len - filled)

        sys.stdout.write(
            f"\r{prefix}{bar} {frac*100:5.1f}%  t = {current_time:6.2f}/{total_time:.2f} s"
        )
        sys.stdout.flush()
        if frac >= 1.0:
            sys.stdout.write("\n")


    @staticmethod
    def var_list_to_str(var_list: list) -> list:
        """Convert a list of sympy variables to their string representations.

        Args:
            var_list (list): List of sympy variables.
        Returns:
            list: List of string representations of the variables.
        """
        return [str(var) for var in var_list]


    @staticmethod
    def _read_metadata_comments(file_path: Path) -> dict:
        """Read metadata lines (prefixed with '#') from a CSV file."""
        metadata = {}
        with open(file_path, 'r') as f:
            for line in f:
                if not line.startswith('#'):
                    continue
                stripped = line.lstrip('#').strip()
                if not stripped:
                    continue
                parts = stripped.split(',', 1)
                if len(parts) != 2:
                    continue
                key, value = parts[0].strip(), parts[1].strip()
                try:
                    metadata[key] = float(value)
                except ValueError:
                    metadata[key] = value
        return metadata


    @staticmethod
    def _append_metadata_comment(file_path: Path, key: str, value):
        """Append a metadata comment line (# key,value) to a CSV if value is provided."""
        if value is None:
            return
        with open(file_path, 'a') as f:
            f.write(f"# {key},{value}\n")
    
    
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
