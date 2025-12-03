from sympy import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from dynamics import Dynamics
from controls import Controls

class Simulation():
    """Class to encapsulate the simulation environment, including dynamics and control systems. Sets, runs, and plots the simulation.
    
    ### Usage
        Call set_dynamics() and set_controls() to set the dynamics and control models before running the simulation.
        
    ### Methods
        set_dynamics(dynamics: Dynamics): Set the dynamics model for the simulation.
        set_controls(controls: Controls): Set the control system for the simulation.
    """
    
    def __init__(self):
        """Initialize the Simulation class.
        
        Attributes:
            dynamics (Dynamics): The dynamics model of the rocket.
            controls (Controls): The control system of the rocket.
            disable_controls (bool): Flag to disable controls during simulation.
        """
        self.dynamics : Dynamics = None # Dynamics model
        self.controls : Controls = None # Control system model
        self.t0 = 0.0 # Initial time
        self.disable_controls : bool = False # Flag to disable controls during simulation
        self.disable_sensors : bool = False # Flag to disable sensors during simulation
        
        ## Logging ##
        
        # Dynamics logs
        self.dynamics_states = []
        self.dynamics_times = []
        
        # Controls logs
        self.controls_states = []
        self.controls_inputs = []
        self.controls_input_moments = []
        self.controls_times = []


    def set_dynamics(self, dynamics: Dynamics):
        """Set the dynamics model for the simulation.

        Args:
            dynamics (Dynamics): An instance of the Dynamics class.
        """
        dynamics.checkParamsSet()
        self.dynamics = dynamics
        
        
    def set_controls(self, controls: Controls):
        """Set the control system for the simulation.

        Args:
            controls (Controls): An instance of the Controls class.
        """
        controls.checkParamsSet()
        self.controls = controls
        
    
    def dynamics_step(self, t: float, xhat: np.ndarray) -> np.ndarray:
        """Perform a single dynamics step. Forward Euler method (no linearization).

        Args:
            t (float): Current time.
            xhat (np.ndarray): Current state vector.

        Returns:
            np.ndarray: Updated state vector after the dynamics step.
        """
        
        print(f"t: {t:.3f}, xhat: {xhat}")

        self.dynamics.set_f(t, xhat)
        f_subs = np.array(self.dynamics.f_subs_full, dtype=float).reshape(-1)
        xhat = xhat + f_subs * self.dynamics.dt
        xhat[6:10] /= np.linalg.norm(xhat[6:10])

        self.dynamics_states.append(xhat)
        self.dynamics_times.append(t)
        if f_subs[5] < 0:
            print("Warning: Longitudinal velocity v3 is negative at time t =", t)
            print(f"t: {t:.3f}, xhat: {xhat}")
            
        return xhat
    
    
    def dynamics_step_rk4(self, t: float, xhat: np.ndarray) -> np.ndarray:
        """Perform a single dynamics step using the RK4 method.

        Args:
            t (float): Current time.
            xhat (np.ndarray): Current state vector.

        Returns:
            np.ndarray: Updated state vector after the dynamics step.
        """
        
        print(f"t: {t:.3f}, xhat: {xhat}")

        dt = self.dynamics.dt
        
        self.dynamics.set_f(t, xhat)
        k1 = np.array(self.dynamics.f_subs_full, dtype=float).reshape(-1)
        
        self.dynamics.set_f(t + dt/2, xhat + k1 * dt/2)
        k2 = np.array(self.dynamics.f_subs_full, dtype=float).reshape(-1)
        
        self.dynamics.set_f(t + dt/2, xhat + k2 * dt/2)
        k3 = np.array(self.dynamics.f_subs_full, dtype=float).reshape(-1)
        
        self.dynamics.set_f(t + dt, xhat + k3 * dt)
        k4 = np.array(self.dynamics.f_subs_full, dtype=float).reshape(-1)
        
        xhat = xhat + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        xhat[6:10] /= np.linalg.norm(xhat[6:10])

        self.dynamics_states.append(xhat)
        self.dynamics_times.append(t)
        if xhat[5] < 0:
            print("Warning: Longitudinal velocity v3 is negative at time t =", t)
            print(f"t: {t:.3f}, xhat: {xhat}")
            
        return xhat


    def controls_step(self, t: float, xhat: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Perform a single simulation step, updating the state based on dynamics and control inputs.
        Forward Euler method with ***linearized dynamics*** :math:`(xdot=Ax+Bu-L(Cx-y))`.

        Args:
            t (float): Current time.
            xhat (np.ndarray): Current state vector.
            u (np.ndarray): Control input vector.
        Returns:
            np.ndarray: Updated state vector after the simulation step.
        """
        print(f"t: {t:.3f}, xhat: {xhat}, u: {np.rad2deg(u)}")

        # Gain scheduling based on vertical velocity
        K = self.controls.K(t, xhat)
        u = np.clip(-K @ (xhat - self.controls.x0) + self.controls.u0, -self.controls.max_input, self.controls.max_input)
        
        # Disable controls if specified
        if self.disable_controls:
            u = np.zeros_like(u)
        
        # Get linearized dynamics matrices
        A, B = self.controls.get_AB(t, xhat, u)
        
        ## Add back thrust and gravity terms (differentiated to 0 in computing A) ##
        xdot = A @ xhat + B @ u + self.controls.get_thrust_accel(t) + self.controls.get_gravity_accel(xhat) \
                
        if not self.disable_sensors:
            # Get C matrix and sensor output
            C = self.controls.get_C(xhat)
            y = self.controls.sensor_model(t, xhat)
            xdot -= self.controls.L @ (C @ xhat - y)
        
        # Forward Euler integration
        xhat = xhat + xdot * self.controls.dt
        
        # Normalize quaternion
        xhat[6:10] /= np.linalg.norm(xhat[6:10])
        
        self.controls_states.append(xhat)
        self.controls_inputs.append(u)
        self.controls_times.append(t)
        
        return xhat, u
    

    def run_dynamics_simulation(self, rk4: bool = False):
        """Run the dynamics simulation until t_estimated_apogee. Uses either RK4 or Forward Euler integration. Default is Forward Euler."""
        t = self.t0
        xhat = self.dynamics.x0.copy()

        while t < self.dynamics.t_estimated_apogee:
            if (t > self.dynamics.t_launch_rail_clearance and xhat[5] < 0.0):
                break
            if rk4:
                xhat = self.dynamics_step_rk4(t, xhat)
            else:
                xhat = self.dynamics_step(t, xhat)
            t += self.dynamics.dt

    
    def run_controls_simulation(self, log_controls_moments: bool = True):
        """Run the control simulation until t_estimated_apogee."""
        t = self.t0
        xhat = self.controls.x0.copy()
        u = self.controls.u0.copy()
        
        while t < self.controls.t_estimated_apogee:
            if (t > self.controls.t_launch_rail_clearance and xhat[5] < 0.0):
                break
            xhat, u = self.controls_step(t, xhat, u)
            if log_controls_moments:
                moments = self.controls.M_controls_func(xhat, u)
                self.controls_input_moments.append(moments)
            t += self.controls.dt


    def save_to_csv(self, filepath: str):
        """Save the logged dynamics and controls data to a CSV file.

        Args:
            filepath (str): Path to the output CSV file.
        """
        dynamics_df = pd.DataFrame(self.dynamics_states, columns=str(self.dynamics.state_vars))
        dynamics_df['time'] = self.dynamics_times
        
        controls_df = pd.DataFrame(self.controls_states, columns=str(self.controls.state_vars))
        controls_df['time'] = self.controls_times
        controls_inputs_df = pd.DataFrame(self.controls_inputs, columns=str(self.controls.input_vars))
        controls_df = pd.concat([controls_df, controls_inputs_df], axis=1)
        
        with pd.ExcelWriter(filepath) as writer:
            dynamics_df.to_excel(writer, sheet_name='Dynamics', index=False)
            controls_df.to_excel(writer, sheet_name='Controls States', index=False)
            controls_inputs_df.to_excel(writer, sheet_name='Controls Inputs', index=False)


    def plot_dynamics(
        self,
        ang_vel: bool = True,
        lin_vel: bool = True,
        attitude: bool = True,
        read_csv : bool = False,
        csv_path : str = None
    ):
        """Plot the state variables over time. Plots angular velocity, linear velocity, and attitude quaternion on separate subplots.\
            Choose to read from CSV file or locally logged data stored in the Dynamics object.
        
        Args:
            ang_vel (bool): Whether to plot angular velocity.
            lin_vel (bool): Whether to plot linear velocity.
            attitude (bool): Whether to plot attitude quaternion.
            read_csv (bool): Whether to read data from CSV file instead of logged data.
            csv_path (str): Path to the CSV file if read_csv is True.
        """
        states = None
        times = None
        if read_csv:
            df = pd.read_csv(csv_path)
            states = df[str(self.dynamics.state_vars)].to_numpy().tolist()
            times = df['time'].to_numpy().tolist()
            if len(states) == 0:
                raise ValueError("No dynamics data to plot. Please run the dynamics simulation first using run_dynamics_simulation() and save to CSV using save_to_csv().")

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
            axs[2].plot(times, states[:, 6], label='qw')
            axs[2].plot(times, states[:, 7], label='qx')
            axs[2].plot(times, states[:, 8], label='qy')
            axs[2].plot(times, states[:, 9], label='qz')
            axs[2].set_title('Attitude Quaternion vs Time')
            axs[2].set_xlabel('Time (s)')
            axs[2].set_ylabel('Quaternion Components')
            axs[2].legend()
            
        # Plot vertical line at motor burnout time
        if self.dynamics.t_motor_burnout is not None:
            for ax in axs:
                ax.axvline(x=self.dynamics.t_motor_burnout, color='r', linestyle='--', label='Motor Burnout')
                ax.legend()
        
        plt.tight_layout()
        plt.show()


    def plot_controls(
        self,
        ang_vel: bool = True,
        lin_vel: bool = True,
        attitude: bool = True,
        inputs: bool = True,
        log_controls_moments: bool = False
    ):
        """Plot the state variables and control inputs over time. Plots angular velocity, linear velocity, attitude quaternion, and control inputs on separate subplots.
        Args:
            ang_vel (bool): Whether to plot angular velocity.
            lin_vel (bool): Whether to plot linear velocity.
            attitude (bool): Whether to plot attitude quaternion.
            inputs (bool): Whether to plot control inputs.
        """
        if len(self.controls_states) == 0:
            raise ValueError("No controls data to plot. Please run the controls simulation first using run_controls_simulation().")

        states = np.array(self.controls_states)
        inputs = np.array(self.controls_inputs)
        times = np.array(self.controls_times)
        fig, axs = plt.subplots(4, 1, figsize=(10, 20))
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
            axs[2].plot(times, states[:, 6], label='qw')
            axs[2].plot(times, states[:, 7], label='qx')
            axs[2].plot(times, states[:, 8], label='qy')
            axs[2].plot(times, states[:, 9], label='qz')
            axs[2].set_title('Attitude Quaternion vs Time')
            axs[2].set_xlabel('Time (s)')
            axs[2].set_ylabel('Quaternion Components')
            axs[2].legend()
        if inputs:
            axs[3].plot(times, np.rad2deg(inputs[:, 0]), label='u1 (deg)')
            axs[3].plot(times, np.rad2deg(inputs[:, 1]), label='u2 (deg)')
            axs[3].plot(times, np.rad2deg(inputs[:, 2]), label='u3 (deg)')
            axs[3].set_title('Control Inputs vs Time')
            axs[3].set_xlabel('Time (s)')
            axs[3].set_ylabel('Control Inputs (degrees)')
            axs[3].legend()

        # Plot vertical line at motor burnout time
        if self.controls.t_motor_burnout is not None:
            for ax in axs:
                ax.axvline(x=self.controls.t_motor_burnout, color='r', linestyle='--', label='Motor Burnout')
                ax.legend()
        
        plt.tight_layout()
        plt.show()
        
    
    def compare_dyn_or(self, or_path: str):
        """Compare simulation results with OpenRocket data.

        Args:
            or_path (str): Path to the OpenRocket CSV data file.
        """
        print("Please ensure that the OpenRocket data file contains the following columns:")
        print("Time (s), Vertical Speed (m/s), Total velocity (m/s), Roll rate (°/s)")        
        # Load OpenRocket data
        or_data = pd.read_csv(or_path)
        or_time = or_data['# Time (s)']
        or_v3 = or_data['Vertical Speed (m/s)']
        or_vmag = or_data['Total velocity (m/s)']
        or_w3 = or_data['Roll rate (°/s)'] * (np.pi / 180.0) # Convert to rad/s
        
        # Extract simulation altitude data
        sim_states = np.array(self.dynamics_states)
        sim_times = np.array(self.dynamics_times)
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

        axs[1].plot(or_time, or_vmag, label='OpenRocket vMag')
        axs[1].plot(sim_times, sim_vmag, label='Simulation vMag')
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
        plt.tight_layout()
        plt.show()
        
        print("If results don't match closely, ensure that the rocket configuration and simulation parameters are consistent between OpenRocket and this simulation.")


    def reset_logs(self):
        """Reset the logged states and times for dynamics and controls."""
        self.dynamics_states = []
        self.dynamics_times = []
        self.controls_states = []
        self.controls_inputs = []
        self.controls_times = []