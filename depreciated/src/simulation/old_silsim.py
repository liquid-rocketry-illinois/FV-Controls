import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from pathlib import Path

import datetime
import numpy as np
import csv

from rocketpy import Environment, SolidMotor, Rocket, Flight
from rocketpy.control.controller import _Controller

from controls.controls import Controls
from src.rocket.fin import Fins

class SilSim:
    def __init__(
            self,
            sampling_rate: float,
            controller: Controls
    ):
        """Initialize the SIL simulation with the given sampling rate and controller.

        Args:
            sampling_rate (float): The sampling rate for the simulation in Hz.
            controller (Controls): The controller object to be used in the simulation.
        """
        self.sampling_rate = sampling_rate
        self.controller = controller
        self.times = []
        self.xhats = [self.controller.x0]
        self.states = []
        self.inputs = []
        self.csv_output_path = "Maurice2/data/testing.csv"
        self.csv_col_title = "input"
        self.DATA_DIR = Path(__file__).resolve().parents[1] / "data"   # => Maurice2/data

    def set_controller(
            self,
            controller: Controls
    ):
        """Setter for the controller object if not initialized in SilSim.

        Args:
            controller (Controls): The controller object to be used in the simulation.
        """
        self.controller = controller


    def rocketpy_state_to_xhat(self, state):
        """Convert RocketPy state vector to state estimate vector xhat.

        Args:
            state (list): RocketPy state vector.

        Returns:
            np.ndarray: State estimate vector xhat.
        """
        # unpack RocketPy state
        v1, v2, v3 = state[3],  state[4],  state[5]
        e0, e1, e2, e3 = state[6],  state[7],  state[8],  state[9]   # quaternion (scalar-first)
        w1, w2, w3 = state[10], state[11], state[12]

        # your convention is [w1 w2 w3 v1 v2 v3 qw qx qy qz]
        return np.array([w1, w2, w3, v1, v2, v3, e0, e1, e2, e3], dtype=float)


    def make_measurement_from_rocketpy(
            self,
            state: list,
            time: float,
        ):
        """Generate measurement vector from RocketPy state.

        Args:
            state (list): RocketPy state vector.
            time (float): Current simulation time.

        Returns:
            np.ndarray: Measurement vector.
        """
        w1, w2, w3 = state[10], state[11], state[12]
        qw, qx, qy, qz = state[6], state[7], state[8], state[9]

        theta, phi, psi = self.controller.quat_to_euler_xyz([qw, qx, qy, qz])

        y = self.controller.deriveSensorModels(time, w1, w2, w3, theta, phi, psi)
        return y


    def controller_function(
            self,
            time,
            sampling_rate,
            state,
            state_history,
            observed_variables,
            interactive_objects,
    ):
        """Initialize the controller function to be called during the simulation.

            Parameters
            ----------
            controller_function : function, callable
                An user-defined function responsible for controlling the simulation.
                This function is expected to take the following arguments, in order:

                1. `time` (float): The current simulation time in seconds.
                2. `sampling_rate` (float): The rate at which the controller
                function is called, measured in Hertz (Hz).
                3. `state` (list): The state vector of the simulation, structured as
                `[x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz]`.
                4. `state_history` (list): A record of the rocket's state at each
                step throughout the simulation. The state_history is organized as
                a list of lists, with each sublist containing a state vector. The
                last item in the list always corresponds to the previous state
                vector, providing a chronological sequence of the rocket's
                evolving states.
                5. `observed_variables` (list): A list containing the variables that
                the controller function returns. The return of each controller
                function call is appended to the observed_variables list. The
                initial value in the first step of the simulation of this list is
                provided by the `initial_observed_variables` argument.
                6. `interactive_objects` (list): A list containing the objects that
                the controller function can interact with. The objects are
                listed in the same order as they are provided in the
                `interactive_objects`.
                7. `sensors` (list): A list of sensors that are attached to the
                    rocket. The most recent measurements of the sensors are provided
                    with the ``sensor.measurement`` attribute. The sensors are
                    listed in the same order as they are added to the rocket

                This function will be called during the simulation at the specified
                sampling rate. The function should evaluate and change the interactive
                objects as needed. The function return statement can be used to save
                relevant information in the `observed_variables` list.

                .. note:: The function will be called according to the sampling rate specified.

            Returns
            -------
            None
            """
        apogee = (time > self.controller.t_launch_rail_clearance + self.controller.t_motor_burnout) and (state[5] <= 0)
        if apogee:
            return np.array([0.0])  # No control after apogee
        self.controller.dt = 1.0 / sampling_rate
        # Convert RocketPy state to xhat
        # xhat = self.rocketpy_state_to_xhat(state)
        xhat = self.xhats[-1]

        # Make measurement y from RocketPy state
        y = self.make_measurement_from_rocketpy(state, time)

        ## Get previous control input ##
        fins : Fins = interactive_objects[0]  # Assuming fins are the first/only interactive object
        # u_prev = fins.aileronAngles
        
        ## Get K and L matrices ##
        K, L = self.controller.control_law(xhat=xhat, t=time), self.controller.L
        xdes = xhat.copy()
        xdes[2] = 0.0 # desired w3 = 0 rad/s
        u = np.clip(-K @ (np.array(xhat) - np.array(xdes)) + self.controller.u0, np.deg2rad(-8), np.deg2rad(8))

        ## Compute control/observer matrices ##
        self.controller.computeAB(t=time, xhat=xhat, u=u)
        self.controller.computeC(xhat=xhat, u=u)
        A, B, C = self.controller.A, self.controller.B, self.controller.C
        A = np.array(A).astype(np.float64)
        B = np.array(B).astype(np.float64)
        C = np.array(C).astype(np.float64)

        accel_T = self.controller.get_thrust_accel(t=time)
        accel_g = self.controller.get_gravity_accel(xhat=xhat)
        # test = xhat + (A @ xhat + B @ u_prev + accel_T + accel_g) * self.controller.dt
        xhatdot = A @ xhat + B @ u + accel_T + accel_g \
                - L @ (C @ xhat - y)
        xhat = xhat + xhatdot * self.controller.dt
        xhat[6:10] /= np.linalg.norm(xhat[6:10])

        # f_subs = np.array(self.controller.f_subs, dtype=float).reshape(-1)
        # xhatdot = f_subs - L @ (C @ xhat - y)
        # xhat = xhat + xhatdot * self.controller.dt
        # xhat[6:10] /= np.linalg.norm(xhat[6:10])
        # K = self.controller.control_law(xhat, time)
        # u = np.clip(-K @ (xhat - self.controller.x0) + self.controller.u0, np.deg2rad(-8), np.deg2rad(8))

        # xhat = self.controller._rk4_step(time, xhat, u_prev) - L @ (C @ xhat - y)
        # qn = np.linalg.norm(xhat[6:10])
        # xhat[6:10] = np.array([1.,0.,0.,0.]) if qn < 1e-12 else xhat[6:10]/qn
        # K = self.controller.control_law(xhat, time)
        # u = np.clip(-K @ (xhat - self.controller.x0) + self.controller.u0, np.deg2rad(-8), np.deg2rad(8))
        
        # u = np.array([0.0]) # Temporary: disable control for testing
        fins.aileronAngles = u
        self.times.append(time)
        self.inputs.append(np.rad2deg(u[0]))
        self.xhats.append(xhat.tolist())
        # self.xhats.append(test.tolist())
        self.states.append(state)

        print("Time: " + str(np.float64(time).round(3)) + " s, v3: " + str(np.float64(state[5]).round(3)) + " m/s, w3: " + str(np.float64(state[12]).round(3)) + " rad/s, u: " + str(np.rad2deg(u).round(3)) + " degrees.")
        # print("Time:" + str(np.float64(time).round(3)) + " s, xhat: " + str(np.array(xhat).round(3)))
        return u

    # TODO: Check rocket params
    def makeOurRocket(self, samplingRate):
        """Create and configure the rocket and its controller.

        Args:
            samplingRate (float): The sampling rate for the controller.

        Returns:
            tuple: A tuple containing the configured Rocket object and its controller.
        """

        ## Define rocket ##
        POWER_OFF_FILE = self.DATA_DIR / "drag_no_thrust.csv"
        POWER_ON_FILE = self.DATA_DIR / "drag_with_thrust.csv"
        maurice2 = Rocket(
            radius=7.87/2/100,
            mass=2328/1000,
            inertia=(0.287, 0.287, 0.0035),
            # power_off_drag=0.6175,
            # power_on_drag=0.633,
            power_off_drag=str(POWER_OFF_FILE),
            power_on_drag=str(POWER_ON_FILE),
            center_of_mass_without_motor=59.7/100,
            coordinate_system_orientation="nose_to_tail",
        )
        
        ## Define solid motor ##
        THRUST_FILE = self.DATA_DIR / "AeroTech_HP-I280DM.eng"
        motor = SolidMotor(
            thrust_source=str(THRUST_FILE),  # Or use a CSV thrust file
            dry_mass=(0.616 - 0.355),  # kg
            burn_time=self.controller.t_motor_burnout,  # Corrected burn time
            dry_inertia=(0.00055, 0.00055, 0.00011),  # kg·m² (approximated)
            # Fixed Geometries to make propellant mass = 0.355 kg
            nozzle_radius= (10 / 1000), 
            grain_number=5,
            grain_density=1800, # Fixed grain density 18 --> 1800 kg·m^3
            grain_outer_radius= 16 / 1000,
            grain_initial_inner_radius= 6 / 1000,  
            grain_initial_height= 57 / 1000,
            grain_separation=0.01,
            grains_center_of_mass_position=-0.1044,  # Estimated
            center_of_dry_mass_position=-0.122,  # Estimated
            nozzle_position=-0.3,
            throat_radius= 3.5 / 1000,  
            coordinate_system_orientation="nozzle_to_combustion_chamber",
        )

        maurice2.add_motor(motor, position=88.2/100)


        nose_cone = maurice2.add_nose(
            length=19/100, kind="Von Karman", position=0
        )

        # Boat tail
        tail = maurice2.add_tail(
            top_radius=7.87/2/100, bottom_radius=5.72/2/100, length=3.81/100, position=(117-3.81)/100
        )

        # Created in fin.py, inherited from RocketPy TrapezoidalFins class
        fins = Fins(
            n=4,
            root_chord=18/100,
            tip_chord=5.97/100,
            span=8.76/100,
            rocket_radius=7.87/2/100,
            cant_angle=0.1,
            sweep_length=14.3/100,
        )

        maurice2.add_surfaces(fins, (92.3)/100)

        # Integrate controller with RocketPy
        rp_controller = _Controller(
            interactive_objects= [fins],
            controller_function= self.controller_function, # Pass our function into rocketpy
            sampling_rate= samplingRate, # How often it runs
            name="MAURICE 2",
        )

        # Commented out to first verify the rocket flies correctly compared to the OpenRocket
        maurice2._add_controllers(rp_controller)

        return maurice2, rp_controller


    def run(self, sampling_rate: float):
        """Run the simulation with the specified sampling rate. Exports RocketPy flight data and state estimations to CSV in the data folder.

        Args:
            sampling_rate (float): The sampling rate for the simulation.

        Returns:
            tuple: A tuple containing the Flight object and its controller.
        """
        self.sampling_rate = sampling_rate
        env = Environment(
            latitude=41.92298772007185,
            longitude=-88.06013490408121,
            elevation=243.43
        )
        tomorrow = datetime.date.today() + datetime.timedelta(days=1)
        env.set_date((tomorrow.year, tomorrow.month, tomorrow.day, 12))  
        env.set_atmospheric_model(type="Forecast", file="GFS")

        rocket, controller = self.makeOurRocket(self.sampling_rate)

        # Flight parameters
        flight = Flight(
            rocket=rocket, environment=env, rail_length=5.2, inclination=85, heading=0
        )
        flight.info()
        # flight.plots.angular_kinematics_data()
        # flight.plots.attitude_data()
        # flight.plots.trajectory_3d()

        export_loc = str(self.DATA_DIR / "rocketpy_output.csv")
        flight.export_data(
            export_loc,
            "w1",
            "w2",
            "w3",
            "alpha1",
            "alpha2",
            "alpha3",
            "vx",
            "vy",
            "vz",
            "x",
            "y",
            "z",
            "e0",
            "e1",
            "e2",
            "e3",
        )
        return flight, controller


    def export_states(self, overwrite: bool = True):
        """
        Save logs to CSV with columns:
        time, state_0..state_{p-1}, xhat_0..xhat_{n-1}, input_0..input_{m-1}

        - Uses the min length across times/xhats/states/inputs.
        - Flattens numpy arrays and lists.
        - Overwrites by default; set overwrite=False to append (adds header only if file doesn't exist).
        """

        # Export to this location
        path = str(self.DATA_DIR / "silsim_output2.csv")

        # Normalize to lists
        times   = list(self.times or [])
        xhats   = list(self.xhats or [])
        states  = list(getattr(self, "states", []) or [])
        inputs  = list(self.inputs or [])

        n = min(len(times), len(xhats), len(states), len(inputs))
        if n == 0:
            raise ValueError("No log data to write (times/xhats/states/inputs are empty or mismatched).")

        # Helper to get flattened length
        def _flat_len(x):
            if x is None:
                return 0
            if isinstance(x, (list, tuple, np.ndarray)):
                return int(np.asarray(x).size)
            return 1

        # Peek sizes (assume consistent sizes within each list)
        state_len = _flat_len(states[0])
        xhat_len  = _flat_len(xhats[0])
        input_len = _flat_len(inputs[0])

        # Build header (swap the two blocks if you prefer xhat_* before state_*)
        header = (["time"]
                + [f"state_{i}" for i in range(state_len)]
                + [f"xhat_{i}" for i in range(xhat_len)]
                + [f"input_{j}" for j in range(input_len)])

        mode = "w" if overwrite else "a"
        write_header = True
        if not overwrite:
            try:
                with open(path, "r"):
                    write_header = False
            except FileNotFoundError:
                write_header = True

        with open(path, mode, newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)

            for i in range(n):
                t  = float(times[i])
                xi = xhats[i]
                si = states[i]
                ui = inputs[i]

                # Flatten to lists with correct lengths
                si_flat = [] if state_len == 0 else np.asarray(si).reshape(-1).tolist()
                xi_flat = [] if xhat_len  == 0 else np.asarray(xi).reshape(-1).tolist()
                ui_flat = [] if input_len == 0 else np.asarray(ui).reshape(-1).tolist()

                # Pad/truncate to header lengths (safety)
                si_flat = (si_flat + [None]*state_len)[:state_len]
                xi_flat = (xi_flat + [None]*xhat_len)[:xhat_len]
                ui_flat = (ui_flat + [None]*input_len)[:input_len]

                # Write row (swap si_flat and xi_flat here too if re-ordering)
                w.writerow([t] + si_flat + xi_flat + ui_flat)

        
# Run SIL simulation, export flight data to CSV
def main():
    ## Define gain matrix ##
    K_pre_max = 1.0e-1
    K_pre_min = 5.0e-4
    # K_pre_max = 5.0e-1
    # K_pre_min = 5.0e-1
    # K_post_max = 5.0e-1
    # K_post_min = 5.0e-1
    K_post_max = 1.0e-1
    K_post_min = 5.0e-3

    pre_width = 3
    post_width = 8

    pre_v3_mid = 100.0
    post_v3_mid = 90.0

    # pre_w3_mid = 0.35
    # post_w3_mid = 0.5

    ## Define initial conditions ##
    xhat0 = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]) # Initial state estimate
    u0 = np.array([np.deg2rad(-0.0)])  # Initial control input
    sampling_rate = 40.0  # Hz
    dt = 1.0 / sampling_rate

    controller = Controls(dt=dt, x0=xhat0, u0=u0, t_launch_rail_clearance=0.308)
    controller.setup_EOM()
    controller.set_K_params(K_pre_max=K_pre_max, K_pre_min=K_pre_min,
                            K_post_max=K_post_max, K_post_min=K_post_min,
                            pre_width=pre_width, post_width=post_width,
                            pre_v3_mid=pre_v3_mid, post_v3_mid=post_v3_mid)
    # controller.set_K_params(K_pre_max=K_pre_max, K_pre_min=K_pre_min,
    #                         K_post_max=K_post_max, K_post_min=K_post_min,
    #                         pre_width=pre_width, post_width=post_width,
    #                         pre_v3_mid=pre_w3_mid, post_v3_mid=post_w3_mid)
    # controller.buildL(lw=10.0, lqw=1.0, lqx=2.0, lqy=2.0, lqz=2.0)
    lw = 1e-3 # any higher makes the simulation unstable for some dumb reason
    lq = 1e-3
    controller.buildL(lw=lw, lqw=lq, lqx=lq, lqy=lq, lqz=lq)
    # controller.buildL(lw=40 , lqw=0.01, lqx=0.5, lqy=0.5, lqz=0.5)
    # controller.buildL(lw=5.0 , lqw=0.5, lqx=1, lqy=1, lqz=1)
    # controller.buildL(lw=0.0, lqw=0.0, lqx=0.0, lqy=0.0, lqz=0.0)

    ## Run SIL simulation ##
    sim = SilSim(sampling_rate=sampling_rate, controller=controller)
    flight, controller = sim.run(sampling_rate=sampling_rate)
    sim.export_states()
    

if __name__ == "__main__":
    main()
    print("SIL simulation complete.")