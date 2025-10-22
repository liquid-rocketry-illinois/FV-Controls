import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import datetime
import numpy as np
import csv

from rocketpy import Environment, SolidMotor, Rocket, Flight
from rocketpy.control.controller import _Controller

from control_algorithm import Controls
from fin import Fins

class SilSim:
    def __init__(
            self,
            sampling_rate: float,
            controller: Controls
    ):
        self.sampling_rate = sampling_rate
        self.controller = controller
        self.times = []
        self.xhats = []
        self.inputs = []
        self.csv_output_path = "Maurice2/data/testing.csv"
        self.csv_col_title = "input"


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
        xhat = self.rocketpy_state_to_xhat(state)

        # Make measurement y from RocketPy state
        y = self.make_measurement_from_rocketpy(state, time)

        ## Get previous control input ##
        fins : Fins = interactive_objects[0]  # Assuming fins are the first/only interactive object
        u_prev = fins.aileronAngles

        ## Compute control/observer matrices ##
        self.controller.computeAB(t=time, xhat=xhat, u=u_prev)
        self.controller.computeC(xhat=xhat, u=u_prev)
        A, B, C = self.controller.A, self.controller.B, self.controller.C
        A = np.array(A).astype(np.float64)
        B = np.array(B).astype(np.float64)
        C = np.array(C).astype(np.float64)

        ## Get K and L matrices ##
        K, L = self.controller.control_law(xhat=xhat, t=time), self.controller.L

        accel_T = self.controller.get_thrust_accel(t=time)
        accel_g = self.controller.get_gravity_accel(xhat=xhat)
        xhatdot = A @ xhat + B @ u_prev + accel_T + accel_g \
                - L @ (C @ xhat - y)
        xhat = xhat + xhatdot * self.controller.dt
        xhat[6:10] /= np.linalg.norm(xhat[6:10])
        u = np.clip(-K @ (xhat - self.controller.x0) + self.controller.u0, np.deg2rad(-8), np.deg2rad(8))
        # u = np.array([0.0]) # Disable control for testing
        fins.aileronAngles = u
        self.times.append(time)
        self.inputs.append(np.rad2deg(u[0]))
        self.xhats.append(xhat.tolist())

        print("At time " + str(time) + " s, the aileron angle is " + str(np.rad2deg(u)) + " degrees.")
        print("The state estimate is " + str(xhat))

        return u

    # TODO: Check rocket params
    def makeOurRocket(self, samplingRate):
        """Create and configure the rocket and its controller.

        Args:
            samplingRate (float): The sampling rate for the controller.

        Returns:
            tuple: A tuple containing the configured Rocket object and its controller.
        """
        maurice2 = Rocket(
            radius=0.079,
            mass=2.259,
            inertia=(0.28, 0.002940, 0.002940),
            power_off_drag=0.560,
            power_on_drag=0.580,
            center_of_mass_without_motor=0.669,
            coordinate_system_orientation="tail_to_nose",
        )
        # Remeasure
        ourMotor = SolidMotor(
            thrust_source="Maurice2/data/AeroTech_HP-I280DM.eng",  # Or use a CSV thrust file
            dry_mass=(0.616 - 0.355),  # kg
            burn_time=self.controller.t_motor_burnout,  # Corrected burn time

            dry_inertia=(0.004, 0.004, 0.287),  # kg·m² (approximated)
            nozzle_radius= (10 / 1000), 
            grain_number=5,
            grain_density=18, 
            grain_outer_radius= 7 / 1000,  
            grain_initial_inner_radius=4 / 1000,  
            grain_initial_height= 360 / 5000,  
            grain_separation=0.01,  
            grains_center_of_mass_position=-0.07,  # Estimated
            center_of_dry_mass_position=0.05,  # Estimated
            nozzle_position=-0.3,
            throat_radius= 3.5 / 1000,  
            coordinate_system_orientation="nozzle_to_combustion_chamber",
        )

        maurice2.add_motor(ourMotor, position=0.01*(117-86.6))

        nose_cone = maurice2.add_nose(
            length=0.19, kind="lvhaack", position=0.01*(117-0.19)
        )

        # Boat Tail
        # Verify that it is von karman
        tail = maurice2.add_tail(
            top_radius=0.0787/2, bottom_radius=0.0572/2, length=0.0381, position=.0381
        )

        # Created in fin.py, inherited from RocketPy TrapezoidalFins class
        ourNewFins = Fins(
            n=4,
            root_chord=0.203,
            tip_chord=0.0762,
            span=0.0737,
            rocket_radius = 7.87/200,
            cant_angle=0.01,
            sweep_angle=62.8
        )

        # Integrate controller with RocketPy
        rpy_controller = _Controller(
            interactive_objects= [ourNewFins],
            controller_function= self.controller_function, # Pass our function into rocketpy
            sampling_rate= samplingRate, # How often it runs
            name="MAURICE 2",
        )

        maurice2.add_surfaces(ourNewFins, 0.01*(117-92.7))
        maurice2._add_controllers(rpy_controller)

        return maurice2, rpy_controller
    

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
        flight.plots.angular_kinematics_data()
        flight.plots.attitude_data()
        flight.plots.trajectory_3d()

        flight.export_data(
            "Maurice2/data/rocketpy_output.csv",
            "w1",
            "w2",
            "w3",
            "alpha1",
            "alpha2",
            "alpha3",
            "vx",
            "vy",
            "vz",
        )
        return flight, controller


    def export_data(self, overwrite: bool = True):
        """
        Save logs to CSV with columns:
        time, xhat_0..xhat_{n-1}, input_0..input_{m-1}

        - Uses the min length across times/xhats/inputs.
        - Flattens numpy arrays and lists.
        - Overwrites by default; set overwrite=False to append (adds header only if file doesn't exist).
        """

        # Export to this location
        path = "Maurice2/data/estimated_output.csv"

        # Normalize to lists
        times   = list(self.times or [])
        xhats   = list(self.xhats or [])
        inputs  = list(self.inputs or [])

        n = min(len(times), len(xhats), len(inputs))
        if n == 0:
            raise ValueError("No log data to write (times/xhats/inputs are empty).")

        # Peek sizes
        def _flat_len(x):
            if x is None:
                return 0
            if isinstance(x, (list, tuple, np.ndarray)):
                return int(np.asarray(x).size)
            return 1

        xhat_len  = _flat_len(xhats[0])
        input_len = _flat_len(inputs[0])

        # Build header
        header = ["time"] + [f"xhat_{i}" for i in range(xhat_len)] + [f"input_{j}" for j in range(input_len)]

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
                t = float(times[i])

                xi = xhats[i]
                ui = inputs[i]

                # flatten to lists with correct lengths
                xi_flat = [] if xhat_len == 0 else np.asarray(xi).reshape(-1).tolist()
                ui_flat = [] if input_len == 0 else np.asarray(ui).reshape(-1).tolist()

                # pad/truncate to header lengths (safety)
                xi_flat = (xi_flat + [None]*xhat_len)[:xhat_len]
                ui_flat = (ui_flat + [None]*input_len)[:input_len]

                w.writerow([t] + xi_flat + ui_flat)

        
# Run SIL simulation, export flight data to CSV
def main():
    ## Define gain matrix ##
    Kmax_preburnout = 100 / 7e1
    Kmin_preburnout = 17.5 / 7e1

    K_max_postburnout = 85 / 6e1
    K_min_postburnout = 17.5 / 6e1

    Ks = np.array([Kmax_preburnout, Kmin_preburnout, K_max_postburnout, K_min_postburnout])  # Gain scheduling based on altitude

    ## Define initial conditions ##
    xhat0 = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]) # Initial state estimate
    u0 = np.array([0])
    sampling_rate = 20.0  # Hz
    dt = 1.0 / sampling_rate

    controller = Controls(Ks=Ks, dt=dt, x0=xhat0, u0=u0, t_launch_rail_clearance=0.308)
    controller.deriveEOM(post_burnout=False)
    controller.deriveEOM(post_burnout=True)
    controller.buildL(lw=5.0, lqw=1.0, lqx=2.0, lqy=2.0, lqz=2.0)
    # controller.buildL(lw=0.0, lqw=0.0, lqx=0.0, lqy=0.0, lqz=0.0)
    sim = SilSim(sampling_rate=sampling_rate, controller=controller)
    flight, controller = sim.run(sampling_rate=sampling_rate)
    sim.export_data()

if __name__ == "__main__":
    main()