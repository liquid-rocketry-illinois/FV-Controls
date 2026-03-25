import csv
from pathlib import Path
from typing import Callable

import numpy as np
from rocketpy import Flight
from rocketpy.rocket.aero_surface import GenericSurface

from controls.controls import Controls
from simulation.simulation import Simulation
from roll_force import RollForce  # noqa: F401 — re-exported for user convenience


class CanardSurface(GenericSurface):
    """Aerodynamic surface that injects canard control moments into RocketPy.

    Delegates to Controls.M_controls_func to compute moments from the current
    aileron deflection angles. Used as a RocketPy interactive object — the
    RocketSim controller updates aileron_angles each timestep.

    No passive aerodynamic forces are computed here. Pair with a plain
    TrapezoidalFins for passive fin aerodynamics.

    ### Usage
        canard = CanardSurface(
            center_of_pressure=0.5,
            reference_area=0.01,
            reference_length=0.1,
            controls=controls,
        )
        rocket.add_aero_surface(canard, position=0.5)
        rocket.add_controller(
            sim.controller_function,
            sampling_rate=sim.sampling_rate,
            initial_observed_variables=[controls.u0],
            interactive_objects=[canard],
        )
    """

    def __init__(
        self,
        center_of_pressure: float,
        reference_area: float,
        reference_length: float,
        controls: Controls,
        name: str = "CanardSurface",
    ):
        """
        Args:
            center_of_pressure (float): Axial position in meters from the nozzle.
            reference_area (float): Reference area in m².
            reference_length (float): Reference length in m (typically rocket diameter).
            controls (Controls): Controls object whose M_controls_func computes moments.
            name (str): Label for this surface in RocketPy output.
        """
        super().__init__(
            reference_area=reference_area,
            reference_length=reference_length,
            coefficients={},
            center_of_pressure=center_of_pressure,
            name=name,
        )
        self.controls = controls
        self.aileron_angles = np.array([0.0])

    def compute_forces_and_moments(
        self,
        stream_velocity,
        stream_speed,
        stream_mach,
        rho,
        cp,
        omega,
        reynolds,
    ):
        """Compute control moments from the current aileron deflection.

        Returns:
            tuple: (R1, R2, R3, M1, M2, M3) — forces are zero, moments from M_controls_func.
        """
        v1, v2, v3 = stream_velocity[0], stream_velocity[1], stream_velocity[2]
        state = np.array([0.0, 0.0, 0.0, v1, v2, v3, 0.0, 0.0, 0.0, 0.0])
        u = np.array([self.aileron_angles[0]])
        Mx, My, Mz = self.controls.M_controls_func(state, u)
        return 0.0, 0.0, 0.0, float(Mx), float(My), float(Mz)


class RocketSim:
    """Simulation runner: RocketPy handles environment physics, FV-Controls runs the control loop.

    ### Usage
        1. Build and configure a Controls object externally (K, L, EOM all set).
        2. Instantiate RocketSim with that Controls object and a sampling rate.
        3. Define create_rocket() and create_env() callables (see contract below).
        4. Call set_rocket() and set_env().
        5. Call run() to execute the simulation.
        6. Call export_states() to save logged control data to CSV.

    ### create_rocket contract
        The callable passed to set_rocket() must:
        - Build a RocketPy Rocket and add passive TrapezoidalFins for passive aero.
        - Instantiate a CanardSurface and add it to the rocket as both an aero surface
          and an interactive object via rocket.add_controller().
        - Optionally instantiate a RollForce and add it as an aero surface.
        - Register sim.controller_function with rocket.add_controller().
        - Return (Rocket, CanardSurface).

    ### Example
        sim = RocketSim(controls=controls, sampling_rate=40.0)

        def create_rocket():
            rocket = Rocket(...)
            fins   = TrapezoidalFins(...)       # passive aero, no subclassing
            canard = CanardSurface(             # control moments
                center_of_pressure=0.5,
                reference_area=0.01,
                reference_length=0.1,
                controls=controls,
            )
            roll = RollForce(                   # CFD roll moment
                center_of_pressure=1.2,
                reference_area=0.01,
                reference_length=0.1,
                controls=controls,
            )
            rocket.add_aero_surface(fins,   position=...)
            rocket.add_aero_surface(canard, position=...)
            rocket.add_aero_surface(roll,   position=...)
            rocket.add_controller(
                sim.controller_function,
                sampling_rate=sim.sampling_rate,
                initial_observed_variables=[controls.u0],
                interactive_objects=[canard],
            )
            return rocket, canard

        def create_env():
            env = Environment(...)
            return env

        sim.set_rocket(create_rocket)
        sim.set_env(create_env)
        flight = sim.run("my_flight", rail_length=5.0)
        sim.export_states("my_flight_states")
    """

    def __init__(self, controls: Controls, sampling_rate: float):
        """
        Args:
            controls (Controls): Fully configured Controls object (K, L, EOM all set).
            sampling_rate (float): Control loop frequency in Hz.
        """
        self.controls = controls
        self.sampling_rate = sampling_rate
        self.simulation = Simulation(controls=controls)

        self.times: list = []
        self.xhats: list = [controls.x0]
        self.states: list = []
        self.inputs: list = [controls.u0]

        self.root = Path(__file__).resolve().parents[1]
        self.output_path = (
            self.root / "rockets" / controls.rocket_name / "data" / "sim_output" / "rocketsim"
        )

        self.create_rocket: Callable = None
        self.create_env: Callable = None

    def set_rocket(self, create_rocket: Callable):
        """Set the callable that builds the RocketPy Rocket.

        Args:
            create_rocket (Callable): No args. Returns (Rocket, CanardSurface).
        """
        self.create_rocket = create_rocket

    def set_env(self, create_env: Callable):
        """Set the callable that builds the RocketPy Environment.

        Args:
            create_env (Callable): No args. Returns Environment.
        """
        self.create_env = create_env

    def _rocketpy_state_to_xhat(self, state) -> np.ndarray:
        """Convert RocketPy state to our convention.

        RocketPy: [x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz]
        Ours:     [w1, w2, w3, v1, v2, v3, qw, qx, qy, qz]
        """
        v1, v2, v3     = state[3],  state[4],  state[5]
        e0, e1, e2, e3 = state[6],  state[7],  state[8],  state[9]
        w1, w2, w3     = state[10], state[11], state[12]
        return np.array([w1, w2, w3, v1, v2, v3, e0, e1, e2, e3], dtype=float)

    def controller_function(
        self,
        time,
        sampling_rate,
        state,
        state_history,
        observed_variables,
        interactive_objects,
    ):
        """RocketPy controller callback. Called by RocketPy at self.sampling_rate Hz.

        Runs one step of the FV-Controls control loop and updates the CanardSurface
        deflection angle for the next physics step.

        interactive_objects[0] must be the CanardSurface.
        """
        self.states.append(state)
        self.times.append(time)

        apogee = (
            time > self.controls.t_launch_rail_clearance + self.controls.t_motor_burnout
            and state[5] <= 0
        )
        if apogee:
            return np.array([0.0])

        self.controls.dt = 1.0 / self.sampling_rate
        xhat, u = self.simulation.controls_step(
            time,
            self.xhats[-1],
            self.inputs[-1],
            self._rocketpy_state_to_xhat(state),
        )

        canard: CanardSurface = interactive_objects[0]
        canard.aileron_angles = u

        self.inputs.append(u.tolist())
        self.xhats.append(xhat.tolist())

        return u

    def run(
        self,
        file_name: str,
        rail_length: float = 1.0,
        inclination: float = 85.0,
        heading: float = 0.0,
        sampling_rate: float = None,
    ) -> Flight:
        """Run the simulation and export RocketPy flight data to CSV.

        Args:
            file_name (str): Output CSV filename (no extension).
            rail_length (float): Launch rail length in meters.
            inclination (float): Launch inclination angle in degrees.
            heading (float): Launch heading in degrees.
            sampling_rate (float): Override the instance sampling rate if provided.

        Returns:
            Flight: The completed RocketPy Flight object.
        """
        if sampling_rate is not None:
            self.sampling_rate = sampling_rate

        self.output_path.mkdir(parents=True, exist_ok=True)

        rocket, _ = self.create_rocket()
        env = self.create_env()

        flight = Flight(
            rocket=rocket,
            environment=env,
            rail_length=rail_length,
            inclination=inclination,
            heading=heading,
        )

        export_loc = str(self.output_path / (file_name + ".csv"))
        flight.export_data(
            export_loc,
            "w1", "w2", "w3",
            "alpha1", "alpha2", "alpha3",
            "vx", "vy", "vz",
            "x", "y", "z",
            "e0", "e1", "e2", "e3",
        )
        print(f"Exported RocketPy flight data to: {export_loc}")

        return flight

    def export_states(self, file_name: str, overwrite: bool = True):
        """Export logged states, state estimates, and control inputs to CSV.

        Args:
            file_name (str): Output CSV filename (no extension).
            overwrite (bool): Overwrite existing file. Default True.
        """
        path = str(self.output_path / (file_name + ".csv"))

        times  = list(self.times)
        xhats  = list(self.xhats)
        states = list(self.states)
        inputs = list(self.inputs)

        n = min(len(times), len(xhats), len(states), len(inputs))
        if n == 0:
            raise ValueError("No log data to export — run the simulation first.")

        def _flat_len(x):
            return int(np.asarray(x).size)

        state_len = _flat_len(states[0])
        xhat_len  = _flat_len(xhats[0])
        input_len = _flat_len(inputs[0])

        header = (
            ["time"]
            + [f"state_{i}" for i in range(state_len)]
            + [f"xhat_{i}"  for i in range(xhat_len)]
            + [f"input_{j}" for j in range(input_len)]
        )

        with open(path, "w" if overwrite else "a", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for i in range(n):
                row = (
                    [float(times[i])]
                    + np.asarray(states[i]).reshape(-1).tolist()
                    + np.asarray(xhats[i]).reshape(-1).tolist()
                    + np.asarray(inputs[i]).reshape(-1).tolist()
                )
                w.writerow(row)

        print(f"Exported simulation states to: {path}")


if __name__ == "__main__":
    # # --- Configure your Controls object ---
    # controls = Controls(IREC_COMPLIANT=True, rocket_name="MyRocket")
    # controls.setup_EOM()
    # controls.setK(...)
    # controls.buildL(...)
    # controls.set_controls_params(u0=np.array([0.0]), max_input=np.deg2rad(8))

    # # --- Build sim ---
    # sim = RocketSim(controls=controls, sampling_rate=40.0)

    # # --- Define rocket and environment builders ---
    # def create_rocket():
    #     from rocketpy import Rocket, TrapezoidalFins
    
    #     rocket = Rocket(...)
    
    #     # Passive fin aerodynamics — plain TrapezoidalFins, no subclassing
    #     fins = TrapezoidalFins(n=4, root_chord=..., tip_chord=..., span=..., rocket_radius=...)
    #     rocket.add_aero_surface(fins, position=...)
    
    #     # Canard control moments
    #     canard = CanardSurface(
    #         center_of_pressure=...,
    #         reference_area=...,
    #         reference_length=...,
    #         controls=controls,
    #     )
    #     rocket.add_aero_surface(canard, position=...)
    
    #     # CFD roll force (optional)
    #     roll = RollForce(
    #         center_of_pressure=...,
    #         reference_area=...,
    #         reference_length=...,
    #         controls=controls,
    #     )
    #     rocket.add_aero_surface(roll, position=...)
    
    #     # Register control loop — canard is the interactive object
    #     rocket.add_controller(
    #         sim.controller_function,
    #         sampling_rate=sim.sampling_rate,
    #         initial_observed_variables=[controls.u0],
    #         interactive_objects=[canard],
    #     )
    #     return rocket, canard
    
    # def create_env():
    #     from rocketpy import Environment
    #     env = Environment(latitude=..., longitude=..., elevation=...)
    #     env.set_atmospheric_model(type="standard_atmosphere")
    #     return env
    
    # sim.set_rocket(create_rocket)
    # sim.set_env(create_env)
    # flight = sim.run("my_flight", rail_length=5.0)
    # sim.export_states("my_flight_states")
    pass
