import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


from flight_computer import Flight_Computer_Sim
from rocketpy_adapter import Adapter
from sim.controls_setup import build_controls_stack
from sim.internal_dynamics_setup import build_internal_dynamics
from sim.rocketpy_setup import build_rocketpy_stack
from sim_outputs import save_and_plot_results


# ================================================================
# SECTION 1 — CHOOSE SIMULATION MODE
# ================================================================
# Pick simulation mode:
#   'dynamics_EKF_compare' — RocketPy truth + EKF with zero control + open-loop internal comparison.
#   'rocketpy_replay'      — backward-compatible alias for dynamics_EKF_compare.
#   'rocketpy_closedloop'  — RocketPy truth with controller feedback injected into RocketPy.
#   'ekf_only'             — Internal dynamics only, EKF only, no control.
#   'ekf_controlled'       — Internal dynamics only, EKF + control.
simulation_mode = "rocketpy_closedloop"


# ================================================================
# SECTION 2 — BUILD SETUPS
# ================================================================
internal_bundle = build_internal_dynamics()
controls_bundle = build_controls_stack(internal_bundle)
rocketpy_bundle = build_rocketpy_stack()

controls = controls_bundle["controls"]
imu = controls_bundle["imu"]
ekf = controls_bundle["ekf"]

env = rocketpy_bundle["env"]
rocket = rocketpy_bundle["rocket"]
flight = rocketpy_bundle["flight"]
canards = rocketpy_bundle["canards"]


# ================================================================
# SECTION 3 — RUN SIMULATION
# ================================================================
controls.set_env_from_rocketpy(env)

sim = Flight_Computer_Sim(controls, imu, ekf)
rocketpy_adapter = Adapter(sim, simulation_type=simulation_mode)

start_time = time.time()
results = rocketpy_adapter.run(
    rocketpy_flight=flight,
    rocket=rocket,
    env=env,
    canard_fin_set=canards,
    rail_length=rocketpy_bundle["rail_length"],
    inclination=rocketpy_bundle["inclination"],
    heading=rocketpy_bundle["heading"],
    terminate_on_apogee=True,
)
end_time = time.time()

results["runtime_s"] = end_time - start_time
print(f"Simulation is done! Simulation runtime: {results['runtime_s']:.3f} seconds")


# ================================================================
# SECTION 4 — SAVE OUTPUTS
# ================================================================
save_and_plot_results(
    results=results,
    controls=controls,
    rocketpy_adapter=rocketpy_adapter,
    env=env,
    cp_func=internal_bundle["cp_func"],
    drag_func=internal_bundle["drag_func"],
    launch_lat=rocketpy_bundle["launch_lat"],
    launch_lon=rocketpy_bundle["launch_lon"],
    upper_button_pos=rocketpy_bundle["upper_button_position"],
    lower_button_pos=rocketpy_bundle["lower_button_position"],
)
