import sys
import time
from pathlib import Path

import numpy as np

control_freak_root = Path(__file__).resolve().parent.parent
project_root = control_freak_root.parents[1]
src_root = project_root / "src"

for path in (control_freak_root, src_root):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


from controls.flight_computer import Flight_Computer_Sim
from rocketpy_adapter import Adapter
from simulation.monte_carlo import MonteCarloConfig, run_monte_carlo
from sim.controls_setup import build_controls_stack
from sim.internal_dynamics_setup import build_internal_dynamics
from sim.rocketpy_setup import build_rocketpy_stack
from sim_outputs import save_and_plot_results


def benchmark_flight_step(sim, results, env, sample_count=250, warmup_count=20):
    """Time the flight-like EKF + control step on representative sim states."""
    t = results["t"]
    if len(t) < 2:
        return None

    x_true = results["x_true"]
    deriv = results["deriv"]
    u_hist = results["u"]
    temp_hist = results.get("temperature")
    pos_hist = results.get("position")

    if temp_hist is None:
        temp_hist = np.full_like(t, 288.15, dtype=float)

    if pos_hist is not None and len(pos_hist) == len(t):
        altitude_hist = pos_hist[:, 2] + float(env.elevation)
    else:
        altitude_hist = np.zeros_like(t, dtype=float) + float(env.elevation)

    usable = len(t) - 1
    total = min(sample_count + warmup_count, usable)
    indices = np.linspace(0, usable - 1, total, dtype=int)

    timings_ms = []
    u_prev = u_hist[0].copy()
    for n, i in enumerate(indices):
        dt = float(t[i + 1] - t[i])
        start = time.perf_counter()
        _, u_prev = sim.step(
            float(t[i]),
            dt,
            x_true[i],
            deriv[i],
            u_prev,
            float(altitude_hist[i]),
            float(temp_hist[i]),
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if n >= warmup_count:
            timings_ms.append(elapsed_ms)

    timings_ms = np.array(timings_ms, dtype=float)
    return {
        "samples": len(timings_ms),
        "mean_ms": float(np.mean(timings_ms)),
        "median_ms": float(np.median(timings_ms)),
        "p95_ms": float(np.percentile(timings_ms, 95)),
        "max_ms": float(np.max(timings_ms)),
    }


# ================================================================
# SECTION 1 — CHOOSE SIMULATION MODE
# ================================================================
# Pick simulation mode:
#   'dynamics_EKF_compare' — RocketPy truth + EKF with zero control + open-loop internal comparison.
#   'rocketpy_replay'      — backward-compatible alias for dynamics_EKF_compare.
#   'rocketpy_closedloop'  — RocketPy truth with controller feedback injected into RocketPy.
#   'ekf_only'             — Internal dynamics only, EKF only, no control.
#   'ekf_controlled'       — Internal dynamics only, EKF + control.
#   'monte_carlo'          — Batch robustness verification after the nominal setup is tuned.
simulation_mode = "ekf_controlled"

monte_carlo_config = MonteCarloConfig(
    base_mode="ekf_controlled",
    num_trials=50,
    seed=4242,
    output_dir=project_root / "results" / "monte_carlo",
)



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

if simulation_mode == "monte_carlo":
    run_monte_carlo(
        config=monte_carlo_config,
        env=env,
        build_internal_dynamics=build_internal_dynamics,
        build_controls_stack=build_controls_stack,
    )
    sys.exit(0)

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

step_benchmark = benchmark_flight_step(sim, results, env)
if step_benchmark is not None:
    frame_budget_ms = 10.0  # 100 Hz control loop
    print(
        "Flight-step benchmark "
        f"({step_benchmark['samples']} warm samples): "
        f"mean={step_benchmark['mean_ms']:.3f} ms, "
        f"median={step_benchmark['median_ms']:.3f} ms, "
        f"p95={step_benchmark['p95_ms']:.3f} ms, "
        f"max={step_benchmark['max_ms']:.3f} ms "
        f"({step_benchmark['mean_ms'] / frame_budget_ms * 100:.1f}% of 100 Hz frame)"
    )


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
