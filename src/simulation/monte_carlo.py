import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from controls.flight_computer import Flight_Computer_Sim
from rocketpy_adapter import Adapter


@dataclass
class MonteCarloConfig:
    """Configuration for post-tuning Monte Carlo verification runs."""

    base_mode: str = "ekf_controlled"
    num_trials: int = 50
    seed: int = 4242
    output_dir: Path = Path("results/monte_carlo")
    imu_noise_scale_bounds: tuple[float, float] = (0.75, 1.50)
    density_scale_bounds: tuple[float, float] = (0.95, 1.05)
    wind_offset_std_m_s: float = 3.0
    temperature_offset_std_K: float = 5.0


class AtmosphericPerturbation:
    """Small wrapper around a RocketPy environment for one randomized trial."""

    def __init__(
        self,
        base_env,
        density_scale: float,
        wind_x_offset_m_s: float,
        wind_y_offset_m_s: float,
        temperature_offset_K: float,
    ):
        self.base_env = base_env
        self.elevation = base_env.elevation
        self.density_scale = density_scale
        self.wind_x_offset_m_s = wind_x_offset_m_s
        self.wind_y_offset_m_s = wind_y_offset_m_s
        self.temperature_offset_K = temperature_offset_K

    def density(self, altitude):
        return self.density_scale * float(self.base_env.density(altitude))

    def gravity(self, altitude):
        return self.base_env.gravity(altitude)

    def wind_velocity_x(self, altitude):
        return float(self.base_env.wind_velocity_x(altitude)) + self.wind_x_offset_m_s

    def wind_velocity_y(self, altitude):
        return float(self.base_env.wind_velocity_y(altitude)) + self.wind_y_offset_m_s

    def temperature(self, altitude):
        return float(self.base_env.temperature(altitude)) + self.temperature_offset_K


def run_monte_carlo(
    config: MonteCarloConfig,
    env,
    build_internal_dynamics: Callable,
    build_controls_stack: Callable,
) -> list[dict]:
    """Run batch Monte Carlo verification and save a trial-summary CSV.

    The Monte Carlo harness intentionally rebuilds the dynamics, controls,
    IMU, and EKF objects for every trial so estimator covariance, random-walk
    state, command history, and cached controller state do not leak across
    trials. Each trial varies only IMU noise and atmospheric conditions.
    """
    if config.base_mode not in ("ekf_controlled", "ekf_only"):
        raise ValueError("Monte Carlo currently supports base_mode='ekf_controlled' or 'ekf_only'.")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"

    rng = np.random.default_rng(config.seed)
    rows = []
    start_time = time.time()

    for trial_idx in range(config.num_trials):
        trial_seed = int(rng.integers(0, 2**32 - 1))
        trial_rng = np.random.default_rng(trial_seed)
        np.random.seed(trial_seed)

        imu_noise_scale = float(trial_rng.uniform(*config.imu_noise_scale_bounds))
        density_scale = float(trial_rng.uniform(*config.density_scale_bounds))
        wind_x_offset = float(trial_rng.normal(0.0, config.wind_offset_std_m_s))
        wind_y_offset = float(trial_rng.normal(0.0, config.wind_offset_std_m_s))
        temperature_offset = float(trial_rng.normal(0.0, config.temperature_offset_std_K))
        trial_env = AtmosphericPerturbation(
            env,
            density_scale,
            wind_x_offset,
            wind_y_offset,
            temperature_offset,
        )

        try:
            internal_bundle = build_internal_dynamics()
            controls_bundle = build_controls_stack(internal_bundle)
            controls = controls_bundle["controls"]
            imu = controls_bundle["imu"]
            ekf = controls_bundle["ekf"]

            controls.set_env_from_rocketpy(trial_env)
            _scale_imu_noise(imu, imu_noise_scale)

            sim = Flight_Computer_Sim(controls, imu, ekf)
            adapter = Adapter(sim, simulation_type=config.base_mode)
            results = adapter.run()

            rows.append(
                _summarize_trial(
                    trial_idx,
                    trial_seed,
                    imu_noise_scale,
                    density_scale,
                    wind_x_offset,
                    wind_y_offset,
                    temperature_offset,
                    results,
                )
            )
        except Exception as exc:
            rows.append(
                _failed_trial_row(
                    trial_idx,
                    trial_seed,
                    imu_noise_scale,
                    density_scale,
                    wind_x_offset,
                    wind_y_offset,
                    temperature_offset,
                    exc,
                )
            )
            print(f"Monte Carlo trial {trial_idx:03d} failed: {exc}")

        print(
            f"Monte Carlo trial {trial_idx + 1}/{config.num_trials} complete "
            f"(seed={trial_seed})"
        )

    _write_summary_csv(summary_path, rows)
    _print_summary(rows, summary_path, time.time() - start_time)
    return rows


def _scale_imu_noise(imu, scale: float) -> None:
    imu.accel_std *= scale
    imu.gyro_std *= scale
    imu.accel_walk_sigma *= scale
    imu.gyro_walk_sigma *= scale
    imu.temp_noise_std *= scale


def _quat_tilt_deg(state_history: np.ndarray) -> np.ndarray:
    if len(state_history) == 0:
        return np.array([])

    q = np.asarray(state_history[:, 6:10], dtype=float)
    norms = np.linalg.norm(q, axis=1)
    norms[norms == 0.0] = 1.0
    q = q / norms[:, None]
    qx = q[:, 1]
    qy = q[:, 2]
    body_z_world_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    body_z_world_z = np.clip(body_z_world_z, -1.0, 1.0)
    return np.degrees(np.arccos(body_z_world_z))


def _summarize_trial(
    trial_idx: int,
    seed: int,
    imu_noise_scale: float,
    density_scale: float,
    wind_x_offset: float,
    wind_y_offset: float,
    temperature_offset: float,
    results: dict,
) -> dict:
    t = results.get("t", np.array([]))
    x_true = results.get("x_true", np.empty((0, 10)))
    xhat = results.get("xhat", np.empty((0, 10)))
    u = results.get("u", np.empty((0, 1)))
    pos = results.get("position", np.empty((0, 3)))

    if len(t) == 0 or len(x_true) == 0:
        raise ValueError("Trial produced no samples.")

    roll_true = x_true[:, 2]
    roll_est = xhat[:, 2] if len(xhat) == len(x_true) else np.full_like(roll_true, np.nan)
    roll_error = roll_est - roll_true
    u_abs = np.abs(u[:, 0]) if u.ndim == 2 and len(u) > 0 else np.array([0.0])
    tilt = _quat_tilt_deg(x_true)
    altitude = pos[:, 2] if pos.ndim == 2 and pos.shape[1] >= 3 and len(pos) == len(t) else np.zeros_like(t)

    return {
        "trial": trial_idx,
        "seed": seed,
        "imu_noise_scale": imu_noise_scale,
        "density_scale": density_scale,
        "wind_x_offset_m_s": wind_x_offset,
        "wind_y_offset_m_s": wind_y_offset,
        "temperature_offset_K": temperature_offset,
        "samples": len(t),
        "sim_time_s": float(t[-1]),
        "apogee_m_agl": float(np.max(altitude)),
        "max_abs_roll_rate_rad_s": float(np.max(np.abs(roll_true))),
        "final_abs_roll_rate_rad_s": float(abs(roll_true[-1])),
        "rms_roll_est_error_rad_s": float(np.sqrt(np.mean(roll_error**2))),
        "max_abs_canard_rad": float(np.max(u_abs)),
        "max_abs_canard_deg": float(np.rad2deg(np.max(u_abs))),
        "max_tilt_deg": float(np.max(tilt)) if len(tilt) else 0.0,
        "success": True,
        "error": "",
    }


def _failed_trial_row(
    trial_idx: int,
    seed: int,
    imu_noise_scale: float,
    density_scale: float,
    wind_x_offset: float,
    wind_y_offset: float,
    temperature_offset: float,
    exc: Exception,
) -> dict:
    return {
        "trial": trial_idx,
        "seed": seed,
        "imu_noise_scale": imu_noise_scale,
        "density_scale": density_scale,
        "wind_x_offset_m_s": wind_x_offset,
        "wind_y_offset_m_s": wind_y_offset,
        "temperature_offset_K": temperature_offset,
        "samples": 0,
        "sim_time_s": 0.0,
        "apogee_m_agl": np.nan,
        "max_abs_roll_rate_rad_s": np.nan,
        "final_abs_roll_rate_rad_s": np.nan,
        "rms_roll_est_error_rad_s": np.nan,
        "max_abs_canard_rad": np.nan,
        "max_abs_canard_deg": np.nan,
        "max_tilt_deg": np.nan,
        "success": False,
        "error": repr(exc),
    }


def _write_summary_csv(summary_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: list[dict], summary_path: Path, runtime_s: float) -> None:
    successful = [row for row in rows if row["success"]]
    print(f"\nMonte Carlo complete: {len(successful)}/{len(rows)} successful trials")
    print(f"Runtime: {runtime_s:.3f} seconds")
    print(f"Summary saved to: {summary_path}")

    if not successful:
        return

    final_roll = np.array([row["final_abs_roll_rate_rad_s"] for row in successful], dtype=float)
    max_canard = np.array([row["max_abs_canard_deg"] for row in successful], dtype=float)
    apogee = np.array([row["apogee_m_agl"] for row in successful], dtype=float)
    print(
        "Key metrics: "
        f"final |roll| mean={np.mean(final_roll):.4f} rad/s, "
        f"p95={np.percentile(final_roll, 95):.4f} rad/s; "
        f"max |canard| p95={np.percentile(max_canard, 95):.2f} deg; "
        f"apogee mean={np.mean(apogee):.1f} m AGL"
    )
