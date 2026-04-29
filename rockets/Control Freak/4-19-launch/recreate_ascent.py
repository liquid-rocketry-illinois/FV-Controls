"""Rebuild the 4/19 Control Freak ascent and estimate main-fin cant.

Project/body axes used here:
    x: up
    y: horizontal west, positive west
    z: horizontal north, positive north

The flight CSV has invalid GPS during ascent, so the measured 3-D trajectory is
baro height plus horizontal range reconstructed from speed and tilt. The rocket
was observed flying mostly west and slightly north, so that range is projected
onto +y/+z with the configurable --north-fraction.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


LAUNCH_DIR = Path(__file__).resolve().parent
CONTROL_FREAK_ROOT = LAUNCH_DIR.parent
PROJECT_ROOT = CONTROL_FREAK_ROOT.parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

os.environ.setdefault("TMPDIR", str(LAUNCH_DIR / ".tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(LAUNCH_DIR / ".matplotlib-cache"))
Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

for path in (CONTROL_FREAK_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim.internal_dynamics_setup import aero_direct_params, fin_params, rocket_params
from sim.rocketpy_setup import build_rocketpy_stack, launch_elevation


DEFAULT_LOG = LAUNCH_DIR / "2026-04-19-serial-11224-flight-0001-via-16793.csv"


@dataclass(frozen=True)
class AscentLog:
    t: np.ndarray
    x_up: np.ndarray
    y_west: np.ndarray
    z_north: np.ndarray
    speed: np.ndarray
    tilt_deg: np.ndarray
    roll_rate_deg_s: np.ndarray
    state: np.ndarray


@dataclass(frozen=True)
class SimAscent:
    t: np.ndarray
    x_up: np.ndarray
    y_west: np.ndarray
    z_north: np.ndarray
    speed: np.ndarray
    roll_rate_deg_s: np.ndarray


def read_ascent_log(csv_path: Path, north_fraction: float) -> AscentLog:
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8"), skipinitialspace=True))

    t = np.array([float(row["time"]) for row in rows])
    height = np.array([float(row["height"]) for row in rows])
    speed = np.array([float(row["speed"]) for row in rows])
    tilt = np.array([float(row["tilt"]) for row in rows])
    roll_rate = np.array([float(row["gyro_roll"]) for row in rows])
    state = np.array([row["state_name"].strip() for row in rows])

    apogee_i = int(np.argmax(height))
    keep = np.arange(len(t)) <= apogee_i
    t, height, speed, tilt, roll_rate, state = (
        t[keep],
        height[keep],
        speed[keep],
        tilt[keep],
        roll_rate[keep],
        state[keep],
    )

    x_up = height - float(np.median(height[: min(8, len(height))]))
    moving = np.flatnonzero((speed > 5.0) | (x_up > 3.0))
    t = t - float(t[moving[0] if len(moving) else 0])

    order = np.argsort(t)
    t, x_up, speed, tilt, roll_rate, state = (
        t[order],
        x_up[order],
        speed[order],
        tilt[order],
        roll_rate[order],
        state[order],
    )
    unique_t, unique_i = np.unique(t, return_index=True)
    t, x_up, speed, tilt, roll_rate, state = (
        unique_t,
        x_up[unique_i],
        speed[unique_i],
        tilt[unique_i],
        roll_rate[unique_i],
        state[unique_i],
    )

    horizontal_range = integrate_range(t, speed, tilt)
    north_fraction = float(np.clip(north_fraction, 0.0, 0.95))
    west_fraction = float(np.sqrt(1.0 - north_fraction**2))

    return AscentLog(
        t=t,
        x_up=x_up,
        y_west=horizontal_range * west_fraction,
        z_north=horizontal_range * north_fraction,
        speed=speed,
        tilt_deg=tilt,
        roll_rate_deg_s=roll_rate,
        state=state,
    )


def integrate_range(t: np.ndarray, speed: np.ndarray, tilt_deg: np.ndarray) -> np.ndarray:
    horizontal_speed = np.maximum(speed, 0.0) * np.sin(np.deg2rad(np.clip(tilt_deg, 0.0, 90.0)))
    horizontal_range = np.zeros_like(t)
    for i in range(1, len(t)):
        dt = max(float(t[i] - t[i - 1]), 0.0)
        horizontal_range[i] = horizontal_range[i - 1] + 0.5 * (
            horizontal_speed[i - 1] + horizontal_speed[i]
        ) * dt
    return horizontal_range


def rocketpy_ascent(main_fin_cant_deg: float) -> SimAscent:
    flight = build_rocketpy_stack(main_fin_cant_angle=main_fin_cant_deg)["flight"]
    solution = np.asarray(flight.solution, dtype=float)

    t = solution[:, 0]
    east = solution[:, 1]
    north = solution[:, 2]
    altitude_asl = solution[:, 3]
    vx = solution[:, 4]
    vy = solution[:, 5]
    vz = solution[:, 6]
    wz = solution[:, 13]

    return SimAscent(
        t=t,
        x_up=altitude_asl - float(launch_elevation),
        y_west=-east,
        z_north=north,
        speed=np.sqrt(vx**2 + vy**2 + vz**2),
        roll_rate_deg_s=np.rad2deg(wz),
    )


def estimate_roll_gyro_bias(log: AscentLog) -> float:
    still = (log.t <= 0.0) | (log.speed <= 1.0)
    return float(np.median(log.roll_rate_deg_s[still])) if np.any(still) else 0.0


def isa_density(height_agl_m: float) -> float:
    altitude_m = float(launch_elevation) + max(float(height_agl_m), 0.0)
    temperature_k = 288.15 - 0.0065 * altitude_m
    pressure_pa = 101325.0 * (temperature_k / 288.15) ** 5.255877
    return pressure_pa / (287.05 * temperature_k)


def estimate_internal_cant(log: AscentLog, gyro_bias_deg_s: float, fit_until_s: float) -> float:
    """Estimate fin_params['delta'] from the internal roll equation."""
    roll_rate = np.deg2rad(log.roll_rate_deg_s - gyro_bias_deg_s)
    roll_accel = np.gradient(roll_rate, log.t)

    I3_0 = float(rocket_params["I_3"])
    I3_f = float(rocket_params["I_3_f"])
    burn_time = float(rocket_params["t_motor_burnout"])
    diameter = float(rocket_params["d"])

    fin_count = float(fin_params["N"])
    root_chord = float(fin_params["Cr"])
    tip_chord = float(fin_params["Ct"])
    span = float(fin_params["s"])
    cnalpha_per_fin = float(aero_direct_params["fin_cn"]) / fin_count

    area = np.pi * (diameter / 2.0) ** 2
    gamma = tip_chord / root_chord
    body_radius = diameter / 2.0
    tau = (span + body_radius) / body_radius
    y_ma = span / 3.0 * (1.0 + 2.0 * gamma) / (1.0 + gamma)

    asin_term = np.arcsin((tau**2 - 1.0) / (tau**2 + 1.0))
    k_forcing = (1.0 / np.pi**2) * (
        (np.pi**2 / 4.0) * ((tau + 1.0) ** 2 / tau**2)
        + (np.pi * (tau**2 + 1.0) ** 2 / (tau**2 * (tau - 1.0) ** 2)) * asin_term
        - (2.0 * np.pi * (tau + 1.0)) / (tau * (tau - 1.0))
        + ((tau**2 + 1.0) ** 2 / (tau**2 * (tau - 1.0) ** 2)) * asin_term**2
        - (4.0 * (tau + 1.0) / (tau * (tau - 1.0))) * asin_term
        + (8.0 / (tau - 1.0) ** 2) * np.log((tau**2 + 1.0) / (2.0 * tau))
    )

    trap_integral = span / 12.0 * (
        (root_chord + 3.0 * tip_chord) * span**2
        + 4.0 * (root_chord + 2.0 * tip_chord) * span * body_radius
        + 6.0 * (root_chord + tip_chord) * body_radius**2
    )
    k_damping = 1.0 + (
        (tau - gamma) / tau - (1.0 - gamma) / (tau - 1.0) * np.log(tau)
    ) / (
        (tau + 1.0) * (tau - gamma) / 2.0
        - (1.0 - gamma) * (tau**3 - 1.0) / (3.0 * (tau - 1.0))
    )

    forcing_coeff = []
    required_moment = []
    fit = (log.t >= 0.0) & (log.t <= fit_until_s) & (log.speed > 50.0)

    for t, height, speed, w, wdot in zip(
        log.t[fit],
        log.x_up[fit],
        log.speed[fit],
        roll_rate[fit],
        roll_accel[fit],
    ):
        q_dyn = 0.5 * isa_density(height) * speed**2
        I3 = I3_0 - (I3_0 - I3_f) / burn_time * t if t <= burn_time else I3_f
        I3_dot = -(I3_0 - I3_f) / burn_time if t <= burn_time else 0.0

        forcing = k_forcing * q_dyn * fin_count * (y_ma + body_radius) * cnalpha_per_fin * area
        c_ldw = 2.0 * fin_count * cnalpha_per_fin / (area * diameter**2) * trap_integral
        damping = k_damping * q_dyn * area * diameter * c_ldw * diameter / (2.0 * speed)

        forcing_coeff.append(forcing)
        required_moment.append(I3 * wdot + I3_dot * w + damping * w)

    forcing_coeff = np.asarray(forcing_coeff)
    required_moment = np.asarray(required_moment)
    delta_rad = float(np.dot(forcing_coeff, required_moment) / np.dot(forcing_coeff, forcing_coeff))
    return float(np.rad2deg(delta_rad))


def roll_rmse(log: AscentLog, sim: SimAscent, gyro_bias_deg_s: float, fit_until_s: float) -> float:
    fit = (log.t >= 0.0) & (log.t <= fit_until_s) & (log.speed > 20.0)
    sim_roll = np.interp(log.t[fit], sim.t, sim.roll_rate_deg_s)
    measured_roll = log.roll_rate_deg_s[fit] - gyro_bias_deg_s
    return float(np.sqrt(np.mean((sim_roll - measured_roll) ** 2)))


def fit_rocketpy_cant(
    log: AscentLog,
    gyro_bias_deg_s: float,
    fit_until_s: float,
    center_deg: float,
    half_width_deg: float,
    step_deg: float,
) -> tuple[float, SimAscent, list[tuple[float, float]]]:
    cant_values = np.arange(center_deg - half_width_deg, center_deg + half_width_deg + 0.5 * step_deg, step_deg)
    sweep = []
    best_cant = float(cant_values[0])
    best_sim = rocketpy_ascent(best_cant)
    best_error = roll_rmse(log, best_sim, gyro_bias_deg_s, fit_until_s)
    sweep.append((best_cant, best_error))

    for cant in cant_values[1:]:
        sim = rocketpy_ascent(float(cant))
        error = roll_rmse(log, sim, gyro_bias_deg_s, fit_until_s)
        sweep.append((float(cant), error))
        if error < best_error:
            best_cant, best_sim, best_error = float(cant), sim, error

    return best_cant, best_sim, sweep


def save_reconstruction_csv(path: Path, log: AscentLog, sim: SimAscent, gyro_bias_deg_s: float) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "t_s",
                "log_x_up_m",
                "log_y_west_m",
                "log_z_north_m",
                "sim_x_up_m",
                "sim_y_west_m",
                "sim_z_north_m",
                "log_speed_m_s",
                "sim_speed_m_s",
                "log_roll_rate_deg_s_bias_corrected",
                "sim_roll_rate_deg_s",
                "state",
            ]
        )
        for i, t in enumerate(log.t):
            writer.writerow(
                [
                    f"{t:.6f}",
                    f"{log.x_up[i]:.6f}",
                    f"{log.y_west[i]:.6f}",
                    f"{log.z_north[i]:.6f}",
                    f"{np.interp(t, sim.t, sim.x_up):.6f}",
                    f"{np.interp(t, sim.t, sim.y_west):.6f}",
                    f"{np.interp(t, sim.t, sim.z_north):.6f}",
                    f"{log.speed[i]:.6f}",
                    f"{np.interp(t, sim.t, sim.speed):.6f}",
                    f"{log.roll_rate_deg_s[i] - gyro_bias_deg_s:.6f}",
                    f"{np.interp(t, sim.t, sim.roll_rate_deg_s):.6f}",
                    log.state[i],
                ]
            )


def save_plots(output_dir: Path, log: AscentLog, sim: SimAscent, sweep: list[tuple[float, float]], gyro_bias_deg_s: float) -> None:
    fig = plt.figure(figsize=(11, 8))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax3d.plot(log.y_west, log.z_north, log.x_up, label="log reconstruction")
    ax3d.plot(sim.y_west, sim.z_north, sim.x_up, label="RocketPy")
    ax3d.set_xlabel("y west (m)")
    ax3d.set_ylabel("z north (m)")
    ax3d.set_zlabel("x up (m)")
    ax3d.legend()

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(log.t, log.x_up, label="log")
    ax.plot(sim.t, sim.x_up, label="RocketPy")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("x up (m)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = fig.add_subplot(2, 2, 3)
    ax.plot(log.t, log.speed, label="log")
    ax.plot(sim.t, sim.speed, label="RocketPy")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("speed (m/s)")
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(2, 2, 4)
    ax.plot(log.t, log.roll_rate_deg_s - gyro_bias_deg_s, label="log")
    ax.plot(log.t, np.interp(log.t, sim.t, sim.roll_rate_deg_s), label="RocketPy")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("roll rate (deg/s)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "ascent_trajectory.png", dpi=180)
    plt.close(fig)

    sweep_arr = np.asarray(sweep)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sweep_arr[:, 0], sweep_arr[:, 1], marker="o", markersize=3)
    ax.set_xlabel("RocketPy main-fin cant (deg)")
    ax.set_ylabel("roll-rate RMSE (deg/s)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "cant_angle_fit.png", dpi=180)
    plt.close(fig)


def save_summary(
    path: Path,
    log: AscentLog,
    sim: SimAscent,
    internal_cant_deg: float,
    rocketpy_cant_deg: float,
    gyro_bias_deg_s: float,
    fit_until_s: float,
    north_fraction: float,
) -> None:
    apogee_i = int(np.argmax(log.x_up))
    sim_apogee_i = int(np.argmax(sim.x_up))
    lines = [
        "Control Freak 4/19 ascent reconstruction",
        "",
        "Coordinate convention: x up, y west, z north",
        f"Assumed north fraction of horizontal travel: {north_fraction:.3f}",
        f"Roll gyro bias removed: {gyro_bias_deg_s:+.3f} deg/s",
        f"Roll fit window: 0 to {fit_until_s:.2f} s",
        "",
        f"Internal dynamics cant estimate for fin_params['delta']: {internal_cant_deg:+.4f} deg",
        f"RocketPy cant sweep best fit: {rocketpy_cant_deg:+.4f} deg",
        "",
        f"Log apogee: x={log.x_up[apogee_i]:.1f} m at t={log.t[apogee_i]:.2f} s",
        f"Log apogee horizontal: y={log.y_west[apogee_i]:.1f} m west, z={log.z_north[apogee_i]:.1f} m north",
        f"RocketPy apogee: x={sim.x_up[sim_apogee_i]:.1f} m at t={sim.t[sim_apogee_i]:.2f} s",
        "",
        "Note: measured y/z are reconstructed from speed and tilt because ascent GPS was invalid.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output-dir", type=Path, default=LAUNCH_DIR)
    parser.add_argument("--north-fraction", type=float, default=0.25)
    parser.add_argument("--fit-until", type=float, default=8.0)
    parser.add_argument("--sweep-half-width", type=float, default=0.5)
    parser.add_argument("--sweep-step", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log = read_ascent_log(args.csv, args.north_fraction)
    gyro_bias = estimate_roll_gyro_bias(log)
    internal_cant = estimate_internal_cant(log, gyro_bias, args.fit_until)
    rocketpy_cant, sim, sweep = fit_rocketpy_cant(
        log,
        gyro_bias,
        args.fit_until,
        center_deg=internal_cant,
        half_width_deg=args.sweep_half_width,
        step_deg=args.sweep_step,
    )

    save_reconstruction_csv(args.output_dir / "ascent_reconstruction.csv", log, sim, gyro_bias)
    save_plots(args.output_dir, log, sim, sweep, gyro_bias)
    save_summary(
        args.output_dir / "cant_angle_summary.txt",
        log,
        sim,
        internal_cant,
        rocketpy_cant,
        gyro_bias,
        args.fit_until,
        args.north_fraction,
    )

    print(f"Internal dynamics delta: {internal_cant:+.4f} deg")
    print(f"RocketPy sweep best fit: {rocketpy_cant:+.4f} deg")
    print(f"Outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
