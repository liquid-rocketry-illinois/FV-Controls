from pathlib import Path
from contextlib import redirect_stdout
import io

import matplotlib.pyplot as plt
import numpy as np



def _body_vel_to_world(state_history: np.ndarray) -> np.ndarray:
    """Convert body-frame velocities [v1, v2, v3] in a state history array to
    world-frame velocities [vx, vy, vz] using the quaternion in each row.

    Args:
        state_history: shape (n, 10) — [w1,w2,w3, v1,v2,v3, qw,qx,qy,qz]

    Returns:
        np.ndarray shape (n, 3): world-frame [vx, vy, vz]
    """
    n = len(state_history)
    v_world = np.zeros((n, 3))
    for i in range(n):
        qw, qx, qy, qz = state_history[i, 6:10]
        norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        if norm > 1e-9:
            qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
        xx, yy, zz = qx*qx, qy*qy, qz*qz
        wx_, wy_, wz_ = qw*qx, qw*qy, qw*qz
        xy, xz, yz = qx*qy, qx*qz, qy*qz
        R_WB = np.array([
            [1-2*(yy+zz), 2*(xy-wz_),  2*(xz+wy_)],
            [2*(xy+wz_),  1-2*(xx+zz), 2*(yz-wx_)],
            [2*(xz-wy_),  2*(yz+wx_),  1-2*(xx+yy)]
        ])
        v_world[i] = R_WB @ state_history[i, 3:6]
    return v_world


def _body_vecs_to_world(state_history: np.ndarray, vec_history: np.ndarray) -> np.ndarray:
    """Rotate a body-frame vector history into world frame using each row's quaternion.

    Args:
        state_history: shape (n, 10) — [w1,w2,w3, v1,v2,v3, qw,qx,qy,qz]
        vec_history: shape (n, 3) — body-frame vectors

    Returns:
        np.ndarray shape (n, 3): world-frame vectors
    """
    n = len(state_history)
    vec_world = np.zeros((n, 3))
    for i in range(n):
        qw, qx, qy, qz = state_history[i, 6:10]
        norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        if norm > 1e-9:
            qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
        xx, yy, zz = qx * qx, qy * qy, qz * qz
        wx_, wy_, wz_ = qw * qx, qw * qy, qw * qz
        xy, xz, yz = qx * qy, qx * qz, qy * qz
        R_WB = np.array([
            [1 - 2 * (yy + zz), 2 * (xy - wz_),     2 * (xz + wy_)],
            [2 * (xy + wz_),     1 - 2 * (xx + zz), 2 * (yz - wx_)],
            [2 * (xz - wy_),     2 * (yz + wx_),     1 - 2 * (xx + yy)]
        ])
        vec_world[i] = R_WB @ vec_history[i]
    return vec_world


def _integrate_world_position(state_history: np.ndarray, t_history: np.ndarray) -> np.ndarray:
    """Reconstruct world-frame position from state history by integrating
    world-frame velocity over time.

    The returned position is relative to launch, so z is altitude AGL.
    """
    v_world = _body_vel_to_world(state_history)
    pos = np.zeros((len(state_history), 3))
    for i in range(1, len(state_history)):
        dt = float(t_history[i] - t_history[i - 1])
        pos[i] = pos[i - 1] + 0.5 * (v_world[i - 1] + v_world[i]) * dt
    return pos


def _plot_environment_profile(env, output_path: Path):
    """Save RocketPy's built-in atmospheric model plot when available."""
    if env is None or not hasattr(env, "plots"):
        return
    env.plots.atmospheric_model(filename=str(output_path / "environment_profile.png"))


def _rocketpy_info_text(rocketpy_adapter):
    """Capture RocketPy's built-in flight.info() output when available."""
    flight = getattr(rocketpy_adapter, "flight", None)
    if flight is None:
        return None

    buf = io.StringIO()
    with redirect_stdout(buf):
        flight.info()

    lines = []
    for line in buf.getvalue().splitlines():
        if "Maximum Dynamic Pressure" in line:
            continue
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


def _temperature_profile_from_altitude(controls, altitude_history: np.ndarray) -> np.ndarray:
    """Return temperature history using the registered environment when available."""
    env_T = getattr(controls, "_env_temperature_func", None)
    if env_T is None:
        return np.maximum(216.65, 288.15 - 0.0065 * altitude_history)

    temps = np.zeros(len(altitude_history))
    for i, alt in enumerate(altitude_history):
        try:
            temps[i] = float(env_T(float(alt)))
        except Exception:
            temps[i] = max(216.65, 288.15 - 0.0065 * float(alt))
    return temps


def _append_summary_block(
    lines,
    title,
    controls,
    state_history,
    pos_history,
    t_history,
    cp_hist,
    mach_hist,
    launch_elevation,
    launch_lat=None,
    launch_lon=None,
):
    """Append one trajectory summary block to the summary text."""
    v_mag = np.linalg.norm(state_history[:, 3:6], axis=1)
    v_world = _body_vel_to_world(state_history)
    a_world = np.gradient(v_world, t_history, axis=0)
    a_mag = np.linalg.norm(a_world, axis=1)

    CG_hist = np.array([controls.get_CG(ti) for ti in t_history])
    SM_hist = (cp_hist - CG_hist) / controls.d
    thrust_hist = np.array([float(controls.get_thrust(ti)[2]) for ti in t_history])
    mass_hist = np.array([controls.get_mass(ti) for ti in t_history])

    rail_t = controls.t_launch_rail_clearance
    rail_idx = int(np.argmin(np.abs(t_history - rail_t)))
    altitude_asl = pos_history[:, 2] + float(launch_elevation)

    v_rail = v_mag[rail_idx]
    SM_rail = SM_hist[rail_idx]
    TW_rail = thrust_hist[rail_idx] / (mass_hist[rail_idx] * controls.g)

    rail_alt = altitude_asl[rail_idx]
    if controls._env_wind_x_func is not None:
        wx_rail = float(controls._env_wind_x_func(rail_alt))
        wy_rail = float(controls._env_wind_y_func(rail_alt))
    else:
        wx_rail, wy_rail = 0.0, 0.0
    va_rail = controls._compute_body_airspeed(state_history[rail_idx], wx_rail, wy_rail)
    aoa_rail = np.degrees(np.arctan2(np.hypot(va_rail[0], va_rail[1]), va_rail[2] + 1e-9))

    apogee_idx = int(np.argmax(pos_history[:, 2]))
    t_apogee = t_history[apogee_idx]
    alt_apogee = pos_history[apogee_idx, 2]
    alt_apogee_asl = altitude_asl[apogee_idx]
    x_apogee = pos_history[apogee_idx, 0]
    y_apogee = pos_history[apogee_idx, 1]

    if controls._env_wind_x_func is not None:
        wx_apogee = float(controls._env_wind_x_func(alt_apogee_asl))
        wy_apogee = float(controls._env_wind_y_func(alt_apogee_asl))
    else:
        wx_apogee, wy_apogee = 0.0, 0.0
    va_apogee = controls._compute_body_airspeed(state_history[apogee_idx], wx_apogee, wy_apogee)
    v_apogee = float(np.linalg.norm(va_apogee))

    if launch_lat is not None and launch_lon is not None:
        lat_apogee = launch_lat + y_apogee / 111_320.0
        lon_apogee = launch_lon + x_apogee / (111_320.0 * np.cos(np.radians(launch_lat)))
    else:
        lat_apogee = None
        lon_apogee = None

    burn_mask = t_history <= controls.t_motor_burnout
    after_mask = ~burn_mask
    max_accel_burn = float(np.max(a_mag[burn_mask])) if burn_mask.any() else 0.0
    max_accel_after = float(np.max(a_mag[after_mask])) if after_mask.any() else 0.0

    W = 62
    SEP = "─" * W

    def row(label, val, unit=""):
        lines.append(f"  {label:<42s}  {val}  {unit}")

    lines.append(f"\n  {title}")
    lines.append(SEP)

    lines.append("\n  Rail Departure")
    lines.append(SEP)
    row("Rail Departure Time",               f"{rail_t:.3f}",   "s")
    row("Rail Departure Velocity",           f"{v_rail:.2f}",   "m/s")
    row("Rail Departure Stability Margin",   f"{SM_rail:.2f}",  "cal")
    row("Rail Departure Angle of Attack",    f"{aoa_rail:.2f}", "°")
    row("Rail Departure Thrust-to-Weight",   f"{TW_rail:.2f}")

    lines.append("\n  Apogee")
    lines.append(SEP)
    row("Apogee Time",                       f"{t_apogee:.3f}",   "s")
    row("Apogee Altitude AGL",               f"{alt_apogee:.1f}", "m")
    row("Apogee Freestream Speed",           f"{v_apogee:.2f}",   "m/s")
    row("Apogee X Position",                 f"{x_apogee:.1f}",   "m")
    row("Apogee Y Position",                 f"{y_apogee:.1f}",   "m")
    if lat_apogee is not None:
        row("Apogee Latitude",               f"{lat_apogee:.6f}", "°")
        row("Apogee Longitude",              f"{lon_apogee:.6f}", "°")

    lines.append("\n  Maximum Values")
    lines.append(SEP)
    row("Maximum Speed",                     f"{np.max(v_mag):.2f}", "m/s")
    row("Maximum Mach Number",               f"{np.max(mach_hist):.4f}")
    row("Maximum Acceleration (motor burn)", f"{max_accel_burn:.2f}", "m/s²")
    row("Maximum G-Load (motor burn)",       f"{max_accel_burn / controls.g:.2f}", "g")
    row("Maximum Acceleration (after burn)", f"{max_accel_after:.2f}", "m/s²")
    row("Maximum Stability Margin",          f"{np.max(SM_hist):.2f}", "cal")


def _print_flight_summary(
    results, controls, rocketpy_adapter, pos, t, mach, v_mag,
    cp_hist, cp_func, output_path, launch_elevation, launch_lat=None, launch_lon=None,
    upper_button_pos=None, lower_button_pos=None,
):
    """Save a flight summary text file to output_path."""
    # ── Write to file ────────────────────────────────────────────────────────
    W   = 62

    lines = []
    lines.append("")
    lines.append("=" * W)
    lines.append("  FLIGHT SUMMARY")
    lines.append("=" * W)

    rocketpy_text = _rocketpy_info_text(rocketpy_adapter)
    if rocketpy_text is not None:
        lines.append("\n  RocketPy Truth")
        lines.append("─" * W)
        lines.append("")
        lines.extend(rocketpy_text.splitlines())

    x_internal = results.get("x_internal")
    if x_internal is not None:
        pos_internal = _integrate_world_position(x_internal, t)
        alt_internal_asl = pos_internal[:, 2] + float(launch_elevation)
        temp_internal = _temperature_profile_from_altitude(controls, alt_internal_asl)
        sos_internal = np.sqrt(1.4 * 287.05 * temp_internal)
        mach_internal = np.linalg.norm(x_internal[:, 3:6], axis=1) / sos_internal
        aoa_internal = np.degrees(
            np.arctan2(np.sqrt(x_internal[:, 3] ** 2 + x_internal[:, 4] ** 2), x_internal[:, 5] + 1e-9)
        )
        cp_internal = np.array([cp_func(a, m) for a, m in zip(aoa_internal, mach_internal)])

        _append_summary_block(
            lines=lines,
            title="Internal Model",
            controls=controls,
            state_history=x_internal,
            pos_history=pos_internal,
            t_history=t,
            cp_hist=cp_internal,
            mach_hist=mach_internal,
            launch_elevation=launch_elevation,
            launch_lat=launch_lat,
            launch_lon=launch_lon,
        )

    lines.append("=" * W)
    lines.append("")

    summary_path = output_path / "flight_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Flight summary saved to {summary_path}")


def save_and_plot_results(
    results,
    controls,
    rocketpy_adapter,
    env,
    cp_func,
    drag_func,
    output_dir="results",
    results_filename="run1.npz",
    launch_lat=None,
    launch_lon=None,
    upper_button_pos=1.41,
    lower_button_pos=2.41,
):
    """Save raw results and generate all post-processing plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rocketpy_adapter.save_results(str(output_path / results_filename))

    t = results["t"]
    x_true = results["x_true"]
    xhat = results["xhat"]
    deriv = results["deriv"]
    P_diag = results["P_diag"]
    pos = results["position"]
    temp_K = results["temperature"]
    truth_label = (
        "RocketPy"
        if getattr(rocketpy_adapter, "sim_type", "") in ("dynamics_EKF_compare", "rocketpy_replay", "rocketpy_closedloop")
        else "Internal model"
    )

    v_mag = np.sqrt(x_true[:, 3] ** 2 + x_true[:, 4] ** 2 + x_true[:, 5] ** 2)
    sos = np.sqrt(1.4 * 287.05 * temp_K)
    mach = v_mag / sos

    aoa_deg_hist = np.degrees(
        np.arctan2(np.sqrt(x_true[:, 3] ** 2 + x_true[:, 4] ** 2), x_true[:, 5] + 1e-9)
    )
    cp_hist = np.array([cp_func(a, m) for a, m in zip(aoa_deg_hist, mach)])
    try:
        cd_hist = np.array([drag_func(m, ti) for m, ti in zip(mach, t)])
    except TypeError:
        cd_hist = np.array([drag_func(m) for m in mach])

    x_internal = results.get("x_internal")   # present in compare / closedloop modes

    fig1, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for i, (ax, label) in enumerate(zip(axes, ["w1 (rad/s)", "w2 (rad/s)", "w3 - Roll (rad/s)"])):
        ax.plot(t, x_true[:, i], label=truth_label)
        ax.plot(t, xhat[:, i], label="EKF estimate", linestyle="--")
        if x_internal is not None:
            ax.plot(t, x_internal[:, i], label="Internal model", linestyle=":", alpha=0.8)
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True)
    axes[-1].set_xlabel("Time (s)")
    plt.suptitle("Angular Velocities")
    plt.tight_layout()
    fig1.savefig(output_path / "angular_velocities.png", dpi=150)
    plt.close(fig1)

    # Convert body-frame velocities to world frame for truth, EKF, and internal model
    v_world_true = _body_vel_to_world(x_true)
    v_world_hat  = _body_vel_to_world(xhat)
    v_world_int  = _body_vel_to_world(x_internal) if x_internal is not None else None
    pos_hat      = _integrate_world_position(xhat, t)
    pos_internal = _integrate_world_position(x_internal, t) if x_internal is not None else None
    ekf_pos_label = "EKF dead-reckoned"

    fig2, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for i, (ax, label) in enumerate(zip(axes, ["vx (m/s)", "vy (m/s)", "vz - Vertical (m/s)"])):
        ax.plot(t, v_world_true[:, i], label=truth_label)
        ax.plot(t, v_world_hat[:, i],  label="EKF estimate", linestyle="--")
        if v_world_int is not None:
            ax.plot(t, v_world_int[:, i], label="Internal model", linestyle=":", alpha=0.8)
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True)
    axes[-1].set_xlabel("Time (s)")
    plt.suptitle("Linear Velocities (World Frame)")
    plt.tight_layout()
    fig2.savefig(output_path / "linear_velocities.png", dpi=150)
    plt.close(fig2)

    # ---- Velocity diagnostic: body-frame vs world-frame ----
    # Speed magnitude is rotation-invariant: if it is smooth during motor burn,
    # thrust is being applied correctly at every step.  Oscillations in vx/vy/vz
    # but a smooth speed curve mean the rocket is pitching/yawing (orientation
    # oscillation), NOT that thrust is missing.
    v3_body_true = x_true[:, 5]          # body-frame axial speed (should be smooth)
    speed_true   = v_mag                  # |v_body| = |v_world|, rotation-invariant
    v3_body_hat  = xhat[:, 5]
    speed_hat    = np.sqrt(xhat[:, 3]**2 + xhat[:, 4]**2 + xhat[:, 5]**2)

    t_burnout = controls.t_motor_burnout

    fig_diag, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(t, speed_true, label="truth |v|")
    axes[0].plot(t, speed_hat,  label="EKF |v|", linestyle="--")
    axes[0].axvline(t_burnout, color='r', linestyle=':', linewidth=0.8, label=f"burnout t={t_burnout}s")
    axes[0].set_ylabel("Speed |v| (m/s)")
    axes[0].set_title("If smooth → thrust OK.  If jagged → force computation issue.")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, v3_body_true, label="truth v3 body")
    axes[1].plot(t, v3_body_hat,  label="EKF v3 body",  linestyle="--")
    axes[1].axvline(t_burnout, color='r', linestyle=':', linewidth=0.8)
    axes[1].set_ylabel("v3 body-frame (m/s)\n(axial speed)")
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle("Velocity Diagnostic — Thrust Verification")
    plt.tight_layout()
    fig_diag.savefig(output_path / "velocity_diagnostic.png", dpi=150)
    plt.close(fig_diag)

    fig3, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    for i, (ax, label) in enumerate(zip(axes, ["qw", "qx", "qy", "qz"])):
        ax.plot(t, x_true[:, 6 + i], label=truth_label)
        ax.plot(t, xhat[:, 6 + i], label="EKF estimate", linestyle="--")
        if x_internal is not None:
            ax.plot(t, x_internal[:, 6 + i], label="Internal model", linestyle=":", alpha=0.8)
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True)
    axes[-1].set_xlabel("Time (s)")
    plt.suptitle("Quaternion Orientation")
    plt.tight_layout()
    fig3.savefig(output_path / "quaternions.png", dpi=150)
    plt.close(fig3)

    ang_accel_world = _body_vecs_to_world(x_true, deriv[:, 0:3])

    # deriv[:, 3:6] = vdot_body = F/mass − ω×v  (body-frame rotating-frame equation).
    # Simply rotating vdot_body to world frame gives R_WB@(F/mass − ω×v), which
    # includes a centripetal term that oscillates at the pitching frequency (~30 m/s²
    # at v=300 m/s, ω=0.1 rad/s) and makes ax/ay look jagged.
    # True world-frame acceleration: a_world = R_WB @ F/mass = R_WB @ (vdot_body + ω×v).
    omega_hist    = x_true[:, 0:3]                          # [w1, w2, w3]
    v_body_hist   = x_true[:, 3:6]                          # [v1, v2, v3]
    omega_cross_v = np.cross(omega_hist, v_body_hist)       # ω×v, shape (n, 3)
    lin_accel_world = _body_vecs_to_world(x_true, deriv[:, 3:6] + omega_cross_v)

    fig4, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    accel_labels = [
        ["alpha_x (rad/s^2)", "alpha_y (rad/s^2)", "alpha_z (rad/s^2)"],
        ["a_x (m/s^2)", "a_y (m/s^2)", "a_z (m/s^2)"],
    ]
    for row in range(2):
        for col in range(3):
            ax = axes[row, col]
            if row == 0:
                ax.plot(t, ang_accel_world[:, col])
            else:
                ax.plot(t, lin_accel_world[:, col])
            ax.set_ylabel(accel_labels[row][col])
            ax.grid(True)
            if row == 1:
                ax.set_xlabel("Time (s)")
    plt.suptitle("Accelerations (World Frame)")
    plt.tight_layout()
    fig4.savefig(output_path / "accelerations.png", dpi=150)
    plt.close(fig4)

    fig5, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(t, np.rad2deg(results["roll_true"]), label=truth_label)
    axes[0].plot(t, np.rad2deg(results["roll_est"]), label="EKF estimate", linestyle="--")
    axes[0].set_ylabel("Roll rate (deg/s)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, np.rad2deg(results["roll_true"] - results["roll_est"]))
    axes[1].set_ylabel("Roll rate error (deg/s)")
    axes[1].grid(True)
    axes[1].axhline(0, color="k", linewidth=0.5)

    axes[2].plot(t, np.rad2deg(np.sqrt(P_diag[:, 2])), color="orange")
    axes[2].set_ylabel("w3 std dev (deg/s)\n[EKF uncertainty]")
    axes[2].grid(True)

    axes[3].plot(t, np.rad2deg(results["u"][:, 0]))
    axes[3].set_ylabel("Canard deflection (deg)")
    axes[3].grid(True)
    axes[3].set_xlabel("Time (s)")

    plt.suptitle("Kalman Filter & Roll Control")
    plt.tight_layout()
    fig5.savefig(output_path / "kalman_filter.png", dpi=150)
    plt.close(fig5)

    canard_deg = np.rad2deg(results["u"][:, 0])
    canard_rate = np.gradient(canard_deg, t)
    max_deg = np.rad2deg(controls.max_input)

    fig6, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(t, canard_deg)
    axes[0].axhline(max_deg, color="r", linestyle="--", linewidth=0.8, label=f"+{max_deg:.1f} deg limit")
    axes[0].axhline(-max_deg, color="r", linestyle="--", linewidth=0.8, label=f"-{max_deg:.1f} deg limit")
    axes[0].axhline(0, color="k", linewidth=0.4)
    axes[0].set_ylabel("Canard deflection (deg)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, canard_rate, color="tab:orange")
    axes[1].axhline(0, color="k", linewidth=0.4)
    axes[1].set_ylabel("Deflection rate (deg/s)")
    axes[1].grid(True)

    axes[2].plot(t, np.rad2deg(results["roll_true"]), label="Roll rate (deg/s)")
    axes[2].plot(t, canard_deg, linestyle="--", label="Canard angle (deg)")
    axes[2].axhline(0, color="k", linewidth=0.4)
    axes[2].set_ylabel("Roll rate vs canard angle")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend()
    axes[2].grid(True)

    plt.suptitle("Canard Deflection")
    plt.tight_layout()
    fig6.savefig(output_path / "canard_angle.png", dpi=150)
    plt.close(fig6)

    fig7, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(t, mach)
    axes[0].set_ylabel("Mach number")
    axes[0].grid(True)

    axes[1].plot(t, cd_hist)
    axes[1].set_ylabel("Drag coefficient Cd")
    axes[1].grid(True)

    axes[2].plot(t, cp_hist)
    axes[2].set_ylabel("CP location (m from nose)")
    axes[2].grid(True)
    axes[2].set_xlabel("Time (s)")

    plt.suptitle("Aerodynamics - Drag and Center of Pressure")
    plt.tight_layout()
    fig7.savefig(output_path / "aero.png", dpi=150)
    plt.close(fig7)

    fig8 = plt.figure(figsize=(15, 5))
    ax1 = fig8.add_subplot(131)
    ax1.plot(t, pos[:, 2], label=truth_label)
    ax1.plot(t, pos_hat[:, 2], label=ekf_pos_label, linestyle="--")
    if pos_internal is not None:
        ax1.plot(t, pos_internal[:, 2], label="Internal model", linestyle=":", alpha=0.8)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Altitude AGL (m)")
    ax1.set_title("Altitude AGL vs Time (EKF via integrated velocity)")
    ax1.grid(True)
    ax1.legend()

    ax2 = fig8.add_subplot(132, projection="3d")
    ax2.plot(pos[:, 0], pos[:, 1], pos[:, 2], label=truth_label)
    ax2.plot(pos_hat[:, 0], pos_hat[:, 1], pos_hat[:, 2], label=ekf_pos_label, linestyle="--")
    if pos_internal is not None:
        ax2.plot(
            pos_internal[:, 0], pos_internal[:, 1], pos_internal[:, 2],
            label="Internal model", linestyle=":", alpha=0.8
        )
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_zlabel("Altitude AGL (m)")
    ax2.set_title("3D Trajectory")
    if pos_internal is not None:
        ax2.legend()

    ax3 = fig8.add_subplot(133)
    ax3.plot(pos[:, 0], pos[:, 1], label=truth_label)
    ax3.plot(pos_hat[:, 0], pos_hat[:, 1], label=ekf_pos_label, linestyle="--")
    if pos_internal is not None:
        ax3.plot(pos_internal[:, 0], pos_internal[:, 1], label="Internal model", linestyle=":", alpha=0.8)
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    ax3.set_title("Ground Track (EKF via integrated velocity)")
    ax3.grid(True)
    ax3.set_aspect("equal")
    ax3.legend()

    plt.suptitle("Trajectory")
    plt.tight_layout()
    fig8.savefig(output_path / "trajectory.png", dpi=150)
    plt.close(fig8)

    # Tilt angle: angle between rocket body z-axis and world vertical (Z-up)
    # Body z in world frame = 3rd column of R_WB, derived from quaternion:
    #   bz_world = [2(qx·qz + qw·qy), 2(qy·qz − qw·qx), 1 − 2(qx² + qy²)]
    qw_h = x_true[:, 6]; qx_h = x_true[:, 7]
    qy_h = x_true[:, 8]; qz_h = x_true[:, 9]
    bz_world_z = 1.0 - 2.0 * (qx_h**2 + qy_h**2)
    tilt_deg = np.degrees(np.arccos(np.clip(bz_world_z, -1.0, 1.0)))

    fig9, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, tilt_deg, label=truth_label)
    if x_internal is not None:
        bz_int = 1.0 - 2.0 * (x_internal[:, 7]**2 + x_internal[:, 8]**2)
        ax.plot(t, np.degrees(np.arccos(np.clip(bz_int, -1.0, 1.0))),
                label="Internal model", linestyle=":", alpha=0.8)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tilt from vertical (deg)")
    ax.set_title("Rocket Orientation — Tilt Angle from Vertical (World Z-axis)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    fig9.savefig(output_path / "tilt_angle.png", dpi=150)
    plt.close(fig9)

    _plot_environment_profile(env, output_path)

    _print_flight_summary(
        results, controls, rocketpy_adapter, pos, t, mach, v_mag,
        cp_hist, cp_func, output_path, float(getattr(env, "elevation", 0.0) or 0.0),
        launch_lat=launch_lat,
        launch_lon=launch_lon,
        upper_button_pos=upper_button_pos,
        lower_button_pos=lower_button_pos,
    )
