"""
Test.py — End-to-end integration test for RocketSim.

Implements a full Controls object + RocketPy simulation using example values
for a small competition rocket. Intended to verify that the full pipeline runs:
    Controls setup → EOM derivation → RocketPy Flight → controller loop → CSV export

Run from project root:
    python src/Test.py
Or from src/:
    python Test.py

NOTE: controls.define_eom() derives symbolic equations and takes ~15-60s on first run.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from rocketpy import Environment, Rocket, SolidMotor
from rocketpy.control import _Controller

from controls.controls import Controls
from rocket import CanardSurface, RocketSim
from roll_force import RollForce


# =============================================================================
# ROCKET PARAMETERS (example values — replace with your measured values)
# =============================================================================
# Coordinate convention: nose at top, +z body axis points out the nozzle (tail).
# All positions below are measured from the NOSE TIP unless noted.

# --- Geometry ---
DIAMETER   = 0.098   # m  — airframe outer diameter
RADIUS     = DIAMETER / 2
L_NE       = 2.50    # m  — total length, nose tip to nozzle exit

# --- Mass (rocket body only, without motor) ---
M_BODY     = 9.5     # kg — rocket dry mass without motor casing

# --- Moments of inertia (rocket body only, without motor) ---
I_BODY_LONG = 3.8    # kg·m² — longitudinal (pitch/yaw), body without motor
I_BODY_ROLL = 0.040  # kg·m² — roll (axial), approximately constant

# --- CG of rocket body without motor, measured from NOSE TIP ---
X_CG_BODY  = 1.05   # m  — typical for a top-heavy tube

# --- Full rocket (body + motor), time-varying during burn ---
M_0        = 15.0   # kg — wet mass at liftoff (body + motor + propellant)
M_F        = 10.5   # kg — dry mass at burnout (body + motor casing, no propellant)
M_P        = 4.5    # kg — propellant mass
I_0        = 5.5    # kg·m² — longitudinal inertia at liftoff
I_F        = 4.0    # kg·m² — longitudinal inertia at burnout
I_3        = 0.050  # kg·m² — roll inertia (approximately constant)

# --- CG of full rocket, measured from NOSE TIP ---
X_CG_0     = 1.20   # m  — at liftoff (propellant shifts CG aft)
X_CG_F     = 1.10   # m  — at burnout

# --- Aerodynamics ---
C_D              = 0.45   # drag coefficient (constant, subsonic)
CNALPHA_ROCKET   = 10.0   # rocket body normal-force coefficient derivative (1/rad)

# CP as a function of AoA in degrees — constant approximation is fine for small AoA.
# Replace with a polynomial from your OpenRocket data if available.
def CP_FUNC(AoA_deg):
    """Center of pressure location from nose tip (m). Constant approximation."""
    return 1.50   # m from nose — gives ~4 caliber stability margin

# --- Fins ---
N_FINS         = 4
FIN_ROOT_CHORD = 0.20   # m
FIN_TIP_CHORD  = 0.08   # m
FIN_SPAN       = 0.12   # m
CNALPHA_FIN    = 2.5    # per fin, normal-force coefficient derivative (1/rad)
FIN_CANT_DEG   = 0.0    # degrees — no passive cant (roll driven by canards)

# --- Timing ---
T_BURNOUT     = 3.5    # s — motor burnout
T_RAIL_CLEAR  = 0.30   # s — time to leave launch rail
T_APOGEE      = 28.0   # s — estimated time to apogee

# --- Thrust curve (example K-class motor, ~2000 N avg) ---
# Format: parallel arrays of time (s) and thrust (N).
THRUST_TIMES  = np.array([0.00, 0.05, 0.20, 1.00, 3.00, 3.40, 3.50, 3.60])
THRUST_FORCES = np.array([0.00, 1800, 2200, 2100, 1900, 1200,  300,  0.00])

# --- Environment ---
V_WIND = [0.0, 0.0]   # m/s [x, y] cross-wind (zero for now)
RHO    = 1.225         # kg/m³ — sea-level air density
G      = 9.81          # m/s²

# --- Control ---
SAMPLING_RATE = 40.0                  # Hz
DT            = 1.0 / SAMPLING_RATE
MAX_DEFLECTION = np.deg2rad(8.0)      # rad — hard canard limit

# Initial state: upright, at rest, unit quaternion
X0 = np.array([
    0.0, 0.0, 0.0,      # w1 w2 w3  (rad/s)
    0.0, 0.0, 0.0,      # v1 v2 v3  (m/s, body frame)
    1.0, 0.0, 0.0, 0.0  # qw qx qy qz  (upright)
])
U0 = np.array([0.0])   # initial canard deflection (rad)

# Motor parameters (example — fill in with your actual motor data)
MOTOR_DRY_MASS = 0.55   # kg — motor casing
MOTOR_NOZZLE_RADIUS    = 0.025  # m
MOTOR_THROAT_RADIUS    = 0.012  # m
MOTOR_GRAIN_NUMBER     = 3
MOTOR_GRAIN_DENSITY    = 1700   # kg/m³  (APCP)
MOTOR_GRAIN_OD         = 0.030  # m  outer radius
MOTOR_GRAIN_ID         = 0.012  # m  inner (burn port) radius
MOTOR_GRAIN_HEIGHT     = 0.080  # m  per grain
MOTOR_GRAIN_SEP        = 0.003  # m  separation between grains
# Positions measured from nozzle exit (RocketPy "nozzle_to_combustion_chamber")
MOTOR_GRAINS_CG_POS    = 0.12   # m
MOTOR_DRY_CG_POS       = 0.14   # m
MOTOR_NOZZLE_POS       = 0.00   # m  (nozzle is position reference)

# Rocket body CG measured from TAIL (for RocketPy "tail_to_nose" frame)
X_CG_BODY_FROM_TAIL = L_NE - X_CG_BODY   # m


# =============================================================================
# BUILD CONTROLS OBJECT
# =============================================================================

print("Setting up Controls object...")

controls = Controls(IREC_COMPLIANT=True, rocket_name="ExampleRocket")

controls.setRocketParams(
    I_0=I_0,
    I_f=I_F,
    I_3=I_3,
    x_CG_0=X_CG_0,
    x_CG_f=X_CG_F,
    m_0=M_0,
    m_f=M_F,
    m_p=M_P,
    d=DIAMETER,
    L_ne=L_NE,
    C_d=C_D,
    Cnalpha_rocket=CNALPHA_ROCKET,
    t_launch_rail_clearance=T_RAIL_CLEAR,
    t_motor_burnout=T_BURNOUT,
    t_estimated_apogee=T_APOGEE,
    CP_func=CP_FUNC,
)

controls.setFinParams(
    N=N_FINS,
    Cr=FIN_ROOT_CHORD,
    Ct=FIN_TIP_CHORD,
    s=FIN_SPAN,
    Cnalpha_fin=CNALPHA_FIN,
    delta=FIN_CANT_DEG,
)

controls.setThrustCurve(
    thrust_times=THRUST_TIMES,
    thrust_forces=THRUST_FORCES,
)

controls.setEnvParams(v_wind=V_WIND, rho=RHO, g=G)
controls.setSimParams(dt=DT, x0=X0)
controls.set_controls_params(u0=U0, max_input=MAX_DEFLECTION)

# --- Derive symbolic equations of motion (slow on first run, ~15-60s) ---
print("Deriving equations of motion (this takes ~15-60s)...")
controls.define_eom()
print("EOM ready.")

# --- Canard moment function ---
# Simple roll moment model: M_roll = C_canard * deflection * axial_velocity
# Replace with your actual aerodynamic model.
C_CANARD = 0.12   # N·m / (rad · m/s) — empirical canard moment coefficient

def M_controls_func(state, u):
    """Roll moment from canard deflection.

    Args:
        state: [w1, w2, w3, v1, v2, v3, qw, qx, qy, qz] or symbolic equivalents
        u:     [zeta] canard deflection angle in radians

    Returns:
        (Mx, My, Mz): moments in body frame N·m. Only Mz (roll) is non-zero.
    """
    try:
        v3 = float(state[5])
        delta = float(u[0])
    except TypeError:
        # Called symbolically during EOM derivation — return symbolic expression
        v3 = state[5]
        delta = u[0]
        return (0, 0, C_CANARD * delta * v3)
    Mz = C_CANARD * delta * abs(v3)
    return (0.0, 0.0, float(Mz))

controls.add_control_surface_moments(M_controls_func)

# --- State feedback gain K(t, xhat) ---
# u = -K @ (xhat - x0), clipped to ±max_input
# Focus on roll rate w3 (index 2) with velocity-adaptive gain.
def K_func(t, xhat):
    """Gain-scheduled proportional controller on roll rate w3.

    Gain scales linearly with axial velocity v3 for proportional authority.
    """
    K = np.zeros((1, 10))
    v3 = abs(float(xhat[5]))
    # Scale gain: 0 at rest → 0.1 rad/rad at v3=100 m/s, capped at 0.12
    k_w3 = float(np.clip(1.0e-3 * v3, 1e-4, 0.12))
    K[0, 2] = k_w3   # index 2 = w3 (roll rate)
    return K

controls.setK(K_func)

# --- Observer gain L (10×10 for full-state feedback) ---
# Small diagonal gains — set to 0 to disable observer (pure state feedback).
# Increase to trust sensors more than the model.
L = np.diag([
    1e-3, 1e-3, 1e-3,    # w1 w2 w3 angular velocity gains
    0.0,  0.0,  0.0,     # v1 v2 v3 velocity — not directly sensed
    1e-3, 1e-3, 1e-3, 1e-3,  # qw qx qy qz quaternion gains
])
controls.setL(L)

# --- Sensor model (full-state, no noise) ---
# sensor_vars must be a subset of controls.state_vars (symbolic variables).
# sensor_model(t, x) receives the true state and returns the measurement vector.
controls.set_sensor_params(
    sensor_vars=controls.state_vars,                          # observe all 10 states
    sensor_model=lambda t, x: np.array(x, dtype=float),      # identity — no noise
)


# =============================================================================
# BUILD ROCKETSIM
# =============================================================================

sim = RocketSim(controls=controls, sampling_rate=SAMPLING_RATE)


# =============================================================================
# DEFINE ROCKET AND ENVIRONMENT BUILDERS
# =============================================================================

def create_rocket():
    """Build and return (Rocket, CanardSurface) for the RocketPy simulation."""

    # ---- Motor ----
    motor = SolidMotor(
        thrust_source=list(zip(THRUST_TIMES.tolist(), THRUST_FORCES.tolist())),
        burn_time=T_BURNOUT,
        dry_mass=MOTOR_DRY_MASS,
        dry_inertia=(0.030, 0.030, 0.002),
        nozzle_radius=MOTOR_NOZZLE_RADIUS,
        throat_radius=MOTOR_THROAT_RADIUS,
        grain_number=MOTOR_GRAIN_NUMBER,
        grain_density=MOTOR_GRAIN_DENSITY,
        grain_outer_radius=MOTOR_GRAIN_OD,
        grain_initial_inner_radius=MOTOR_GRAIN_ID,
        grain_initial_height=MOTOR_GRAIN_HEIGHT,
        grain_separation=MOTOR_GRAIN_SEP,
        grains_center_of_mass_position=MOTOR_GRAINS_CG_POS,
        center_of_dry_mass_position=MOTOR_DRY_CG_POS,
        nozzle_position=MOTOR_NOZZLE_POS,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )

    # ---- Rocket body ----
    # mass = dry mass of rocket WITHOUT motor
    # inertia = (I_11, I_22, I_33) of the rocket body without motor
    # center_of_mass_without_motor = distance from tail in "tail_to_nose" frame
    rocket = Rocket(
        radius=RADIUS,
        mass=M_BODY,
        inertia=(I_BODY_LONG, I_BODY_LONG, I_BODY_ROLL),
        power_off_drag=C_D,
        power_on_drag=C_D,
        center_of_mass_without_motor=X_CG_BODY_FROM_TAIL,
        coordinate_system_orientation="tail_to_nose",
    )

    # Motor mounted at tail (position 0 in tail_to_nose frame)
    rocket.add_motor(motor, position=0.0)

    # ---- Passive fins (plain TrapezoidalFins — no subclassing) ----
    # position = distance from tail to the root chord's highest point
    rocket.add_trapezoidal_fins(
        n=N_FINS,
        root_chord=FIN_ROOT_CHORD,
        tip_chord=FIN_TIP_CHORD,
        span=FIN_SPAN,
        position=0.30,         # m from tail — near the tail
        cant_angle=FIN_CANT_DEG,
        name="PassiveFins",
    )

    # ---- Canard control surface (moments only, no passive aero) ----
    # Position ~0.5m from nose tip = L_NE - 0.5m from tail
    canard_pos_from_tail = L_NE - 0.50
    canard = CanardSurface(
        center_of_pressure=canard_pos_from_tail,
        reference_area=np.pi * RADIUS ** 2,
        reference_length=DIAMETER,
        controls=controls,
        name="CanardSurface",
    )
    rocket.add_surfaces(canard, canard_pos_from_tail)

    # ---- CFD roll force (returns 0 until cfd_roll_moment is implemented) ----
    roll_pos_from_tail = L_NE - 0.30
    roll = RollForce(
        center_of_pressure=roll_pos_from_tail,
        reference_area=np.pi * RADIUS ** 2,
        reference_length=DIAMETER,
        controls=controls,
        name="RollForce",
    )
    rocket.add_surfaces(roll, roll_pos_from_tail)

    # ---- Register control loop ----
    controller = _Controller(
        interactive_objects=[canard],
        controller_function=sim.controller_function,
        sampling_rate=SAMPLING_RATE,
        initial_observed_variables=[U0],
        name="RollController",
    )
    rocket._add_controllers(controller)

    return rocket, canard


def create_env():
    """Build and return a RocketPy Environment."""
    env = Environment(
        latitude=40.1020,    # Champaign, IL (example launch site)
        longitude=-88.2272,
        elevation=228,       # m ASL
    )
    env.set_atmospheric_model(type="standard_atmosphere")
    return env


sim.set_rocket(create_rocket)
sim.set_env(create_env)


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    print("\nStarting RocketPy simulation...")
    flight = sim.run(
        file_name="example_flight",
        rail_length=5.5,
        inclination=85.0,
        heading=0.0,
    )

    print("\nExporting state estimates and control inputs...")
    sim.export_states("example_flight_states")

    print("\n--- Flight summary ---")
    print(f"  Apogee:       {flight.apogee:.1f} m")
    print(f"  Max speed:    {flight.max_speed:.1f} m/s")
    print(f"  Flight time:  {flight.t_final:.1f} s")
    print(f"  Control steps logged: {len(sim.times)}")
    if sim.inputs:
        inputs_deg = np.rad2deg([i[0] for i in sim.inputs[1:]])
        print(f"  Max canard deflection: {np.max(np.abs(inputs_deg)):.2f} deg")

    print("\nDone.")
