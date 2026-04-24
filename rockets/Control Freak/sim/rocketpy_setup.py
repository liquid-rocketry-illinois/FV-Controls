envfrom datetime import datetime
from pathlib import Path

import rocketpy

from sim.internal_dynamics_setup import (
    motor_burn_time,
    rail_button_angular_position_deg,
)


SIM_DIR = Path(__file__).resolve().parent
DATA_DIR = SIM_DIR.parent / "data"

thrust_source = DATA_DIR / "motor_file" / "AeroTech_M2400T.eng"
power_off_drag = DATA_DIR / "drag" / "CD_Power-off.csv"
power_on_drag = DATA_DIR / "drag" / "CD_Power-on.csv"

# Launch-site inputs from rockets/Control Freak/Rocketpy/rocketpy.ipynb
launch_lat = 40.386772
launch_lon = -87.511283
launch_elevation = 216.10
launch_date = datetime(2026, 4, 18, 12, 0, 0)
launch_timezone = "America/New_York"

# Flight inputs from rocketpy.ipynb
rail_length = 5.18
inclination = 90
heading = 0
upper_button_position = 1.727
lower_button_position = 2.692


def build_rocketpy_stack():
    """Create RocketPy environment, rocket, and flight objects."""
    env = rocketpy.Environment(
        latitude=launch_lat,
        longitude=launch_lon,
        elevation=launch_elevation,
        date=launch_date,
        timezone=launch_timezone,
    )
    env.set_atmospheric_model(type="Windy", file="GFS")

    control_freak = rocketpy.Rocket(
        radius=0.131 / 2,
        mass=14.1067,
        inertia=(8.27, 8.27, 0.056),
        power_off_drag=str(power_off_drag),
        power_on_drag=str(power_on_drag),
        center_of_mass_without_motor=1.5875,
        coordinate_system_orientation="nose_to_tail",
    )

    nose_cone = control_freak.add_nose(
        length=0.724,
        kind="von karman",
        position=0,
    )

    fin_set = control_freak.add_trapezoidal_fins(
        n=4,
        root_chord=0.305,
        tip_chord=0.152,
        span=0.133,
        position=2.459,
        cant_angle=0,
        airfoil=None,
        sweep_length=0.234,
    )

    canards = control_freak.add_trapezoidal_fins(
        n=2,
        root_chord=0.0508,
        tip_chord=0.0127,
        span=0.0635,
        position=1.04,
        sweep_angle=0.001,
        cant_angle=0,
    )

    tail = control_freak.add_tail(
        top_radius=0.131 / 2,
        bottom_radius=0.102 / 2,
        length=0.0762,
        position=2.7938,
    )

    rail_buttons = control_freak.set_rail_buttons(
        upper_button_position=upper_button_position,
        lower_button_position=lower_button_position,
        angular_position=rail_button_angular_position_deg,
    )

    main = control_freak.add_parachute(
        name="Main",
        cd_s=7.868,
        trigger=304.8,
        sampling_rate=105,
    )

    drogue = control_freak.add_parachute(
        name="Drogue",
        cd_s=0.684,
        trigger="apogee",
        sampling_rate=105,
    )

    motor = rocketpy.SolidMotor(
        thrust_source=str(thrust_source),
        dry_mass=2.799,
        dry_inertia=(0.08481, 0.08481, 0.00336),
        nozzle_radius=0.024,
        grain_number=3,
        grain_density=1672.16,
        grain_outer_radius=0.04153,
        grain_initial_inner_radius=0.01429,
        grain_initial_height=0.1524,
        grain_separation=0.005,
        grains_center_of_mass_position=0.2985,
        center_of_dry_mass_position=0.2985,
        nozzle_position=0,
        burn_time=motor_burn_time,
        throat_radius=0.0127,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )

    control_freak.add_motor(motor, position=2.870)

    flight = rocketpy.Flight(
        rocket=control_freak,
        environment=env,
        rail_length=rail_length,
        inclination=inclination,
        heading=heading,
        terminate_on_apogee=False,
    )

    return {
        "env": env,
        "rocket": control_freak,
        "flight": flight,
        "canards": canards,
        "nose_cone": nose_cone,
        "fin_set": fin_set,
        "tail": tail,
        "rail_buttons": rail_buttons,
        "main": main,
        "drogue": drogue,
        "launch_lat": launch_lat,
        "launch_lon": launch_lon,
        "upper_button_position": upper_button_position,
        "lower_button_position": lower_button_position,
        "rail_length": rail_length,
        "inclination": inclination,
        "heading": heading,
    }
