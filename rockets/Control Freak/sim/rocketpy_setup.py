from datetime import datetime

import rocketpy
from sim.internal_dynamics_setup import (
    motor_burn_time,
    rail_button_angular_position_deg,
    thrust_curve,
)

# Launch-site inputs
launch_lat = 40.386772  # deg, launch-site latitude
launch_lon = -87.511283  # deg, launch-site longitude
launch_elevation = 216.10  # m ASL, launch-site elevation above sea level
launch_date = datetime(2026, 4, 18, 9, 0, 0)  # local launch datetime
launch_timezone = "America/New_York"  # IANA timezone string for RocketPy date handling



rail_length = 5.18  # m
inclination = 90  # deg, rail inclination from horizontal
heading = 0  # deg clockwise from north
upper_button_position = 1.41  # m from nose tip
lower_button_position = 2.41  # m from nose tip


environment_params = {
    "latitude": launch_lat,  # deg
    "longitude": launch_lon,  # deg
    "elevation": launch_elevation,  # m ASL
    "date": launch_date,  # datetime
    "timezone": launch_timezone,  # timezone string
}

# everything without motor
rocket_params = {
    "radius": 0.131 / 2,  # m, rocket body radius
    "mass": 16.326,  # kg, dry airframe mass without motor
    "inertia": (6.93, 6.93, 0.057),  # kg*m^2, dry airframe inertia tuple (I11, I22, I33)
    "power_off_drag": "CD_Power-off_copy.csv",  # CSV
    "power_on_drag": "CD_Power-on_copy.csv",  # CSV
    "center_of_mass_without_motor": 1.34,  # m from nose tip, dry airframe CG without motor
    "coordinate_system_orientation": "nose_to_tail",
}


motor_params = {
    "thrust_source": thrust_curve,  # .eng file or other RocketPy-compatible thrust source
    "dry_mass": 2.799,  # kg, motor hardware mass without propellant
    "dry_inertia": (0.08481, 0.08481, 0.00336),  # kg*m^2, dry motor inertia tuple (I11, I22, I33)
    "nozzle_radius": 0.024,  # m, nozzle exit radius
    "grain_number": 3,  # number of propellant grains
    "grain_density": 1672.16,  # kg/m^3
    "grain_outer_radius": 0.04153,  # m
    "grain_initial_inner_radius": 0.01429,  # m
    "grain_initial_height": 0.1524,  # m
    "grain_separation": 0.005,  # m, spacing between grains
    "grains_center_of_mass_position": 0.2985,  # m from nozzle plane
    "center_of_dry_mass_position": 0.2985,  # m from nozzle plane
    "nozzle_position": 0,  # m
    "burn_time": motor_burn_time,  # s
    "throat_radius": 0.0127,  # m, nozzle throat radius
    "coordinate_system_orientation": "nozzle_to_combustion_chamber",  # RocketPy motor-axis convention
}

# Main fin-set geometry
main_fin_params = {
    "n": 4,  # count, number of main fins
    "root_chord": 0.305,  # m
    "tip_chord": 0.152,  # m
    "span": 0.133,  # m, from fin root to tip
    "position": 2.1768,  # m from nose tip, main-fin root leading-edge position
    "cant_angle": 0.2,  # deg
    "airfoil": None,  # RocketPy airfoil definition; None uses default flat-plate-style model
    "sweep_length": 0.234,  # m
}

# Canard geometry
canard_params = {
    "n": 2,  # count, number of canards
    "root_chord": 0.0508,  # m
    "tip_chord": 0.0127,  # m
    "span": 0.0635,  # m
    "position": 0.787,  # m from nose tip, canard root leading-edge position
    "sweep_angle": 0.001,  # deg, canard sweep angle
}

# Tail / boattail geometry
tail_params = {
    "top_radius": 0.131 / 2,  # m
    "bottom_radius": 0.102 / 2,  # m
    "length": 0.076,  # m
    "position": 2.5146,  # m from nose tip, forward edge of tail section
}

# Rail-button geometry
rail_button_params = {
    "upper_button_position": upper_button_position,  # m from nose tip
    "lower_button_position": lower_button_position,  # m from nose tip
    "angular_position": rail_button_angular_position_deg,  # deg about body axis, relative to RocketPy reference
}

# Flight-object inputs passed to rocketpy.Flight(...)
flight_params = {
    "rail_length": rail_length,  # m
    "inclination": inclination,  # deg
    "heading": heading,  # deg
    "terminate_on_apogee": True,  # bool, stop the RocketPy run at apogee
}


def build_rocketpy_stack():
    """Create RocketPy environment, rocket, and flight objects."""
    env = rocketpy.Environment(**environment_params)
    env.set_atmospheric_model(type="Windy", file="GFS")

    rocket = rocketpy.Rocket(**rocket_params)

    motor = rocketpy.SolidMotor(**motor_params)

    rocket.add_motor(motor, position=2.5908)
    rocket.add_nose(length=0.6096, kind="von karman", position=0)
    rocket.add_trapezoidal_fins(**main_fin_params)
    canards = rocket.add_trapezoidal_fins(**canard_params)
    rocket.add_tail(**tail_params)
    rocket.set_rail_buttons(**rail_button_params)

    flight = rocketpy.Flight(
        rocket=rocket,
        environment=env,
        **flight_params,
    )

    return {
        "env": env,
        "rocket": rocket,
        "flight": flight,
        "canards": canards,
        "launch_lat": launch_lat,
        "launch_lon": launch_lon,
        "upper_button_position": upper_button_position,
        "lower_button_position": lower_button_position,
        "rail_length": rail_length,
        "inclination": inclination,
        "heading": heading,
    }
