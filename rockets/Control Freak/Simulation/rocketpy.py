import rocketpy
from datetime import datetime, timedelta

tomorrow = datetime.now() + timedelta(days=1)

env = rocketpy.Environment(
    latitude=40.386772,
    longitude=-87.511283,
    elevation=216.10
    date = tomorrow
)
env.set_atmospheric_model(type="forecast", file="NAM")
#
########
########
#
# )
Control_Freak = rocketpy.Rocket(
    radius = 0.131/2,    # m
    mass   = ...,      # kg dry mass
    inertia=(..., ..., ...),
    power_off_drag="...",
    power_on_drag="...",
    center_of_mass_without_motor=...,
    coordinate_system_orientation=, # "tail_to_nose", "nose_to_tail"
 )


motor = rocketpy.SolidMotor(
    thrust_source = 'AeroTech_M2400T.eng',
    dry_mass=...,
    dry_inertia=(..., ..., ...),
    nozzle_radius=...,
    grain_number=...,
    grain_density=...,
    grain_outer_radius=...,
    grain_initial_inner_radius= ...,
    grain_initial_height=...,
    grain_separation=...,
    grains_center_of_mass_position=...,
    center_of_dry_mass_position=...,
    nozzle_position=...,
    burn_time= ...,
    throat_radius=...,
    coordinate_system_orientation="combustion_chamber_to_nozzle",
    )


Control_Freak.add_motor(motor, position=...)

nose_cone = Control_Freak.add_nose(
    length=..., kind="von karman", position=...
)

fin_set = Control_Freak.add_trapezoidal_fins(
    n=4,
    root_chord=0.305,
    tip_chord=0.152,
    span=0.133,
    position=...,
    cant_angle=...,
    airfoil= None,
)

tail = Control_Freak.add_tail(
    top_radius = 0.131/2, bottom_radius = 0.102/2, length = 0.076, position = 0
)


main = Control_Freak.add_parachute(
    name="Main",
    cd_s=...,
    trigger=...,
    sampling_rate=...,
    radius=...,
    height=...,
    porosity= ...,
)

drogue = Control_Freak.add_parachute(
    name="Drogue",
    cd_s= ...,
    trigger="apogee",
    sampling_rate= ...,
    radius=...,
    height=...,
    porosity= ...,
)


flight = rocketpy.Flight (
     rocket      = Control_Freak,
     environment = env,
     rail_length = ...,
     inclination = ...,    # deg — vertical launch
     heading     = 0,
)