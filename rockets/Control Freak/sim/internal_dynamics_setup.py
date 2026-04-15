from dynamics import build_power_state_drag_model
from parameter import Parameter

# Shared motor / launch-reference inputs
thrust_curve = "AeroTech_M2400T.eng"  # .eng thrust-curve file
motor_burn_time = 3.28  # s
rail_button_angular_position_deg = 45.0  # deg, about the fins axis



rocket_params = {
    "I_0": 11.31,  # kg*m^2, transverse inertia at ignition
    "I_f": 9.18,  # kg*m^2, transverse inertia after burnout
    "I_3": 0.065,  # kg*m^2, roll-axis inertia at ignition
    "I_3_f": 0.06,  # kg*m^2, roll-axis inertia after burnout
    "x_CG_0": 1.6095,  # m from nose tip, CG at ignition
    "x_CG_f": 1.4795,  # m from nose tip, CG after burnout
    "m_0": 22.777,  # kg, total mass at ignition
    "m_f": 19.12454,  # kg, total mass after burnout
    "m_p": 22.777 - 19.12454,  # kg, propellant mass
    "d": 0.131,  # m, body outer diameter
    "L_ne": 2.59,  # m, nose-to-nozzle reference length, rocekt length in most case
    "t_launch_rail_clearance": 0.322,  # s, expected rail-clearance time
    "t_motor_burnout": motor_burn_time,  # s, burnout time seen  
    "t_estimated_apogee": 24.959,  # s, expected apogee time
}


fin_params = {
    "N": 4,  # count, number of main fins
    "Cr": 0.305,  # m, fin root chord
    "Ct": 0.152,  # m, fin tip chord
    "s": 0.133,  # m, fin span
    "delta": 0.2,  # deg, main-fin cant angle
}


canard_params = {
    "N": 2,  # count, number of canards
    "Cr": 0.0508,  # m, canard root chord
    "Ct": 0.0127,  # m, canard tip chord
    "s": 0.0635,  # m, canard span
    "x_le": 0.787,  # m from nose tip, canard root leading-edge position
    "sweep_angle": 0.001,  # deg, canard sweep angle
    "plane_angle_deg": 0 #rail_button_angular_position_deg + 90.0,  # deg, canard-plane azimuth; perpendicular to the rail-button axis
}


aero_direct_params = {
    "nose_cn": 2.000,  # 1/rad, nose normal-force slope contribution
    "nose_cp": 0.305,  # m from nose tip, nose CP location
    "canard_cn": 0.855,  # 1/rad, canard normal-force slope contribution
    "canard_cp": 0.796,  # m from nose tip, canard CP location
    "fin_cn": 9.346,  # 1/rad, main-fin normal-force slope contribution
    "fin_cp": 2.340,  # m from nose tip, main-fin CP location
    "tail_cn": -0.787,  # 1/rad, tail normal-force slope contribution
    "tail_cp": 2.551,  # m from nose tip, tail CP location
}

 ##fall back option when not using rocketpy environment
env_params = {
    "v_wind": [0.0, 0.0],  # m/s, fallback horizontal wind [east, north] for non-RocketPy use
    "rho": 1.225,  # kg/m^3, fallback air density
    "g": 9.81,  # m/s^2, fallback gravitational acceleration
}


sim_params = {
    "dt": 0.01,  # s, internal integrator/sample timestep
    "rail_button_angular_position_deg": rail_button_angular_position_deg,  # deg, aligns initial body-axis roll angle with RocketPy
    "coordinate_system_orientation": "nose_to_tail",
}


drag_model_params = {
    "power_on_csv": "CD_Power-on_copy.csv",  # CSV, drag-vs-Mach curve used while motor is burning, get from RAS Aero
    "power_off_csv": "CD_Power-off_copy.csv",  # CSV, drag-vs-Mach curve used after burnout
    "burnout_time": motor_burn_time,  # s, switches the internal drag model from power-on to power-off
}

def build_internal_dynamics():
    """Create the internal-dynamics parameter object and drag model."""
    p = Parameter()

    p.setRocketParams(**rocket_params)
    p.setFinParams(**fin_params)
    p.set_aero_direct(**aero_direct_params)
    p.setEnvParams(**env_params)
    p.setSimParamsFromRailAngle(**sim_params)
    p.canard_plane_angle_deg = canard_params["plane_angle_deg"]

    p.setThrustCurveFromFile(thrust_curve)
    p.checkParamsSet()

    drag_func = build_power_state_drag_model(**drag_model_params)

    return {
        "parameter": p,
        "drag_func": drag_func,
        "cp_func": p.cp_func_for_plots,
        "thrust_curve": thrust_curve,
        "motor_burn_time": motor_burn_time,
        "rail_button_angular_position_deg": rail_button_angular_position_deg,
        "canard_plane_angle_deg": canard_params["plane_angle_deg"],
    }
