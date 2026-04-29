from pathlib import Path

from dynamics import build_power_state_drag_model
from dynamics import Parameter

SIM_DIR = Path(__file__).resolve().parent
DATA_DIR = SIM_DIR.parent / "data"

# Shared motor / launch-reference inputs
thrust_curve = DATA_DIR / "motor_file" / "AeroTech_M2400T.eng"  # .eng thrust-curve file from rocketpy.ipynb
motor_burn_time = 3.28  # s
rail_button_angular_position_deg = 45.0  # deg, about the fins axis



rocket_params = {
    "I_0": 13.147498862800127,  # kg*m^2, transverse inertia at ignition from RocketPy
    "I_f": 10.616251553628745,  # kg*m^2, transverse inertia after burnout from RocketPy
    "I_3": 0.06288225431082371,  # kg*m^2, roll-axis inertia at ignition from RocketPy
    "I_3_f": 0.05936000000111055,  # kg*m^2, roll-axis inertia after burnout from RocketPy
    "x_CG_0": 1.8962789255921615,  # m from nose tip, CG at ignition from RocketPy
    "x_CG_f": 1.7504164128075148,  # m from nose tip, CG after burnout from RocketPy
    "m_0": 20.557700768112838,  # kg, total mass at ignition from RocketPy
    "m_f": 16.9057,  # kg, total mass after burnout from RocketPy
    "m_p": 3.6520007681128384,  # kg, propellant mass from motor grain geometry
    "d": 0.131,  # m, body outer diameter
    "L_ne": 2.870,  # m, nose-to-nozzle reference length from rocketpy.ipynb motor placement
    "t_launch_rail_clearance": 0.323,  # s, rail departure time from RocketPy
    "t_motor_burnout": motor_burn_time,  # s, burnout time seen
    "t_estimated_apogee": 24.301,  # s, apogee time from RocketPy
}


fin_params = {
    "N": 4,  # count, number of main fins
    "Cr": 0.305,  # m, fin root chord
    "Ct": 0.152,  # m, fin tip chord
    "s": 0.133,  # m, fin span
    "delta": 0.5,  # deg, main-fin cant angle from rocketpy.ipynb
}


canard_params = {
    "N": 2,  # count, number of canards
    "Cr": 0.0508,  # m, canard root chord
    "Ct": 0.0127,  # m, canard tip chord
    "s": 0.0635,  # m, canard span
    "x_le": 1.04,  # m from nose tip, canard root leading-edge position from rocketpy.ipynb
    "sweep_angle": 0.001,  # deg, canard sweep angle
    "plane_angle_deg": 0 #rail_button_angular_position_deg + 90.0,  # deg, canard-plane azimuth; perpendicular to the rail-button axis
}


aero_direct_params = {
    "nose_cn": 2.000,  # 1/rad, nose normal-force slope contribution
    "nose_cp": 0.362,  # m from nose tip, nose CP location from RocketPy
    "canard_cn": 0.855,  # 1/rad, canard normal-force slope contribution
    "canard_cp": 1.049,  # m from nose tip, canard CP location from RocketPy
    "fin_cn": 9.346,  # 1/rad, main-fin normal-force slope contribution
    "fin_cp": 2.622,  # m from nose tip, main-fin CP location from RocketPy
    "tail_cn": -0.787,  # 1/rad, tail normal-force slope contribution
    "tail_cp": 2.830,  # m from nose tip, tail CP location from RocketPy
}

 ##fall back option when not using rocketpy environment
env_params = {
    "v_wind": [0.0, 0.0],  # m/s, fallback horizontal wind [east, north] for non-RocketPy use
    "rho": 1.213,  # kg/m^3, surface air density from RocketPy environment
    "g": 9.8014,  # m/s^2, surface gravity from RocketPy environment
}


sim_params = {
    "dt": 0.01,  # s, internal integrator/sample timestep
    "rail_button_angular_position_deg": rail_button_angular_position_deg,  # deg, aligns initial body-axis roll angle with RocketPy
    "coordinate_system_orientation": "nose_to_tail",
}


drag_model_params = {
    "power_on_csv": DATA_DIR / "drag" / "CD_Power-on.csv",  # CSV, drag-vs-Mach curve used while motor is burning
    "power_off_csv": DATA_DIR / "drag" / "CD_Power-off.csv",  # CSV, drag-vs-Mach curve used after burnout
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

    p.setThrustCurveFromFile(str(thrust_curve))
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
