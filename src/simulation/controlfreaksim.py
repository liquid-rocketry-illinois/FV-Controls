import numpy as np
from rocketpy import Environment, SolidMotor, Rocket, Flight
from sensors.sensor_model import IMU
import matplotlib.pyplot as plt

def run_rocketpy_sensor_test():
    env = Environment(latitude=40.106264, longitude=-88.223366, elevation=100)
    env.set_atmospheric_model(type="standard_atmosphere")

    #copy pasted from rocketpy

    Pro75M1670 = SolidMotor(
        thrust_source=np.array([[0, 1500], [3.0, 1500], [3.1, 0]]), 
        dry_mass=1.815,
        dry_inertia=(0.125, 0.125, 0.002),
        center_of_dry_mass_position=0.317,
        grains_center_of_mass_position=-0.397,
        burn_time=3.1,
        grain_number=5,
        grain_separation=0.005,
        grain_density=1815,
        grain_outer_radius=0.033,
        grain_initial_inner_radius=0.015,
        grain_initial_height=0.12,
        nozzle_radius=0.033,
        throat_radius=0.011,
        interpolation_method="linear",
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )

    calisto = Rocket(
        radius=0.0635,
        mass=14.426,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag=0.4,
        power_on_drag=0.4,
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )
    calisto.add_motor(Pro75M1670, position=-1.255)
    calisto.add_nose(length=0.55829, kind="vonKarman", position=0.71971)
    calisto.add_trapezoidal_fins(n=4, root_chord=0.120, tip_chord=0.040, span=0.100, position=-1.04956)

    test_flight = Flight(rocket=calisto, environment=env, rail_length=5.2, inclination=85, heading=0)

    #dummy imu
    my_imu = IMU(
        update_rate=100.0, # 100 Hz
        accel_range=157.0, gyro_range=35.0,
        accel_noise_density=10, gyro_noise_density=0.5,
        accel_random_walk=0.001, gyro_random_walk=0.001
    )

    dt = my_imu.dt
    #launch to apogee
    time_steps = np.arange(0, test_flight.apogee_time, dt)
    
    true_z_accel_history = []
    noisy_z_accel_history = []
    timestamps = []

    for t in time_steps:
        #get state from rocketpy
        
        # RocketPy Euler parameters: e0 (scalar), e1, e2, e3 (vector)
        qw = test_flight.e0(t)
        qx = test_flight.e1(t)
        qy = test_flight.e2(t)
        qz = test_flight.e3(t)
        
        #angular velocities (body frame)
        w1, w2, w3 = test_flight.w1(t), test_flight.w2(t), test_flight.w3(t)
        
        #linear velocities (inertial frame -> mapped to our array)
        v1, v2, v3 = test_flight.vx(t), test_flight.vy(t), test_flight.vz(t)
        
        #map to our 10state array
        rocket_state = np.array([w1, w2, w3, v1, v2, v3, qw, qx, qy, qz])

        #get acceleration from rocketpy

        #rocketpy calcs inertial accelerations
        #extract_physics() in IMU expects body frame kinematic acceleration at derivatives[3:6]
        #have to rotate rocketpys inertial acceleration into body frame
        ax_inertial, ay_inertial, az_inertial = test_flight.ax(t), test_flight.ay(t), test_flight.az(t)
        a_inertial = np.array([ax_inertial, ay_inertial, az_inertial])
        
        #convert quaternion to rotation matrix (world -> body)
        #same one:
        xx, yy, zz = qx*qx, qy*qy, qz*qz
        wx, wy, wz = qw*qx, qw*qy, qw*qz
        xy, xz, yz = qx*qy, qx*qz, qy*qz
        R_WB = np.array([
            [1 - 2*(yy + zz),   2*(xy + wz),       2*(xz - wy)],
            [2*(xy - wz),       1 - 2*(xx + zz),   2*(yz + wx)],
            [2*(xz + wy),       2*(yz - wx),       1 - 2*(xx + yy)]
        ])
        
        a_body = R_WB @ a_inertial
        
        #derivatives array: [wdot (0:3), vdot (3:6), qdot (6:10)]
        derivatives = np.zeros(13)
        derivatives[3:6] = a_body

        #send "clean" data to imu
        y_noisy = my_imu.read(t, rocket_state, derivatives)

        #conversion from g to m/s2, accel reads in terms of g
        #copy to not overwrite original data
        y_ms = np.copy(y_noisy)
        y_ms[0:3] = y_ms[0:3] * 9.81

        #for plot
        timestamps.append(t)
        
        #true sensor feel to compare against IMU
        #kinematic acceleration in body Z - gravity in body Z
        g_world = np.array([0, 0, -9.81])
        g_body = R_WB @ g_world
        true_sensor_z = a_body[2] - g_body[2] 
        
        true_z_accel_history.append(true_sensor_z)
        noisy_z_accel_history.append(y_ms[2]) # y[2] is accel z, use converted version
        
    print("Generating Plot...")
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, true_z_accel_history, label="Perfect RocketPy Data", color='black', linewidth=2)
    plt.plot(timestamps, noisy_z_accel_history, label="Noisy IMU Output", color='red', alpha=0.6)
    
    # Draw a line where the motor burns out
    plt.axvline(x=3.1, color='blue', linestyle='--', label="Motor Burnout")
    
    plt.title("RocketPy 6-DOF vs. Custom IMU Sensor Model (Z-Axis Acceleration)")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s^2)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_rocketpy_sensor_test()

    #python -m simulation.controlfreaksim in src