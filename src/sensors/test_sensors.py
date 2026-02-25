import numpy as np
from sympy import Matrix

# Adjust these imports to match your folder structure!
from sensors.sensor_model import IMU
from sensors.sensor_fusion import SensorFusion
from controls.controls import Controls 

# ==============================================================================
# TEST 1: IMU SENSOR LOGIC
# ==============================================================================
def test_imu_sensor(imu):
    print("\n--- TEST 1: IMU Hardware & Physics ---")
    
    t = 0.0
    # 10 states: [w1, w2, w3, v1, v2, v3, qw, qx, qy, qz]
    rocket_state = np.zeros(10)
    rocket_state[6] = 1.0 # Identity quat (qw = 1.0)
    derivatives  = np.zeros(10) 
    
    # 1A. Gravity Test
    reading = imu.read(t, rocket_state, derivatives)
    accel_z = reading[2]
    print(f"Static Z-Accel (Expect ~9.81): {accel_z:.4f} m/s^2")
    
    if 9.7 < accel_z < 9.9:
        print("  [PASS] Gravity Correct.")
    else:
        print("  [FAIL] Gravity incorrect.")

    # 1B. Saturation Test
    # Apply 200 m/s^2 acceleration to Z-axis (index 5 of derivatives is v3dot)
    derivatives[5] = 200.0 
    reading = imu.read(0.01, rocket_state, derivatives)
    
    if 156.0 < abs(reading[2]) < 158.0:
        print(f"  [PASS] Output saturated correctly at {reading[2]:.2f}")
    else:
        print(f"  [FAIL] Output did not saturate: {reading[2]}")
        
    if hasattr(imu, 'saturation_warning_triggered') and imu.saturation_warning_triggered:
         print("  [PASS] Warning flag triggered successfully.")
    else:
         print("  [FAIL] Warning flag missing or failed.")

# ==============================================================================
# TEST 2: SENSOR FUSION (EKF) USING REAL CONTROLS
# ==============================================================================
def test_ekf_fusion(shared_imu):
    print("\n--- TEST 2: EKF Sensor Fusion (Real Controls) ---")

    try:
        # 1. INITIALIZE REAL CONTROLS
        real_controls = Controls(IREC_COMPLIANT=True, rocket_name="TestRocket")
        
        # 2. LOAD DUMMY PHYSICS DATA (Crucial to avoid "mpf from None")
        real_controls.setRocketParams(
            I_0=0.5, I_f=0.4, I_3=0.01,
            x_CG_0=1.5, x_CG_f=1.4,
            m_0=10.0, m_f=8.0, m_p=2.0,
            d=0.1, L_ne=2.0, C_d=0.5, Cnalpha_rocket=2.0,
            t_launch_rail_clearance=0.5, t_motor_burnout=3.0, t_estimated_apogee=15.0,
            CP_func=lambda aoa: 1.2 # Dummy CP location at 1.2m
        )
        real_controls.setFinParams(N=4, Cr=0.1, Ct=0.05, s=0.05, Cnalpha_fin=2.0, delta=0.0)
        real_controls.setThrustCurve(
            thrust_times=np.array([0, 3, 3.1]), 
            thrust_forces=np.array([100, 100, 0])
        )
        real_controls.setEnvParams(v_wind=[0.0, 0.0], rho=1.225, g=9.81)

        # 3. SET SYMBOLS
        real_controls.set_symbols()

        # 4. SATISFY CONTROLS GUARDRAILS (Must happen before define_eom)
        real_controls.set_controls_params(u0=np.array([0.0]), max_input=0.26) 
        
        # Return a SymPy Matrix of zeros so it can be added to the dynamics equations
        real_controls.add_control_surface_moments(lambda state, u: Matrix([0, 0, 0]))
        
        real_controls.setK(lambda t, x: np.zeros((1, 10))) 
        real_controls.setL(np.zeros((10, 6))) 

        # 5. COMPILE EOM
        # Now it has the M_controls_func it needs to build the equations!
        real_controls.define_eom() 

        # 6. DEFINE THE 'C' MATRIX
        w1, w2, w3, v1, v2, v3, qw, qx, qy, qz = real_controls.state_vars
        my_sensor_vars = [v1, v2, v3, w1, w2, w3] 
        
        def imu_wrapper(t, x):
            return shared_imu.read(t, x, derivatives=np.zeros(len(x)))
            
        real_controls.set_sensor_params(
            sensor_vars=my_sensor_vars,
            sensor_model=imu_wrapper
        )

        # 7. INITIALIZE THE EKF
        STATE_SIZE = 10
        R_matrix = np.diag([shared_imu.accel_std**2]*3 + [shared_imu.gyro_std**2]*3)
        Q_matrix = np.eye(STATE_SIZE) * 0.001 
        P_matrix = np.eye(STATE_SIZE) * 0.01 

        initial_state = np.zeros(STATE_SIZE)
        initial_state[6] = 1.0 # qw = 1.0

        ekf = SensorFusion(
            initial_state=initial_state,
            initial_covariance=P_matrix,
            Q=Q_matrix,
            R=R_matrix,
            controls_model=real_controls 
        )

        t = 0.0
        dt = shared_imu.dt
        dummy_u = np.array([0.0]) # 1 fin input (zeta) based on your input_vars

        print("Running 50-step EKF Loop with real Jacobians...")
        for i in range(50):
            t += dt
            
            # Simulated Reality
            true_state = np.zeros(STATE_SIZE)
            true_state[6] = 1.0 
            true_derivs = np.zeros(STATE_SIZE)
            true_derivs[5] = -9.81 
            
            # Sensor distorts reality
            y_noisy = shared_imu.read(t, true_state, true_derivs)
            
            # EKF cleans it up
            x_clean = ekf.update(t, dt, y_meas=y_noisy, u=dummy_u)
            
        print("  [PASS] EKF Loop completed using real Controls f_numeric and get_AB!")
        print(f"  Final Filtered State Vector Size: {len(x_clean)}")

    except Exception as e:
        import traceback
        print(f"\n  [FAIL] The test crashed. Error details:")
        print(f"  {e}")
        print("\n--- Traceback ---")
        traceback.print_exc()

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("=== STARTING ESTIMATION PIPELINE TESTS ===")
    
    # Initialize one shared IMU for both tests. 
    shared_imu = IMU(
        update_rate=100.0,
        accel_range=157.0, gyro_range=35.0,
        accel_noise_density=0.01, 
        gyro_noise_density=0.001,
        accel_random_walk=0.0, gyro_random_walk=0.0
    )
    
    # Run the tests
    test_imu_sensor(shared_imu)
    test_ekf_fusion(shared_imu)
    
    print("\n=== ALL TESTS COMPLETE ===")