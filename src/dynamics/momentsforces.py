from sympy import *
import numpy as np
import pandas as pd
from typing import Callable

class MomentsForces():
    def __init__(self):
        # super().__init__()

        self.M = None

        self.params : list = None

        # Environmental parameters
        self.v_wind : list = [0.0, 0.0]
        self.rho : float = 1.225 # Air density kg/m^3
        self.g : float = 9.81 # Gravitational acceleration m/s^2

        self.F : Matrix = None # Forces matrix
        self.M : Matrix = None # Moments matrix

        self.f : Matrix = None
        self.state_vars : list = None

        # Thrust curve data
        self.thrust_times : np.ndarray = None
        self.thrust_forces : np.ndarray = None

    def _get_directional_aero_split(self):
        """Return base and canard aerodynamic component data.

        The "base" rocket is the symmetric contribution from nose, main fins,
        and tail. The canard contribution is kept separate so a 2-canard
        configuration can add directional stiffness only in the canard plane.

        Returns:
            tuple[float, float, float, float]:
                (base_cn, base_cp, canard_cn, canard_cp)
        """
        canard_cn = float(getattr(self, "CN_alpha_canards", 0.0) or 0.0)
        canard_cp = float(getattr(self, "CP_canards", 0.0) or 0.0)

        base_components = []
        for cn_attr, cp_attr in (
            ("CN_alpha_nose", "CP_nose"),
            ("CN_alpha_fins", "CP_fins"),
            ("CN_alpha_tail", "CP_tail"),
        ):
            cn_val = getattr(self, cn_attr, None)
            cp_val = getattr(self, cp_attr, None)
            if cn_val is None or cp_val is None:
                continue
            base_components.append((float(cn_val), float(cp_val)))

        if base_components:
            base_cn = sum(cn for cn, _ in base_components)
            if abs(base_cn) > 1e-9:
                base_cp = sum(cn * cp for cn, cp in base_components) / base_cn
            else:
                base_cp = float(canard_cp)
            return base_cn, float(base_cp), canard_cn, canard_cp

        total_cn = float(getattr(self, "Cnalpha_rocket", 0.0) or 0.0)
        if callable(getattr(self, "CP_func", None)):
            total_cp = float(self.CP_func(0.0))
        else:
            total_cp = canard_cp

        base_cn = total_cn - canard_cn
        if abs(base_cn) > 1e-9:
            base_cp = (total_cn * total_cp - canard_cn * canard_cp) / base_cn
        else:
            base_cp = total_cp
        return float(base_cn), float(base_cp), canard_cn, canard_cp

    def set_symbols(self):
        w1, w2, w3, v1, v2 = symbols('w_1 w_2 w_3 v_1 v_2', real=True)
        v3 = symbols('v_3', real=True, positive=True)
        qw, qx, qy, qz = symbols('q_w q_x q_y q_z', real=True)

        # --- ALL params must be real SymPy symbols ---
        I1, I2, I3 = symbols('I_1 I_2 I_3', real=True, positive=True)
        T1, T2, T3 = symbols('T_1 T_2 T_3', real=True, positive=True)
        mass, rho, d, g, CG = symbols('m rho d g CG', real=True, positive=True)
        delta = symbols('delta', real=True)
        C_d = symbols('C_d', real=True, positive=True)
        Cnalpha_fin, Cnalpha_rocket = symbols('C_n_alpha_fin C_n_alpha_rocket', real=True, positive=True)
        Cr, Ct, s = symbols('Cr Ct s', real=True, positive=True)
        N_fins = symbols('N_fins', real=True, positive=True)   # <-- renamed from N
        v_wind1, v_wind2 = symbols('v_wind_1 v_wind_2', real=True)
        t_sym = symbols('t', real=True, positive=True)
        CP = symbols('CP', real=True, positive=True)   # Center of pressure location

        self.state_vars = [w1, w2, w3, v1, v2, v3, qw, qx, qy, qz]
        self.params = [
            I1, I2, I3,                          # indices 0-2
            T1, T2, T3,                          # indices 3-5
            mass, rho, d, g, CG,                 # indices 6-10
            delta, C_d,                          # indices 11-12
            Cnalpha_fin, Cnalpha_rocket,         # indices 13-14
            Cr, Ct, s,                           # indices 15-17
            N_fins,                              # index 18
            v_wind1, v_wind2,                     # indices 19-20
            CP                                   #index 21
        ]
        self.t_sym = t_sym

    def get_mass(self, t: float) -> float:
        """Get the mass of the rocket at time t.

        Args:
            t (float): The time in seconds.

        Returns:
            float: The mass of the rocket at time t in kg.
        """
        mass_rocket = self.m_0 - self.m_p / self.t_motor_burnout * t if t <= self.t_motor_burnout else self.m_f
        return mass_rocket

    def get_thrust_accel(self, t: float) -> np.ndarray:
        """Get the thrust acceleration at time t. Does this by dividing thrusts by m.

        Args:
            t (float): The time in seconds.

        Returns:
            np.array: The thrust acceleration vector as a numpy array.
        """
        thrust = self.get_thrust(t)
        m = self.get_mass(t)
        a_thrust = np.zeros(10)
        a_thrust[3] = thrust[0] / m
        a_thrust[4] = thrust[1] / m
        a_thrust[5] = thrust[2] / m
        return a_thrust


    def get_gravity_accel(self, xhat: np.array):
        """Get the gravity acceleration in body frame at time t.

        Args:
            xhat (np.array): The current state estimate as a numpy array.

        Returns:
            np.array: The gravity acceleration vector as a numpy array.
        """
        g = np.array([0.0, 0.0, -self.g])
        qw, qx, qy, qz = xhat[6], xhat[7], xhat[8], xhat[9]
        R_world_to_body = np.array(self.R_BW_from_q(qw, qx, qy, qz)).astype(np.float64)
        g_body = R_world_to_body @ g
        a_gravity = np.zeros(10)
        a_gravity[3:6] = g_body
        return a_gravity

    def set_forces(self) -> Matrix:
        """Get the forces for the rocket. Sets self.F.
        """
        w1, w2, w3, v1, v2, v3, qw, qx, qy, qz = self.state_vars
        I1, I2, I3, T1, T2, T3, mass, rho, d, g, CG, delta, C_d, Cnalpha_fin, Cnalpha_rocket, Cr, Ct, s, N_fins, v_wind1, v_wind2, CP = self.params
        t_sym = self.t_sym
        base_cn_val, base_cp_val, canard_cn_val, canard_cp_val = self._get_directional_aero_split()
        base_cn = Float(base_cn_val)
        canard_cn = Float(canard_cn_val)
        canard_plane_angle_rad = Float(
            np.deg2rad(float(getattr(self, "canard_plane_angle_deg", 0.0) or 0.0))
        )
        canard_dir_1 = cos(canard_plane_angle_rad)
        canard_dir_2 = sin(canard_plane_angle_rad)

        H = Heaviside(t_sym - Float(self.t_launch_rail_clearance), 0)  # 0 if t < t_launch_rail_clearance, 1 if t >= t_launch_rail_clearance

        ## World-to-body rotation matrix (unit quaternion assumed in EOM) ##
        _xx, _yy, _zz = qx*qx, qy*qy, qz*qz
        _wx, _wy, _wz = qw*qx, qw*qy, qw*qz
        _xy, _xz, _yz = qx*qy, qx*qz, qy*qz
        R_BW = Matrix([
            [1-2*(_yy+_zz),   2*(_xy+_wz),   2*(_xz-_wy)],
            [2*(_xy-_wz),     1-2*(_xx+_zz), 2*(_yz+_wx)],
            [2*(_xz+_wy),     2*(_yz-_wx),   1-2*(_xx+_yy)]
        ])

        ## Air-relative velocity in body frame: v_air = v_body - R_BW @ [v_wind1, v_wind2, 0] ##
        v_wind_body = R_BW * Matrix([v_wind1, v_wind2, Float(0)])
        va1 = v1 - v_wind_body[0]
        va2 = v2 - v_wind_body[1]
        va3 = v3 - v_wind_body[2]

        epsAoA = Float(1e-9)
        AoA = atan2(sqrt(va1**2 + va2**2), va3 + epsAoA)  # AoA using airspeed
        AoA = Piecewise(
            (0,   Abs(AoA) <= epsAoA),                # inside deadband
            (Min(Abs(AoA), 15 * pi / 180) * (AoA/Abs(AoA)), True)  # ±15°
        )

        eps = Float(1e-9)
        v_air = Matrix([va1, va2, va3])
        v_air_mag = sqrt(va1**2 + va2**2 + va3**2 + eps**2)  # airspeed magnitude
        vahat = v_air / v_air_mag  # unit airspeed vector

        ## Rocket reference area ##
        A = pi * (d/2)**2  # m^2

        ## Thrust ##
        Ft : Matrix = Matrix([T1, T2, T3])  # Thrust vector, T1 and T2 are assumed 0

        ## Gravity (world -> body) ##
        Fg_world = Matrix([0, 0, -mass * g])
        Fg : Matrix = R_BW * Fg_world

        ## Drag Force (uses airspeed) ##
        D = C_d * Rational(1,2) * rho * v_air_mag**2 * A
        Fd : Matrix = -D * vahat

        ## Lift Force (uses airspeed and Cnalpha_rocket for consistency with pitching moment) ##
        # Regularise the lateral-speed denominator so cos/sin(beta) never divide by zero
        # during axial flight (va1 ≈ va2 ≈ 0).  eps_vxy is tiny relative to any real speed.
        eps_vxy = Float(1e-9)
        v_xy_mag = sqrt(va1**2 + va2**2 + eps_vxy**2)
        L_base = H * Rational(1,2) * rho * v_air_mag**2 * base_cn * AoA * A
        nL_base = Matrix([
            -cos(AoA) * va1 / v_xy_mag,
            -cos(AoA) * va2 / v_xy_mag,
            sin(AoA)
        ])  # lift direction unit vector (perpendicular to airspeed, regularised)

        # Two canards are not azimuthally symmetric like the four main fins.
        # We model their added normal force only in the configured canard plane.
        eps_vcan = Float(1e-9)
        v_can_lat = va1 * canard_dir_1 + va2 * canard_dir_2
        v_can_lat_mag = sqrt(v_can_lat**2 + eps_vcan**2)
        v_air_can_mag = sqrt(v_can_lat**2 + va3**2 + eps**2)
        AoA_can = atan2(v_can_lat_mag, va3 + epsAoA)
        AoA_can = Piecewise(
            (0,   Abs(AoA_can) <= epsAoA),
            (Min(Abs(AoA_can), 15 * pi / 180) * (AoA_can/Abs(AoA_can)), True)
        )
        L_can = H * Rational(1,2) * rho * v_air_can_mag**2 * canard_cn * AoA_can * A
        nL_can = Matrix([
            -cos(AoA_can) * v_can_lat / v_can_lat_mag * canard_dir_1,
            -cos(AoA_can) * v_can_lat / v_can_lat_mag * canard_dir_2,
            sin(AoA_can)
        ])

        Fl : Matrix = L_base * nL_base + L_can * nL_can

        ## Total Forces ##
        F = Ft + Fd + Fl + Fg  # Thrust + Drag + Lift + Gravity

        self.F = F

    def get_forces(self):
        """Get the forces for the rocket.
        Returns:
            Matrix: The forces vector.
        """
        self.set_forces()
        return self.F

    def set_moments(self) -> Matrix:
        """Get the moments for the rocket.
        Returns:
            Matrix: The moments vector.
        """
        w1, w2, w3, v1, v2, v3, qw, qx, qy, qz = self.state_vars
        I1, I2, I3, T1, T2, T3, mass, rho, d, g, CG, delta, C_d, Cnalpha_fin, Cnalpha_rocket, Cr, Ct, s, N_fins, v_wind1, v_wind2, CP = self.params
        t_sym = self.t_sym
        base_cn_val, base_cp_val, canard_cn_val, canard_cp_val = self._get_directional_aero_split()
        base_cn = Float(base_cn_val)
        base_cp = Float(base_cp_val)
        canard_cn = Float(canard_cn_val)
        canard_cp = Float(canard_cp_val)
        canard_plane_angle_rad = Float(
            np.deg2rad(float(getattr(self, "canard_plane_angle_deg", 0.0) or 0.0))
        )
        canard_dir_1 = cos(canard_plane_angle_rad)
        canard_dir_2 = sin(canard_plane_angle_rad)

        H = Heaviside(t_sym - Float(self.t_launch_rail_clearance), 0)  # 0 if t < t_launch_rail_clearance, 1 if t >= t_launch_rail_clearance

        ## World-to-body rotation matrix (unit quaternion assumed in EOM) ##
        _xx, _yy, _zz = qx*qx, qy*qy, qz*qz
        _wx, _wy, _wz = qw*qx, qw*qy, qw*qz
        _xy, _xz, _yz = qx*qy, qx*qz, qy*qz
        R_BW = Matrix([
            [1-2*(_yy+_zz),   2*(_xy+_wz),   2*(_xz-_wy)],
            [2*(_xy-_wz),     1-2*(_xx+_zz), 2*(_yz+_wx)],
            [2*(_xz+_wy),     2*(_yz-_wx),   1-2*(_xx+_yy)]
        ])

        ## Air-relative velocity in body frame ##
        v_wind_body = R_BW * Matrix([v_wind1, v_wind2, Float(0)])
        va1 = v1 - v_wind_body[0]
        va2 = v2 - v_wind_body[1]
        va3 = v3 - v_wind_body[2]

        epsAoA = Float(1e-9)
        AoA = atan2(sqrt(va1**2 + va2**2), va3 + epsAoA)  # AoA using airspeed
        AoA = Piecewise(
            (0,   Abs(AoA) <= epsAoA),                # inside deadband
            (Min(Abs(AoA), 15 * pi / 180) * (AoA/Abs(AoA)), True)  # ±15°
        )

        eps = Float(1e-9)
        v_air_mag = sqrt(va1**2 + va2**2 + va3**2 + eps**2)  # airspeed magnitude

        ## Rocket reference area ##
        A = pi * (d/2)**2  # m^2

        ## Corrective (restoring) moments — split into symmetric base + directional canards ##
        C_raw_base = v_air_mag**2 * A * base_cn * AoA * (base_cp - CG) * rho / 2

        # Regularise lateral-speed denominator (same reason as in set_forces)
        eps_vxy = Float(1e-9)
        v_xy_mag = sqrt(va1**2 + va2**2 + eps_vxy**2)
        # cos(AoA) = va3/v_air_mag — correct moment magnitude at larger AoA
        cos_AoA = va3 / v_air_mag
        M_f_pitch_base = -C_raw_base * cos_AoA * va2 / v_xy_mag
        M_f_yaw_base   =  C_raw_base * cos_AoA * va1 / v_xy_mag

        eps_vcan = Float(1e-9)
        v_can_lat = va1 * canard_dir_1 + va2 * canard_dir_2
        v_can_lat_mag = sqrt(v_can_lat**2 + eps_vcan**2)
        v_air_can_mag = sqrt(v_can_lat**2 + va3**2 + eps**2)
        AoA_can = atan2(v_can_lat_mag, va3 + epsAoA)
        AoA_can = Piecewise(
            (0,   Abs(AoA_can) <= epsAoA),
            (Min(Abs(AoA_can), 15 * pi / 180) * (AoA_can/Abs(AoA_can)), True)
        )
        C_raw_can = v_air_can_mag**2 * A * canard_cn * AoA_can * (canard_cp - CG) * rho / 2
        cos_AoA_can = va3 / v_air_can_mag
        canard_lat_sign = v_can_lat / v_can_lat_mag
        M_f_pitch_can = -C_raw_can * cos_AoA_can * canard_lat_sign * canard_dir_2
        M_f_yaw_can   =  C_raw_can * cos_AoA_can * canard_lat_sign * canard_dir_1

        ## Propulsive Damping (Cdp) — only active during motor burn ##
        H_burn = Heaviside(Float(self.t_motor_burnout) - t_sym, 0)  # 1 during burn, 0 after
        mdot = self.m_p / self.t_motor_burnout  # kg/s, average mass flow rate
        Cdp = H_burn * mdot * (self.L_ne - CG)**2  # kg·m²/s

        ## Aerodynamic pitch/yaw damping (Cda) — directional canard contribution projected by canard plane ##
        Cda_base = (rho * v_air_mag * A / 2) * (base_cn * (base_cp - CG)**2)
        Cda_can  = (rho * v_air_can_mag * A / 2) * (canard_cn * (canard_cp - CG)**2)

        M_d_pitch = Cdp + Cda_base + Cda_can * canard_dir_2**2
        M_d_yaw   = Cdp + Cda_base + Cda_can * canard_dir_1**2

        ## Moment due to fin cant angle (roll) ##
        gamma = Ct/Cr
        r_t = d/2
        tau = (s + r_t) / r_t

        # Roll forcing moment — Barrowman/RocketPy
        Y_MA = (s/3) * (1 + 2*gamma)/(1+gamma)  # spanwise aerodynamic centre
        K_f = (1/pi**2) * \
            ((pi**2/4)*((tau+1)**2/tau**2) \
            + (pi*(tau**2+1)**2/(tau**2*(tau-1)**2))*asin((tau**2-1)/(tau**2+1)) \
            - (2*pi*(tau+1))/(tau*(tau-1)) \
            + ((tau**2+1)**2/(tau**2*(tau-1)**2))*asin((tau**2-1)/(tau**2+1))**2 \
            - (4*(tau+1)/(tau*(tau-1)))*asin((tau**2-1)/(tau**2+1)) \
            + (8/(tau-1)**2)*log((tau**2+1)/(2*tau)))
        M_f_roll = K_f * (Rational(1,2) * rho * v_air_mag**2) * \
            (N_fins * (Y_MA + r_t) * Cnalpha_fin * delta * A)

        # Roll damping moment (uses airspeed)
        trap_integral = s/12 * ((Cr + 3*Ct)*s**2 + 4*(Cr+2*Ct)*s*r_t + 6*(Cr + Ct)*r_t**2)
        C_ldw = 2 * N_fins * Cnalpha_fin / (A * d**2) * cos(delta) * trap_integral
        K_d = 1 + ((tau-gamma)/tau - (1-gamma)/(tau-1)*ln(tau))/ \
            ((tau+1)*(tau-gamma)/2 - (1-gamma)*(tau**3-1)/(3*(tau-1)))
        M_d_roll = K_d * (Rational(1,2) * rho * v_air_mag**2) * A * d * C_ldw * (d / (2 * v_air_mag))

        M_f = Matrix([
            M_f_pitch_base + M_f_pitch_can,
            M_f_yaw_base + M_f_yaw_can,
            M_f_roll
        ])
        M_d = Matrix([M_d_pitch, M_d_yaw, M_d_roll])

        M1 = M_f[0] - M_d[0] * w1
        M2 = M_f[1] - M_d[1] * w2
        M3 = M_f[2] - M_d[2] * w3

        M = H * Matrix([M1, M2, M3])

        self.M = M

    def get_moments(self) -> Matrix:
        """Get the moments for the rocket.
        Returns:
            Matrix: The moments vector.
        """
        self.set_moments()
        return self.M
    
    def setThrustCurve(self, thrust_times: np.ndarray, thrust_forces: np.ndarray):
        """Set the thrust curve data.

        Args:
            thrust_times (np.ndarray): Array of time points in seconds.
            thrust_force (np.ndarray): Array of thrust values in Newtons corresponding to the time points.
        """
        self.thrust_times = thrust_times
        self.thrust_forces = thrust_forces

    def get_thrust(self, t: float) -> Matrix:
        """Get the thrust for the rocket at time t.

        Args:
            t (float): The time in seconds.

        Returns:
            dict: A dictionary containing inertia, mass, CG, and thrust at time t. WRONG. We return T = [0.,0.,thrust]
        """

        T = Matrix([0., 0., 0.])  # N
        motor_burnout = t > self.t_motor_burnout
        if not motor_burnout:
            # Enforce zero thrust outside the sampled burn window.
            T[2] = np.interp(
                t,
                self.thrust_times,
                self.thrust_forces,
                left=0.0,
                right=0.0,
            )  # Thrust acting in z direction
            
        return T

    ## Helper function to print thrust curve ##
    def printThrustCurve(self, thrust_file: str):
        """Print the thrust curve data from a .csv or .eng file. Copy cell output to code block to set thrust curve parameters.
        Replace 'your_object_name' with whatever you name your Dynamics object as (e.g. dynamics = Dynamics(), your object name
        would be 'dynamics').

        Args:
            thrust_file (str): Path to the .csv or .eng file containing thrust curve data. Can be from OpenRocket or thrustcurve.org.
        """
        df = None
        if thrust_file.endswith('.csv'):
            df = pd.read_csv(thrust_file)
        elif thrust_file.endswith('.eng'):
            rows = []
            with open(thrust_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    # skip empty lines and comments
                    if not line or line.startswith(';'):
                        continue

                    parts = line.split()

                    # Data lines in .eng files are usually: "<time> <thrust>"
                    # Header/metadata has more columns, so we ignore those.
                    if len(parts) == 2:
                        try:
                            t = float(parts[0])
                            F = float(parts[1])
                            rows.append((t, F))
                        except ValueError:
                            # In case something weird slips through, just skip the line
                            continue

            df = pd.DataFrame(rows, columns=["# Time (s)", "Thrust (N)"])
        else:
            raise ValueError("Unsupported file format. Please provide a .csv or .eng file.")

        times = df["# Time (s)"]
        thrust = df["Thrust (N)"]
        stop_index = np.argmax(thrust[1:] == 0.0)
        times = times[:stop_index + 2]
        thrust = thrust[:stop_index + 2]
        
        print(f"thrust_times = np.array({times.tolist()})")
        print(f"thrust_forces = np.array({thrust.tolist()})")
        print("your_object_name.setThrustCurve(thrust_times=thrust_times, thrust_forces=thrust_forces)")
    

if __name__ == "__main__":
    x = MomentsForces()
    x.set_moments()
    x.set_forces()
    print(x.get_forces(),x.get_moments())
    print("ssdfds")
