import numpy as np
from controls import Controls
from sensors.sensor_model import IMU
from sensors.sensor_fusion import SensorFusion


class Flight_Computer_Sim:
    """Core simulation loop. Takes true state at each timestep,
    runs the IMU model, EKF, and control law, and logs results.
    Agnostic to where the true state comes from (RocketPy or otherwise)."""

    def __init__(self, controls: Controls, imu: IMU, ekf: SensorFusion):
        """
        Args:
            controls: Fully configured Controls object.
            imu:      Fully configured IMU object.
            ekf:      Fully configured SensorFusion object.
        """
        self.controls = controls
        self.imu      = imu
        self.ekf      = ekf

        # Logs — filled during run()
        self.t_log      = []   # time at each step
        self.xhat_log   = []   # EKF state estimate [w1,w2,w3,v1,v2,v3,qw,qx,qy,qz]
        self.u_log      = []   # control input (canard deflection angle)
        self.u_applied_log = []  # control input actually used for truth propagation
        self.x_true_log = []   # true state passed in from RocketPy
        self.roll_log   = []   # w3 true vs estimated for quick plotting
        self.deriv_log  = []   # true state derivatives [w1dot..v3dot] (first 6)
        self.fin_roll_moment_log = []  # main-fin roll moment from dynamics EOM (N*m)
        self.canard_roll_moment_log = []  # canard roll moment actually applied (N*m)
        self.total_roll_moment_log = []   # total roll moment inferred from w3dot (N*m)
        self.P_diag_log = []   # EKF covariance diagonal (uncertainty per state)

    def step(self, t, dt, true_state, true_derivs, u_prev, altitude_asl=0.0, true_temperature=288.15):
        # Tell controls the current ASL altitude before any param evaluation
        self.controls.set_current_altitude(altitude_asl)

        imu_output = self.imu.read(t, true_state, true_derivs, true_temperature)
        y_meas = imu_output[:6]   # [a1,a2,a3, w1,w2,w3]
        measured_temp = imu_output[6]     # temperature channel

        # Feed measured temperature into controls for Mach computation
        self.controls.set_current_temperature(measured_temp)

        xhat = self.ekf.update(t, dt, y_meas, u_prev)
        u = self.controls.compute_control(t, xhat)
        return xhat, u

    def _canard_roll_moment(self, t: float, x: np.ndarray, u: np.ndarray) -> float:
        """Return the numeric canard roll moment applied by controls.f_numeric."""
        canard_func = getattr(self.controls, "_canard_cfd_func", None)
        if canard_func is None or u is None or len(u) == 0:
            return 0.0

        param_vals = self.controls._gather_param_values(
            t,
            x,
            getattr(self.controls, "_current_altitude", None),
        )
        v_wind_x = param_vals[19]
        v_wind_y = param_vals[20]
        va = self.controls._compute_body_airspeed(x, v_wind_x, v_wind_y)
        v_air_mag = float(np.linalg.norm(va))
        return float(canard_func(v_air_mag, float(u[0])))

    def _roll_moment_from_derivative(self, t: float, x: np.ndarray, w3dot: float) -> float:
        """Infer total body-axis roll moment from Euler's roll equation."""
        I1, I2, I3 = self.controls.get_inertia(t)
        _, _, I3dot = self.controls.get_inertia_dot(t)
        w1, w2, w3 = x[:3]
        return float(I3 * w3dot - (I1 - I2) * w1 * w2 + I3dot * w3)

    def run_ekf_controlled(self, t_total=None, initial_altitude=0.0):
        """Standalone EKF + control simulation — no RocketPy truth data needed.

        Truth state is propagated using controls.f_numeric (RK4 integration).
        EKF and control law both run at every timestep. World-frame position is
        tracked by rotating body-frame velocity through the quaternion each step.

        Args:
            t_total:          Simulation end time (s). Defaults to controls.t_estimated_apogee.
            initial_altitude: Launch site altitude ASL (m). Used for env lookups at t=0.
        """
        self._reset_logs()

        dt      = self.controls.dt
        t_total = t_total if t_total is not None else self.controls.t_estimated_apogee

        x_true   = self.controls.x0.copy()
        u        = self.controls.u0.copy()
        position = np.array([0.0, 0.0, 0.0])
        altitude_asl = float(initial_altitude)

        pos_log  = []
        temp_log = []
        t = 0.0

        while t < t_total:
            temperature = self._env_temperature(altitude_asl)

            self.controls.set_current_altitude(altitude_asl)
            self.controls.set_current_temperature(temperature)
            u_applied = u.copy()

            # True derivatives — RK4 propagation.
            # Euler at dt=0.01s is unstable for this rocket's pitch/yaw oscillator
            # (damping ratio ~1.7%, ω_n ~6.6 rad/s → Euler stable limit ~0.005s).
            # u and altitude are held constant over the interval.
            k1   = self.controls.f_numeric(t,        x_true,              u_applied)
            k2   = self.controls.f_numeric(t + dt/2, x_true + dt/2 * k1, u_applied)
            k3   = self.controls.f_numeric(t + dt/2, x_true + dt/2 * k2, u_applied)
            k4   = self.controls.f_numeric(t + dt,   x_true + dt   * k3, u_applied)
            xdot = k1  # start-of-step derivative used for logging and IMU
            true_derivs = np.zeros(10)
            true_derivs[:3] = xdot[:3]
            true_derivs[3:6] = xdot[3:6] + np.cross(x_true[0:3], x_true[3:6])
            fin_roll_moment = self.controls.fin_roll_moment_numeric(t, x_true, altitude_asl)
            canard_roll_moment = self._canard_roll_moment(t, x_true, u_applied)
            total_roll_moment = self._roll_moment_from_derivative(t, x_true, xdot[2])

            # IMU model: inject noise into true state + derivs
            imu_output    = self.imu.read(t, x_true, true_derivs, temperature)
            y_meas        = imu_output[:6]
            measured_temp = imu_output[6]
            self.controls.set_current_temperature(measured_temp)

            # EKF predict + correct, then control law
            xhat = self.ekf.update(t, dt, y_meas, u)
            u    = self.controls.compute_control(t, xhat)

            # --- log ---
            self.t_log.append(t)
            self.xhat_log.append(xhat.copy())
            self.u_log.append(u.copy())
            self.u_applied_log.append(u_applied.copy())
            self.x_true_log.append(x_true.copy())
            self.roll_log.append((x_true[2], xhat[2]))
            self.deriv_log.append(xdot[:6].copy())
            self.fin_roll_moment_log.append(fin_roll_moment)
            self.canard_roll_moment_log.append(canard_roll_moment)
            self.total_roll_moment_log.append(total_roll_moment)
            self.P_diag_log.append(self.ekf.P.diagonal().copy())
            pos_log.append(position.copy())
            temp_log.append(temperature)

            # RK4 propagation of truth state
            x_true = x_true + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
            q = x_true[6:10]
            norm = np.linalg.norm(q)
            if norm > 0:
                x_true[6:10] = q / norm

            # Launch rail constraint: lock orientation and zero lateral motion
            x_true = self._apply_rail_constraint(x_true, t)

            # Integrate world-frame position: rotate body velocity through quaternion
            qw, qx, qy, qz = x_true[6:10]
            xx, yy, zz = qx*qx, qy*qy, qz*qz
            wx_, wy_, wz_ = qw*qx, qw*qy, qw*qz
            xy, xz, yz = qx*qy, qx*qz, qy*qz
            R_WB = np.array([
                [1-2*(yy+zz), 2*(xy-wz_),  2*(xz+wy_)],
                [2*(xy+wz_),  1-2*(xx+zz), 2*(yz-wx_)],
                [2*(xz-wy_),  2*(yz+wx_),  1-2*(xx+yy)]
            ])
            position = position + R_WB @ x_true[3:6] * dt
            altitude_asl = float(initial_altitude) + float(position[2])

            t += dt

        results = self._package_logs()
        n = len(results['t'])
        results['position']    = np.array(pos_log[:n])
        results['temperature'] = np.array(temp_log[:n])
        return results

    def _env_temperature(self, altitude_asl: float) -> float:
        """Temperature (K) at ASL altitude — uses RocketPy env if registered, else ISA."""
        env_T = getattr(self.controls, '_env_temperature_func', None)
        if env_T is not None:
            try:
                return float(env_T(altitude_asl))
            except Exception:
                pass
        return max(216.65, 288.15 - 0.0065 * altitude_asl)

    def run_ekf_only(self, t_total=None, initial_altitude=0.0):
        """Standalone EKF-only simulation using internal dynamics.

        Truth is propagated with controls.f_numeric (RK4) using u=0 at every step.
        The EKF estimates the state while the control input remains zero.
        """
        self._reset_logs()

        dt      = self.controls.dt
        t_total = t_total if t_total is not None else self.controls.t_estimated_apogee

        x_true   = self.controls.x0.copy()
        u        = np.zeros_like(self.controls.u0)
        position = np.array([0.0, 0.0, 0.0])
        altitude_asl = float(initial_altitude)

        pos_log  = []
        temp_log = []
        t = 0.0

        while t < t_total:
            temperature = self._env_temperature(altitude_asl)

            self.controls.set_current_altitude(altitude_asl)
            self.controls.set_current_temperature(temperature)
            u_applied = u.copy()

            # RK4: u=0 throughout, altitude held constant over the interval
            k1   = self.controls.f_numeric(t,        x_true,              u_applied)
            k2   = self.controls.f_numeric(t + dt/2, x_true + dt/2 * k1, u_applied)
            k3   = self.controls.f_numeric(t + dt/2, x_true + dt/2 * k2, u_applied)
            k4   = self.controls.f_numeric(t + dt,   x_true + dt   * k3, u_applied)
            xdot = k1  # start-of-step derivative for logging and IMU
            true_derivs = np.zeros(10)
            true_derivs[:3] = xdot[:3]
            true_derivs[3:6] = xdot[3:6] + np.cross(x_true[0:3], x_true[3:6])
            fin_roll_moment = self.controls.fin_roll_moment_numeric(t, x_true, altitude_asl)
            canard_roll_moment = self._canard_roll_moment(t, x_true, u_applied)
            total_roll_moment = self._roll_moment_from_derivative(t, x_true, xdot[2])

            imu_output    = self.imu.read(t, x_true, true_derivs, temperature)
            y_meas        = imu_output[:6]
            measured_temp = imu_output[6]
            self.controls.set_current_temperature(measured_temp)

            xhat = self.ekf.update(t, dt, y_meas, u)

            self.t_log.append(t)
            self.xhat_log.append(xhat.copy())
            self.u_log.append(u.copy())
            self.u_applied_log.append(u_applied.copy())
            self.x_true_log.append(x_true.copy())
            self.roll_log.append((x_true[2], xhat[2]))
            self.deriv_log.append(xdot[:6].copy())
            self.fin_roll_moment_log.append(fin_roll_moment)
            self.canard_roll_moment_log.append(canard_roll_moment)
            self.total_roll_moment_log.append(total_roll_moment)
            self.P_diag_log.append(self.ekf.P.diagonal().copy())
            pos_log.append(position.copy())
            temp_log.append(temperature)

            x_true = x_true + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
            q = x_true[6:10]
            norm = np.linalg.norm(q)
            if norm > 0:
                x_true[6:10] = q / norm

            # Launch rail constraint: lock orientation and zero lateral motion
            x_true = self._apply_rail_constraint(x_true, t)

            qw, qx, qy, qz = x_true[6:10]
            xx, yy, zz = qx*qx, qy*qy, qz*qz
            wx_, wy_, wz_ = qw*qx, qw*qy, qw*qz
            xy, xz, yz = qx*qy, qx*qz, qy*qz
            R_WB = np.array([
                [1-2*(yy+zz), 2*(xy-wz_),  2*(xz+wy_)],
                [2*(xy+wz_),  1-2*(xx+zz), 2*(yz-wx_)],
                [2*(xz-wy_),  2*(yz+wx_),  1-2*(xx+yy)]
            ])
            position = position + R_WB @ x_true[3:6] * dt
            altitude_asl = float(initial_altitude) + float(position[2])

            t += dt

        results = self._package_logs()
        n = len(results['t'])
        results['position']    = np.array(pos_log[:n])
        results['temperature'] = np.array(temp_log[:n])
        return results

    def run(self, t_history, state_history, deriv_history, altitude_history=None, temperature_history=None):
        self._reset_logs()
        u = self.controls.u0.copy()

        for i in range(len(t_history) - 1):
            t           = t_history[i]
            dt          = t_history[i+1] - t_history[i]
            x_true      = state_history[i]
            xd_true     = deriv_history[i]
            altitude_asl = altitude_history[i] if altitude_history is not None else 0.0
            temperature = temperature_history[i] if temperature_history is not None else 288.15

            xhat, u = self.step(t, dt, x_true, xd_true, u, altitude_asl, temperature)

            # Log
            self.t_log.append(t)
            self.xhat_log.append(xhat.copy())
            self.u_log.append(u.copy())
            self.x_true_log.append(x_true.copy())
            self.roll_log.append((x_true[2], xhat[2]))   # (w3_true, w3_est)
            self.deriv_log.append(xd_true[:6].copy())    # angular + linear accels
            self.P_diag_log.append(self.ekf.P.diagonal().copy())

        return self._package_logs()

    def _apply_rail_constraint(self, x: np.ndarray, t: float) -> np.ndarray:
        """Constrain truth state to the launch rail while t < t_launch_rail_clearance.

        On the rail the rocket cannot pitch, yaw, or move laterally — only axial
        translation (v3) and roll (w3) are unconstrained.  The quaternion is locked
        to the initial launch orientation to prevent numeric drift from corrupting
        the orientation before aerodynamic forces take over.
        """
        if t >= self.controls.t_launch_rail_clearance:
            return x
        x = x.copy()
        x[0] = 0.0   # w1 — pitch rate
        x[1] = 0.0   # w2 — yaw rate
        x[3] = 0.0   # v1 — lateral velocity
        x[4] = 0.0   # v2 — lateral velocity
        # Clamp axial velocity to ≥ 0: rail holds rocket until thrust exceeds gravity.
        # Without this, gravity alone would accelerate the rocket backward before ignition.
        x[5] = max(0.0, float(x[5]))   # v3 — axial speed (clamp to forward-only)
        q0 = self.controls.x0[6:10].copy()
        norm = np.linalg.norm(q0)
        if norm > 0:
            q0 /= norm
        x[6:10] = q0
        return x

    def _reset_logs(self):
        self.t_log      = []
        self.xhat_log   = []
        self.u_log      = []
        self.u_applied_log = []
        self.x_true_log = []
        self.roll_log   = []
        self.deriv_log  = []
        self.fin_roll_moment_log = []
        self.canard_roll_moment_log = []
        self.total_roll_moment_log = []
        self.P_diag_log = []

    def _package_logs(self) -> dict:
        roll = np.array(self.roll_log)
        results = {
            't':         np.array(self.t_log),
            'xhat':      np.array(self.xhat_log),
            'u':         np.array(self.u_log),
            'x_true':    np.array(self.x_true_log),
            'roll_true': roll[:, 0],
            'roll_est':  roll[:, 1],
            'deriv':     np.array(self.deriv_log),    # shape (n, 6): [w1dot..v3dot]
            'P_diag':    np.array(self.P_diag_log),   # shape (n, 10): EKF covariance diag
        }
        if len(self.u_applied_log) == len(self.t_log):
            results['u_applied'] = np.array(self.u_applied_log)
        if len(self.fin_roll_moment_log) == len(self.t_log):
            results['fin_roll_moment'] = np.array(self.fin_roll_moment_log)
        if len(self.canard_roll_moment_log) == len(self.t_log):
            results['canard_roll_moment'] = np.array(self.canard_roll_moment_log)
        if len(self.total_roll_moment_log) == len(self.t_log):
            results['total_roll_moment'] = np.array(self.total_roll_moment_log)
        return results
