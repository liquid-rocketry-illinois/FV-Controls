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
        self.x_true_log = []   # true state passed in from RocketPy
        self.roll_log   = []   # w3 true vs estimated for quick plotting
        self.deriv_log  = []   # true state derivatives [w1dot..v3dot] (first 6)
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

            # True derivatives — RK4 propagation.
            # Euler at dt=0.01s is unstable for this rocket's pitch/yaw oscillator
            # (damping ratio ~1.7%, ω_n ~6.6 rad/s → Euler stable limit ~0.005s).
            # u and altitude are held constant over the interval.
            k1   = self.controls.f_numeric(t,        x_true,              u)
            k2   = self.controls.f_numeric(t + dt/2, x_true + dt/2 * k1, u)
            k3   = self.controls.f_numeric(t + dt/2, x_true + dt/2 * k2, u)
            k4   = self.controls.f_numeric(t + dt,   x_true + dt   * k3, u)
            xdot = k1  # start-of-step derivative used for logging and IMU
            true_derivs = np.zeros(10)
            true_derivs[:3] = xdot[:3]
            true_derivs[3:6] = xdot[3:6] + np.cross(x_true[0:3], x_true[3:6])

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
            self.x_true_log.append(x_true.copy())
            self.roll_log.append((x_true[2], xhat[2]))
            self.deriv_log.append(xdot[:6].copy())
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

            # RK4: u=0 throughout, altitude held constant over the interval
            k1   = self.controls.f_numeric(t,        x_true,              u)
            k2   = self.controls.f_numeric(t + dt/2, x_true + dt/2 * k1, u)
            k3   = self.controls.f_numeric(t + dt/2, x_true + dt/2 * k2, u)
            k4   = self.controls.f_numeric(t + dt,   x_true + dt   * k3, u)
            xdot = k1  # start-of-step derivative for logging and IMU
            true_derivs = np.zeros(10)
            true_derivs[:3] = xdot[:3]
            true_derivs[3:6] = xdot[3:6] + np.cross(x_true[0:3], x_true[3:6])

            imu_output    = self.imu.read(t, x_true, true_derivs, temperature)
            y_meas        = imu_output[:6]
            measured_temp = imu_output[6]
            self.controls.set_current_temperature(measured_temp)

            xhat = self.ekf.update(t, dt, y_meas, u)

            self.t_log.append(t)
            self.xhat_log.append(xhat.copy())
            self.u_log.append(u.copy())
            self.x_true_log.append(x_true.copy())
            self.roll_log.append((x_true[2], xhat[2]))
            self.deriv_log.append(xdot[:6].copy())
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
        self.x_true_log = []
        self.roll_log   = []
        self.deriv_log  = []
        self.P_diag_log = []

    def _package_logs(self) -> dict:
        roll = np.array(self.roll_log)
        return {
            't':         np.array(self.t_log),
            'xhat':      np.array(self.xhat_log),
            'u':         np.array(self.u_log),
            'x_true':    np.array(self.x_true_log),
            'roll_true': roll[:, 0],
            'roll_est':  roll[:, 1],
            'deriv':     np.array(self.deriv_log),    # shape (n, 6): [w1dot..v3dot]
            'P_diag':    np.array(self.P_diag_log),   # shape (n, 10): EKF covariance diag
        }
