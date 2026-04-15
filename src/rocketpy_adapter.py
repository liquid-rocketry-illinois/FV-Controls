import numpy as np
import rocketpy
from controls.flight_computer import Flight_Computer_Sim
from controlled_flight import ControlledMomentFlight


class Adapter:
    """Outer simulation harness. Handles RocketPy truth data extraction
    and feeds it into the Simulation loop.

    simulation_type options:
        'dynamics_EKF_compare' — One-way comparison mode: RocketPy truth is extracted,
                                 the EKF runs against that truth with control forced to
                                 zero, and the internal model is propagated open-loop
                                 with zero control input.
        'rocketpy_replay'      — Backward-compatible alias for dynamics_EKF_compare.
        'rocketpy_closedloop' — Two-way: canard deflection is injected back into
                                RocketPy at each timestep via a Controller, so the
                                canard roll moment truly affects the trajectory.
        'ekf_only'            — Standalone: internal dynamics and integration.
                                No RocketPy is used. EKF runs with u=0 to validate
                                estimator accuracy on the internal model.
        'ekf_controlled'      — Standalone: truth state propagated via the internal
                                dynamics model (no RocketPy). EKF and control law
                                both run at every timestep. Use this to test EKF +
                                control performance without a full RocketPy flight.
    """

    def __init__(self, simulation: Flight_Computer_Sim, simulation_type: str = 'dynamics_EKF_compare'):
        self.sim      = simulation
        self.sim_type = simulation_type
        self.results  = None
        self.flight   = None

    def run(self, rocketpy_flight=None, rocket=None, env=None,
            canard_fin_set=None, **flight_kwargs) -> dict:
        """Run the SILSIM. Mode is determined by simulation_type set at construction.

        Args (replay mode):
            rocketpy_flight: a completed RocketPy Flight object.

        Args (closedloop mode):
            rocket:         RocketPy Rocket object.
            env:            RocketPy Environment object.
            canard_fin_set: the fin set object whose cant_angle the controller drives.
            **flight_kwargs: passed directly to rocketpy.Flight()
                             e.g. rail_length, inclination, heading, terminate_on_apogee.
        """
        if self.sim_type in ('dynamics_EKF_compare', 'rocketpy_replay'):
            if rocketpy_flight is None:
                raise ValueError("dynamics_EKF_compare requires a rocketpy_flight object.")
            return self._run_replay(rocketpy_flight)
        elif self.sim_type == 'rocketpy_closedloop':
            if rocket is None or env is None or canard_fin_set is None:
                raise ValueError(
                    "rocketpy_closedloop requires rocket, env, and canard_fin_set."
                )
            return self._run_closedloop(rocket, env, canard_fin_set, **flight_kwargs)
        elif self.sim_type == 'ekf_only':
            return self._run_ekf_only()
        elif self.sim_type == 'ekf_controlled':
            return self._run_ekf_controlled()
        else:
            raise ValueError(f"Unknown simulation_type: {self.sim_type}")

    def _run_closedloop(self, rocket, env, canard_fin_set, **flight_kwargs):
        """Two-way coupled simulation.

        A RocketPy Controller calls our flight computer at each IMU timestep.
        The computed canard command is converted to an external roll moment and
        injected into RocketPy's 6-DOF dynamics in the next integration step.
        """
        self.sim._reset_logs()

        # Mutable state shared with the closure
        u_state = [self.sim.controls.u0.copy()]
        last_t  = [0.0]
        flight_ref = {"flight": None}
        position_log = []
        temperature_log = []
        altitude_log = []

        def _controller_fn(time, sampling_rate, state, state_history,
                           observed_variables, interactive_objects, sensors):
            dt = time - last_t[0]
            if dt <= 0.0:
                return interactive_objects, {}

            # RocketPy state: [x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz]
            rpy_state  = np.array(state)
            body_state = self._rocketpy_to_body(rpy_state)
            altitude   = float(rpy_state[2])
            position_agl = np.array(rpy_state[0:3], dtype=float)
            position_agl[2] -= float(env.elevation)

            # Temperature for Mach/speed-of-sound computation
            try:
                temperature = float(env.temperature(altitude))
            except Exception:
                temperature = max(216.65, 288.15 - 0.0065 * altitude)

            # Body-frame acceleration: finite-difference world-frame velocity,
            # then rotate to body frame to avoid Coriolis contamination.
            if len(state_history) >= 2:
                prev_state = np.array(state_history[-1])
                v_world_now  = np.array(rpy_state[3:6])
                v_world_prev = np.array(prev_state[3:6])
                a_world = (v_world_now - v_world_prev) / max(dt, 1e-9)
                e0, e1, e2, e3 = rpy_state[6], rpy_state[7], rpy_state[8], rpy_state[9]
                q = np.array([e0, e1, e2, e3]); q /= np.linalg.norm(q)
                qw_, qx_, qy_, qz_ = q
                xx, yy, zz = qx_*qx_, qy_*qy_, qz_*qz_
                wx__, wy__, wz__ = qw_*qx_, qw_*qy_, qw_*qz_
                xy_, xz_, yz_ = qx_*qy_, qx_*qz_, qy_*qz_
                R_BW = np.array([
                    [1-2*(yy+zz), 2*(xy_+wz__),  2*(xz_-wy__)],
                    [2*(xy_-wz__), 1-2*(xx+zz),  2*(yz_+wx__)],
                    [2*(xz_+wy__), 2*(yz_-wx__),  1-2*(xx+yy)]
                ])
                ang_accel = (body_state[:3] - self._rocketpy_to_body(prev_state)[:3]) / max(dt, 1e-9)
                body_deriv = np.zeros(10)
                body_deriv[:3]  = ang_accel
                body_deriv[3:6] = R_BW @ a_world
            else:
                body_deriv = np.zeros(10)

            # Run EKF + control law
            xhat, u = self.sim.step(
                time, dt, body_state, body_deriv, u_state[0], altitude, temperature
            )
            u_state[0] = u
            last_t[0]  = time

            # ---- Two-way coupling: inject external body moment into RocketPy ----
            wx = float(env.wind_velocity_x(altitude))
            wy = float(env.wind_velocity_y(altitude))
            va = self.sim.controls._compute_body_airspeed(body_state, wx, wy)
            v_air_mag = float(np.linalg.norm(va))
            if self.sim.controls._canard_cfd_func is not None:
                m3 = float(self.sim.controls._canard_cfd_func(v_air_mag, float(u[0])))
            else:
                m3 = 0.0
            if flight_ref["flight"] is not None:
                flight_ref["flight"].set_external_body_moment([0.0, 0.0, m3])

            # Log (mirrors what Flight_Computer_Sim.run() does)
            self.sim.t_log.append(time)
            self.sim.xhat_log.append(xhat.copy())
            self.sim.u_log.append(u.copy())
            self.sim.x_true_log.append(body_state.copy())
            self.sim.roll_log.append((body_state[2], xhat[2]))
            self.sim.deriv_log.append(body_deriv[:6].copy())
            self.sim.P_diag_log.append(self.sim.ekf.P.diagonal().copy())
            position_log.append(position_agl)
            temperature_log.append(temperature)
            altitude_log.append(altitude)

            return interactive_objects, {}

        # Attach controller — sampling rate matches IMU rate
        controller = rocketpy._Controller(
            interactive_objects    = [canard_fin_set],
            controller_function    = _controller_fn,
            sampling_rate          = 1.0 / self.sim.imu.dt,
            name                   = "CanardRollController",
        )

        # Attach controller to rocket, then run flight
        rocket._add_controllers([controller])
        flight = ControlledMomentFlight(
            rocket      = rocket,
            environment = env,
            **flight_kwargs
        )
        flight_ref["flight"] = flight
        self.flight = flight

        self.results = self.sim._package_logs()
        self.results['position'] = np.array(position_log)
        self.results['temperature'] = np.array(temperature_log)
        initial_alt = float(altitude_log[0]) if len(altitude_log) > 0 else 0.0
        self.results['x_internal']  = self._propagate_internal(
            self.results['t'], initial_alt
        )
        return self.results

    def _run_ekf_controlled(self):
        self.results = self.sim.run_ekf_controlled()
        return self.results

    def _run_ekf_only(self):
        self.results = self.sim.run_ekf_only()
        return self.results

    def _propagate_internal(self, t_array: np.ndarray,
                            initial_altitude: float = 0.0) -> np.ndarray:
        """RK4-propagate the internal dynamics model with u=0, no EKF.

        Returns state history shape (n, 10) for plotting alongside RocketPy truth
        and EKF estimate in replay / closedloop modes.
        """
        controls = self.sim.controls
        x        = controls.x0.copy()
        u        = np.zeros_like(controls.u0)
        altitude = float(initial_altitude)
        x_hist   = []

        for i in range(len(t_array) - 1):
            t  = float(t_array[i])
            dt = float(t_array[i + 1] - t_array[i])

            controls.set_current_altitude(altitude)

            k1 = controls.f_numeric(t,        x,              u)
            k2 = controls.f_numeric(t + dt/2, x + dt/2 * k1, u)
            k3 = controls.f_numeric(t + dt/2, x + dt/2 * k2, u)
            k4 = controls.f_numeric(t + dt,   x + dt   * k3, u)

            x_hist.append(x.copy())

            x = x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
            q_norm = np.linalg.norm(x[6:10])
            if q_norm > 0:
                x[6:10] /= q_norm

            x = self.sim._apply_rail_constraint(x, t)

            # Update altitude for next env lookup
            qw_, qx_, qy_, qz_ = x[6:10]
            xx_, yy_, zz_ = qx_*qx_, qy_*qy_, qz_*qz_
            wx__, wy__, wz__ = qw_*qx_, qw_*qy_, qw_*qz_
            xy_, xz_, yz_ = qx_*qy_, qx_*qz_, qy_*qz_
            R_WB = np.array([
                [1-2*(yy_+zz_), 2*(xy_-wz__),  2*(xz_+wy__)],
                [2*(xy_+wz__),  1-2*(xx_+zz_), 2*(yz_-wx__)],
                [2*(xz_-wy__),  2*(yz_+wx__),  1-2*(xx_+yy_)]
            ])
            altitude = max(0.0, altitude + (R_WB @ x[3:6])[2] * dt)

        x_hist.append(x.copy())
        return np.array(x_hist[:len(t_array)])

    def _run_replay(self, flight):
        self.flight = flight
        t_history, state_history, deriv_history, altitude_history, \
            temperature_history, position_history = self._extract_rocketpy_data(flight)

        # Replay mode remains a no-feedback comparison pass, but we still run the
        # EKF against RocketPy truth. Control is forced to zero so estimation can
        # be evaluated without actuation affecting the comparison.
        original_compute_control = self.sim.controls.compute_control
        try:
            self.sim.controls.compute_control = (
                lambda t, xhat: np.zeros_like(self.sim.controls.u0)
            )
            self.results = self.sim.run(
                t_history, state_history, deriv_history, altitude_history, temperature_history
            )
        finally:
            self.sim.controls.compute_control = original_compute_control

        n = len(self.results['t'])
        self.results['position']    = position_history[:n]
        self.results['temperature'] = temperature_history[:n]
        initial_alt = float(altitude_history[0]) if len(altitude_history) > 0 else 0.0
        self.results['x_internal']  = self._propagate_internal(
            self.results['t'], initial_alt
        )
        return self.results

    def _extract_rocketpy_data(self, flight):
        solution    = np.array(flight.solution)
        t_history   = solution[:, 0]
        rpy_states  = solution[:, 1:]

        n = len(t_history)
        state_history    = np.zeros((n, 10))
        deriv_history    = np.zeros((n, 10))
        altitude_history = rpy_states[:, 2]   # index 2 = z = altitude in meters

        for i in range(n):
            state_history[i] = self._rocketpy_to_body(rpy_states[i])

        # Compute body-frame acceleration correctly:
        # Finite-difference world-frame velocity (rpy_states[:, 3:6]) to get
        # world-frame acceleration, then rotate to body frame using the quaternion.
        # This avoids the Coriolis contamination that occurs when differencing
        # body-frame velocities directly (which adds the omega x v term).
        world_vel = rpy_states[:, 3:6]   # vx, vy, vz in world frame

        for i in range(n):
            if i == 0:
                dt = t_history[1] - t_history[0]
                a_world = (world_vel[1] - world_vel[0]) / dt
            elif i == n - 1:
                dt = t_history[-1] - t_history[-2]
                a_world = (world_vel[-1] - world_vel[-2]) / dt
            else:
                dt = t_history[i+1] - t_history[i-1]
                a_world = (world_vel[i+1] - world_vel[i-1]) / dt

            # Rotate world-frame acceleration into body frame
            e0, e1, e2, e3 = rpy_states[i, 6], rpy_states[i, 7], rpy_states[i, 8], rpy_states[i, 9]
            q = np.array([e0, e1, e2, e3]); q /= np.linalg.norm(q)
            qw, qx, qy, qz = q
            xx, yy, zz = qx*qx, qy*qy, qz*qz
            wx_, wy_, wz_ = qw*qx, qw*qy, qw*qz
            xy, xz, yz = qx*qy, qx*qz, qy*qz
            R_BW = np.array([
                [1-2*(yy+zz), 2*(xy+wz_),  2*(xz-wy_)],
                [2*(xy-wz_),  1-2*(xx+zz), 2*(yz+wx_)],
                [2*(xz+wy_),  2*(yz-wx_),  1-2*(xx+yy)]
            ])
            a_body = R_BW @ a_world

            # Angular acceleration: finite difference of angular rates (already body-frame)
            if i == 0:
                ang_vel_now  = state_history[1, :3]
                ang_vel_prev = state_history[0, :3]
                dt_ang = t_history[1] - t_history[0]
            elif i == n - 1:
                ang_vel_now  = state_history[-1, :3]
                ang_vel_prev = state_history[-2, :3]
                dt_ang = t_history[-1] - t_history[-2]
            else:
                ang_vel_now  = state_history[i+1, :3]
                ang_vel_prev = state_history[i-1, :3]
                dt_ang = t_history[i+1] - t_history[i-1]
            ang_accel = (ang_vel_now - ang_vel_prev) / dt_ang

            deriv_history[i, :3] = ang_accel   # w1dot, w2dot, w3dot
            deriv_history[i, 3:6] = a_body      # v1dot, v2dot, v3dot (body frame, no Coriolis)

        # World-frame position (x, y, z) from RocketPy solution
        position_history = self._position_history_agl(
            rpy_states[:, 0:3], flight.env.elevation
        )

        # Temperature from RocketPy env; fall back to ISA lapse rate if unavailable
        try:
            temperature_history = np.array([float(flight.env.temperature(alt)) for alt in altitude_history])
        except Exception:
            temperature_history = 288.15 - 0.0065 * altitude_history  # ISA troposphere

        return t_history, state_history, deriv_history, altitude_history, \
               temperature_history, position_history

    @staticmethod
    def _position_history_agl(position_history_asl: np.ndarray, elevation_m: float) -> np.ndarray:
        """Convert RocketPy world positions to an output-friendly AGL form.

        RocketPy's raw z position carries the launch-site elevation offset.
        For user-facing plots and summaries we want z=0 at launch, so only the
        vertical component is shifted here. Atmosphere lookups still use the
        original altitude history elsewhere in the adapter.
        """
        position_history_agl = np.array(position_history_asl, copy=True)
        position_history_agl[:, 2] -= float(elevation_m)
        return position_history_agl

    def _rocketpy_to_body(self, rpy_state: np.ndarray) -> np.ndarray:
        """Convert a single RocketPy state vector to your body-frame state vector.

        Args:
            rpy_state: RocketPy state [x,y,z, vx,vy,vz, e0,e1,e2,e3, wx,wy,wz]

        Returns:
            body_state: [w1, w2, w3, v1, v2, v3, qw, qx, qy, qz]
        """
        vx, vy, vz = rpy_state[3], rpy_state[4], rpy_state[5]
        e0, e1, e2, e3 = rpy_state[6], rpy_state[7], rpy_state[8], rpy_state[9]
        wx, wy, wz = rpy_state[10], rpy_state[11], rpy_state[12]

        # Normalize quaternion
        q = np.array([e0, e1, e2, e3])
        q = q / np.linalg.norm(q)
        qw, qx, qy, qz = q

        # Build R_BW to rotate world-frame velocity to body frame
        xx, yy, zz = qx*qx, qy*qy, qz*qz
        wx_, wy_, wz_ = qw*qx, qw*qy, qw*qz
        xy, xz, yz = qx*qy, qx*qz, qy*qz

        R_BW = np.array([
            [1-2*(yy+zz),   2*(xy+wz_),   2*(xz-wy_)],
            [2*(xy-wz_),    1-2*(xx+zz),  2*(yz+wx_)],
            [2*(xz+wy_),    2*(yz-wx_),   1-2*(xx+yy)]
        ])

        v_world = np.array([vx, vy, vz])
        v_body  = R_BW @ v_world

        return np.array([wx, wy, wz,
                         v_body[0], v_body[1], v_body[2],
                         qw, qx, qy, qz])

    def save_results(self, path: str):
        """Save logged results to a .npz file.

        Args:
            path: File path to save to (e.g. 'results/run1.npz')
        """
        if self.results is None:
            raise RuntimeError("No results to save. Run the simulation first.")
        np.savez(path, **self.results)
        print(f"Results saved to {path}")
