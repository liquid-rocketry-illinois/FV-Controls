import numpy as np
from rocketpy.simulation.flight import Flight
from rocketpy.mathutils.vector_matrix import Matrix, Vector


class ControlledMomentFlight(Flight):
    """RocketPy Flight subclass with an externally injected body moment."""

    def __init__(self, *args, external_body_moment_provider=None, **kwargs):
        self.external_body_moment_provider = external_body_moment_provider
        self.external_body_moment = np.zeros(3, dtype=float)
        super().__init__(*args, **kwargs)

    def set_external_body_moment(self, moment_body):
        self.external_body_moment = np.asarray(moment_body, dtype=float).reshape(3)

    def _get_external_body_moment(self, t, u):
        if self.external_body_moment_provider is not None:
            moment = self.external_body_moment_provider(t, u)
            if moment is not None:
                return np.asarray(moment, dtype=float).reshape(3)
        return np.asarray(self.external_body_moment, dtype=float).reshape(3)

    def u_dot_generalized(self, t, u, post_processing=False):  # pylint: disable=too-many-locals,too-many-statements
        """RocketPy 6-DOF dynamics with an additional externally injected body moment."""
        _, _, z, vx, vy, vz, e0, e1, e2, e3, omega1, omega2, omega3 = u

        v = Vector([vx, vy, vz])
        e = [e0, e1, e2, e3]
        w = Vector([omega1, omega2, omega3])

        total_mass = self.rocket.total_mass.get_value_opt(t)
        total_mass_dot = self.rocket.total_mass_flow_rate.get_value_opt(t)
        total_mass_ddot = self.rocket.total_mass_flow_rate.differentiate_complex_step(t)

        r_CM_z = self.rocket.com_to_cdm_function
        r_CM_t = r_CM_z.get_value_opt(t)
        r_CM = Vector([0, 0, r_CM_t])
        r_CM_dot = Vector([0, 0, r_CM_z.differentiate_complex_step(t)])
        r_CM_ddot = Vector([0, 0, r_CM_z.differentiate(t, order=2)])

        r_NOZ = Vector([0, 0, self.rocket.nozzle_to_cdm])
        S_nozzle = self.rocket.nozzle_gyration_tensor
        inertia_tensor = self.rocket.get_inertia_tensor_at_time(t)
        I_dot = self.rocket.get_inertia_tensor_derivative_at_time(t)

        H = (r_CM.cross_matrix @ -r_CM.cross_matrix) * total_mass
        I_CM = inertia_tensor - H

        K = Matrix.transformation(e)
        Kt = K.transpose

        R1, R2, R3, M1, M2, M3 = 0, 0, 0, 0, 0, 0

        rho = self.env.density.get_value_opt(z)
        wind_velocity_x = self.env.wind_velocity_x.get_value_opt(z)
        wind_velocity_y = self.env.wind_velocity_y.get_value_opt(z)
        wind_velocity = Vector([wind_velocity_x, wind_velocity_y, 0])
        free_stream_speed = abs((wind_velocity - Vector(v)))
        speed_of_sound = self.env.speed_of_sound.get_value_opt(z)
        free_stream_mach = free_stream_speed / speed_of_sound

        if self.rocket.motor.burn_start_time < t < self.rocket.motor.burn_out_time:
            pressure = self.env.pressure.get_value_opt(z)
            net_thrust = max(
                self.rocket.motor.thrust.get_value_opt(t)
                + self.rocket.motor.pressure_thrust(pressure),
                0,
            )
            drag_coeff = self.rocket.power_on_drag.get_value_opt(free_stream_mach)
        else:
            net_thrust = 0
            drag_coeff = self.rocket.power_off_drag.get_value_opt(free_stream_mach)

        R3 += -0.5 * rho * (free_stream_speed**2) * self.rocket.area * drag_coeff
        for air_brakes in self.rocket.air_brakes:
            if air_brakes.deployment_level > 0:
                air_brakes_cd = air_brakes.drag_coefficient.get_value_opt(
                    air_brakes.deployment_level, free_stream_mach
                )
                air_brakes_force = (
                    -0.5
                    * rho
                    * (free_stream_speed**2)
                    * air_brakes.reference_area
                    * air_brakes_cd
                )
                if air_brakes.override_rocket_drag:
                    R3 = air_brakes_force
                else:
                    R3 += air_brakes_force

        velocity_in_body_frame = Kt @ v
        for aero_surface, _ in self.rocket.aerodynamic_surfaces:
            comp_cp = self.rocket.surfaces_cp_to_cdm[aero_surface]
            comp_vb = velocity_in_body_frame + (w ^ comp_cp)
            comp_z = z + (K @ comp_cp).z
            comp_wind_vx = self.env.wind_velocity_x.get_value_opt(comp_z)
            comp_wind_vy = self.env.wind_velocity_y.get_value_opt(comp_z)
            comp_wind_vb = Kt @ Vector([comp_wind_vx, comp_wind_vy, 0])
            comp_stream_velocity = comp_wind_vb - comp_vb
            comp_stream_speed = abs(comp_stream_velocity)
            comp_stream_mach = comp_stream_speed / speed_of_sound
            comp_reynolds = (
                self.env.density.get_value_opt(comp_z)
                * comp_stream_speed
                * aero_surface.reference_length
                / self.env.dynamic_viscosity.get_value_opt(comp_z)
            )
            X, Y, Z, M, N, L = aero_surface.compute_forces_and_moments(
                comp_stream_velocity,
                comp_stream_speed,
                comp_stream_mach,
                rho,
                comp_cp,
                w,
                comp_reynolds,
            )
            R1 += X
            R2 += Y
            R3 += Z
            M1 += M
            M2 += N
            M3 += L

        M1 += (
            self.rocket.cp_eccentricity_y * R3
            + self.rocket.thrust_eccentricity_y * net_thrust
        )
        M2 -= (
            self.rocket.cp_eccentricity_x * R3
            + self.rocket.thrust_eccentricity_x * net_thrust
        )
        M3 += self.rocket.cp_eccentricity_x * R2 - self.rocket.cp_eccentricity_y * R1

        ext_moment = self._get_external_body_moment(t, u)
        M1 += float(ext_moment[0])
        M2 += float(ext_moment[1])
        M3 += float(ext_moment[2])

        weight_in_body_frame = Kt @ Vector(
            [0, 0, -total_mass * self.env.gravity.get_value_opt(z)]
        )

        T00 = total_mass * r_CM
        T03 = 2 * total_mass_dot * (r_NOZ - r_CM) - 2 * total_mass * r_CM_dot
        T04 = (
            Vector([0, 0, net_thrust])
            - total_mass * r_CM_ddot
            - 2 * total_mass_dot * r_CM_dot
            + total_mass_ddot * (r_NOZ - r_CM)
        )
        T05 = total_mass_dot * S_nozzle - I_dot

        T20 = (
            ((w ^ T00) ^ w)
            + (w ^ T03)
            + T04
            + weight_in_body_frame
            + Vector([R1, R2, R3])
        )

        T21 = (
            ((inertia_tensor @ w) ^ w)
            + T05 @ w
            - (weight_in_body_frame ^ r_CM)
            + Vector([M1, M2, M3])
        )

        w_dot = I_CM.inverse @ (T21 + (T20 ^ r_CM))
        v_dot = K @ (T20 / total_mass - (r_CM ^ w_dot))

        e_dot = [
            0.5 * (-omega1 * e1 - omega2 * e2 - omega3 * e3),
            0.5 * (omega1 * e0 + omega3 * e2 - omega2 * e3),
            0.5 * (omega2 * e0 - omega3 * e1 + omega1 * e3),
            0.5 * (omega3 * e0 + omega2 * e1 - omega1 * e2),
        ]
        r_dot = [vx, vy, vz]
        u_dot = [*r_dot, *v_dot, *e_dot, *w_dot]

        if post_processing:
            self._Flight__post_processed_variables.append(  # pylint: disable=protected-access
                [t, *v_dot, *w_dot, R1, R2, R3, M1, M2, M3, net_thrust]
            )

        return u_dot
