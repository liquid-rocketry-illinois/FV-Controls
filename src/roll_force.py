from typing import Callable

from rocketpy.rocket.aero_surface import GenericSurface

from controls.controls import Controls


class RollForce(GenericSurface):
    """Custom aerodynamic surface that applies a CFD-derived roll moment to a RocketPy simulation.

    The base roll moment is provided by Controls.cfd_roll_moment(velocity) — a stub
    to be implemented with a CFD lookup table. Additional roll contributions
    (e.g. vortex-induced effects) can be stacked with add_roll_func().

    Only outputs M3 (roll moment about the body axial axis).
    All other forces and moments returned are zero.

    ### Usage
        roll = RollForce(
            center_of_pressure=1.2,
            reference_area=0.01,
            reference_length=0.1,
            controls=controls,
        )
        roll.add_roll_func(my_vortex_moment_func)
        rocket.add_aero_surface(roll, position=1.2)
    """

    def __init__(
        self,
        center_of_pressure: float,
        reference_area: float,
        reference_length: float,
        controls: Controls,
        name: str = "RollForce",
    ):
        """
        Args:
            center_of_pressure (float): Axial position along the rocket body in meters
                (measured from the nozzle, same convention as other RocketPy surfaces).
            reference_area (float): Reference area in m².
            reference_length (float): Reference length in m (typically rocket diameter).
            controls (Controls): Controls object whose cfd_roll_moment() provides the
                base CFD roll moment.
            name (str): Label for this surface in RocketPy output.
        """
        super().__init__(
            reference_area=reference_area,
            reference_length=reference_length,
            coefficients={},
            center_of_pressure=center_of_pressure,
            name=name,
        )
        self.controls = controls
        self._extra_roll_funcs: list[Callable] = []

    def add_roll_func(self, func: Callable):
        """Register an additional roll moment contribution.

        All registered functions are evaluated and summed at every timestep,
        on top of the base CFD moment from Controls.cfd_roll_moment().

        Args:
            func (Callable): Signature (velocity: float) -> float.
                             Receives the freestream speed in m/s.
                             Returns an additional roll moment in N·m.
        """
        self._extra_roll_funcs.append(func)

    def compute_forces_and_moments(
        self,
        stream_velocity,
        stream_speed,
        stream_mach,
        rho,
        cp,
        omega,
        reynolds,
    ):
        """Compute the roll moment at the current timestep.

        Called by RocketPy at every integration step.

        Returns:
            tuple: (R1, R2, R3, M1, M2, M3) — all zero except M3 (roll moment).
        """
        M3 = self.controls.cfd_roll_moment(stream_speed)

        for func in self._extra_roll_funcs:
            M3 += func(stream_speed)

        return 0.0, 0.0, 0.0, 0.0, 0.0, M3
