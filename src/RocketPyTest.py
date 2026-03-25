"""
RocketPyTest.py — Tests for RollForce, CanardSurface, RocketSim, and cfd_roll_moment.

Run with:
    pytest src/RocketPyTest.py -v
or from src/:
    pytest RocketPyTest.py -v
"""
import csv
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from controls.controls import Controls
from roll_force import RollForce
from rocket import CanardSurface, RocketSim


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_mock_controls(cfd_roll_return=0.0, m_controls_return=(0.0, 0.0, 0.0)):
    """Return a MagicMock shaped like a Controls object."""
    ctrl = MagicMock(spec=Controls)
    ctrl.x0 = np.zeros(10)
    ctrl.u0 = np.array([0.0])
    ctrl.rocket_name = "TestRocket"
    ctrl.t_launch_rail_clearance = 0.3
    ctrl.t_motor_burnout = 3.0
    ctrl.dt = 0.025
    ctrl.cfd_roll_moment = MagicMock(return_value=cfd_roll_return)
    ctrl.M_controls_func = MagicMock(return_value=m_controls_return)
    return ctrl


def rocketpy_stream_args(velocity=None):
    """Return the positional args RocketPy passes to compute_forces_and_moments."""
    if velocity is None:
        velocity = np.array([0.1, 0.2, 50.0])
    speed = float(np.linalg.norm(velocity))
    mach = speed / 343.0
    rho = 1.225
    cp = MagicMock()
    omega = (0.0, 0.0, 0.1)
    reynolds = 1e6
    return velocity, speed, mach, rho, cp, omega, reynolds


def make_rocketpy_state(vz=50.0, w3=0.1):
    """Return a 13-element RocketPy state vector [x,y,z,vx,vy,vz,e0,e1,e2,e3,wx,wy,wz]."""
    return [0, 0, 100, 0, 0, vz, 1, 0, 0, 0, 0, 0, w3]


# ---------------------------------------------------------------------------
# RollForce
# ---------------------------------------------------------------------------

class TestRollForce:

    def _make(self, **kw):
        ctrl = make_mock_controls(**kw)
        roll = RollForce(
            center_of_pressure=1.2,
            reference_area=0.01,
            reference_length=0.1,
            controls=ctrl,
        )
        return roll, ctrl

    def test_init_stores_controls(self):
        roll, ctrl = self._make()
        assert roll.controls is ctrl

    def test_init_extra_funcs_empty(self):
        roll, _ = self._make()
        assert roll._extra_roll_funcs == []

    def test_forces_are_zero(self):
        roll, _ = self._make(cfd_roll_return=99.0)
        R1, R2, R3, M1, M2, _ = roll.compute_forces_and_moments(*rocketpy_stream_args())
        assert R1 == 0.0
        assert R2 == 0.0
        assert R3 == 0.0
        assert M1 == 0.0
        assert M2 == 0.0

    def test_M3_from_cfd_roll_moment(self):
        roll, _ = self._make(cfd_roll_return=12.5)
        _, _, _, _, _, M3 = roll.compute_forces_and_moments(*rocketpy_stream_args())
        assert M3 == pytest.approx(12.5)

    def test_cfd_roll_moment_called_with_stream_speed(self):
        roll, ctrl = self._make()
        args = rocketpy_stream_args()
        expected_speed = args[1]
        roll.compute_forces_and_moments(*args)
        ctrl.cfd_roll_moment.assert_called_once_with(expected_speed)

    def test_add_roll_func_stored(self):
        roll, _ = self._make()
        f1, f2 = lambda v: 1.0, lambda v: 2.0
        roll.add_roll_func(f1)
        roll.add_roll_func(f2)
        assert roll._extra_roll_funcs == [f1, f2]

    def test_extra_funcs_summed_with_cfd(self):
        roll, _ = self._make(cfd_roll_return=1.0)
        roll.add_roll_func(lambda v: 2.0)
        roll.add_roll_func(lambda v: 3.0)
        _, _, _, _, _, M3 = roll.compute_forces_and_moments(*rocketpy_stream_args())
        assert M3 == pytest.approx(6.0)   # 1 + 2 + 3

    def test_extra_funcs_receive_stream_speed(self):
        roll, _ = self._make(cfd_roll_return=0.0)
        received = []
        roll.add_roll_func(lambda v: received.append(v) or 0.0)
        args = rocketpy_stream_args()
        roll.compute_forces_and_moments(*args)
        assert len(received) == 1
        assert received[0] == pytest.approx(args[1])

    def test_no_extra_funcs_M3_is_only_cfd(self):
        roll, _ = self._make(cfd_roll_return=7.77)
        _, _, _, _, _, M3 = roll.compute_forces_and_moments(*rocketpy_stream_args())
        assert M3 == pytest.approx(7.77)

    def test_zero_cfd_and_zero_extras_gives_zero_M3(self):
        roll, _ = self._make(cfd_roll_return=0.0)
        _, _, _, _, _, M3 = roll.compute_forces_and_moments(*rocketpy_stream_args())
        assert M3 == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# CanardSurface
# ---------------------------------------------------------------------------

class TestCanardSurface:

    def _make(self, m_controls_return=(0.1, 0.2, 0.3)):
        ctrl = make_mock_controls(m_controls_return=m_controls_return)
        canard = CanardSurface(
            center_of_pressure=0.5,
            reference_area=0.01,
            reference_length=0.1,
            controls=ctrl,
        )
        return canard, ctrl

    def test_init_stores_controls(self):
        canard, ctrl = self._make()
        assert canard.controls is ctrl

    def test_init_aileron_angles_zero(self):
        canard, _ = self._make()
        np.testing.assert_array_equal(canard.aileron_angles, np.array([0.0]))

    def test_forces_are_zero(self):
        canard, _ = self._make()
        R1, R2, R3, _, _, _ = canard.compute_forces_and_moments(*rocketpy_stream_args())
        assert R1 == 0.0
        assert R2 == 0.0
        assert R3 == 0.0

    def test_moments_match_M_controls_func(self):
        canard, _ = self._make(m_controls_return=(1.0, 2.0, 3.0))
        _, _, _, M1, M2, M3 = canard.compute_forces_and_moments(*rocketpy_stream_args())
        assert M1 == pytest.approx(1.0)
        assert M2 == pytest.approx(2.0)
        assert M3 == pytest.approx(3.0)

    def test_aileron_angles_forwarded_to_M_controls_func(self):
        canard, ctrl = self._make()
        canard.aileron_angles = np.array([0.07])
        canard.compute_forces_and_moments(*rocketpy_stream_args())
        _, call_kwargs = ctrl.M_controls_func.call_args
        u_passed = ctrl.M_controls_func.call_args[0][1]
        assert u_passed[0] == pytest.approx(0.07)

    def test_stream_velocity_components_in_state(self):
        canard, ctrl = self._make()
        velocity = np.array([3.0, 4.0, 50.0])
        canard.compute_forces_and_moments(*rocketpy_stream_args(velocity=velocity))
        state_passed = ctrl.M_controls_func.call_args[0][0]
        # state[3:6] must be v1, v2, v3 from stream_velocity
        np.testing.assert_array_equal(state_passed[3:6], velocity)

    def test_aileron_angle_updates_propagate(self):
        canard, ctrl = self._make()
        for angle in [0.0, 0.05, -0.05, 0.1]:
            canard.aileron_angles = np.array([angle])
            canard.compute_forces_and_moments(*rocketpy_stream_args())
            u_passed = ctrl.M_controls_func.call_args[0][1]
            assert u_passed[0] == pytest.approx(angle)


# ---------------------------------------------------------------------------
# RocketSim
# ---------------------------------------------------------------------------

class TestRocketSim:

    def _make(self):
        ctrl = make_mock_controls()
        sim = RocketSim(controls=ctrl, sampling_rate=40.0)
        return sim, ctrl

    # --- init ---

    def test_init_stores_controls_and_rate(self):
        sim, ctrl = self._make()
        assert sim.controls is ctrl
        assert sim.sampling_rate == 40.0

    def test_init_logs_empty(self):
        sim, ctrl = self._make()
        assert sim.times == []
        assert sim.states == []
        np.testing.assert_array_equal(sim.xhats[0], ctrl.x0)
        np.testing.assert_array_equal(sim.inputs[0], ctrl.u0)

    def test_set_rocket_and_env(self):
        sim, _ = self._make()
        cr = lambda: None
        ce = lambda: None
        sim.set_rocket(cr)
        sim.set_env(ce)
        assert sim.create_rocket is cr
        assert sim.create_env is ce

    # --- state conversion ---

    def test_rocketpy_state_to_xhat_mapping(self):
        sim, _ = self._make()
        # [x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz]
        rpy = [1, 2, 3,  4, 5, 6,  0.9, 0.1, 0.2, 0.3,  7, 8, 9]
        xhat = sim._rocketpy_state_to_xhat(rpy)
        expected = np.array([7, 8, 9, 4, 5, 6, 0.9, 0.1, 0.2, 0.3])
        np.testing.assert_array_equal(xhat, expected)

    def test_rocketpy_state_to_xhat_length(self):
        sim, _ = self._make()
        xhat = sim._rocketpy_state_to_xhat(make_rocketpy_state())
        assert len(xhat) == 10

    # --- controller_function ---

    def test_controller_logs_state_and_time(self):
        sim, ctrl = self._make()
        sim.simulation.controls_step = MagicMock(
            return_value=(np.zeros(10), np.array([0.01]))
        )
        canard = MagicMock()
        sim.controller_function(1.0, 40.0, make_rocketpy_state(), [], [], [canard])
        assert len(sim.times) == 1
        assert sim.times[0] == 1.0
        assert len(sim.states) == 1

    def test_controller_updates_canard_aileron_angles(self):
        sim, ctrl = self._make()
        u_new = np.array([0.05])
        sim.simulation.controls_step = MagicMock(
            return_value=(np.zeros(10), u_new)
        )
        canard = MagicMock()
        sim.controller_function(1.0, 40.0, make_rocketpy_state(), [], [], [canard])
        np.testing.assert_array_equal(canard.aileron_angles, u_new)

    def test_controller_returns_control_input(self):
        sim, ctrl = self._make()
        u_new = np.array([0.03])
        sim.simulation.controls_step = MagicMock(
            return_value=(np.zeros(10), u_new)
        )
        result = sim.controller_function(
            1.0, 40.0, make_rocketpy_state(), [], [], [MagicMock()]
        )
        np.testing.assert_array_equal(result, u_new)

    def test_controller_apogee_returns_zero(self):
        sim, ctrl = self._make()
        # time past burnout, vz negative → apogee
        state = make_rocketpy_state(vz=-1.0)
        result = sim.controller_function(
            time=10.0,
            sampling_rate=40.0,
            state=state,
            state_history=[],
            observed_variables=[],
            interactive_objects=[MagicMock()],
        )
        np.testing.assert_array_equal(result, np.array([0.0]))

    def test_controller_apogee_does_not_call_controls_step(self):
        sim, ctrl = self._make()
        sim.simulation.controls_step = MagicMock()
        state = make_rocketpy_state(vz=-1.0)
        sim.controller_function(10.0, 40.0, state, [], [], [MagicMock()])
        sim.simulation.controls_step.assert_not_called()

    def test_controller_not_apogee_when_ascending(self):
        """Past burnout time but still ascending — should NOT trigger apogee."""
        sim, ctrl = self._make()
        sim.simulation.controls_step = MagicMock(
            return_value=(np.zeros(10), np.array([0.0]))
        )
        state = make_rocketpy_state(vz=10.0)   # vz > 0
        sim.controller_function(10.0, 40.0, state, [], [], [MagicMock()])
        sim.simulation.controls_step.assert_called_once()

    def test_controller_appends_xhat_and_input(self):
        sim, ctrl = self._make()
        xhat_new = np.ones(10)
        u_new = np.array([0.02])
        sim.simulation.controls_step = MagicMock(return_value=(xhat_new, u_new))
        sim.controller_function(1.0, 40.0, make_rocketpy_state(), [], [], [MagicMock()])
        np.testing.assert_array_equal(np.array(sim.xhats[-1]), xhat_new)
        np.testing.assert_array_equal(np.array(sim.inputs[-1]), u_new)

    # --- export_states ---

    def test_export_states_creates_file(self):
        sim, _ = self._make()
        sim.times  = [0.0, 0.025]
        sim.states = [make_rocketpy_state(), make_rocketpy_state(vz=55.0)]
        sim.xhats  = [np.zeros(10), np.ones(10), np.ones(10)]
        sim.inputs = [np.array([0.0]), np.array([0.05]), np.array([0.05])]

        with tempfile.TemporaryDirectory() as tmp:
            sim.output_path = Path(tmp)
            sim.export_states("test_export")
            assert (Path(tmp) / "test_export.csv").exists()

    def test_export_states_correct_row_count(self):
        sim, _ = self._make()
        n = 3
        sim.times  = [i * 0.025 for i in range(n)]
        sim.states = [make_rocketpy_state() for _ in range(n)]
        sim.xhats  = [np.zeros(10)] * (n + 1)
        sim.inputs = [np.array([0.0])] * (n + 1)

        with tempfile.TemporaryDirectory() as tmp:
            sim.output_path = Path(tmp)
            sim.export_states("test_rows")
            with open(Path(tmp) / "test_rows.csv") as f:
                rows = list(csv.reader(f))
            assert rows[0][0] == "time"
            assert len(rows) == n + 1  # header + n data rows

    def test_export_states_time_values_correct(self):
        sim, _ = self._make()
        sim.times  = [0.0, 0.025, 0.05]
        sim.states = [make_rocketpy_state()] * 3
        sim.xhats  = [np.zeros(10)] * 4
        sim.inputs = [np.array([0.0])] * 4

        with tempfile.TemporaryDirectory() as tmp:
            sim.output_path = Path(tmp)
            sim.export_states("test_time")
            with open(Path(tmp) / "test_time.csv") as f:
                rows = list(csv.reader(f))
            times = [float(r[0]) for r in rows[1:]]
            assert times == pytest.approx([0.0, 0.025, 0.05])

    def test_export_states_raises_if_no_data(self):
        sim, _ = self._make()
        with tempfile.TemporaryDirectory() as tmp:
            sim.output_path = Path(tmp)
            with pytest.raises(ValueError):
                sim.export_states("empty")

    # --- run ---

    def test_run_creates_flight_with_correct_args(self):
        sim, _ = self._make()
        mock_rocket, mock_canard, mock_env, mock_flight = (
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        sim.set_rocket(lambda: (mock_rocket, mock_canard))
        sim.set_env(lambda: mock_env)

        with tempfile.TemporaryDirectory() as tmp:
            sim.output_path = Path(tmp)
            with patch("rocket.Flight", return_value=mock_flight) as mock_cls:
                sim.run("flight", rail_length=5.0, inclination=85.0, heading=0.0)

            mock_cls.assert_called_once_with(
                rocket=mock_rocket,
                environment=mock_env,
                rail_length=5.0,
                inclination=85.0,
                heading=0.0,
            )

    def test_run_calls_export_data(self):
        sim, _ = self._make()
        mock_flight = MagicMock()
        sim.set_rocket(lambda: (MagicMock(), MagicMock()))
        sim.set_env(lambda: MagicMock())

        with tempfile.TemporaryDirectory() as tmp:
            sim.output_path = Path(tmp)
            with patch("rocket.Flight", return_value=mock_flight):
                sim.run("flight")
            mock_flight.export_data.assert_called_once()

    def test_run_returns_flight_object(self):
        sim, _ = self._make()
        mock_flight = MagicMock()
        sim.set_rocket(lambda: (MagicMock(), MagicMock()))
        sim.set_env(lambda: MagicMock())

        with tempfile.TemporaryDirectory() as tmp:
            sim.output_path = Path(tmp)
            with patch("rocket.Flight", return_value=mock_flight):
                result = sim.run("flight")
            assert result is mock_flight

    def test_run_sampling_rate_override(self):
        sim, _ = self._make()
        assert sim.sampling_rate == 40.0
        sim.set_rocket(lambda: (MagicMock(), MagicMock()))
        sim.set_env(lambda: MagicMock())

        with tempfile.TemporaryDirectory() as tmp:
            sim.output_path = Path(tmp)
            with patch("rocket.Flight", return_value=MagicMock()):
                sim.run("flight", sampling_rate=100.0)
        assert sim.sampling_rate == 100.0


# ---------------------------------------------------------------------------
# Controls.cfd_roll_moment stub
# ---------------------------------------------------------------------------

class TestCfdRollMomentStub:

    def _get_unbound(self):
        """Return the real method without instantiating Controls."""
        return Controls.cfd_roll_moment

    def test_returns_zero(self):
        method = self._get_unbound()
        assert method(MagicMock(spec=Controls), velocity=100.0) == 0.0

    def test_returns_float(self):
        method = self._get_unbound()
        result = method(MagicMock(spec=Controls), velocity=50.0)
        assert isinstance(result, float)

    @pytest.mark.parametrize("v", [0.0, 50.0, 340.0, 1000.0])
    def test_always_zero_for_any_velocity(self, v):
        method = self._get_unbound()
        assert method(MagicMock(spec=Controls), velocity=v) == 0.0
