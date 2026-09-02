"""Tests for CrosstalkCompensatedSQRB QUA probe generation."""

from types import SimpleNamespace
import numpy as np
import pytest

from scqo_qm.experiments.crosstalk_compensated_sqrb import (
    QMCrosstalkCompensatedSQRB,
    build_program,
    compute_theoretical_phase_rate,
)


from qm.qua import declare, declare_stream, fixed


def _mock_machine():
    """Mock QUAM machine with probe and drive qubits."""
    def _decl():
        return [declare(fixed)], [declare_stream()], [declare(fixed)], [declare_stream()], declare(int), declare_stream()

    q1_xy = SimpleNamespace(
        name="q1_xy",
        operations={"x180": SimpleNamespace(length=20)},
        wait=lambda *a: None,
        play=lambda *a: None,
    )
    q2_xy = SimpleNamespace(
        name="q2_xy",
        operations={"x180": SimpleNamespace(length=20)},
        wait=lambda *a: None,
        play=lambda *a: None,
    )

    q1_res = SimpleNamespace(name="q1_res", measure=lambda *a, **kw: None)
    q2_res = SimpleNamespace(name="q2_res", measure=lambda *a, **kw: None)

    q1 = SimpleNamespace(
        name="q1",
        xy=q1_xy,
        resonator=q1_res,
        reset=lambda *a: None,
        readout_state=lambda *a: None,
    )
    q2 = SimpleNamespace(
        name="q2",
        xy=q2_xy,
        resonator=q2_res,
        reset=lambda *a: None,
        readout_state=lambda *a: None,
    )

    mach = SimpleNamespace(
        declare_qua_variables=_decl,
        initialize_qpu=lambda **kw: None,
        qubits={"q1": q1, "q2": q2},
    )
    return mach, q1, q2


def test_theoretical_formula():
    f_d = 5.2e9
    f_p = 5.1e9
    slot_ns = 40.0
    rate = compute_theoretical_phase_rate(f_d, f_p, slot_ns)
    # Expected: 2 * pi * 100 MHz * 40 ns = 8 * pi
    assert np.isclose(rate, 2 * np.pi * 100e6 * 40e-9)


def test_build_program_benchmark_mode():
    mach, q1, q2 = _mock_machine()

    prog, sweep_axes = build_program(
        mach,
        q1,
        q2,
        mode="benchmark",
        depths=[1, 2, 4],
        num_sequences=2,
        num_shots=10,
        cancel_amp=0.035,
        init_phase=0.85,
        phase_rate_turns=0.25,
        cancel_amps=[0.01, 0.05],
        init_phases=[0.0, 1.0],
        cal_depth=10,
        reset_type="thermal",
        use_state_discrimination=False,
    )

    assert "condition" in sweep_axes
    assert "depth" in sweep_axes
    assert list(sweep_axes["condition"].values) == ["isolated", "simultaneous", "compensated"]
    assert len(sweep_axes["depth"]) == 3
    assert prog is not None


def test_build_program_calibrate_mode():
    mach, q1, q2 = _mock_machine()

    prog, sweep_axes = build_program(
        mach,
        q1,
        q2,
        mode="calibrate",
        depths=[1, 2, 4],
        num_sequences=2,
        num_shots=10,
        cancel_amp=0.035,
        init_phase=0.85,
        phase_rate_turns=0.25,
        cancel_amps=[0.01, 0.05, 0.1],
        init_phases=[-0.5, 0.0, 0.5],
        cal_depth=15,
        reset_type="thermal",
        use_state_discrimination=False,
    )

    assert "cancel_amp" in sweep_axes
    assert "init_phase" in sweep_axes
    assert len(sweep_axes["cancel_amp"]) == 3
    assert len(sweep_axes["init_phase"]) == 3
    assert prog is not None
