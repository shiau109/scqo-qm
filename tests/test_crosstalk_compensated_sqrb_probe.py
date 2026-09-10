"""Tests for CrosstalkCompensatedSQRB QUA probe generation."""

from types import SimpleNamespace
import numpy as np
import pytest

from scqo_qm.experiments.crosstalk_compensated_sqrb import (
    CLIFFORD_3SUBSLOTS,
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
        name="q1.xy",
        operations={"x180": SimpleNamespace(length=20)},
        wait=lambda *a: None,
        play=lambda *a: None,
    )
    q2_xy = SimpleNamespace(
        name="q2.xy",
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
        target_gate="x180",
        cal_stage_repetitions=[10, 20, 30],
        num_shots=10,
        cancel_amps=[0.01, 0.05, 0.1],
        init_phases=[-0.5, 0.0, 0.5],
        sub_phase_rate_turns=0.25,
        reset_type="thermal",
        use_state_discrimination=False,
    )

    assert "cancel_amp" in sweep_axes
    assert "init_phase" in sweep_axes
    assert len(sweep_axes["cancel_amp"]) == 3
    assert len(sweep_axes["init_phase"]) == 3
    assert prog is not None

    from qm import generate_qua_script
    script = generate_qua_script(prog)
    assert 'frame_rotation_2pi(-0.25, "q2.xy")' in script or "frame_rotation_2pi(-0.25, 'q2.xy')" in script
    assert 'frame_rotation_2pi(0.25, "q1.xy_cancel")' not in script and "frame_rotation_2pi(0.25, 'q1.xy_cancel')" not in script
    assert 'frame_rotation_2pi(-0.25, "q1.xy_cancel")' not in script and "frame_rotation_2pi(-0.25, 'q1.xy_cancel')" not in script


def test_clifford_3subslots_structure():
    assert len(CLIFFORD_3SUBSLOTS) == 24
    for idx, subslots in enumerate(CLIFFORD_3SUBSLOTS):
        assert len(subslots) == 3, f"Gate {idx} does not have 3 subslots"
        for slot in subslots:
            if slot is not None:
                assert len(slot) == 3
                op_name, pulse_kind, axis_phase = slot
                assert pulse_kind in ("x90", "x180")
                assert 0.0 <= axis_phase < 1.0


def test_build_program_with_sub_phase_rate():
    mach, q1, q2 = _mock_machine()
    prog, sweep_axes = build_program(
        mach,
        q1,
        q2,
        mode="benchmark",
        depths=[1, 2],
        num_sequences=2,
        num_shots=5,
        cancel_amp=0.02,
        init_phase=0.5,
        sub_phase_rate_turns=0.08,
        cancel_amps=[0.01, 0.05],
        init_phases=[0.0, 1.0],
        cal_depth=5,
        reset_type="thermal",
    )
    assert prog is not None


def test_patch_preview_config():
    from scqo.experiments.crosstalk_compensated_sqrb import CrosstalkCompensatedSQRBParameters

    mach, q1, q2 = _mock_machine()
    exp = QMCrosstalkCompensatedSQRB(
        SimpleNamespace(machine=mach, device=None),
        CrosstalkCompensatedSQRBParameters(probe_qubit="q1", drive_qubit="q2"),
    )

    cfg = {
        "elements": {
            "q1.xy": {"core": "a", "operations": {"x180": "q1_x180_pulse"}, "MWInput": {"port": ("con1", 6, 2)}},
            "q2.xy": {"core": "b", "operations": {"x180": "q2_x180_pulse"}, "MWInput": {"port": ("con1", 6, 3)}},
        }
    }

    patched = exp.patch_preview_config(cfg)
    cancel_key = "q1.xy_cancel" if "q1.xy_cancel" in patched["elements"] else "q1_xy_cancel"
    assert cancel_key in patched["elements"]
    assert patched["elements"][cancel_key]["core"] != patched["elements"]["q1.xy"]["core"]
    assert patched["elements"][cancel_key]["core"] not in ["a", "b"]
    assert patched["elements"][cancel_key]["operations"]["x180"] == "q2_x180_pulse"


def test_preview_program():
    from scqo.experiments.crosstalk_compensated_sqrb import CrosstalkCompensatedSQRBParameters
    from qm import generate_qua_script

    mach, q1, q2 = _mock_machine()
    exp = QMCrosstalkCompensatedSQRB(
        SimpleNamespace(machine=mach, device=None),
        CrosstalkCompensatedSQRBParameters(probe_qubit="q1", drive_qubit="q2", mode="benchmark"),
    )
    prog = exp.preview_program()
    assert prog is not None
    script = generate_qua_script(prog)
    assert "q1.xy_cancel" in script or "q1_xy_cancel" in script
    assert "cond_idx" not in script
    assert "case_(0)" not in script
    assert "case_(1)" not in script


def test_acquire_benchmark_batched(monkeypatch):
    import xarray as xr
    from scqo.experiments.crosstalk_compensated_sqrb import CrosstalkCompensatedSQRBParameters

    mach, q1, q2 = _mock_machine()
    exp = QMCrosstalkCompensatedSQRB(
        SimpleNamespace(machine=mach, device=None),
        CrosstalkCompensatedSQRBParameters(
            probe_qubit="q1",
            drive_qubit="q2",
            mode="benchmark",
            depths=[2, 4, 8],
            benchmark_batch_size=1,
            num_random_sequences=2,
        ),
    )

    call_count = 0
    def mock_acquire(m, p, s, **kw):
        nonlocal call_count
        call_count += 1
        d_vals = s["depth"].values
        c_vals = s["condition"].values
        n_seq = len(s["sequence_idx"].values)
        return xr.Dataset(
            data_vars={"I": (("qubit", "condition", "sequence_idx", "depth"), np.zeros((1, len(c_vals), n_seq, len(d_vals))))},
            coords={
                "qubit": ["q1"],
                "condition": c_vals,
                "sequence_idx": s["sequence_idx"].values,
                "depth": d_vals,
            },
        )

    import scqo_qm.experiments.crosstalk_compensated_sqrb as mod
    monkeypatch.setattr(mod, "_acquire", mock_acquire)

    res = exp._acquire_benchmark(mach, None, None, num_shots=10, probe_qubit=q1, drive_qubit=q2)
    assert call_count == 9
    assert list(res.coords["depth"].values) == [2, 4, 8]
    assert list(res.coords["condition"].values) == ["isolated", "simultaneous", "compensated"]
    assert list(res.coords["sequence_idx"].values) == [0, 1]


def test_strict_timing_toggle():
    from scqo.experiments.crosstalk_compensated_sqrb import CrosstalkCompensatedSQRBParameters
    from qm import generate_qua_script

    mach, q1, q2 = _mock_machine()
    exp_strict = QMCrosstalkCompensatedSQRB(
        SimpleNamespace(machine=mach, device=None),
        CrosstalkCompensatedSQRBParameters(probe_qubit="q1", drive_qubit="q2", mode="benchmark", strict_timing=True),
    )
    prog_strict = exp_strict.preview_program()
    script_strict = generate_qua_script(prog_strict)
    assert "strict_timing_" in script_strict

    exp_loose = QMCrosstalkCompensatedSQRB(
        SimpleNamespace(machine=mach, device=None),
        CrosstalkCompensatedSQRBParameters(probe_qubit="q1", drive_qubit="q2", mode="benchmark", strict_timing=False),
    )
    prog_loose = exp_loose.preview_program()
    script_loose = generate_qua_script(prog_loose)
    assert "strict_timing_" not in script_loose


def test_compensation_3subslot_decomposition():
    from scqo.experiments.crosstalk_compensated_sqrb import CrosstalkCompensatedSQRBParameters
    from qm import generate_qua_script

    assert len(CLIFFORD_3SUBSLOTS) == 24
    for gate_subslots in CLIFFORD_3SUBSLOTS:
        assert len(gate_subslots) == 3

    mach, q1, q2 = _mock_machine()
    exp = QMCrosstalkCompensatedSQRB(
        SimpleNamespace(machine=mach, device=None),
        CrosstalkCompensatedSQRBParameters(
            probe_qubit="q1",
            drive_qubit="q2",
            mode="benchmark",
            strict_timing=True,
            phase_rate=0.5,
        ),
    )
    prog = exp.preview_program()
    script = generate_qua_script(prog)
    assert "q1.xy_cancel" in script or "q1_xy_cancel" in script
    assert "frame_rotation_2pi" in script



