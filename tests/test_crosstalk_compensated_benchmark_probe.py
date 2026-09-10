"""Tests for CrosstalkCompensatedBenchmark QUA probe generation."""

from types import SimpleNamespace
import numpy as np
import pytest

from scqo_qm.experiments.crosstalk_compensated_benchmark import (
    QMCrosstalkCompensatedBenchmark,
    build_program,
)

from qm.qua import declare, declare_stream, fixed, play, amp


def _mock_machine():
    """Mock QUAM machine with probe and drive qubits."""
    def _decl():
        return [declare(fixed)], [declare_stream()], [declare(fixed)], [declare_stream()], declare(int), declare_stream()

    q1_xy = SimpleNamespace(
        name="q1.xy",
        operations={"x180": SimpleNamespace(length=16)},
        wait=lambda *a, **kw: None,
        play=lambda op, amplitude_scale=None, **kw: play(
            op * amp(amplitude_scale) if amplitude_scale is not None else op, "q1.xy"
        ),
    )
    q2_xy = SimpleNamespace(
        name="q2.xy",
        operations={"x180": SimpleNamespace(length=16)},
        wait=lambda *a, **kw: None,
        play=lambda op, amplitude_scale=None, **kw: play(
            op * amp(amplitude_scale) if amplitude_scale is not None else op, "q2.xy"
        ),
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


def test_build_program_calibrate_mode():
    mach, q1, q2 = _mock_machine()

    prog, sweep_axes = build_program(
        mach,
        q1,
        q2,
        mode="calibrate",
        target_gate="x180",
        cal_repetitions=[10, 20, 30],
        repetitions=[0, 4, 8],
        num_shots=10,
        cancel_amps=[0.01, 0.03, 0.05],
        init_phases=[0.0, 0.5, 1.0],
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


def test_build_program_benchmark_mode():
    mach, q1, q2 = _mock_machine()

    prog, sweep_axes = build_program(
        mach,
        q1,
        q2,
        mode="benchmark",
        target_gate="x180",
        probe_gate="x180",
        repetitions=[0, 4, 8, 16],
        num_shots=10,
        cancel_amp=0.035,
        init_phase=0.85,
        sub_phase_rate_turns=0.25,
        reset_type="thermal",
        use_state_discrimination=False,
    )

    assert "condition" in sweep_axes
    assert "repetitions" in sweep_axes
    assert list(sweep_axes["condition"].values) == ["isolated", "simultaneous", "compensated"]
    assert len(sweep_axes["repetitions"]) == 4
    assert prog is not None


def test_patch_preview_config():
    from scqo.experiments.crosstalk_compensated_benchmark import CrosstalkCompensatedBenchmarkParameters

    mach, q1, q2 = _mock_machine()
    exp = QMCrosstalkCompensatedBenchmark(
        SimpleNamespace(machine=mach, device=None),
        CrosstalkCompensatedBenchmarkParameters(probe_qubit="q1", drive_qubit="q2"),
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
    from scqo.experiments.crosstalk_compensated_benchmark import CrosstalkCompensatedBenchmarkParameters
    from qm import generate_qua_script

    mach, q1, q2 = _mock_machine()
    exp = QMCrosstalkCompensatedBenchmark(
        SimpleNamespace(machine=mach, device=None),
        CrosstalkCompensatedBenchmarkParameters(probe_qubit="q1", drive_qubit="q2", mode="benchmark"),
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
    from scqo.experiments.crosstalk_compensated_benchmark import CrosstalkCompensatedBenchmarkParameters

    mach, q1, q2 = _mock_machine()
    exp = QMCrosstalkCompensatedBenchmark(
        SimpleNamespace(machine=mach, device=None),
        CrosstalkCompensatedBenchmarkParameters(
            probe_qubit="q1",
            drive_qubit="q2",
            mode="benchmark",
            repetitions=[2, 4, 8],
            benchmark_batch_size=1,
        ),
    )

    call_count = 0
    def mock_acquire(m, p, s, **kw):
        nonlocal call_count
        call_count += 1
        reps = s["repetitions"].values
        return xr.Dataset(
            data_vars={"I": (("qubit", "condition", "repetitions"), np.zeros((1, 3, len(reps))))},
            coords={
                "qubit": ["q1"],
                "condition": ["isolated", "simultaneous", "compensated"],
                "repetitions": reps,
            },
        )

    import scqo_qm.experiments.crosstalk_compensated_benchmark as mod
    monkeypatch.setattr(mod, "_acquire", mock_acquire)
    res = exp._acquire_benchmark(mach, None, None, num_shots=10, probe_qubit=q1, drive_qubit=q2)
    assert call_count == 3
    assert list(res.coords["repetitions"].values) == [2, 4, 8]


def test_build_program_alternating_combinations():
    from qm import generate_qua_script

    mach, q1, q2 = _mock_machine()

    def has_probe_neg(script: str) -> bool:
        return "play('x180'*amp(-1.0), 'q1.xy')" in script or 'play("x180"*amp(-1.0), "q1.xy")' in script

    def has_target_neg(script: str) -> bool:
        return "play('x180'*amp(-1.0), 'q2.xy')" in script or 'play("x180"*amp(-1.0), "q2.xy")' in script

    def has_cancel_neg(script: str) -> bool:
        return "amp(v6)" in script and "q1.xy_cancel" in script

    # Case 1: alternate_probe=True, alternate_target=False
    prog_probe_only, _ = build_program(
        mach,
        q1,
        q2,
        mode="benchmark",
        target_gate="x180",
        probe_gate="x180",
        repetitions=[2],
        num_shots=10,
        cancel_amp=0.035,
        alternate_probe=True,
        alternate_target=False,
    )
    script_p = generate_qua_script(prog_probe_only)
    assert has_probe_neg(script_p)
    assert not has_target_neg(script_p)
    assert not has_cancel_neg(script_p)

    # Case 2: alternate_probe=False, alternate_target=True
    prog_target_only, _ = build_program(
        mach,
        q1,
        q2,
        mode="benchmark",
        target_gate="x180",
        probe_gate="x180",
        repetitions=[2],
        num_shots=10,
        cancel_amp=0.035,
        alternate_probe=False,
        alternate_target=True,
    )
    script_t = generate_qua_script(prog_target_only)
    assert not has_probe_neg(script_t)
    assert has_target_neg(script_t)
    assert has_cancel_neg(script_t)

    # Case 3: alternate_probe=True, alternate_target=True
    prog_both, _ = build_program(
        mach,
        q1,
        q2,
        mode="benchmark",
        target_gate="x180",
        probe_gate="x180",
        repetitions=[2],
        num_shots=10,
        cancel_amp=0.035,
        alternate_probe=True,
        alternate_target=True,
    )
    script_both = generate_qua_script(prog_both)
    assert has_probe_neg(script_both)
    assert has_target_neg(script_both)
    assert has_cancel_neg(script_both)



