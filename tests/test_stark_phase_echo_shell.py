"""Offline build proof for the AC-Stark phase echo QM probe.

The committed quam_state carries no ``stark`` xy op (it is an operator action via
quam_config/register_stark.py), so nothing else in the suite exercises this
builder. These tests register a ``stark`` SquarePulse IN MEMORY on a live qubit
and render the QUA script, checking that:

* the off-resonant bracket (``update_frequency``) and BOTH closing bases
  (x90 / -y90) appear;
* arm 1's idle equals the stark op's OWN length (balanced echo) and the tone is
  played at its natural length -- NO ``duration=`` override, so the arbitrary
  stark waveform is never zero-padded into free evolution;
* the op / amplitude-factor guards refuse by name before any QUA is built.

``Quam.load`` returns a shared in-memory tree, so every test sets the exact op
state it needs and the fixture always REMOVES the op it added (no leakage into
the rest of the suite).
"""

from pathlib import Path

import numpy as np
import pytest

quam_config = pytest.importorskip("quam_config")
pytest.importorskip("qm")

STATE = str(Path(__file__).resolve().parents[1] / "quam_state")
STARK_LEN_NS = 64          # a distinctive length -> arm 1 idles 64 ns / 4 = 16 cycles
STARK_LEN_CYCLES = STARK_LEN_NS // 4


def _load():
    return quam_config.Quam.load(STATE)


def _first_qubit_with_gates(machine):
    for name, q in machine.qubits.items():
        xy = getattr(q, "xy", None)
        if xy is not None and all(op in xy.operations for op in ("x90", "x180", "y90")):
            return name
    return None


def _drop_stark(xy):
    if "stark" in xy.operations:
        del xy.operations["stark"]


@pytest.fixture
def machine_with_stark():
    """A live machine with a ``stark`` SquarePulse registered in memory, removed on teardown."""
    from quam.components.pulses import SquarePulse

    machine = _load()
    target = _first_qubit_with_gates(machine)
    if target is not None:
        _drop_stark(machine.qubits[target].xy)
        machine.qubits[target].xy.operations["stark"] = SquarePulse(
            length=STARK_LEN_NS, amplitude=0.25, axis_angle=0.0)
    yield machine, target
    if target is not None:
        _drop_stark(machine.qubits[target].xy)


def test_build_program_renders_the_echo_stark_sequence(machine_with_stark):
    from qm import generate_qua_script

    from scqo_qm.experiments._lib import select_qubits
    from scqo_qm.experiments.qubit_stark_phase_echo import build_program

    machine, target = machine_with_stark
    if target is None:
        pytest.skip("no live qubit carrying x90/x180/y90")

    qubits = select_qubits(machine, [target], multiplexed=True)
    prog, axes = build_program(
        machine, qubits, stark_amps=np.linspace(-1.0, 1.0, 5),
        stark_detuning_hz=50e6, stark_operation="stark",
        num_shots=10, reset_type="thermal",
    )
    assert list(axes) == ["qubit", "stark_amp", "meas_basis"]

    text = generate_qua_script(prog, machine.generate_config())
    assert "update_frequency" in text            # the off-resonant detune/restore bracket
    assert "y90" in text                          # the -y90 closing basis

    # Arm 1 idles for the stark tone's OWN length (balanced echo).
    q = target
    assert f'wait({STARK_LEN_CYCLES}, "{q}.xy")' in text

    # The stark tone plays at its natural length -- NO duration override (which
    # would zero-pad the arbitrary waveform, leaving free evolution in arm 2).
    stark_lines = [ln for ln in text.splitlines() if 'play("stark"' in ln]
    assert stark_lines, "stark tone is not played"
    assert all("duration" not in ln for ln in stark_lines), \
        f"stark must play at its natural length, got: {stark_lines}"


def test_build_program_refuses_missing_stark_op():
    from scqo_qm.experiments._lib import select_qubits
    from scqo_qm.experiments.qubit_stark_phase_echo import build_program

    machine = _load()
    target = _first_qubit_with_gates(machine)
    if target is None:
        pytest.skip("no live qubit carrying x90/x180/y90")
    _drop_stark(machine.qubits[target].xy)  # ensure absent (shared tree may carry it)
    qubits = select_qubits(machine, [target], multiplexed=True)
    with pytest.raises(ValueError, match="stark"):
        build_program(machine, qubits, stark_amps=np.linspace(-1.0, 1.0, 5),
                      stark_detuning_hz=50e6, stark_operation="stark",
                      num_shots=10, reset_type="thermal")


def test_baked_stark_amps_reports_the_absolute_reference(machine_with_stark):
    """The swept factor multiplies the op's BAKED amplitude, and only this driver
    can read a named operation's own amplitude — so the report is what gives the run
    an absolute amplitude scale at all (scqo turns it into `digital_amp`)."""
    from scqo_qm.experiments._lib import select_qubits
    from scqo_qm.experiments.qubit_stark_phase_echo import _baked_stark_amps

    machine, target = machine_with_stark
    if target is None:
        pytest.skip("no live qubit carrying x90/x180/y90")
    qubits = select_qubits(machine, [target], multiplexed=True)
    assert _baked_stark_amps(qubits, "stark") == {target: 0.25}  # the fixture's bake


def test_baked_stark_amps_skips_a_pulse_without_a_scalar_amplitude(machine_with_stark):
    """Provenance degrades instead of raising: a waveform with no scalar amplitude
    has no absolute reference to report, and scqo just leaves the axis off."""
    from scqo_qm.experiments._lib import select_qubits
    from scqo_qm.experiments.qubit_stark_phase_echo import _baked_stark_amps

    machine, target = machine_with_stark
    if target is None:
        pytest.skip("no live qubit carrying x90/x180/y90")
    op = machine.qubits[target].xy.operations["stark"]
    op.amplitude = None
    qubits = select_qubits(machine, [target], multiplexed=True)
    assert _baked_stark_amps(qubits, "stark") == {}


def test_build_program_refuses_amplitude_factor_ge_two(machine_with_stark):
    from scqo_qm.experiments._lib import select_qubits
    from scqo_qm.experiments.qubit_stark_phase_echo import build_program

    machine, target = machine_with_stark
    if target is None:
        pytest.skip("no live qubit carrying x90/x180/y90")
    qubits = select_qubits(machine, [target], multiplexed=True)
    with pytest.raises(ValueError, match="amplitude"):
        build_program(machine, qubits, stark_amps=np.array([0.0, 2.5]),
                      stark_detuning_hz=50e6, stark_operation="stark",
                      num_shots=10, reset_type="thermal")
