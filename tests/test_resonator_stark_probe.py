"""``qubit_resonator_stark``: the Stark tone, the drive and the readout in the emitted QUA.

The QM half of the backend-parity rule for this experiment (SCQO CLAUDE.md,
*Backend parity*; the sequence itself lives in scqo ``experiments/_stark_tone.py``):
a tone on the resonator runs one depletion wait BEFORE the saturation drive and
ends WITH it, and the standard readout follows one more depletion wait later.

Every one of those claims is invisible anywhere but the generated QUA:

* CONCURRENT is the ABSENCE of an ``align()`` between the Stark play and the drive
  play, plus the drive's ring-up ``wait`` — the two then END together because the
  tone is exactly ring-up + drive long.
* THE READOUT IS UNTOUCHED is a SECOND ``align()`` after both, a depletion ``wait``
  on the resonator, and a ``measure`` with no ``amp()`` — the same readout in every
  row, which is the whole point of a separate Stark tone.

Live-QUAM: the Stark tone plays the REAL ``readout`` operation through
``Channel.play`` with a dynamic ``amp()``, and whether that stretches and scales is a
property of the actual tree, not of a stub.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

quam_config = pytest.importorskip("quam_config")
pytest.importorskip("qm")

from conftest import recording_device  # noqa: E402
from test_qm_backend import roster_toml_for  # noqa: E402

from scqo_qm.backend.qm_backend import QMBackend  # noqa: E402
from scqo.roster import parse_components  # noqa: E402

TARGET = "q4"


@pytest.fixture(scope="module")
def machine():
    return quam_config.Quam.load(str(Path(__file__).resolve().parents[1] / "quam_state"))


@pytest.fixture(scope="module")
def live_roster(machine):
    return parse_components(roster_toml_for(machine))


@pytest.fixture(scope="module")
def config(machine):
    return machine.generate_config()


def _script(prog, config) -> list[str]:
    from qm import generate_qua_script

    return [ln.strip().replace(chr(34), chr(39))
            for ln in generate_qua_script(prog, config).splitlines()]


def _body(lines) -> list[str]:
    """One sweep point, from the detuning update to the first save."""
    start = next(i for i, ln in enumerate(lines) if ln.startswith("update_frequency"))
    tail = lines[start:]
    return tail[:next(i for i, ln in enumerate(tail) if ln.startswith("save("))]


def _index(body, prefix) -> int:
    return next(i for i, ln in enumerate(body) if ln.startswith(prefix))


def _duration(line: str) -> int:
    return int(re.search(r"duration=(\d+)", line).group(1))


def _wait_cycles(line: str) -> int:
    return int(re.search(r"wait\((\d+)", line).group(1))


def _build(machine, live_roster, targets=(TARGET,), **params):
    from scqo.experiments import get

    import scqo_qm.experiments  # noqa: F401  (registers the QM probes)

    backend = QMBackend(machine, roster=live_roster)
    cls = get("qubit_resonator_stark")
    kwargs = dict(num_drive_freq_points=5, num_amp_points=3, num_averages=10, **params)
    exp = cls(backend, cls.Parameters(targets=list(targets), **kwargs))
    exp.device = recording_device(backend, live_roster)
    exp.sweep_axes = exp.define_sweep()
    prog, _axes = exp.probe()
    return exp, prog


def test_the_stark_tone_is_the_readout_op_scaled_and_stretched(machine, live_roster, config):
    """One play of the readout operation on the resonator with a DYNAMIC amp()
    (the swept prefactor) and a duration of ring-up + drive — before the measure."""
    exp, prog = _build(machine, live_roster, drive_len_ns=2000.0,
                       readout_depletion_ns=400.0)
    body = _body(_script(prog, config))
    tone = [ln for ln in body if ln.startswith("play('readout'*amp(")]
    assert len(tone) == 1, f"expected one Stark tone, got {tone}"
    assert f"'{TARGET}.resonator'" in tone[0]
    assert _duration(tone[0]) == (400 + 2000) // 4
    assert body.index(tone[0]) < _index(body, "measure('readout'")
    assert exp.resolved_windows().tone_len_ns == 2400.0


def test_the_drive_rings_up_and_ends_with_the_tone(machine, live_roster, config):
    """THE concurrency claim: no align() between the tone and the drive, the drive
    waits the ring-up on its OWN element, and ring-up + drive = the tone."""
    _e, prog = _build(machine, live_roster, drive_len_ns=2000.0,
                      readout_depletion_ns=400.0)
    body = _body(_script(prog, config))
    tone = _index(body, "play('readout'*amp(")
    drive = _index(body, "play('saturation'")
    assert tone < drive
    between = body[tone + 1:drive]
    assert not [ln for ln in between if ln.startswith("align(")], between
    ring = [ln for ln in between if ln.startswith("wait(") and f"'{TARGET}.xy'" in ln]
    assert len(ring) == 1 and _wait_cycles(ring[0]) == 400 // 4
    assert _duration(body[drive]) == 2000 // 4


def test_the_readout_waits_out_the_photons_and_is_unscaled(machine, live_roster, config):
    """After the tone: a barrier, the depletion wait on the resonator, then the
    STANDARD measure — no amp() on it, so every row reads out the same way."""
    _e, prog = _build(machine, live_roster, drive_len_ns=2000.0,
                      readout_depletion_ns=400.0)
    body = _body(_script(prog, config))
    drive = _index(body, "play('saturation'")
    measure = _index(body, "measure('readout'")
    tail = body[drive + 1:measure]
    assert [ln for ln in tail if ln.startswith("align(")], tail
    settle = [ln for ln in tail if ln.startswith("wait(") and f"'{TARGET}.resonator'" in ln]
    assert len(settle) == 1 and _wait_cycles(settle[0]) == 400 // 4
    assert "amp(" not in body[measure]


def test_the_standing_knob_times_the_run_without_an_override(machine, live_roster, config):
    """No readout_depletion_ns: the tree's governed depletion_time is the wait."""
    exp, prog = _build(machine, live_roster, drive_len_ns=2000.0)
    knob_ns = exp.device.channel(TARGET, "readout").readout_depletion_s * 1e9
    assert exp.resolved_windows().depletion_ns == pytest.approx(knob_ns)
    body = _body(_script(prog, config))
    assert _duration(body[_index(body, "play('readout'*amp(")]) == int(knob_ns + 2000) // 4


def test_the_quam_factory_depletion_is_refused_by_name(machine, live_roster):
    """QUAM's depletion_time is never None; an uncalibrated one sits at 16 ns. The
    neutral helper cannot tell that from a measurement, so this shell refuses it —
    unless the run states its own wait."""
    from scqo.experiments import get

    import scqo_qm.experiments  # noqa: F401

    resonator = machine.qubits[TARGET].resonator
    standing = resonator.depletion_time
    resonator.depletion_time = 16
    try:
        backend = QMBackend(machine, roster=live_roster)
        cls = get("qubit_resonator_stark")
        exp = cls(backend, cls.Parameters(targets=[TARGET], num_drive_freq_points=5,
                                          num_amp_points=3, num_averages=10))
        exp.device = recording_device(backend, live_roster)
        exp.sweep_axes = exp.define_sweep()
        with pytest.raises(ValueError, match=r"16 ns factory default"):
            exp.probe()
        exp = cls(backend, cls.Parameters(targets=[TARGET], num_drive_freq_points=5,
                                          num_amp_points=3, num_averages=10,
                                          readout_depletion_ns=400.0))
        exp.device = recording_device(backend, live_roster)
        exp.sweep_axes = exp.define_sweep()
        exp.probe()  # the stated wait is honoured
    finally:
        resonator.depletion_time = standing


def test_targets_run_one_at_a_time(machine, live_roster, config):
    """Two targets are two sequential blocks (the Qblox probe's per-target
    sub-schedules), never N concurrent Stark tones on one feedline."""
    _one, single = _build(machine, live_roster, readout_depletion_ns=400.0,
                          drive_len_ns=400.0)
    _two, double = _build(machine, live_roster, targets=(TARGET, "q5"),
                          readout_depletion_ns=400.0, drive_len_ns=400.0)
    loops = [sum(ln.startswith("with for_(") for ln in _script(prog, config))
             for prog in (single, double)]
    assert loops[1] == 2 * loops[0]
