"""``qubit_spectroscopy``: where the saturation drive sits against the readout.

This is the QM half of the backend-parity rule (SCQO CLAUDE.md, *Backend parity*).
The drive ENDS at an anchor and starts ``drive_len_ns`` earlier; ``readout_overlap``
picks the anchor — the readout tone's START (the default) or its END.

Both claims are invisible everywhere except in the emitted QUA:

* SEQUENTIAL is the PRESENCE of an ``align()`` between the drive and the
  measurement, plus a FINITE ``duration=`` on the saturation play. Drop either
  and the drive is still live while the ADC integrates, which is the exact
  divergence this experiment used to have against the Qblox backend: a
  latched continuous tone there, a finite pulse here, one ``drive_freq_hz``
  written back from two different physical measurements.
* OVERLAP is the ABSENCE of that barrier, plus a ``wait`` on whichever element
  starts second so the two END together.

So this reads the generated QUA and asserts the shape directly, one mode as the
other's contrast.

Live-QUAM: the pre-tone plays a REAL ``readout`` operation through
``Channel.play``, and whether a ``BaseReadoutPulse`` can be played without being
measured is a property of the actual tree, not of a stub.
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


def _body(prog, config) -> list[str]:
    """The stripped lines of one sweep point's QUA, from the detuning update on."""
    from qm import generate_qua_script

    lines = [ln.strip() for ln in generate_qua_script(prog, config).splitlines()]
    starts = [i for i, ln in enumerate(lines) if ln.startswith("update_frequency")]
    assert starts, "expected the detuning update that opens each sweep point"
    tail = lines[starts[0]:]
    stop = next(i for i, ln in enumerate(tail) if ln.startswith("save("))
    return tail[:stop]


def _q(line: str) -> str:
    """Double quotes normalised to single, so a renderer change cannot break a match."""
    return line.replace(chr(34), chr(39))


def _index(body, prefix) -> int:
    return next(i for i, ln in enumerate(body) if _q(ln).startswith(prefix))


def _after_the_shared_align(body) -> list[str]:
    """Only the timed block: the reset's own wait sits before the shared align in
    the overlap sequence, and counting it as a lead would pass by accident."""
    drive = _index(body, "play('saturation'")
    last_align = max(i for i in range(drive) if body[i].startswith("align("))
    return body[last_align + 1:]


def _duration(line: str) -> int:
    return int(re.search(r"duration=(\d+)", line).group(1))


def _build(machine, live_roster, **params):
    from scqo.experiments import get

    import scqo_qm.experiments  # noqa: F401  (registers the QM probes)

    backend = QMBackend(machine, roster=live_roster)
    cls = get("qubit_spectroscopy")
    kwargs = dict(num_drive_freq_points=5, num_averages=10, **params)
    exp = cls(backend, cls.Parameters(targets=[TARGET], **kwargs))
    exp.device = recording_device(backend, live_roster)
    exp.sweep_axes = exp.define_sweep()
    prog, _axes = exp.probe()
    return exp, prog


def _readout_ns(exp) -> float:
    return exp.device.channel(TARGET, "readout").readout_duration_s * 1e9


# ------------------------------------------------------------------ sequential

def test_the_drive_is_a_finite_pulse_that_ends_before_the_readout(
        machine, live_roster, config):
    """THE parity claim. A finite ``duration=`` plus the ``align()`` barrier is
    what makes the readout happen with the drive already off — the assumption
    the experiment rests on (T1 outlasts the readout), and the thing the Qblox
    backend used to break by latching a continuous tone instead."""
    exp, prog = _build(machine, live_roster, drive_len_ns=8000.0)
    body = _body(prog, config)

    drive = _index(body, "play('saturation'")
    measure = _index(body, "measure('readout'")
    assert drive < measure
    assert _duration(body[drive]) == 2000  # 8000 ns / 4
    assert [ln for ln in body[drive + 1:measure] if ln.startswith("align(")], \
        f"no align() between the drive and the measurement: {body[drive:measure + 1]}"


def test_the_sequential_mode_plays_no_pre_tone(machine, live_roster, config):
    """``acq_start_ns`` is refused outright in this mode (there is no steady
    state to wait for), so the resonator's timeline is the readout op alone."""
    _e, prog = _build(machine, live_roster, drive_len_ns=8000.0)
    body = _body(prog, config)
    assert not [ln for ln in body if _q(ln).startswith("play('readout', '")]


# --------------------------------------------------------------------- overlap

def test_no_align_between_the_drive_and_the_measurement(machine, live_roster, config):
    """The contrast: with the overlap asked for, the barrier is gone and both
    elements run from the one shared align above them."""
    _o, overlap = _build(machine, live_roster, readout_overlap=True, drive_len_ns=8000.0)
    _s, sequential = _build(machine, live_roster, drive_len_ns=8000.0)

    for prog, expect_barrier in ((sequential, True), (overlap, False)):
        body = _body(prog, config)
        drive = _index(body, "play('saturation'")
        measure = _index(body, "measure('readout'")
        assert drive < measure
        between = [ln for ln in body[drive + 1:measure] if ln.startswith("align(")]
        assert bool(between) is expect_barrier, body[drive:measure + 1]


def test_the_adc_lead_is_a_pre_tone_on_the_resonator(machine, live_roster, config):
    """QUAM's ``measure()`` has no acquisition-delay argument, so the lead is the
    same readout operation played back-to-back into it — one seamless tone whose
    tail is what gets integrated. Durations are QUA clock cycles."""
    _e, prog = _build(machine, live_roster, readout_overlap=True,
                      acq_start_ns=400.0, drive_len_ns=600.0)
    body = _body(prog, config)

    pre = [ln for ln in body if _q(ln).startswith("play('readout', '")]
    assert len(pre) == 1, f"expected one readout pre-tone, got {pre}"
    assert _duration(pre[0]) == 100  # 400 ns / 4
    assert body.index(pre[0]) < _index(body, "measure('readout'")
    assert _duration(body[_index(body, "play('saturation'")]) == 150  # 600 ns / 4


def test_a_short_drive_makes_the_drive_element_wait(machine, live_roster, config):
    """Tone longer than the drive: they END together, so the drive starts late
    and it is the XY element that sits out the difference."""
    exp, prog = _build(machine, live_roster, readout_overlap=True, drive_len_ns=400.0)
    tone_ns = _readout_ns(exp)  # acq_start_ns = 0, so the tone is the knob
    assert tone_ns > 400.0, "fixture too short to exercise this branch"
    block = _after_the_shared_align(_body(prog, config))

    waits = [ln for ln in block if ln.startswith("wait(")]
    assert len(waits) == 1, f"expected exactly one lead wait, got {waits}"
    assert ".xy" in waits[0], f"the lead belongs on the drive element: {waits[0]}"
    assert int(re.search(r"wait\((\d+)", waits[0]).group(1)) == round((tone_ns - 400.0) / 4)


def test_a_long_drive_makes_the_resonator_wait(machine, live_roster, config):
    """The normal case, and the one the old co-start definition could not express:
    a 20 us saturation against a ~us tone simply starts first and runs through
    it, so the RESONATOR is the element that waits."""
    exp, prog = _build(machine, live_roster, readout_overlap=True, drive_len_ns=20000.0)
    tone_ns = _readout_ns(exp)
    assert tone_ns < 20000.0, "fixture too long to exercise this branch"
    block = _after_the_shared_align(_body(prog, config))

    waits = [ln for ln in block if ln.startswith("wait(")]
    assert len(waits) == 1, f"expected exactly one lead wait, got {waits}"
    assert ".resonator" in waits[0], f"the lead belongs on the readout element: {waits[0]}"
    assert int(re.search(r"wait\((\d+)", waits[0]).group(1)) == round((20000.0 - tone_ns) / 4)


def test_equal_lengths_need_no_wait_at_all(machine, live_roster, config):
    """The boundary between the two branches — both tones start and end together,
    so neither element is given a lead."""
    backend_exp, _p = _build(machine, live_roster, readout_overlap=True, drive_len_ns=400.0)
    tone_ns = _readout_ns(backend_exp)
    _e, prog = _build(machine, live_roster, readout_overlap=True, drive_len_ns=tone_ns)
    block = _after_the_shared_align(_body(prog, config))
    assert not [ln for ln in block if ln.startswith("wait(")]


def test_targets_with_different_readout_windows_are_refused(machine, live_roster):
    """One multiplexed program plays ONE set of times. Taking the first target's
    silently would put every other target's ADC in the wrong place — a weaker,
    shifted peak on exactly the qubits nobody was watching."""
    from scqo.experiments import get

    import scqo_qm.experiments  # noqa: F401

    backend = QMBackend(machine, roster=live_roster)
    cls = get("qubit_spectroscopy")
    others = [q for q in ("q4", "q5") if q in live_roster.entities]
    if len(others) < 2:
        pytest.skip("needs two qubits on the live tree")

    exp = cls(backend, cls.Parameters(targets=others, readout_overlap=True,
                                      num_drive_freq_points=5, num_averages=10))
    exp.device = recording_device(backend, live_roster)
    exp.sweep_axes = exp.define_sweep()
    view = exp.device.channel(others[1], "readout")
    before = view.readout_duration_s
    view.readout_duration_s = before + 400e-9
    try:
        with pytest.raises(ValueError, match="different concurrent-tone windows"):
            exp.probe()
    finally:
        view.readout_duration_s = before
