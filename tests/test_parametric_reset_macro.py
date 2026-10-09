"""The parametric-reset macro: what it emits, in what order, and what it refuses.

Two halves, as in ``test_unidirectional_trotter_probe``:

* the ORDER and the REFUSALS are pinned against recording stubs - no QUAM tree, no
  QUA program. The macro only ever talks to its qubit and that qubit's z line, so a
  stub that writes down each call is the whole observable;
* the COMPILED PROGRAM is rendered from the live ``quam_state``, because "a later
  flux pulse on the same line plays as stored" is a statement about generated QUA.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from scqo_qm.components.macros.parametric_reset_macro import ParametricReset

STATE = str(Path(__file__).resolve().parents[1] / "quam_state")

DRIVE_HZ = 355_000_000
PULSE_NS = 100


# ------------------------------------------------------------------ stubs


class _Line:
    """A z line that records what is asked of it."""

    def __init__(self, calls, intermediate_frequency):
        self.name = "q9.z"
        self.intermediate_frequency = intermediate_frequency
        self.operations = {"parametric_reset": SimpleNamespace(length=PULSE_NS)}
        self._calls = calls

    def update_frequency(self, frequency):
        self._calls.append(("update_frequency", frequency))

    def reset_if_phase(self):
        self._calls.append(("reset_if_phase",))

    def play(self, operation):
        self._calls.append(("play", operation))


class _Qubit:
    name = "q9"

    def __init__(self, calls, intermediate_frequency):
        self.z = _Line(calls, intermediate_frequency)
        self._calls = calls

    def align(self):
        self._calls.append(("align",))

    def wait(self, cycles):
        self._calls.append(("wait", cycles))


@pytest.fixture
def attach(monkeypatch):
    """Build a macro whose ``qubit`` is a recording stub; returns (macro, calls)."""

    def _attach(*, intermediate_frequency=0, **fields):
        calls = []
        qubit = _Qubit(calls, intermediate_frequency)
        monkeypatch.setattr(ParametricReset, "qubit", property(lambda self: qubit))
        return ParametricReset(drive_frequency=DRIVE_HZ, **fields), calls

    return _attach


#: the pulse itself, and the oscillator handed back: every apply() emits these
PULSE = [("update_frequency", DRIVE_HZ), ("reset_if_phase",), ("play", "parametric_reset")]


def test_the_line_returns_to_its_stored_if_after_the_pulse(attach):
    """Without the last statement every later pulse on this z line in the same
    program plays modulated at the drive frequency. And with no ring-down stored,
    that is ALL the macro emits: no wait, no align."""
    macro, calls = attach()
    macro.apply()
    assert calls == PULSE + [("update_frequency", 0)]


def test_a_nonzero_stored_if_is_what_the_line_returns_to(attach):
    macro, calls = attach(intermediate_frequency=50e6)
    macro.apply()
    assert calls[-1] == ("update_frequency", 50_000_000)


def test_the_ringdown_waits_on_the_whole_qubit_after_aligning_it(attach):
    """The pulse moved the z line's timeline only. A wait counts from each
    channel's own, so the qubit is aligned first: otherwise a gate on its xy line
    could start before the pulse has even ended."""
    macro, calls = attach(ringdown_ns=32)
    macro.apply()
    assert calls == PULSE + [("update_frequency", 0), ("align",), ("wait", 8)]


def test_a_call_can_replace_the_stored_ringdown(attach):
    """``qubit_ramsey_reset_ringdown`` plays the reset with 0 to measure what the
    wait has to be; the stored value is not touched."""
    macro, calls = attach(ringdown_ns=32)
    macro.apply(ringdown_ns=0)
    assert calls == PULSE + [("update_frequency", 0)]
    assert macro.ringdown_ns == 32

    macro, calls = attach()
    macro.apply(ringdown_ns=48)
    assert calls[-2:] == [("align",), ("wait", 12)]

    macro, calls = attach()
    with pytest.raises(ValueError, match="ringdown_ns must be 0 or a multiple of 4 ns"):
        macro.apply(ringdown_ns=10)
    assert calls == []


@pytest.mark.parametrize("ringdown_ns", [4, 12, 18, 30, -16, 16.5])
def test_a_ringdown_qua_cannot_wait_is_refused_by_name(attach, ringdown_ns):
    """Below 16 ns QUA's wait() is a compile error naming an internal variable,
    and off the 4 ns grid it would be rounded. Nothing is emitted first."""
    macro, calls = attach(ringdown_ns=ringdown_ns)
    with pytest.raises(ValueError, match="ringdown_ns must be 0 or a multiple of 4 ns"):
        macro.apply()
    assert calls == []


def test_a_z_line_without_an_oscillator_is_refused_by_name(attach):
    """intermediate_frequency None means the generated element has no oscillator:
    update_frequency would fail at compile, naming nothing an operator can fix."""
    macro, calls = attach(intermediate_frequency=None)
    with pytest.raises(ValueError, match="q9.z has no intermediate_frequency"):
        macro.apply()
    assert calls == []


@pytest.mark.parametrize("ringdown_ns", [0, 32])
def test_the_duration_is_the_pulse_plus_the_ringdown(attach, ringdown_ns):
    macro, _calls = attach(ringdown_ns=ringdown_ns)
    assert macro.inferred_duration == pytest.approx((PULSE_NS + ringdown_ns) * 1e-9)


def test_a_reset_without_a_ringdown_saves_as_it_did_before_the_field():
    """A state.json written before ``ringdown_ns`` existed must stay what a save
    writes, or every staged save on that tree (register-partial-swap,
    adopt-channel) finds a key it did not ask for and refuses; the setup snapshot
    of an untouched tree would stop matching its files the same way."""
    bare = ParametricReset(drive_frequency=DRIVE_HZ).to_dict(include_defaults=True)
    assert "ringdown_ns" not in bare and bare["drive_frequency"] == DRIVE_HZ
    stored = ParametricReset(drive_frequency=DRIVE_HZ, ringdown_ns=32)
    assert stored.to_dict(include_defaults=True)["ringdown_ns"] == 32
    assert stored.to_dict()["ringdown_ns"] == 32


# ------------------------------------------------------------ requires QUAM

quam_config = pytest.importorskip("quam_config")
pytest.importorskip("qm")

RESET_QUBIT = "q2"
RESET_MACRO = "reset"


@pytest.fixture(scope="module")
def live_reset():
    """The live qubit carrying a parametric reset, a second operation on its z
    line, and the machine - or a skip naming what is absent."""
    machine = quam_config.Quam.load(STATE)
    qubit = machine.qubits.get(RESET_QUBIT)
    if qubit is None or RESET_MACRO not in qubit.macros:
        pytest.skip(f"live {RESET_QUBIT} has no {RESET_MACRO!r} macro (register_reset_macro.py)")
    macro = qubit.macros[RESET_MACRO]
    others = [op for op in qubit.z.operations if op != macro.flux_pulse]
    if not others:
        pytest.skip(f"live {RESET_QUBIT}.z carries no operation besides {macro.flux_pulse!r}")
    return machine, qubit, macro, others[0]


def _program_lines(machine, qubit, macro, later):
    """Generated QUA for: reset, then another stored pulse on the same z line."""
    from qm import generate_qua_script
    from qm.qua import program

    with program() as prog:
        macro.apply()
        qubit.z.play(later)
    text = generate_qua_script(prog, machine.generate_config())
    body = text[:text.index("config = ")]
    return [ln.strip() for ln in body.splitlines() if "q2." in ln]


def test_a_later_flux_pulse_plays_at_the_stored_if(live_reset):
    """BACKLOG I33, on the compiled program: between the reset's pulse and the
    next pulse on the same z line the oscillator goes back to the stored IF."""
    machine, qubit, macro, later = live_reset
    lines = _program_lines(machine, qubit, macro, later)
    assert lines == [
        f'update_frequency("q2.z", {macro.drive_frequency}, "Hz", False)',
        'reset_if_phase("q2.z")',
        f'play("{macro.flux_pulse}", "q2.z")',
        f'update_frequency("q2.z", {int(qubit.z.intermediate_frequency)}, "Hz", False)',
        f'play("{later}", "q2.z")',
    ]


def test_the_stored_ringdown_reaches_the_compiled_program(live_reset):
    """The wait names every channel of the qubit, after an align of the same
    channels, and sits between the reset and whatever plays next."""
    machine, qubit, macro, later = live_reset
    saved = macro.ringdown_ns
    try:
        macro.ringdown_ns = 32
        lines = _program_lines(machine, qubit, macro, later)
        assert macro.duration == pytest.approx(
            (qubit.z.operations[macro.flux_pulse].length + 32) * 1e-9)
    finally:
        macro.ringdown_ns = saved
    channels = {channel.name for channel in qubit.channels.values()}
    align, wait = lines[-3], lines[-2]
    assert lines[-4].startswith('update_frequency("q2.z"')
    assert align.startswith("align(") and all(f'"{name}"' in align for name in channels)
    assert wait.startswith("wait(8, ") and all(f'"{name}"' in wait for name in channels)
    assert lines[-1] == f'play("{later}", "q2.z")'
