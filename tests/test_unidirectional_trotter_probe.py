"""Offline build proof + guard census for the unidirectional-Trotter QM probe.

Two halves, deliberately:

* the GUARDS run before a single QUA statement is emitted, so they are pinned
  against plain stubs — no QUAM, no config, no QOP. Every one of them is a
  refusal an operator can act on (register this macro, lower that factor);
* the BUILD is rendered from the live ``quam_state``, because a QUA program is
  made out of the vendor's own macros and there is no honest stand-in for
  ``pair.macros["iswap"].apply()``. Each build test skips by name when the
  committed state does not carry what the chain needs, exactly as
  ``test_stark_phase_echo_shell`` does.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from scqo_qm.experiments.qc_unidirectional_trotter import build_program

STATE = str(Path(__file__).resolve().parents[1] / "quam_state")

#: the chain the live 5Q4C state can express: q1 -> q2 -> q3, resetting q2.
CHAIN = ("q1", "q2", "q3")
PAIRS = ("q1_q2", "q2_q3")
SWAP = "iswap"          # the one swap macro BOTH live pairs carry
RESET_MACRO = "reset"   # ParametricReset, registered on q2


# --------------------------------------------------------------- guard stubs


def _qubit(name, ops=("x180", "stark"), macros=(), if_hz=100_000_000):
    return SimpleNamespace(
        name=name,
        macros={m: SimpleNamespace() for m in macros},
        xy=SimpleNamespace(operations={op: SimpleNamespace() for op in ops},
                           intermediate_frequency=if_hz),
    )


#: the flux pulse a stub pair's swap macro plays, and its length. An idle step
#: copies that length, so the stubs have to carry one for the idle guards to
#: reach anything but "there is no duration here".
FLUX_PULSE = "flattop_cosine"
PULSE_NS = 64


def _pair(name, macros=(SWAP,), pulse_ns=PULSE_NS):
    """A stub pair. ``pulse_ns=None`` registers the macro with NO flux pulse,
    which is the shape an idle step cannot take its duration from."""
    ops = {} if pulse_ns is None else {FLUX_PULSE: SimpleNamespace(length=pulse_ns)}
    return SimpleNamespace(
        name=name,
        macros={m: SimpleNamespace(flux_pulse=FLUX_PULSE) for m in macros},
        qubit_control=SimpleNamespace(z=SimpleNamespace(operations=ops)),
        coupler=SimpleNamespace(operations=ops),
    )


def _reset_ready(name):
    """A stub qubit carrying a reset macro whose ring-down a call can set - the
    shape a reset at the start of a shot needs."""
    qubit = _qubit(name)
    qubit.macros[RESET_MACRO] = SimpleNamespace(ringdown_ns=0)
    return qubit


def _kwargs(**overrides):
    """The smallest legal call; overrides replace one piece at a time."""
    q1, q2, q3 = (_qubit(n) for n in CHAIN)
    q2.macros[RESET_MACRO] = SimpleNamespace()
    base = dict(
        machine=None,
        measure_qubits=[q1, q2, q3],
        first_pair=_pair("q1_q2"),
        second_pair=_pair("q2_q3"),
        reset_qubit=q2,
        prep=[(q1, "x180")],
        first_operation=SWAP,
        second_operation=SWAP,
        idle_reference_operation=SWAP,
        reset_operation=RESET_MACRO,
        stark_operation="stark",
        stark_detuning_hz=50e6,
        compensation=[(q1, 0.3), (q2, 0.2), (q3, 0.25)],
        rounds_array=np.arange(0, 5),
        num_shots=10,
        reset_type="thermal",
        keep_shots=False,
    )
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    "override, message",
    [
        ({"operation_gap_ns": 6}, "multiple of 4"),
        ({"operation_gap_ns": -4}, "multiple of 4"),
        ({"first_pair": _pair("q1_q2", macros=())}, "no macro 'iswap'"),
        ({"second_pair": _pair("q2_q3", macros=("cz",))}, "no macro 'iswap'"),
        ({"reset_operation": "paramreset"}, "no macro 'paramreset'"),
        ({"prep": [(_qubit("q1"), "x270")]}, "no xy operation 'x270'"),
        # every prep entry is checked, not just the first
        ({"prep": [(_qubit("q1"), "x180"), (_qubit("q3", ops=("stark",)), "x180")]},
         "prep qubit q3 has no xy operation 'x180'"),
        ({"stark_operation": "tone"}, "no xy operation 'tone'"),
        # an idle step still needs a DURATION, and it comes from a real macro:
        # a pair that does not carry the reference cannot idle for the right
        # length, and idling for the wrong one is the silent failure.
        ({"first_operation": None, "first_pair": _pair("q1_q2", macros=())},
         "has no macro 'iswap', so an 'idle' step"),
        ({"first_operation": None, "first_pair": _pair("q1_q2", pulse_ns=None)},
         "no flux pulse with a length"),
        ({"first_operation": None, "first_pair": _pair("q1_q2", pulse_ns=6)},
         "not a multiple of the 4 ns clock"),
        ({"first_operation": None, "first_pair": _pair("q1_q2", pulse_ns=8)},
         "below QUA's 16 ns minimum wait"),
        # the angle knob has nothing to turn on a step that plays no pulse
        ({"first_operation": None, "first_coupler_amp": 0.04},
         "is an idle step but was given a coupler amplitude"),
        # with the tones DURING the reset, one on the reset qubit (q2 is in the
        # default compensation) would detune the qubit the reset is acting on
        ({"stark_with_reset": True},
         "q2 is the reset qubit and also plays a stark tone"),
        # the reset at the start of a shot needs the stored macro on EVERY qubit
        # it names: skipping one would start the chain from another state, and
        # nothing in the data would say which
        ({"init_resets": [_qubit("q1")]},
         "init_reset_qubits names q1, which has no macro 'reset'"),
        # ...and a macro whose ring-down the call can set
        ({"init_resets": [_qubit("q3", macros=(RESET_MACRO,))]},
         "is a SimpleNamespace, not a parametric reset"),
        # the ring-down is a QUA wait: on the 4 ns grid, 16 ns at least, or none
        ({"init_resets": [_reset_ready("q3")], "init_reset_ringdown_ns": 6},
         "init_reset_ringdown_ns must be 0 or a multiple of 4"),
        ({"init_resets": [_reset_ready("q3")], "init_reset_ringdown_ns": 8},
         "init_reset_ringdown_ns must be 0 or a multiple of 4"),
    ],
)
def test_guards_refuse_by_name_before_any_qua(override, message):
    """Every refusal names the thing to fix, and fires with no machine at all —
    proof it happens before the builder touches the vendor tree."""
    with pytest.raises(ValueError, match=message):
        build_program(**_kwargs(**override))


def test_an_unrepresentable_compensation_factor_is_refused():
    """QUA's amplitude_scale spans (-2, 2); the refusal must name the neutral
    knob, not an internal QUA variable (scqo_qm/experiments/_amp_limits.py)."""
    q1, q2, q3 = (_qubit(n) for n in CHAIN)
    q2.macros[RESET_MACRO] = SimpleNamespace()
    with pytest.raises(ValueError, match="compensation_amps"):
        build_program(**_kwargs(compensation=[(q1, 0.3), (q2, 2.5)],
                                measure_qubits=[q1, q2, q3], reset_qubit=q2,
                                prep=[(q1, "x180")]))


def test_init_resets_start_together_and_close_with_one_align(monkeypatch):
    """Each named qubit plays its own macro with the run's ring-down, nothing
    orders them against each other, and ONE global align follows - so the prep
    waits for the longest of them. With no qubit named nothing is emitted, not
    even that align: the default shot stays the program it was."""
    from scqo_qm.experiments import _chain_round

    calls = []

    def reset_qubit(name):
        def apply(*, ringdown_ns):
            calls.append((name, ringdown_ns))
        return SimpleNamespace(
            name=name, macros={RESET_MACRO: SimpleNamespace(ringdown_ns=0, apply=apply)})

    monkeypatch.setattr(_chain_round, "align", lambda: calls.append("align"))
    _chain_round.play_init_resets(
        [reset_qubit("q2"), reset_qubit("q3")], RESET_MACRO, 400)
    assert calls == [("q2", 400), ("q3", 400), "align"]

    calls.clear()
    _chain_round.play_init_resets([], RESET_MACRO, 400)
    assert calls == []


def test_reduce_raw_names_the_readout_form():
    """`state` means per-shot integer LEVELS, so only the FPGA-averaged stream
    is renamed to `population` — the contract accepts both, which is exactly why
    the backend's own rename cannot do it."""
    from scqo_qm.experiments.qc_unidirectional_trotter import (
        QMQcUnidirectionalTrotter,
    )

    raw = xr.Dataset(
        {"state": (("qubit", "round_count"), np.zeros((2, 3)))},
        coords={"qubit": ["q1", "q2"], "round_count": np.arange(3)},
    )
    shot = SimpleNamespace(params=SimpleNamespace(readout_mode="shot"))
    average = SimpleNamespace(params=SimpleNamespace(readout_mode="average"))
    assert "state" in QMQcUnidirectionalTrotter.reduce_raw(shot, raw).data_vars
    reduced = QMQcUnidirectionalTrotter.reduce_raw(average, raw)
    assert "population" in reduced.data_vars and "state" not in reduced.data_vars


# ------------------------------------------------------------ requires QUAM

quam_config = pytest.importorskip("quam_config")
pytest.importorskip("qm")


@pytest.fixture(scope="module")
def live_chain():
    """The live machine plus the chain objects, or a skip naming what is absent.

    The chain needs a lot of the committed state at once (two pair macros, a
    parametric-reset macro, a stark tone on every member), so the skip says
    which piece is missing rather than failing as though the builder were
    wrong."""
    machine = quam_config.Quam.load(STATE)
    missing = [name for name in CHAIN if name not in machine.qubits]
    missing += [name for name in PAIRS if name not in machine.qubit_pairs]
    if missing:
        pytest.skip(f"live state has no {missing}")
    for name in PAIRS:
        if SWAP not in machine.qubit_pairs[name].macros:
            pytest.skip(f"live pair {name} has no {SWAP!r} macro")
    if RESET_MACRO not in machine.qubits["q2"].macros:
        pytest.skip(f"live q2 has no {RESET_MACRO!r} macro (register_reset_macro.py)")
    for name in CHAIN:
        if "stark" not in machine.qubits[name].xy.operations:
            pytest.skip(f"live {name} has no 'stark' xy op (register_stark.py)")
    return machine


def _live_kwargs(machine, **overrides):
    qubits = [machine.qubits[name] for name in CHAIN]
    base = dict(
        machine=machine,
        measure_qubits=qubits,
        first_pair=machine.qubit_pairs[PAIRS[0]],
        second_pair=machine.qubit_pairs[PAIRS[1]],
        reset_qubit=machine.qubits["q2"],
        prep=[(machine.qubits["q1"], "x180")],
        first_operation=SWAP,
        second_operation=SWAP,
        idle_reference_operation=SWAP,
        reset_operation=RESET_MACRO,
        stark_operation="stark",
        stark_detuning_hz=50e6,
        compensation=[(q, amp) for q, amp in zip(qubits, (0.3, 0.2, 0.25))],
        rounds_array=np.arange(0, 6),
        num_shots=10,
        reset_type="thermal",
        keep_shots=True,
    )
    base.update(overrides)
    return base


def test_build_program_renders_the_trotter_round(live_chain):
    from qm import generate_qua_script

    machine = live_chain
    prog, axes = build_program(**_live_kwargs(machine))
    assert list(axes) == ["qubit", "shot_idx", "round_count"]
    assert list(axes["qubit"].values) == list(CHAIN)
    assert list(axes["round_count"].values) == list(range(6))

    text = generate_qua_script(prog, machine.generate_config())
    # ONE prep, on the chain source only — the excitation is watched, not pumped
    assert text.count('play("x180"') == 1
    # the mid-circuit parametric reset rides the RELAY's z line
    assert 'play("parametric_reset", "q2.z")' in text
    # a stark tone per compensated qubit, each at its own amplitude factor
    for name, amp in zip(CHAIN, (0.3, 0.2, 0.25)):
        assert f'play("stark"*amp({amp}), "{name}.xy")' in text
    # the off-resonant bracket: each compensated qubit is detuned once and
    # restored once, and the detune is exactly stark_detuning_hz
    for name in CHAIN:
        shifts = [float(ln.split(",")[1]) for ln in text.splitlines()
                  if f'update_frequency("{name}.xy"' in ln]
        assert len(shifts) == 2, f"{name}: {shifts}"
        assert shifts[0] - shifts[1] == pytest.approx(50e6)


def test_active_reset_initializes_every_involved_qubit(live_chain):
    """reset_type='active' EXECUTES QUAM's reset_qubit_active for each involved
    qubit, so the build proves the kwarg threads through (max_attempts) and the
    whole program serialises. Thresholds are written in memory and restored;
    whether the live state is calibrated is check_reset_method's question."""
    from qm import generate_qua_script

    machine = live_chain
    pulses = [machine.qubits[name].resonator.operations["readout"] for name in CHAIN]
    saved = [(p.threshold, p.rus_exit_threshold) for p in pulses]
    try:
        for pulse in pulses:
            pulse.threshold, pulse.rus_exit_threshold = -1.0e-4, -2.0e-4
        thermal, _ = build_program(**_live_kwargs(machine))
        active, _ = build_program(**_live_kwargs(
            machine, reset_type="active", reset_max_attempts=2))
        config = machine.generate_config()
        text_thermal = generate_qua_script(thermal, config)
        text_active = generate_qua_script(active, config)
    finally:
        for pulse, (threshold, rus) in zip(pulses, saved):
            pulse.threshold, pulse.rus_exit_threshold = threshold, rus
    # thermal measures only in the final readout; active adds QUAM's reset
    # measurements (how many per qubit is QUAM's business) in front of the
    # prep, on every involved qubit
    assert text_active.count("measure(") > text_thermal.count("measure(")
    assert text_thermal[:text_thermal.index('play("x180"')].count("measure(") == 0
    before_prep = text_active[:text_active.index('play("x180"')]
    for name in CHAIN:
        assert f'"{name}.resonator"' in before_prep


def test_the_round_is_swap_swap_reset_stark_in_that_order(live_chain):
    """The relay is dumped AFTER both swaps: reset it earlier and the second
    swap would carry nothing, which is the whole mechanism. The compensation
    closes the round, so its phase correction covers the swaps it follows."""
    from qm import generate_qua_script

    machine = live_chain
    prog, _axes = build_program(**_live_kwargs(machine))
    text = generate_qua_script(prog, machine.generate_config())
    body = text[text.index('play("x180"'):text.index("stream_processing")]
    marks = [body.index(needle) for needle in (
        f'"{PAIRS[0]}")',                    # first swap rides the q1_q2 element
        f'"{PAIRS[1]}")',                    # then the q2_q3 one
        'play("parametric_reset", "q2.z")',  # then the relay is dumped
        'play("stark"',                      # then the compensation tones
    )]
    assert marks == sorted(marks), body


def _ends(machine):
    """A tone on each chain END and none on the relay: the compensation a
    ``stark_with_reset`` round can carry."""
    return [(machine.qubits["q1"], 0.3), (machine.qubits["q3"], 0.25)]


def test_stark_with_reset_drops_exactly_the_align_before_the_tones(live_chain):
    """The whole difference between the two timings is ONE statement: the
    ``align()`` that makes the tones wait for the relay's reset and its gap.
    Without it they start with the reset macro (other elements), and the
    round's closing align waits for whichever ends later. Everything else -
    prep, both swaps, the gap, the readout - is the same program, which is what
    lets one estimator read both."""
    from qm import generate_qua_script

    machine = live_chain
    config = machine.generate_config()

    def lines(**overrides):
        prog, _axes = build_program(**_live_kwargs(
            machine, compensation=_ends(machine), operation_gap_ns=20, **overrides))
        return [ln for ln in generate_qua_script(prog, config).splitlines()
                if "generated at" not in ln]

    after, together = lines(), lines(stark_with_reset=True)
    assert len(after) == len(together) + 1
    at = next(i for i, (a, b) in enumerate(zip(after, together)) if a != b)
    assert after[:at] + after[at + 1:] == together
    assert after[at].strip() == "align()"
    # ...and it is the one between the reset's gap and the first tone. The reset
    # macro ends by returning its z line to the stored IF; the gap follows it.
    assert 'play("parametric_reset", "q2.z")' in after[at - 3]
    assert after[at - 2].strip().startswith('update_frequency("q2.z", 0,')
    assert after[at - 1].strip().startswith("wait(5,") and '"q2.z"' in after[at - 1]
    assert 'play("stark"' in after[at + 1]
    # in the with_reset round nothing separates the reset's gap from the tones
    reset = next(i for i, ln in enumerate(together) if "parametric_reset" in ln)
    assert 'play("stark"*amp(0.3), "q1.xy")' in together[reset + 3]
    assert 'play("stark"*amp(0.25), "q3.xy")' in together[reset + 4]
    assert together[reset + 5].strip() == "align()"


def test_stark_with_reset_without_a_tone_is_the_same_program(live_chain):
    """No tone, nothing to move: scqo documents the two timings as ONE round
    then, so they must not differ by a stray align either."""
    from qm import generate_qua_script

    machine = live_chain
    config = machine.generate_config()

    def script(**overrides):
        prog, _axes = build_program(**_live_kwargs(
            machine, compensation=[], operation_gap_ns=20, **overrides))
        return [ln for ln in generate_qua_script(prog, config).splitlines()
                if "generated at" not in ln]

    assert script() == script(stark_with_reset=True)


def _init_reset_block(plain, reset):
    """The statements ``reset`` has and ``plain`` lacks, and where they start.
    Asserts they are ONE contiguous block and that nothing else moved."""
    at = next(i for i, (a, b) in enumerate(zip(plain, reset)) if a != b)
    added = len(reset) - len(plain)
    assert added > 0 and reset[:at] + reset[at + added:] == plain
    return at, [ln.strip() for ln in reset[at:at + added]]


def test_an_init_reset_plays_once_between_the_shot_reset_and_the_prep(live_chain):
    """init_resets adds one block to a shot and changes nothing else: the named
    qubit's reset macro, the run's ring-down in place of the stored one, and a
    global align. It sits after the between-shots reset and before the prep, so
    the prep pulse meets a qubit whose resonator has emptied."""
    from qm import generate_qua_script

    machine = live_chain
    config = machine.generate_config()
    q2 = machine.qubits["q2"]

    def lines(**overrides):
        prog, _axes = build_program(**_live_kwargs(machine, **overrides))
        return [ln for ln in generate_qua_script(prog, config).splitlines()
                if "generated at" not in ln]

    plain = lines()
    # no qubit named: the program it was, whatever the ring-down says
    assert lines(init_resets=[], init_reset_ringdown_ns=400) == plain

    reset = lines(init_resets=[q2], init_reset_ringdown_ns=400)
    at, block = _init_reset_block(plain, reset)
    drive = q2.macros[RESET_MACRO].drive_frequency
    assert block[0].startswith(f'update_frequency("q2.z", {drive},')
    assert block[1] == 'reset_if_phase("q2.z")'
    assert block[2] == 'play("parametric_reset", "q2.z")'
    assert block[3].startswith('update_frequency("q2.z", 0,')
    # the ring-down holds every channel of the qubit, for the run's 400 ns
    assert block[4].startswith("align(") and '"q2.xy"' in block[4]
    assert block[5].startswith("wait(100,")
    for channel in ("q2.xy", "q2.z", "q2.resonator"):
        assert f'"{channel}"' in block[5]
    assert block[6:] == ["align()"]
    # before it: the between-shots reset's align; after it: the prep
    assert reset[at - 1].strip() == "align()"
    assert reset[at + len(block)].strip() == 'play("x180", "q1.xy")'
    # once per shot, not per round: the rounds still carry their own reset
    assert sum("reset_if_phase" in ln for ln in reset) == 2

    # a ring-down of 0 plays the macro bare and still closes with the align
    _at, bare = _init_reset_block(
        plain, lines(init_resets=[q2], init_reset_ringdown_ns=0))
    assert bare[:4] == block[:4] and bare[4:] == ["align()"]


def test_shot_and_average_modes_differ_only_in_the_stream(live_chain):
    """One program, two stream terminals: the shot loop is identical and only
    what stream_processing does with the states changes, so a mode switch can
    never quietly change the circuit."""
    from qm import generate_qua_script

    machine = live_chain
    config = machine.generate_config()

    def script(prog):
        return "\n".join(ln for ln in generate_qua_script(prog, config).splitlines()
                         if "generated at" not in ln)

    shot, shot_axes = build_program(**_live_kwargs(machine, keep_shots=True))
    avg, avg_axes = build_program(**_live_kwargs(machine, keep_shots=False))
    assert "shot_idx" in shot_axes and "shot_idx" not in avg_axes

    shot_text, avg_text = script(shot), script(avg)
    # the circuit half is byte-identical up to stream_processing
    assert (shot_text[:shot_text.index("stream_processing")]
            == avg_text[:avg_text.index("stream_processing")])
    assert ".average()" in avg_text and ".average()" not in shot_text


def test_a_qubit_without_a_compensation_amp_gets_no_tone(live_chain):
    """compensation_amps={} is a real configuration (no phase correction), and
    a qubit merely ABSENT from the map must not be detuned either."""
    from qm import generate_qua_script

    machine = live_chain
    prog, _axes = build_program(**_live_kwargs(machine, compensation=[]))
    text = generate_qua_script(prog, machine.generate_config())
    assert 'play("stark"' not in text
    # no xy frame is touched at all. (The z one still is: the parametric-reset
    # macro sets its own drive frequency, which is the macro's business.)
    assert not [ln for ln in text.splitlines()
                if "update_frequency" in ln and ".xy" in ln]


def _swap_pulse(machine, pair_name):
    """``(pulse name, length in ns)`` the pair's swap macro plays — read off the
    live tree rather than hard-coded, so a re-registered chip does not silently
    turn the idle test into a test of the number 64."""
    macro = machine.qubit_pairs[pair_name].macros[SWAP]
    name = macro.flux_pulse
    return name, machine.qubit_pairs[pair_name].qubit_control.z.operations[name].length


def test_an_idle_step_waits_the_swap_out_instead_of_playing_it(live_chain):
    """The control arm. Idling the FIRST step must remove its pulses and put a
    wait of EXACTLY the swap's length on the same channels — the round has to
    keep its duration, or the comparison stops being about the swap and starts
    being about when everything else happened."""
    from qm import generate_qua_script

    machine = live_chain
    pulse, length_ns = _swap_pulse(machine, PAIRS[0])
    prog, _axes = build_program(**_live_kwargs(machine, first_operation=None))
    text = generate_qua_script(prog, machine.generate_config())

    # the first step plays nothing: not on the control z line, not on the coupler
    assert f'play("{pulse}", "q1.z")' not in text
    assert f'play("{pulse}", "{PAIRS[0]}")' not in text
    # ...it waits instead, for the swap's own length, on the pair's channels
    waits = [ln for ln in text.splitlines()
             if ln.strip().startswith(f"wait({length_ns // 4},")
             and f'"{PAIRS[0]}"' in ln]
    assert len(waits) == 1, text
    for channel in ("q1.z", "q2.z", PAIRS[0]):
        assert f'"{channel}"' in waits[0]
    # and the SECOND step is untouched — one step idled, not the round. It plays
    # its OWN pulse: the two pairs carry different flux pulses under the same
    # macro key on this chip, which is the whole reason a step names its own
    # operation rather than sharing one.
    second_pulse, _ns = _swap_pulse(machine, PAIRS[1])
    assert f'play("{second_pulse}", "{PAIRS[1]}")' in text


# ------------------------------------------------------- the registered shell


def _composite_over(roster, members):
    """The roster composite whose high/low members are exactly ``members``.

    Looked up by MEMBERSHIP, never by name: the roster orders a pair's name by
    design-nominal frequency (the q2-q3 couple is the composite ``q3_q2``) while
    QUAM keys the same pair ``q2_q3``, so a name-based join would be false here
    on purpose — that mismatch is what ``_vendor.vendor_pair`` exists to absorb.
    """
    want = set(members)
    for name, entity in roster.composites().items():
        roles = getattr(entity, "roles", {}) or {}
        if {m for role in ("high", "low") for m in roles.get(role, ())} == want:
            return name
    pytest.skip(f"generated roster has no composite over {sorted(want)}")


def _live_experiment(machine, **params):
    """The shell wired the way the Session wires one, on the live tree."""
    from scqo.roster import parse_components

    from scqo_qm.backend.qm_backend import QMBackend
    from scqo_qm.backend.roster_gen import roster_toml_for
    from scqo_qm.experiments.qc_unidirectional_trotter import (
        QMQcUnidirectionalTrotter,
    )

    roster = parse_components(roster_toml_for(machine))
    backend = QMBackend(machine, roster=roster)
    settings = dict(
        targets=list(CHAIN),
        first_pair={"pair": _composite_over(roster, CHAIN[:2]), "operation": SWAP},
        second_pair={"pair": _composite_over(roster, CHAIN[1:]), "operation": SWAP},
        reset_qubit="q2", idle_reference_operation=SWAP,
        reset_operation=RESET_MACRO,
        compensation_amps={"q1": 0.3, "q2": 0.2, "q3": 0.25},
        max_rounds=5, num_averages=10)
    settings.update(params)
    exp = QMQcUnidirectionalTrotter(
        backend, QMQcUnidirectionalTrotter.Parameters(**settings))
    exp.sweep_axes = exp.define_sweep()   # the Session's job before the hook
    return backend, exp


def test_probe_matches_the_direct_build(live_chain):
    """probe() must produce the same QUA program as calling build_program with
    the mapped kwargs — that mapping (chain roles resolved through scqo, target
    ORDER preserved, readout_mode -> keep_shots) is the whole adapter."""
    from qm import generate_qua_script

    machine = live_chain
    config = machine.generate_config()

    def script(prog):
        return "\n".join(ln for ln in generate_qua_script(prog, config).splitlines()
                         if "generated at" not in ln)

    _backend, exp = _live_experiment(machine, readout_mode="shot")
    from_probe, probe_axes = exp.probe()
    direct, direct_axes = build_program(**_live_kwargs(machine))
    assert script(from_probe) == script(direct)
    assert list(probe_axes) == list(direct_axes)
    # the prep qubit was DERIVED: q1 is the first_pair member absent from
    # second_pair, and nothing in the params named it
    assert script(from_probe).count('play("x180"') == 1


def test_a_two_qubit_prep_plays_both_before_the_first_round(live_chain):
    """prep_operations={q1: x180, q3: x180} starts the chain in |101>: each end
    plays its pulse ONCE, before the first swap, and probe() hands the neutral
    map to the builder as the same list a direct call takes."""
    from qm import generate_qua_script

    machine = live_chain
    config = machine.generate_config()

    def script(prog):
        return "\n".join(ln for ln in generate_qua_script(prog, config).splitlines()
                         if "generated at" not in ln)

    ends = [(machine.qubits["q1"], "x180"), (machine.qubits["q3"], "x180")]
    direct, _axes = build_program(**_live_kwargs(machine, prep=ends))
    text = script(direct)
    assert text.count('play("x180"') == 2
    body = text[text.index('play("x180"'):]
    first_swap = body.index(f'"{PAIRS[0]}")')
    for name in ("q1", "q3"):
        assert body.index(f'play("x180", "{name}.xy")') < first_swap, name

    _backend, exp = _live_experiment(machine, readout_mode="shot",
                                     prep_operations={"q1": "x180", "q3": "x180"})
    from_probe, _axes = exp.probe()
    assert script(from_probe) == text


def test_probe_translates_the_neutral_idle_into_the_builders_none(live_chain):
    """scqo spells the reserved step 'idle' and the builder spells it None —
    that translation is the adapter's, and it is the only place the two
    vocabularies meet. Same program either way, or one of them is wrong."""
    from qm import generate_qua_script

    machine = live_chain
    config = machine.generate_config()

    def script(prog):
        return "\n".join(ln for ln in generate_qua_script(prog, config).splitlines()
                         if "generated at" not in ln)

    _backend, exp = _live_experiment(machine, readout_mode="shot")
    exp.params.first_pair = {**exp.params.first_pair, "operation": "idle"}
    from_probe, _axes = exp.probe()
    direct, _ = build_program(**_live_kwargs(machine, first_operation=None))
    assert script(from_probe) == script(direct)


def test_probe_translates_stark_timing_into_the_builders_flag(live_chain):
    """scqo spells the setting stark_timing='with_reset' and the builder takes a
    flag; the adapter is the one place the two meet. And scqo's own gate comes
    first: a tone on the reset qubit never reaches the builder."""
    from qm import generate_qua_script

    machine = live_chain
    config = machine.generate_config()

    def script(prog):
        return "\n".join(ln for ln in generate_qua_script(prog, config).splitlines()
                         if "generated at" not in ln)

    _backend, exp = _live_experiment(
        machine, readout_mode="shot", stark_timing="with_reset",
        compensation_amps={"q1": 0.3, "q3": 0.25})
    from_probe, _axes = exp.probe()
    direct, _ = build_program(**_live_kwargs(
        machine, compensation=_ends(machine), stark_with_reset=True))
    after, _ = build_program(**_live_kwargs(machine, compensation=_ends(machine)))
    assert script(from_probe) == script(direct) != script(after)

    with pytest.raises(ValueError, match="names the reset qubit 'q2'"):
        _live_experiment(machine, stark_timing="with_reset")


def test_probe_hands_init_reset_qubits_to_the_builder(live_chain):
    """scqo names the qubits and the wait; the adapter joins the names to their
    QUAM objects through the flux door and passes the wait on. scqo's own gates
    come first: a name off the chain never reaches the builder, and a chain
    qubit that carries no reset macro is refused by the builder, by name."""
    from qm import generate_qua_script

    machine = live_chain
    config = machine.generate_config()

    def script(prog):
        return "\n".join(ln for ln in generate_qua_script(prog, config).splitlines()
                         if "generated at" not in ln)

    _backend, exp = _live_experiment(
        machine, readout_mode="shot", init_reset_qubits=["q2"])
    from_probe, _axes = exp.probe()
    # 400 ns is scqo's default ring-down
    direct, _ = build_program(**_live_kwargs(
        machine, init_resets=[machine.qubits["q2"]], init_reset_ringdown_ns=400))
    plain, _ = build_program(**_live_kwargs(machine))
    assert script(from_probe) == script(direct) != script(plain)

    with pytest.raises(ValueError, match="'q4', which is not on the chain"):
        _live_experiment(machine, init_reset_qubits=["q4"])
    if RESET_MACRO not in machine.qubits["q1"].macros:
        _backend, exp = _live_experiment(machine, init_reset_qubits=["q1", "q2"])
        with pytest.raises(ValueError,
                           match="init_reset_qubits names q1, which has no macro"):
            exp.probe()


def test_preview_renders_the_chain_without_touching_the_network(live_chain,
                                                                tmp_path,
                                                                monkeypatch):
    """One program for the whole chain means the shell does NOT self-acquire,
    so `scqo run --preview` works with no opt-out and no extra code path."""
    import socket

    monkeypatch.setattr(socket, "create_connection",
                        lambda *a, **k: pytest.fail(
                            "no_simulate must never touch the network"))
    backend, exp = _live_experiment(live_chain)
    out_dir = tmp_path / "prev"
    files = backend.preview(exp, out_dir, no_simulate=True)
    assert files == [out_dir / "qua_script.py"]
    text = files[0].read_text(encoding="utf-8")
    assert text.startswith("# scqo preview: qc_unidirectional_trotter\n# backend: qm\n")
    assert 'play("parametric_reset", "q2.z")' in text
