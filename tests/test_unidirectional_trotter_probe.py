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
        prep_qubit=q1,
        prep_operation="x180",
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
        ({"prep_operation": "x270"}, "no xy operation 'x270'"),
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
                                prep_qubit=q1))


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
        prep_qubit=machine.qubits["q1"],
        prep_operation="x180",
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
    exp = QMQcUnidirectionalTrotter(
        backend,
        QMQcUnidirectionalTrotter.Parameters(
            targets=list(CHAIN),
            first_pair={"pair": _composite_over(roster, CHAIN[:2]),
                        "operation": SWAP},
            second_pair={"pair": _composite_over(roster, CHAIN[1:]),
                         "operation": SWAP},
            reset_qubit="q2", idle_reference_operation=SWAP,
            reset_operation=RESET_MACRO,
            compensation_amps={"q1": 0.3, "q2": 0.2, "q3": 0.25},
            max_rounds=5, num_averages=10, **params))
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
