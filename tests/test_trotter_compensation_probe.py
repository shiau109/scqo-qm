"""Offline build proof + guard census for the Trotter compensation-scan probe.

Same two halves as ``test_unidirectional_trotter_probe``: the guards are pinned
against plain stubs (they fire before any QUA is emitted, with no machine at
all), and the build is rendered from the live ``quam_state`` because there is no
honest stand-in for the vendor's own macros.

The scan's own hazards, over and above the chain's: the swept tone must reach
the FPGA as a QUA variable while the fixed ones stay compile-time constants, and
the swept qubit must not ALSO be a fixed tone -- which would play twice per round
and double the very phase being measured, silently.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from scqo_qm.experiments.qc_trotter_compensation import build_program

STATE = str(Path(__file__).resolve().parents[1] / "quam_state")

#: the chain the live state can express: q1 -> q2 -> q3, resetting q2.
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
#: copies that length, so the stubs carry one.
FLUX_PULSE = "flattop_cosine"
PULSE_NS = 64


def _pair(name, macros=(SWAP,), pulse_ns=PULSE_NS):
    """A stub pair. ``pulse_ns=None`` registers the macro with NO flux pulse,
    which is the shape an idle step cannot take its duration from."""
    ops = {} if pulse_ns is None else {FLUX_PULSE: SimpleNamespace(length=pulse_ns)}
    control = _qubit(f"{name}_c")
    control.z = SimpleNamespace(operations=ops)
    return SimpleNamespace(
        name=name,
        qubit_control=control,
        qubit_target=_qubit(f"{name}_t"),
        coupler=SimpleNamespace(operations=ops),
        macros={m: SimpleNamespace(flux_pulse=FLUX_PULSE) for m in macros},
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
        prep=[(q1, "x180")],
        first_operation=SWAP,
        second_operation=SWAP,
        idle_reference_operation=SWAP,
        reset_operation=RESET_MACRO,
        stark_operation="stark",
        stark_detuning_hz=50e6,
        compensation=[(q3, 0.25)],
        swept_qubit=q1,
        compensation_amps=np.linspace(0.0, 1.0, 5),
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
        # an idle step still needs a DURATION, and it comes from a real macro
        ({"first_operation": None, "first_pair": _pair("q1_q2", macros=())},
         "has no macro 'iswap', so an 'idle' step"),
        ({"first_operation": None, "first_pair": _pair("q1_q2", pulse_ns=None)},
         "no flux pulse with a length"),
        # the angle knob has nothing to turn on a step that plays no pulse
        ({"first_operation": None, "first_coupler_amp": 0.04},
         "is an idle step but was given a coupler amplitude"),
        ({"reset_operation": "paramreset"}, "no macro 'paramreset'"),
        ({"prep": [(_qubit("q1"), "x270")]}, "no xy operation 'x270'"),
        ({"stark_operation": "tone"}, "no xy operation 'tone'"),
        # with the tones DURING the reset, none may sit on the reset qubit (q2):
        # not a fixed one, and not the swept one
        ({"stark_with_reset": True, "compensation": [(_qubit("q2"), 0.2)]},
         "q2 is the reset qubit and also plays a stark tone"),
        ({"stark_with_reset": True, "swept_qubit": _qubit("q2")},
         "q2 is the reset qubit and also plays a stark tone"),
    ],
)
def test_guards_refuse_by_name_before_any_qua(override, message):
    """Every refusal names the thing to fix, and fires with no machine at all."""
    with pytest.raises(ValueError, match=message):
        build_program(**_kwargs(**override))


def test_the_swept_qubit_may_not_also_be_a_fixed_tone():
    """A doubled tone plays twice per round and doubles the phase under
    measurement -- invisible in the data, so it is refused at build time. scqo
    refuses it upstream too; this is the backstop for a direct builder call."""
    q1, q2, q3 = (_qubit(n) for n in CHAIN)
    q2.macros[RESET_MACRO] = SimpleNamespace()
    with pytest.raises(ValueError, match="play TWICE per round"):
        build_program(**_kwargs(measure_qubits=[q1, q2, q3], reset_qubit=q2,
                                prep=[(q1, "x180")], swept_qubit=q1,
                                compensation=[(q1, 0.3)]))


def test_an_unrepresentable_swept_amplitude_names_its_own_knob():
    """QUA's amplitude_scale spans (-2, 2). The FIXED tones and the SWEPT window
    are separate knobs on the scqo surface, so they must be refused by separate
    names -- max_compensation_amp, not compensation_amps."""
    with pytest.raises(ValueError, match="max_compensation_amp"):
        build_program(**_kwargs(compensation_amps=np.linspace(0.0, 2.5, 5)))
    with pytest.raises(ValueError, match="compensation_amps"):
        build_program(**_kwargs(compensation=[(_qubit("q3"), 2.5)]))


def test_a_missing_stark_op_on_the_SWEPT_qubit_is_caught():
    """The swept qubit is not in the `compensation` list, so it needs its own
    membership in the operation check -- otherwise the omission surfaces only
    when the QUA is emitted."""
    q1, q2, q3 = (_qubit(n) for n in CHAIN)
    q2.macros[RESET_MACRO] = SimpleNamespace()
    bare = _qubit("q1", ops=("x180",))          # no 'stark'
    with pytest.raises(ValueError, match="no xy operation 'stark'"):
        build_program(**_kwargs(measure_qubits=[q1, q2, q3], reset_qubit=q2,
                                prep=[(q1, "x180")], swept_qubit=bare,
                                compensation=[(q3, 0.25)]))


def test_reduce_raw_names_the_readout_form():
    """`state` means per-shot integer LEVELS, so only the FPGA-averaged stream
    is renamed to `population` -- the contract accepts both, which is exactly why
    the backend's own rename cannot do it."""
    from scqo_qm.experiments.qc_trotter_compensation import QMQcTrotterCompensation

    raw = xr.Dataset(
        {"state": (("qubit", "compensation_amp", "round_count"),
                   np.zeros((2, 4, 3)))},
        coords={"qubit": ["q1", "q2"],
                "compensation_amp": np.linspace(0, 1, 4),
                "round_count": np.arange(3)},
    )
    shot = SimpleNamespace(params=SimpleNamespace(readout_mode="shot"))
    average = SimpleNamespace(params=SimpleNamespace(readout_mode="average"))
    assert "state" in QMQcTrotterCompensation.reduce_raw(shot, raw).data_vars
    reduced = QMQcTrotterCompensation.reduce_raw(average, raw)
    assert "population" in reduced.data_vars and "state" not in reduced.data_vars


# ------------------------------------------------------------ requires QUAM

quam_config = pytest.importorskip("quam_config")
pytest.importorskip("qm")


@pytest.fixture(scope="module")
def live_chain():
    """The live machine, or a skip naming exactly which piece is absent."""
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
        compensation=[(machine.qubits["q3"], 0.25)],
        swept_qubit=machine.qubits["q1"],
        compensation_amps=np.linspace(0.0, 1.0, 5),
        rounds_array=np.arange(0, 6),
        num_shots=10,
        reset_type="thermal",
        keep_shots=False,
    )
    base.update(overrides)
    return base


def test_build_program_renders_the_scan(live_chain):
    from qm import generate_qua_script

    prog, axes = build_program(**_live_kwargs(live_chain))
    script = generate_qua_script(prog, live_chain.generate_config())
    # the swept axis is OUTER of the round axis and INSIDE the shot loop
    assert list(axes) == ["qubit", "compensation_amp", "round_count"]
    assert axes["compensation_amp"].attrs["units"] == ""
    # both chain qubits carrying a tone get detuned and restored
    assert "update_frequency" in script


def test_stark_with_reset_drops_exactly_the_align_before_the_tones(live_chain):
    """The scan plays the chain probe's own round (``_chain_round``), so the two
    timings differ here by the same ONE statement: the ``align()`` between the
    relay's reset (with its gap) and the tones. The swept tone moves with the
    fixed one - a scan that left it behind would calibrate another round."""
    from qm import generate_qua_script

    config = live_chain.generate_config()

    def lines(**overrides):
        prog, _axes = build_program(**_live_kwargs(
            live_chain, operation_gap_ns=20, **overrides))
        return [ln for ln in generate_qua_script(prog, config).splitlines()
                if "generated at" not in ln]

    after, together = lines(), lines(stark_with_reset=True)
    assert len(after) == len(together) + 1
    at = next(i for i, (a, b) in enumerate(zip(after, together)) if a != b)
    assert after[:at] + after[at + 1:] == together
    assert after[at].strip() == "align()"
    # the reset macro ends by returning its z line to the stored IF; the gap follows it
    assert 'play("parametric_reset", "q2.z")' in after[at - 3]
    assert after[at - 2].strip().startswith('update_frequency("q2.z", 0,')
    assert after[at - 1].strip().startswith("wait(5,") and '"q2.z"' in after[at - 1]
    # fixed tone first, then the swept one at its QUA variable, then the align
    reset = next(i for i, ln in enumerate(together) if "parametric_reset" in ln)
    assert 'play("stark"*amp(0.25), "q3.xy")' in together[reset + 3]
    assert 'play("stark"*amp(v' in together[reset + 4] and '"q1.xy"' in together[reset + 4]
    assert together[reset + 5].strip() == "align()"


def test_the_scan_plays_the_chains_init_reset(live_chain):
    """The scan calibrates the shot the chain runs, so a reset at the start of
    the shot has to be in both: the same block (the macro, the run's ring-down,
    a global align) between the between-shots reset and the prep, and nothing
    else moved. With no qubit named the scan is the program it was."""
    from qm import generate_qua_script

    config = live_chain.generate_config()
    q2 = live_chain.qubits["q2"]

    def lines(**overrides):
        prog, _axes = build_program(**_live_kwargs(live_chain, **overrides))
        return [ln for ln in generate_qua_script(prog, config).splitlines()
                if "generated at" not in ln]

    plain = lines()
    assert lines(init_resets=[], init_reset_ringdown_ns=400) == plain

    reset = lines(init_resets=[q2], init_reset_ringdown_ns=400)
    at = next(i for i, (a, b) in enumerate(zip(plain, reset)) if a != b)
    added = len(reset) - len(plain)
    assert reset[:at] + reset[at + added:] == plain
    block = [ln.strip() for ln in reset[at:at + added]]
    assert block[2] == 'play("parametric_reset", "q2.z")'
    assert block[5].startswith("wait(100,") and '"q2.z"' in block[5]
    assert block[-1] == "align()" and len(block) == 7
    assert reset[at + added].strip() == 'play("x180", "q1.xy")'

    # the guards are the chain probe's own (_chain_round.check_init_resets)
    with pytest.raises(ValueError, match="init_reset_qubits names q1, which has no macro"):
        build_program(**_kwargs(init_resets=[_qubit("q1")]))


def test_shot_mode_adds_the_shot_axis(live_chain):
    _prog, axes = build_program(**_live_kwargs(live_chain, keep_shots=True))
    assert list(axes) == ["qubit", "shot_idx", "compensation_amp", "round_count"]


def test_swap_coupler_flux_reaches_the_macro(live_chain):
    """The angle knob threads through unchanged: setting it must build, and a
    pair left unset must still build (the pre-existing bare-apply sequence)."""
    from qm import generate_qua_script

    bare, _axes = build_program(**_live_kwargs(live_chain))
    bare_script = generate_qua_script(bare, live_chain.generate_config())

    driven, _axes = build_program(**_live_kwargs(
        live_chain, first_coupler_amp=0.01, second_coupler_amp=0.012))
    driven_script = generate_qua_script(driven, live_chain.generate_config())
    # driving the couplers changes the emitted program; leaving them alone does not
    assert driven_script != bare_script


def test_a_coupler_baked_at_zero_is_refused_here_too(live_chain):
    """The chain shells share ``_coupler_knob`` with pair_swap_angle, so the
    unsettable-coupler refusal is the same one -- and it must fire from the chain
    path as well, not only from the sweeping probe."""
    pair = live_chain.qubit_pairs[PAIRS[0]]
    stored = pair.coupler.operations[pair.macros[SWAP].flux_pulse]
    original = stored.amplitude
    try:
        stored.amplitude = 0.0
        with pytest.raises(ValueError, match="register_flattop_cosine.py"):
            build_program(**_live_kwargs(live_chain, first_coupler_amp=0.01))
    finally:
        stored.amplitude = original
