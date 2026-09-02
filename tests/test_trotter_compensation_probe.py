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


def _pair(name, macros=(SWAP,)):
    return SimpleNamespace(
        name=name,
        qubit_control=_qubit(f"{name}_c"),
        qubit_target=_qubit(f"{name}_t"),
        macros={m: SimpleNamespace(flux_pulse="flattop_cosine") for m in macros},
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
        swap_operation=SWAP,
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
        ({"reset_operation": "paramreset"}, "no macro 'paramreset'"),
        ({"prep_operation": "x270"}, "no xy operation 'x270'"),
        ({"stark_operation": "tone"}, "no xy operation 'tone'"),
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
                                prep_qubit=q1, swept_qubit=q1,
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
                                prep_qubit=q1, swept_qubit=bare,
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
        prep_qubit=machine.qubits["q1"],
        prep_operation="x180",
        swap_operation=SWAP,
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
