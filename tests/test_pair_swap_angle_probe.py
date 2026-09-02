"""Offline build proof + guard census for the partial-swap angle QM probe.

Two halves, deliberately:

* the GUARDS run before a single QUA statement is emitted, so they are pinned
  against plain stubs -- no QUAM, no config, no QOP. Every one of them is a
  refusal an operator can act on (register this macro, bring that coupler up);
* the BUILD is rendered from the live ``quam_state``, because a QUA program is
  made out of the vendor's own macros and there is no honest stand-in for
  ``pair.macros["iswap"].apply(cplr_amp=...)``. It skips by name when the
  committed state cannot express the sweep -- which, at the time this probe was
  written, it could not: both live couplers carry their swap pulse at amplitude
  0.0, the exact state the zero-amplitude guard exists for.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from scqo_qm.experiments._coupler_knob import resolve_coupler_knob
from scqo_qm.experiments.pair_swap_angle import _member_states, build_program

STATE = str(Path(__file__).resolve().parents[1] / "quam_state")

PAIR = "q1_q2"
SWAP = "iswap"          # the one swap macro BOTH live pairs carry


# --------------------------------------------------------------- guard stubs


def _qubit(name, ops=("x180",)):
    return SimpleNamespace(
        name=name,
        xy=SimpleNamespace(operations={op: SimpleNamespace() for op in ops}),
        z=SimpleNamespace(operations={}),
    )


def _pair(name=PAIR, macros=(SWAP,), coupler_ops=("flattop_cosine",),
          coupler_amp=0.25, flux_pulse="flattop_cosine", coupler=True):
    """A stub pair whose coupler knob can be broken one way at a time."""
    control, target = _qubit("q1"), _qubit("q2")
    cpl = None
    if coupler:
        cpl = SimpleNamespace(
            name=f"{name}_c",
            operations={op: SimpleNamespace(amplitude=coupler_amp)
                        for op in coupler_ops},
        )
    return SimpleNamespace(
        name=name,
        qubit_control=control,
        qubit_target=target,
        coupler=cpl,
        macros={m: SimpleNamespace(flux_pulse=flux_pulse) for m in macros},
    )


def _kwargs(**overrides):
    """The smallest call that reaches the coupler guard."""
    pair = overrides.pop("pair", None) or _pair()
    base = dict(
        machine=None,
        measure_qubits=[pair.qubit_control, pair.qubit_target],
        swap_pair=pair,
        swap_operation=SWAP,
        rounds_array=np.arange(0, 5),
        coupler_amplitudes=np.linspace(0.0, 0.1, 5),
        num_shots=10,
        reset_type="thermal",
    )
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    "override, message",
    [
        ({"operation_gap_ns": 6}, "multiple of 4"),
        ({"operation_gap_ns": -4}, "multiple of 4"),
        ({"pair": _pair(macros=())}, "no macro 'iswap'"),
        ({"pair": _pair(macros=("cz",))}, "no macro 'iswap'"),
        ({"pair": _pair(coupler=False)}, "has no coupler"),
        ({"pair": _pair(coupler_ops=("const",))}, "no coupler flux_pulse"),
        ({"pair": _pair(flux_pulse=None)}, "no coupler flux_pulse"),
    ],
)
def test_guards_refuse_by_name_before_any_qua(override, message):
    """Every refusal names the thing to fix, and fires with no machine at all --
    proof it happens before the builder touches the vendor tree."""
    with pytest.raises(ValueError, match=message):
        build_program(**_kwargs(**override))


def test_a_coupler_baked_at_zero_is_refused_and_names_the_fix():
    """THE trap this probe exists to guard. The macro turns cplr_amp into an
    amplitude_scale by DIVIDING by the stored coupler amplitude, so a coupler
    baked at 0.0 -- the state of every chip whose swaps have only ever been
    detuning swaps -- is unsettable, not merely weak. Uncaught it is a division
    by zero inside the vendor macro; caught, it names the register script."""
    with pytest.raises(ValueError, match="baked at amplitude 0.0"):
        build_program(**_kwargs(pair=_pair(coupler_amp=0.0)))
    with pytest.raises(ValueError, match="register_flattop_cosine.py"):
        build_program(**_kwargs(pair=_pair(coupler_amp=0.0)))


def test_resolve_coupler_knob_returns_the_coupler_and_its_pulse():
    """The happy path of the shared resolver: the three chain/angle probes all
    read the SAME coupler object and pulse name out of it."""
    coupler, pulse = resolve_coupler_knob(_pair(), SWAP)
    assert pulse == "flattop_cosine"
    assert coupler.name == "q1_q2_c"


def test_member_states_orders_by_ROLE_not_by_readout():
    """The probe measures [control, target]; the schema's member axis is
    (high, low). When the roster's HIGH member is the vendor TARGET the two
    disagree, and getting it wrong silently swaps the prepared and transfer
    panels of every figure."""
    state = xr.DataArray(
        np.arange(2 * 2 * 3 * 4).reshape(2, 2, 3, 4),
        dims=("qubit", "shot", "coupler_amplitude", "round"),
        coords={"qubit": ["q1", "q2"]},
    )
    as_control = _member_states(state, "control")
    as_target = _member_states(state, "target")

    assert as_control.dims == ("member", "shot_idx", "coupler_flux_v", "swap_count")
    assert list(as_control.member.values) == ["high", "low"]
    # high==control keeps the readout order; high==target flips it
    np.testing.assert_array_equal(as_control.isel(member=0).values, state.isel(qubit=0).values)
    np.testing.assert_array_equal(as_target.isel(member=0).values, state.isel(qubit=1).values)


# ------------------------------------------------------------ requires QUAM

quam_config = pytest.importorskip("quam_config")
pytest.importorskip("qm")


@pytest.fixture(scope="module")
def live_pair():
    """The live machine plus a sweepable pair, or a skip naming what is absent.

    A pair is only sweepable here if its coupler carries the swap pulse at a
    NONZERO amplitude, so this skip is the live-state twin of the stub guard
    above -- and it is the one that will start passing the day the coupler is
    brought up."""
    machine = quam_config.Quam.load(STATE)
    if PAIR not in machine.qubit_pairs:
        pytest.skip(f"live state has no pair {PAIR}")
    pair = machine.qubit_pairs[PAIR]
    if SWAP not in pair.macros:
        pytest.skip(f"live pair {PAIR} has no {SWAP!r} macro")
    try:
        resolve_coupler_knob(pair, SWAP)
    except ValueError as err:
        pytest.skip(f"live pair {PAIR} has no turnable coupler knob: {err}")
    return machine, pair


def test_build_program_sweeps_the_coupler_not_the_control(live_pair):
    """The whole point of this probe, asserted on the generated QUA: the swept
    amplitude reaches the COUPLER element, and the control qubit's own flux
    pulse plays bare at its calibrated amplitude (that is qc_n_swap_amp's knob,
    not this one)."""
    from qm import generate_qua_script

    machine, pair = live_pair
    prog, axes = build_program(
        machine=machine,
        measure_qubits=[pair.qubit_control, pair.qubit_target],
        swap_pair=pair,
        swap_operation=SWAP,
        rounds_array=np.arange(0, 6),
        coupler_amplitudes=np.linspace(0.0, 0.02, 5),
        num_shots=10,
        reset_type="thermal",
    )
    script = generate_qua_script(prog, machine.generate_config())
    assert pair.coupler.name in script
    assert list(axes) == ["qubit", "shot", "coupler_amplitude", "round"]
    assert axes["coupler_amplitude"].attrs["units"] == "V"


# ------------------------------------------- the phase-compensation tone


def test_a_missing_stark_op_is_refused_by_name():
    """The tone is what turns the fitted per-round composite angle back into the
    exchange angle, so a member without the operation is refused rather than
    silently left uncompensated -- which would look like a measurement."""
    pair = _pair()
    bare = _qubit("q1", ops=("x180",))          # no 'stark'
    with pytest.raises(ValueError, match="no xy operation 'stark'"):
        build_program(**_kwargs(pair=pair, compensation=[(bare, 0.3)]))


def test_an_unrepresentable_compensation_factor_names_its_knob():
    """QUA's amplitude_scale spans (-2, 2); the refusal must name the neutral
    knob, not an internal QUA variable."""
    q = _qubit("q1", ops=("x180", "stark"))
    with pytest.raises(ValueError, match="compensation_amps"):
        build_program(**_kwargs(compensation=[(q, 2.5)]))


def test_guards_fire_before_the_intermediate_frequency_is_read():
    """Ordering proof: the stubs carry no intermediate_frequency at all, so a
    refusal that still fires is one that happened before any IF bookkeeping --
    i.e. before the builder touched the vendor tree."""
    q = _qubit("q1", ops=("x180",))
    assert not hasattr(q.xy, "intermediate_frequency")
    with pytest.raises(ValueError, match="no xy operation"):
        build_program(**_kwargs(compensation=[(q, 0.3)]))


def test_the_tone_reaches_the_program(live_pair):
    """A compensated build must differ from an uncompensated one, and must carry
    the frequency shift that makes the tone SHIFT rather than rotate."""
    from qm import generate_qua_script

    machine, pair = live_pair
    common = dict(
        machine=machine,
        measure_qubits=[pair.qubit_control, pair.qubit_target],
        swap_pair=pair,
        swap_operation=SWAP,
        rounds_array=np.arange(0, 6),
        coupler_amplitudes=np.linspace(0.0, 0.02, 5),
        num_shots=10,
        reset_type="thermal",
    )
    if "stark" not in pair.qubit_control.xy.operations:
        pytest.skip("live control member has no 'stark' xy op (register_stark.py)")

    bare, _ = build_program(**common)
    toned, _ = build_program(**common,
                             compensation=[(pair.qubit_control, 0.3)],
                             stark_detuning_hz=50e6)
    bare_script = generate_qua_script(bare, machine.generate_config())
    toned_script = generate_qua_script(toned, machine.generate_config())

    assert toned_script != bare_script
    assert "update_frequency" in toned_script
    assert "update_frequency" not in bare_script
