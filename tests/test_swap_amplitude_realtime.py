"""No arithmetic in front of a swap's flux plays.

A probe that sweeps a swap's flux amplitude has to turn volts into a QUA
``amplitude_scale`` somewhere. Done IN FRONT of the play -- between a round's
``align()`` and the pulse -- it is FPGA arithmetic on the swap's own timeline,
which can hold that element back against the other pulse of the same swap and
against the bare gate the calibration is for. On 5Q4C q1_q2 (2026-09-21) the
swept-amplitude flux map found its per-round phase about 0.40 turn away from
the bare-gate map at the same flux, the size of two clock cycles of round time
at that pair's detuning; its QUA read ``amp((v12/0.1))`` where the bare gate
read no amplitude at all.

Two halves:

* the macro's rule (:func:`resolve_amplitude_scale`): volts must be a Python
  number and are divided in Python, a scale passes through verbatim, and a QUA
  variable offered as volts is refused;
* every swept-amplitude swap probe, BUILT on the live ``quam_state`` and read
  back as QUA script: each ``play`` carries ``amp(<one variable>)`` or a
  literal, never an expression -- and the division, where a probe still needs
  one, is an ``assign`` of its own.

``Quam.load`` returns a shared in-memory tree, so what a fixture registers it
removes again.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

quam_config = pytest.importorskip("quam_config")
pytest.importorskip("qm")

from qm import generate_qua_script  # noqa: E402
from qm.qua import declare, fixed, program  # noqa: E402

from scqo_qm.components.macros.iswap_macro import resolve_amplitude_scale  # noqa: E402

STATE = str(Path(__file__).resolve().parents[1] / "quam_state")
PAIR = "q1_q2"
SWAP = "iswap"

#: an ``amp()`` argument that is ONE QUA variable or a numeric literal -- the
#: only two things a swap's play may carry.
PLAIN = re.compile(r"^(v\d+|-?\d+(\.\d+)?(e-?\d+)?)$")


def play_amplitudes(script: str) -> list:
    """Every ``amp(...)`` argument on a ``play`` line, exactly as scripted."""
    found = []
    for line in script.splitlines():
        if "play(" not in line:
            continue
        for match in re.finditer(r"amp\(", line):
            depth, i = 1, match.end()
            while depth and i < len(line):
                depth += {"(": 1, ")": -1}.get(line[i], 0)
                i += 1
            found.append(line[match.end():i - 1].strip())
    return found


def assert_plain_plays(script: str, what: str) -> None:
    args = play_amplitudes(script)
    assert args, f"{what}: no amplitude-scaled play at all -- the sweep is gone"
    arithmetic = [arg for arg in args if not PLAIN.match(arg)]
    assert not arithmetic, f"{what}: arithmetic in front of a play: {arithmetic}"


def test_the_checker_catches_what_the_old_map_played():
    """The pattern the fix removes, verbatim from the pre-fix script -- so a
    green run of the builds below means something."""
    old = 'play("flattop_cosine"*amp((v12/0.1)), "q1.z")'
    assert play_amplitudes(old) == ["(v12/0.1)"]
    assert not PLAIN.match("(v12/0.1)")
    assert PLAIN.match("v12") and PLAIN.match("0.08") and PLAIN.match("-0.5")
    assert play_amplitudes('play("x180", "q1.xy")') == []


# ------------------------------------------- the macro's rule, no QUAM tree


def test_volts_are_divided_in_python():
    assert resolve_amplitude_scale(-0.1497, reference=-0.15, what="z") == pytest.approx(0.998)
    # numpy scalars are Python numbers for this purpose
    assert resolve_amplitude_scale(np.float64(0.02), reference=0.25,
                                   what="coupler") == pytest.approx(0.08)


def test_a_scale_passes_verbatim_and_bare_stays_bare():
    token = object()
    assert resolve_amplitude_scale(scale=token, reference=0.1, what="z") is token
    assert resolve_amplitude_scale(reference=0.1, what="z") is None


def test_volts_and_a_scale_together_are_refused():
    with pytest.raises(ValueError, match="not both"):
        resolve_amplitude_scale(0.1, 1.0, reference=0.1, what="z")


def test_volts_against_a_zero_reference_are_refused():
    with pytest.raises(ValueError, match="0.0"):
        resolve_amplitude_scale(0.1, reference=0.0, what="coupler")


def test_a_qua_variable_is_refused_as_volts():
    """The exact call the old probes made. As a SCALE the same variable is
    the supported path."""
    with program():
        v = declare(fixed)
        with pytest.raises(TypeError, match="scale"):
            resolve_amplitude_scale(v, reference=0.1, what="q1.z")
        assert resolve_amplitude_scale(scale=v, reference=0.1, what="q1.z") is v


@pytest.fixture
def machine():
    return quam_config.Quam.load(STATE)


@pytest.fixture
def pair(machine):
    if PAIR not in machine.qubit_pairs:
        pytest.skip(f"live state has no pair {PAIR}")
    qp = machine.qubit_pairs[PAIR]
    if SWAP not in qp.macros:
        pytest.skip(f"live pair {PAIR} has no {SWAP!r} macro")
    return qp


@pytest.fixture
def stark(pair):
    """A ``stark`` op on the control's xy, registered IN MEMORY for the builds
    that play one, and removed again."""
    from quam.components.pulses import SquarePulse

    xy = pair.qubit_control.xy
    had = "stark" in xy.operations
    if not had:
        xy.operations["stark"] = SquarePulse(length=64, amplitude=0.1, axis_angle=0.0)
    yield "stark"
    if not had:
        del xy.operations["stark"]


def _script(machine, prog, config=None):
    return generate_qua_script(prog, config or machine.generate_config())


def test_the_macro_refuses_a_qua_variable_as_volts(pair):
    with program():
        v = declare(fixed)
        with pytest.raises(TypeError, match="scale"):
            pair.macros[SWAP].apply(ctrl_amp=v)
        with pytest.raises(TypeError, match="scale"):
            pair.macros[SWAP].apply(cplr_amp=v)


def test_the_chain_shells_volts_become_a_constant(machine, pair):
    """``qc_unidirectional_trotter`` / ``qc_trotter_compensation`` hand the
    macro a Python float through ``swap_coupler_flux``: it must reach QUA as a
    literal, with the control's pulse still bare."""
    ref = float(pair.coupler.operations[pair.macros[SWAP].flux_pulse].amplitude)
    with program() as prog:
        pair.macros[SWAP].apply(cplr_amp=0.5 * ref)
    assert play_amplitudes(_script(machine, prog)) == ["0.5"]


def test_qc_swap_flux_stark_plays_plain_scales(machine, pair, stark):
    from scqo_qm.experiments.qc_swap_flux_stark import build_program

    volts = np.linspace(0.0, 0.05, 5)
    prog, axes = build_program(
        machine, [pair.qubit_control, pair.qubit_target], pair,
        swap_operation=SWAP, stark_operation=stark, stark_detuning_hz=50e6,
        swap_count=3, qubit_amplitudes=volts, stark_amps=np.linspace(0.0, 1.0, 5),
        num_shots=10, reset_type="thermal", use_state_discrimination=True,
        operation_gap_ns=20)
    assert_plain_plays(_script(machine, prog), "qc_swap_flux_stark")
    # the loop variable became a scale; the data axis still carries VOLTS
    np.testing.assert_allclose(axes["qubit_amplitude"].values, volts)


def test_qc_n_swap_amp_plays_plain_scales(machine, pair):
    from scqo_qm.experiments.qc_n_swap_amp import build_program

    volts = np.linspace(0.0, 0.05, 5)
    prog, axes = build_program(
        machine, [pair.qubit_control, pair.qubit_target], pair,
        swap_operation=SWAP, rounds_array=np.arange(0, 4), qubit_amplitudes=volts,
        num_shots=10, reset_type="thermal", use_state_discrimination=True,
        operation_gap_ns=20)
    assert_plain_plays(_script(machine, prog), "qc_n_swap_amp")
    np.testing.assert_allclose(axes["qubit_amplitude"].values, volts)


def test_pair_swap_angle_plays_plain_scales(machine, pair):
    from scqo_qm.experiments.pair_swap_angle import build_program

    volts = np.linspace(0.0, 0.02, 5)
    prog, axes = build_program(
        machine=machine, measure_qubits=[pair.qubit_control, pair.qubit_target],
        swap_pair=pair, swap_operation=SWAP, rounds_array=np.arange(0, 4),
        coupler_amplitudes=volts, num_shots=10, reset_type="thermal",
        operation_gap_ns=20)
    assert_plain_plays(_script(machine, prog), "pair_swap_angle")
    np.testing.assert_allclose(axes["coupler_amplitude"].values, volts)


@pytest.mark.parametrize("via_macro", [False, True])
def test_pair_swap_flux_map_divides_ahead_of_the_swap(machine, pair, via_macro):
    """Multiplexed pairs share the volts loop variable but not their references,
    so this map keeps a division -- as its own ``assign``, ahead of the reset."""
    from scqo_qm.experiments._lib import select_qubit_pairs
    from scqo_qm.experiments.pair_swap_flux_map import build_program

    prog, _axes = build_program(
        machine, select_qubit_pairs(machine, [PAIR]),
        coupler_amplitudes=np.linspace(-0.02, 0.02, 5),
        qubit_amplitudes=np.linspace(0.0, 0.02, 5),
        flux_time=None if via_macro else 44, amp_mode="absolute",
        num_shots=10, reset_type="thermal", use_state_discrimination=True,
        swap_via_macro=via_macro, swap_operation=SWAP)
    script = _script(machine, prog)
    assert_plain_plays(script, "pair_swap_flux_map")
    assert re.search(r"assign\(v\d+, \(v\d+/", script), \
        "the volts -> scale division is no longer assigned ahead of the swap"


@pytest.fixture
def square_swap(pair):
    """A swap macro whose coupler pulse is SQUARE -- the only shape the
    chevron's coupled path can stretch -- registered in memory and removed."""
    from scqo_qm.components.macros.iswap_macro import ISwapImplementation

    name = "test_square_swap"
    pair.macros[name] = ISwapImplementation(flux_pulse="const")
    yield name
    del pair.macros[name]


@pytest.mark.parametrize("coupled", [False, True])
def test_pair_swap_chevron_plays_plain_scales(machine, pair, square_swap, coupled):
    """The stretched branch's ``(base_level / denom) * a`` is assigned per pair
    ahead of the reset, on both paths -- on the coupled one it used to hold the
    z pulse back while the coupler pulse beside it, a constant, did not wait."""
    from scqo_qm.experiments._lib import select_qubit_pairs
    from scqo_qm.experiments.pair_swap_chevron import build_program

    extra = dict(coupler_amp=0.05, swap_operation=square_swap) if coupled else {}
    prog, _axes, config = build_program(
        machine, select_qubit_pairs(machine, [PAIR]),
        amplitudes=np.linspace(0.01, 0.05, 5),
        times_cycles=np.array([16, 20, 24, 28, 32]) if coupled else np.arange(1, 33),
        num_shots=10, reset_type="thermal", use_state_discrimination=True,
        amp_mode="absolute", **extra)
    script = _script(machine, prog, config)
    assert_plain_plays(script, "pair_swap_chevron")
    assert re.search(r"assign\(v\d+, \(-?[\d.e-]+\*v\d+\)\)", script), \
        "the stretched-branch scale is no longer assigned ahead of the swap"
