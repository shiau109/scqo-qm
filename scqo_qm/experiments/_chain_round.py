"""The chain shells' shared rules: the Trotter ROUND itself, how long an
``idle`` step waits, which qubits a shot resets, and the optional parametric
reset at the START of a shot (``check_init_resets`` / ``play_init_resets``,
scqo's ``init_reset_qubits``).

THE ROUND lives here, once (``play_chain_round``). ``qc_trotter_compensation``
exists to calibrate the exact round ``qc_unidirectional_trotter`` runs, so a
second copy of the round body would be a second sequence waiting to drift from
the first - and then the scan calibrates something the chain does not play.

WHY A GLOBAL ``align()`` AFTER EACH SWAP. The two pairs SHARE their relay member,
so their own ``apply()`` aligns overlap only partially: the first swap's align
covers source+relay, the second's covers relay+sink, and the sink's timeline is
not in the first. Relying on the pair aligns to serialize the round would make
correctness depend on which channels each macro happens to touch, so the round
is aligned explicitly.

WHERE THE STARK TONES GO is the one thing a run chooses (scqo's
``stark_timing``, a flag here). After the reset: an ``align()`` separates them
from the reset and its gap, and the round is strictly sequential. WITH the
reset: that one ``align()`` is left out, so the tones start at the instant the
reset macro does - they ride other elements (xy lines against the reset qubit's
z line), which is all QUA needs to play them together - and the round's closing
``align()`` waits for whichever ends later, the reset with its gap or the tones.
The gap stays the reset's in both: it lets a FLUX pulse settle before the next
round's swap, and a tone on an xy line has nothing to settle. The tones of one
round are concurrent with each other in both settings.

THE GAP AFTER THE RESET can be set apart from the other two (scqo's
``reset_gap_ns`` -> ``reset_gap_cycles`` here): it is the time between the
relay's reset and the next round's first swap, and a run that asks whether that
time matters must not move the gaps the swaps settle in. Unset, all three gaps
are ``operation_gap_ns`` and the round is the program it was.

Both chain probes (``qc_unidirectional_trotter``, ``qc_trotter_compensation``)
accept ``operation="idle"`` on either of their two round steps. An idle step
plays no pulse and waits instead, for exactly as long as that pair's swap would
have taken -- so an idle round and a swapping round have the SAME duration and
the two runs differ by the exchange alone. That is the whole point of the control
arm: a shorter round would change every phase the chain accumulates and the
comparison would no longer be about the swap.

The duration therefore has to come from a REAL registered operation --
``idle_reference_operation`` names it -- and resolving it lives here rather than
in each probe, because two copies of a "how many clock cycles?" rule are two
chances to get it wrong.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Optional

from qm.qua import align

from ._coupler_knob import find_coupler_pulse

__all__ = ["chain_reset_qubits", "check_init_resets", "check_tones_clear_of_reset",
           "idle_wait_cycles", "play_chain_round", "play_init_resets",
           "reset_gap_cycles"]

#: QUA's floor for ``wait()``. A shorter wait is not a short wait -- it is a
#: compile error naming an internal variable, which is how the xy-z delay shell
#: shipped a bug that looked like anything but a too-short idle.
MIN_WAIT_CYCLES = 4


def idle_wait_cycles(pair, reference_operation: str) -> int:
    """Clock cycles an ``idle`` step on ``pair`` waits, or refuse BY NAME.

    ``ISwapImplementation`` plays ONE named ``flux_pulse`` on both the control
    qubit's z line and the coupler, and each channel holds its own copy of that
    operation. They are meant to agree (``register_swap_macro.py`` says the
    control copy "must match the swap duration"), so the honest occupancy of the
    step is the LONGER of the two: taking the control copy alone would under-wait
    a chip whose coupler pulse is longer and silently shorten the round this is
    supposed to leave untouched.
    """
    macros = getattr(pair, "macros", {}) or {}
    if reference_operation not in macros:
        raise ValueError(
            f"Pair {pair.name} has no macro {reference_operation!r}, so an "
            f"'idle' step on it has no duration to copy; available: "
            f"{sorted(macros)}. Either register it (quam_config/"
            f"register_swap_macro.py) or point idle_reference_operation at an "
            f"operation this pair does carry.")
    macro = macros[reference_operation]

    lengths: list[int] = []
    flux_pulse = getattr(macro, "flux_pulse", None)
    control_z = getattr(getattr(pair, "qubit_control", None), "z", None)
    control_ops = getattr(control_z, "operations", {}) or {}
    if isinstance(flux_pulse, str) and flux_pulse in control_ops:
        lengths.append(int(getattr(control_ops[flux_pulse], "length", 0) or 0))
    coupler_pulse = find_coupler_pulse(macro, getattr(pair, "coupler", None))
    if coupler_pulse is not None:
        lengths.append(int(getattr(coupler_pulse, "length", 0) or 0))

    length_ns = max(lengths, default=0)
    if length_ns <= 0:
        raise ValueError(
            f"Macro {reference_operation!r} on {pair.name} has no flux pulse with "
            f"a length (flux_pulse={flux_pulse!r}), so an 'idle' step on this "
            f"pair cannot be made to last as long as its swap. Name an operation "
            f"whose pulse is registered on the control z line or the coupler.")
    if length_ns % 4:
        raise ValueError(
            f"Macro {reference_operation!r} on {pair.name} has a {length_ns} ns "
            f"flux pulse, which is not a multiple of the 4 ns clock — an 'idle' "
            f"step cannot match it exactly. Re-register the pulse on the grid.")
    cycles = length_ns // 4
    if cycles < MIN_WAIT_CYCLES:
        raise ValueError(
            f"Macro {reference_operation!r} on {pair.name} has a {length_ns} ns "
            f"flux pulse, below QUA's {MIN_WAIT_CYCLES * 4} ns minimum wait, so "
            f"an 'idle' step on this pair cannot be expressed at all.")
    return cycles


def reset_gap_cycles(reset_gap_ns: Optional[int], gap_cycles: int) -> int:
    """Clock cycles a round idles after the relay's reset, or refuse BY NAME.

    ``reset_gap_ns`` is scqo's setting of that one gap: None leaves it at the
    round's common gap (``gap_cycles``, from ``operation_gap_ns``), 0 removes it,
    and any other value has to be a wait QUA can play.
    """
    if reset_gap_ns is None:
        return int(gap_cycles)
    if reset_gap_ns != 0 and (reset_gap_ns < MIN_WAIT_CYCLES * 4 or reset_gap_ns % 4):
        raise ValueError(
            f"reset_gap_ns must be 0 or a multiple of 4 ns of at least "
            f"{MIN_WAIT_CYCLES * 4} ns (QUA's shortest wait), got {reset_gap_ns!r}.")
    return int(reset_gap_ns) // 4


def check_tones_clear_of_reset(reset_qubit, toned: Iterable) -> None:
    """Refuse BY NAME a stark tone on the reset qubit when the tones play WITH
    its reset. ``toned`` is every qubit that plays a tone, a swept one included.

    After the reset such a tone is inert - it meets a qubit that has just been
    emptied. During the reset it Stark-shifts the qubit the reset is acting on,
    and a parametric reset is a narrow resonance: the reset would quietly stop
    emptying the relay, and nothing in the data would name the tone. scqo
    refuses this upstream; this is the backstop for a direct builder call.
    """
    if any(qubit.name == reset_qubit.name for qubit in toned):
        raise ValueError(
            f"{reset_qubit.name} is the reset qubit and also plays a stark tone, "
            f"while the tones play WITH the reset (stark_timing='with_reset'): the "
            f"tone would shift it off the reset's resonance during its own reset. "
            f"Remove it from compensation_amps (and do not sweep it), or play the "
            f"tones after the reset (stark_timing='after_reset').")


def check_init_resets(qubits: Iterable, reset_operation: str,
                      ringdown_ns: int) -> None:
    """Refuse BY NAME what ``play_init_resets`` could not play, before any QUA.

    ``qubits`` are the ones scqo's ``init_reset_qubits`` names. Each needs the
    stored reset macro, and one whose ring-down a call can set: the wait before
    the prep is the run's, not the one stored for the rounds. A macro of
    another kind under that name is refused rather than played with whatever
    wait it has.
    """
    qubits = list(qubits)
    if not qubits:
        return
    if ringdown_ns != 0 and (ringdown_ns < MIN_WAIT_CYCLES * 4 or ringdown_ns % 4):
        raise ValueError(
            f"init_reset_ringdown_ns must be 0 or a multiple of 4 ns of at least "
            f"{MIN_WAIT_CYCLES * 4} ns (QUA's shortest wait), got {ringdown_ns!r}.")
    for qubit in qubits:
        macros = getattr(qubit, "macros", {}) or {}
        if reset_operation not in macros:
            raise ValueError(
                f"init_reset_qubits names {qubit.name}, which has no macro "
                f"{reset_operation!r}; available: {sorted(macros)}. Store its reset "
                f"tone first (quam_config/register_reset_macro.py; SCQO "
                f"procedures/qubit-resonator-sideband-swap, Step 5), or leave the "
                f"qubit out.")
        macro = macros[reset_operation]
        if not hasattr(macro, "ringdown_ns"):
            raise ValueError(
                f"init_reset_qubits names {qubit.name}, whose macro "
                f"{reset_operation!r} is a {type(macro).__name__}, not a parametric "
                f"reset: the reset at the start of a shot is played with "
                f"init_reset_ringdown_ns as its ring-down, which only a "
                f"ParametricReset takes.")


def play_init_resets(qubits: Iterable, reset_operation: str,
                     ringdown_ns: int) -> None:
    """Emit the reset at the START of a shot: every named qubit's stored reset,
    all starting together, then one global ``align()``.

    Called after the between-shots reset and its align, before the prep. Each
    macro plays on its own qubit's z line, so nothing orders them against each
    other, and each waits ``ringdown_ns`` on its own qubit in place of its
    stored ring-down. The closing align makes the prep wait for the longest of
    them, so every reset is followed by at least that ring-down.

    With no qubit named this emits NOTHING - not even the align - so a run that
    does not ask for it is the program it was, statement for statement.
    """
    qubits = list(qubits)
    if not qubits:
        return
    for qubit in qubits:
        qubit.macros[reset_operation].apply(ringdown_ns=int(ringdown_ns))
    align()


def play_chain_round(
    first_pair,
    second_pair,
    reset_qubit,
    *,
    first_operation: Optional[str],
    second_operation: Optional[str],
    idle_cycles: Mapping[str, int],
    first_swap_kwargs: Mapping,
    second_swap_kwargs: Mapping,
    reset_operation: str,
    stark_operation: str,
    tones: Sequence[tuple],
    gap_cycles: int,
    stark_with_reset: bool,
    after_reset_cycles: Optional[int] = None,
) -> None:
    """Emit ONE Trotter round: first step, second step, relay reset, stark tones.

    Called inside the caller's round loop; everything that can refuse has
    already refused (the callers' guard blocks run before any QUA is built).

    A step whose operation is ``None`` is IDLE: it waits ``idle_cycles[<step>]``
    on the SAME channels the swap would have occupied (``pair.wait`` covers both
    members and the coupler), so an idle round and a swapping round share a
    timeline and only the exchange differs. A pair with swap kwargs is driven at
    that coupler amplitude (the angle knob); empty kwargs play the macro bare.

    ``tones`` is ``(qubit, amplitude_scale)`` per stark tone, in play order; the
    scale may be a Python float or a QUA variable (the compensation scan's swept
    one). ``stark_with_reset`` picks where they go - see the module docstring.
    Ends with a global ``align()``.

    ``gap_cycles`` idles after each of the three operations. ``after_reset_cycles``
    replaces it after the relay's reset alone (``reset_gap_cycles`` resolves it
    from scqo's ``reset_gap_ns``); None leaves all three alike.
    """
    if first_operation is None:
        first_pair.wait(idle_cycles["first_pair"])
    else:
        first_pair.macros[first_operation].apply(**first_swap_kwargs)
    if gap_cycles > 0:
        first_pair.wait(gap_cycles)
    align()
    if second_operation is None:
        second_pair.wait(idle_cycles["second_pair"])
    else:
        second_pair.macros[second_operation].apply(**second_swap_kwargs)
    if gap_cycles > 0:
        second_pair.wait(gap_cycles)
    align()
    reset_qubit.macros[reset_operation].apply()
    if after_reset_cycles is None:
        after_reset_cycles = gap_cycles
    if after_reset_cycles > 0:
        reset_qubit.wait(after_reset_cycles)
    # The one statement stark_timing decides. With no tone there is nothing to
    # move, so the two settings stay the same program, statement for statement.
    if not (stark_with_reset and tones):
        align()
    # Concurrent by design: one compensation instant per round.
    for qubit, amplitude_scale in tones:
        qubit.xy.play(stark_operation, amplitude_scale=amplitude_scale)
    align()


def chain_reset_qubits(params, source: str, relay: str, sink: str,
                       prep: Iterable[str]) -> list[str]:
    """Every qubit a chain shot RESETS, by roster name, targets first.

    ``prep`` is the resolved prep map from scqo's ``chain_roles`` (or any
    iterable of its qubit names). The builders reset the measured qubits, both
    pairs' members, the reset qubit and every prep qubit — whether or not all
    of them are read out. An
    ACTIVE reset thresholds each one against its own readout discriminator, so
    ``check_reset_method`` must see this list rather than the targets alone: a
    sink left out of ``targets`` is still reset, and an uncalibrated threshold
    there would die inside the QUA DSL naming nothing.
    """
    names = [*params.targets, source, relay, sink, params.reset_qubit, *prep]
    return list(dict.fromkeys(names))
