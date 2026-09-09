"""The chain shells' round step: how long an ``idle`` step waits.

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

from ._coupler_knob import find_coupler_pulse

__all__ = ["idle_wait_cycles"]

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
