"""The swap ANGLE knob: resolving and guarding a pair macro's coupler pulse.

Three probes drive a swap through its coupler amplitude -- ``pair_swap_angle``
sweeps it, and both chain shells (``qc_unidirectional_trotter``,
``qc_trotter_compensation``) set it per pair through ``swap_coupler_flux`` -- so
the refusals live here once rather than three times.

WHY A ZERO STORED AMPLITUDE IS THE INTERESTING CASE. ``ISwapImplementation.apply``
turns a ``cplr_amp`` in volts into a QUA ``amplitude_scale`` by dividing by the
coupler pulse's own stored amplitude. A pair whose coupler pulse is baked at
0.0 V therefore cannot have its coupler driven at all -- and that is not a rare
misconfiguration, it is exactly the state of a chip whose swaps have only ever
been driven by detuning the control qubit into resonance, with the coupler parked
and never brought up. Caught here, it names ``register_flattop_cosine.py``;
uncaught, it is a division by zero (or an infinite amplitude_scale) surfacing as
a QUA build error about an internal variable.

The split from ``_flux_limits`` is the same one that module already states: this
answers "is there a coupler knob to turn, and is it turnable?", while
``check_flux_pulse_relative`` answers "can this PORT emit these volts?". Both run,
in that order -- there is no point checking the rail on a knob that does not exist.
"""

from __future__ import annotations

from typing import Iterable, Optional

from ._flux_limits import check_flux_pulse_relative, declared_idle_offset_v

__all__ = ["resolve_coupler_knob", "guard_coupler_amplitudes"]


def resolve_coupler_knob(swap_pair, swap_operation: str, *, why: str = ""):
    """``(coupler, flux_pulse_name)`` for a pair macro, or refuse BY NAME.

    Four separate failures, each with its own message because each has its own
    fix: no such macro (register it), no coupler on the pair (wrong experiment),
    no coupler-side flux pulse (the macro is not the shape this drives), and a
    coupler pulse baked at ZERO amplitude (bring the coupler up first).

    ``why`` is appended to the no-coupler message so each caller can say what it
    wanted the coupler FOR.
    """
    if swap_operation not in (getattr(swap_pair, "macros", {}) or {}):
        raise ValueError(
            f"Pair {swap_pair.name} has no macro {swap_operation!r}; available: "
            f"{sorted(getattr(swap_pair, 'macros', {}) or {})}. Register it first "
            f"(quam_config/register_swap_macro.py).")
    coupler = getattr(swap_pair, "coupler", None)
    if coupler is None:
        tail = f" {why}" if why else ""
        raise ValueError(
            f"Pair {swap_pair.name} has no coupler, so it has no swap ANGLE knob.{tail}")
    flux_pulse_name = getattr(swap_pair.macros[swap_operation], "flux_pulse", None)
    ops = getattr(coupler, "operations", {}) or {}
    if not isinstance(flux_pulse_name, str) or flux_pulse_name not in ops:
        raise ValueError(
            f"Macro {swap_operation!r} on {swap_pair.name} has no coupler flux_pulse "
            f"playable with cplr_amp (flux_pulse={flux_pulse_name!r}; coupler "
            f"operations: {sorted(ops)}).")
    stored = float(getattr(ops[flux_pulse_name], "amplitude", 0.0) or 0.0)
    if stored == 0.0:
        raise ValueError(
            f"{swap_pair.name}: the coupler pulse {flux_pulse_name!r} is baked at "
            f"amplitude 0.0, so the swap angle CANNOT be driven on this pair -- the "
            f"macro converts cplr_amp to an amplitude_scale by dividing by that "
            f"stored value. A zero here means the swap has only ever been driven by "
            f"detuning the control qubit and the coupler was never brought up. Fix: "
            f"re-run quam_config/register_flattop_cosine.py with a NONZERO coupler "
            f"amplitude for this pair (TUTORIAL section 12 step 2), then re-run.")
    return coupler, flux_pulse_name


def guard_coupler_amplitudes(swap_pair, swap_operation: str,
                             amps_v: Iterable[float], *,
                             why: str = "", label: Optional[str] = None):
    """Resolve the coupler knob AND check the volts it is asked to emit.

    ``amps_v`` is every coupler amplitude the program will play (one value for a
    per-pair setting, a whole sweep for ``pair_swap_angle``). The macro's coupler
    pulse is the amplitude_scale REFERENCE -- not a ``const``, so the rail/2
    convention deliberately does not apply to it -- and the volts are an
    excursion on top of whatever standing bias ``initialize_qpu`` applied, so the
    RELATIVE frame is the right one. The returned reference is unused by callers:
    the macro does its own ``cplr_amp/ref`` rescaling internally, and this call is
    for its refusals.
    """
    coupler, flux_pulse_name = resolve_coupler_knob(
        swap_pair, swap_operation, why=why)
    check_flux_pulse_relative(
        coupler,
        name=label or f"{swap_pair.name} macro {swap_operation!r} on its coupler",
        idle_v=declared_idle_offset_v(coupler),
        amps_v=list(amps_v),
        operation=flux_pulse_name,
    )
    return coupler, flux_pulse_name
