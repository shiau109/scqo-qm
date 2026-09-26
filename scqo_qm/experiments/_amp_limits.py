"""What QUA's dynamic ``amp()`` can express — the DRIVE/READOUT amplitude bound.

``play("op" * amp(v))`` scales the stored waveform by a fixed-point multiplier
that is only defined on ``(-2, 2)``. A sweep that asks for more is a BUILD-TIME
error on the QOP compiler, i.e. after instrument time has already been booked —
and the simulator shows nothing, so it looks like a hardware mystery rather than
a parameter mistake.

The flux probes have guarded this for a while (``_flux_limits``); the
drive/readout amplitude sweeps did not, which is why this module exists.

:data:`MAX_AMP_SCALE` is the ONE home for the bound — ``_flux_limits`` and the
four flux/parametric probes that each carried their own ``_MAX_AMP_SCALE = 2.0``
now import it from here. It is a property of ``amp()``, not of any port, so a
per-probe copy could drift from the vendor while every test still passed. (Same
rule ``_flux_limits`` already states for the DAC rail: no probe carries its own
constant.)

The split with ``_flux_limits`` is by WHAT IS BEING BOUNDED: this module answers
"can ``amp()`` represent this multiplier?", while the rail, the ``const``
convention and the idle-sum arithmetic answer "can this PORT emit these volts?".
"""

from typing import Iterable

import numpy as np

__all__ = ["MAX_AMP_SCALE", "check_amp_scale_window"]

#: QUA's bound on a DYNAMIC amplitude_scale. The fixed-point multiplier spans
#: ``[-2, 2 - 2**-16]``, so 2.0 itself is already unrepresentable — the check is
#: ``>=``, not ``>``.
MAX_AMP_SCALE = 2.0


def check_amp_scale_window(prefactors: Iterable[float], *, name: str,
                           knob: str = "start_amp_factor/end_amp_factor") -> None:
    """Refuse a prefactor sweep QUA cannot express, BY NAME and before building.

    Args:
        prefactors: the swept amplitude factors (scqo's ``amp_prefactor`` axis).
        name: the qubit/target the window belongs to, for the message.
        knob: the neutral parameter the caller should lower — naming the knob is
            the point, because the compiler's own error names an internal QUA
            variable the operator has never heard of.

    Raises:
        ValueError: when any ``|factor| >= 2``.
    """
    values = np.asarray(list(prefactors), dtype=float)
    if values.size == 0:
        return
    peak = float(np.max(np.abs(values)))
    if peak >= MAX_AMP_SCALE:
        raise ValueError(
            f"{name}: amplitude factor {peak:.4f} is outside QUA's "
            f"+/-{MAX_AMP_SCALE} amplitude_scale range, so the program cannot be "
            f"built. Lower {knob} below {MAX_AMP_SCALE}."
        )
