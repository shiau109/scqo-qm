"""
This script is use to set instrument limits and prevent it from outputting too much power, saturating and overflowing
in the case of multiplexing. These setpoints are used in the analysis sections of the nodes in order to cap the
waveform amplitude to its limit.
Feel free to add additional limits as you see fit.

The LIMITS below are node policy (subjective "safe" headroom for the qualibrate
path); WHICH FAMILY a channel belongs to is not, and is answered by
``scqo_qm._family.rf_chain`` so this module and the driver cannot drift into two
different opinions. That also makes it work on the plain stubs the driver's tests
build, which an ``isinstance`` check against the quam classes never could.
"""

from dataclasses import dataclass
from typing import Any

from scqo_qm._family import RF_MW_FEM, RF_OCTAVE, rf_chain


@dataclass(frozen=True)
class InstrumentLimits:
    max_wf_amplitude: float
    max_x180_wf_amplitude: float
    max_readout_amplitude: float
    units: str


def instrument_limits(channel: Any) -> InstrumentLimits:
    chain = rf_chain(channel)
    if chain not in (RF_MW_FEM, RF_OCTAVE):
        raise TypeError(
            f"Expected a channel on an MW-FEM or Octave RF chain, got {chain!r} "
            f"for {type(channel)}."
        )

    if chain == RF_MW_FEM:
        limits = InstrumentLimits(
            # MW-FEM max normalized amplitude
            max_wf_amplitude=1,
            # A subjective "safe" value for x180 pulses
            max_x180_wf_amplitude=0.6,
            # A subjective "safe" value assuming up to 10 qubits on the same channel
            max_readout_amplitude=0.1,
            units="(scaled by `full_scale_power_dbm`)",
        )
    else:
        limits = InstrumentLimits(
            # The OPX+ DAC rail feeding the Octave (an LF-FEM in "direct" mode
            # reaches the same 0.5 V; in "amplified" it reaches 2.5 V, but a flux
            # line is not what this function is asked about)
            max_wf_amplitude=0.5,
            # A subjective "safe" value for x180 pulses
            max_x180_wf_amplitude=0.3,
            # A subjective "safe" value assuming up to 10 qubits on the same channel
            max_readout_amplitude=0.05,
            units="V",
        )

    return limits
