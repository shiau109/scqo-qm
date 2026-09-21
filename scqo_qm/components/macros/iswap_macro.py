import numbers

from quam.core import quam_dataclass
from quam.components.pulses import Pulse
# from quam.components.macro import QubitPairMacro
from scqo_qm.components.macros.two_qubit_pair_macro import QubitPairMacro


def resolve_amplitude_scale(amp=None, scale=None, *, reference: float, what: str):
    """The ``amplitude_scale`` one element of the swap plays at, or None for bare.

    Two ways to rescale, and the difference between them is WHERE the arithmetic
    runs:

    * ``amp`` is an absolute amplitude in VOLTS and must be a Python number. The
      conversion ``amp / reference`` happens here, in Python, so the play carries
      a compile-time constant.
    * ``scale`` is the ``amplitude_scale`` itself, used verbatim. This is the one
      that takes a QUA variable -- a probe that sweeps the amplitude in real time
      precomputes ``volts / reference`` and hands the result over.

    A QUA variable as ``amp`` is REFUSED. Dividing it would run on the FPGA inside
    every round, between the round's ``align()`` and this play. On 5Q4C q1_q2
    (2026-09-21) a map that did it (``qc_swap_flux_stark``
    ``20260921-123347-429``) found its per-round phase about 0.40 turn away from a
    bare-gate map at the same flux (``qc_n_stark_amp`` ``20260921-124233-223``) --
    the size of two clock cycles of round time at the pair's 301 MHz detuning. A
    calibration taken that way does not transfer to the gate it calibrates.
    """
    if amp is not None and scale is not None:
        raise ValueError(
            f"{what}: pass an absolute amplitude OR an amplitude_scale, not both "
            f"(got amp={amp!r}, scale={scale!r}).")
    if scale is not None:
        return scale
    if amp is None:
        return None
    if not isinstance(amp, numbers.Real):
        raise TypeError(
            f"{what}: the absolute amplitude must be a Python number, got "
            f"{type(amp).__name__}. Converting a QUA variable to an amplitude_scale "
            f"would divide on the FPGA inside every round and shift the round's "
            f"timing; precompute volts / {reference} in Python and pass the result "
            f"as the scale instead.")
    if float(reference) == 0.0:
        raise ValueError(
            f"{what}: the stored pulse amplitude is 0.0, so an absolute amplitude "
            f"cannot be converted to an amplitude_scale.")
    return float(amp) / float(reference)


@quam_dataclass
class ISwapImplementation(QubitPairMacro):
    """ISWAP Operation for a qubit pair"""

    # flux_pulse: Pulse
    flux_pulse: str

    phase_shift_control: float = 0.0
    phase_shift_target: float = 0.0

    def apply(self, *, ctrl_amp=None, cplr_amp=None, ctrl_scale=None, cplr_scale=None):
        """Play the swap: the control's flux pulse and the coupler's, then align the pair.

        A bare call plays both at their stored amplitudes -- the gate itself.
        Each element may be rescaled by ``*_amp`` (absolute volts, Python numbers
        only) or by ``*_scale`` (an amplitude_scale, which may be a QUA variable);
        see :func:`resolve_amplitude_scale` for why a QUA variable is refused as
        a volts value.
        """
        ctrl = resolve_amplitude_scale(
            ctrl_amp, ctrl_scale,
            reference=self.qubit_control.z.operations[self.flux_pulse].amplitude,
            what=f"{self.qubit_control.name}.z {self.flux_pulse!r}")
        cplr = resolve_amplitude_scale(
            cplr_amp, cplr_scale,
            reference=self.coupler.operations[self.flux_pulse].amplitude,
            what=f"coupler {self.flux_pulse!r}")

        if ctrl is None:
            self.qubit_control.z.play(self.flux_pulse)
        else:
            self.qubit_control.z.play(self.flux_pulse, amplitude_scale=ctrl)

        if cplr is None:
            self.coupler.play(self.flux_pulse)
        else:
            self.coupler.play(self.flux_pulse, amplitude_scale=cplr)

        # wait(operation_gap_ns//4)

        self.qubit_pair.align()

        # Copy from quam
        # self.flux_pulse.play(amplitude_scale=amplitude_scale)
        # self.qubit_control.align(self.qubit_target)
        # self.qubit_control.xy.frame_rotation(self.phase_shift_control)
        # self.qubit_target.xy.frame_rotation(self.phase_shift_target)
        # self.qubit_pair.align()
