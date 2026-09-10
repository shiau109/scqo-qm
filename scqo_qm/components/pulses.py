from quam.core import QuamComponent, quam_dataclass
from quam.components.pulses import Pulse, ReadoutPulse

import numpy as np
from typing import List, Tuple, Union

@quam_dataclass
class RampPulse(Pulse):
    """Gaussian pulse with flat top QUAM component.

    Args:
        length (int): The total length of the pulse in samples.
        amplitude (float): The amplitude of the pulse in volts.
        axis_angle (float, optional): IQ axis angle of the output pulse in radians.
            If None (default), the pulse is meant for a single channel or the I port
                of an IQ channel
            If not None, the pulse is meant for an IQ channel (0 is X, pi/2 is Y).

    """

    axis_angle: float = None
    start_value: float = 0
    end_value: float = 0

    def waveform_function(self):

        waveform = np.linspace(self.start_value, self.end_value, self.length)


        if self.axis_angle is not None:
            waveform = waveform * np.exp(1j * self.axis_angle)

        return waveform


@quam_dataclass
class ParaCosinePulse(Pulse):
    """Cosine pulse QUAM component.

    Args:
        length (int): The total length of the pulse in samples.
        amplitude (float): The amplitude of the pulse in volts.
        frequency (float): The frequency of the cosine wave in units of cycles per sample.
        phase (float): The phase offset of the cosine wave in radians.

    """

    amplitude: float = 0.1
    frequency: float = 0.1
    phase: float = 0.0

    def waveform_function(self):
        # Generate time array
        t = np.linspace( 0, self.length ,self.length, endpoint=False)  # An array of size pulse length in ns
        
        # Generate cosine waveform
        waveform = self.amplitude * np.cos(2 * np.pi * self.frequency * t + self.phase)

        return waveform


@quam_dataclass
class CascadeFlatTopGaussianPulse(Pulse):
    """Gaussian pulse with flat top QUAM component.

    Args:
        length (int): The total length of the pulse in samples.
        amplitude (float): The amplitude of the pulse in volts.
        axis_angle (float, optional): IQ axis angle of the output pulse in radians.
            If None (default), the pulse is meant for a single channel or the I port
                of an IQ channel
            If not None, the pulse is meant for an IQ channel (0 is X, pi/2 is Y).
        flat_length (int): The length of the pulse's flat top in samples.
            The rise and fall lengths are calculated from the total length and the
            flat length.
    """

    amplitude: float
    axis_angle: float = None
    flat_length: int
    def waveform_function(self):
        from qualang_tools.config.waveform_tools import flattop_gaussian_waveform

        rise_fall_length = (self.length - self.flat_length) // 2
        if not self.flat_length + 2 * rise_fall_length == self.length:
            raise ValueError(
                "FlatTopGaussianPulse rise_fall_length (=length-flat_length) must be"
                f" a multiple of 2 ({self.length} - {self.flat_length} ="
                f" {self.length - self.flat_length})"
            )

        waveform = flattop_gaussian_waveform(
            amplitude=self.amplitude,
            flat_length=self.flat_length,
            rise_fall_length=rise_fall_length,
            return_part="all",
        )
        waveform = np.array(waveform)

        if self.axis_angle is not None:
            waveform = waveform * np.exp(1j * self.axis_angle)

        return waveform


@quam_dataclass
class TwoStepPulse(Pulse):
    """Two-step function pulse QUAM component.

    Args:
        length (int): The total length of the pulse in samples.
        A1 (float): The amplitude of the first step in volts.
        A2 (float): The amplitude of the second step in volts.
        W1 (int): The length of the first amplitude step in samples.
            The second step length is calculated as (length - W1).
        axis_angle (float, optional): IQ axis angle of the output pulse in radians.
            If None (default), the pulse is meant for a single channel or the I port
                of an IQ channel
            If not None, the pulse is meant for an IQ channel (0 is X, pi/2 is Y).

    """

    up_amp_ratio: float = 5.0
    amplitude: float = 0.1
    up_width: int = 5
    axis_angle: float = None

    def waveform_function(self):
        if self.up_width > self.length:
            raise ValueError(
                f"First step length W1 ({self.up_width}) cannot be greater than total "
                f"pulse length ({self.length})"
            )

        # Create the two-step waveform
        waveform = np.concatenate([
            np.full(self.up_width, self.up_amp_ratio * self.amplitude),  # First step with amplitude A1
            np.full(self.length - self.up_width, self.amplitude)  # Second step with amplitude A2
        ])

        if self.axis_angle is not None:
            waveform = waveform * np.exp(1j * self.axis_angle)

        return waveform

@quam_dataclass
class TwoStepReadoutPulse(ReadoutPulse, TwoStepPulse):
    """Two-step function pulse QUAM component.

    Args:
        length (int): The total length of the pulse in samples.
        up_amp_ratio (float): The amplitude ratio of the first step to the second step.
        amplitude (float): The amplitude of the second step in volts.
        up_width (int): The length of the first amplitude step in samples.
            The second step length is calculated as (length - up_width).
        axis_angle (float, optional): IQ axis angle of the output pulse in radians.
            If None (default), the pulse is meant for a single channel or the I port
                of an IQ channel
            If not None, the pulse is meant for an IQ channel (0 is X, pi/2 is Y).

    """

    pass


@quam_dataclass
class ThreeStepPulse(Pulse):
    """Three-step function pulse QUAM component.

    Args:
        length (int): The total length of the pulse in samples.
        up_amp_ratio (float): The amplitude ratio of the first step to the second step.
        amplitude (float): The amplitude of the second (middle) step in volts.
        down_amp_ratio (float): The amplitude ratio of the third step to the second step.
        up_width (int): The length of the first step in samples.
        down_width (int): The length of the third step in samples.
            The second step length is calculated as (length - up_width - down_width).
        axis_angle (float, optional): IQ axis angle of the output pulse in radians.
            If None (default), the pulse is meant for a single channel or the I port
                of an IQ channel
            If not None, the pulse is meant for an IQ channel (0 is X, pi/2 is Y).

    """

    up_amp_ratio: float = 5.0
    amplitude: float = 0.1
    down_amp_ratio: float = 0.5
    up_width: int = 5
    down_width: int = 5
    axis_angle: float = None

    def waveform_function(self):
        mid_width = self.length - self.up_width - self.down_width
        if mid_width < 0:
            raise ValueError(
                f"up_width ({self.up_width}) + down_width ({self.down_width}) "
                f"cannot be greater than total pulse length ({self.length})"
            )

        # Create the three-step waveform
        waveform = np.concatenate([
            np.full(self.up_width, self.up_amp_ratio * self.amplitude),
            np.full(mid_width, self.amplitude),
            np.full(self.down_width, self.down_amp_ratio * self.amplitude),
        ])

        if self.axis_angle is not None:
            waveform = waveform * np.exp(1j * self.axis_angle)

        return waveform


@quam_dataclass
class ThreeStepReadoutPulse(ReadoutPulse, ThreeStepPulse):
    """Three-step function readout pulse QUAM component.

    Args:
        length (int): The total length of the pulse in samples.
        up_amp_ratio (float): The amplitude ratio of the first step to the second step.
        amplitude (float): The amplitude of the second (middle) step in volts.
        down_amp_ratio (float): The amplitude ratio of the third step to the second step.
        up_width (int): The length of the first step in samples.
        down_width (int): The length of the third step in samples.
            The second step length is calculated as (length - up_width - down_width).
        axis_angle (float, optional): IQ axis angle of the output pulse in radians.
            If None (default), the pulse is meant for a single channel or the I port
                of an IQ channel
            If not None, the pulse is meant for an IQ channel (0 is X, pi/2 is Y).

    """
    # pass
    integration_weights: Union[List[float], List[Tuple[float, int]]] = (
        "#./default_integration_weights"
    )
    integration_weights_angle: float = 0

    @property
    def default_integration_weights(self) -> List[Tuple[float, int]]:
        return [(1, self.length - self.down_width), (0, self.down_width)]


@quam_dataclass
class FlatTopCosinePulse(Pulse):
    """Flat-top pulse with sine (raised-cosine) edges QUAM component.

    The waveform is composed of three parts: a rising edge that follows a sine
    from 0 to pi/2 (0 -> amplitude), a constant flat top at ``amplitude``, and a
    falling edge that follows a sine from pi/2 to pi (amplitude -> 0).

    Args:
        length (int): The total length of the pulse in samples.
        amplitude (float): The peak (flat-top) amplitude of the pulse in volts.
        edge_width (int): The length of each sine edge in samples. The edge spans
            a quarter of the sine period (pi/2 radians). The flat-top length is
            calculated as (length - 2 * edge_width).
        axis_angle (float, optional): IQ axis angle of the output pulse in radians.
            If None (default), the pulse is meant for a single channel or the I port
                of an IQ channel
            If not None, the pulse is meant for an IQ channel (0 is X, pi/2 is Y).

    """

    # keep < 0.5: the LF-FEM "direct" rail AND the OPX+ rail (which has no other
    # mode); >= 0.5 is clipped on HW, and the simulator shows nothing.
    amplitude: float = 0.25
    edge_width: int = 5
    axis_angle: float = None

    def waveform_function(self):
        flat_length = self.length - 2 * self.edge_width
        if flat_length < 0:
            raise ValueError(
                f"edge_width ({self.edge_width}) * 2 cannot be greater than total "
                f"pulse length ({self.length})"
            )

        rise = np.sin(np.linspace(0, np.pi / 2, self.edge_width, endpoint=False))
        flat = np.ones(flat_length)
        fall = np.sin(np.linspace(np.pi / 2, np.pi, self.edge_width, endpoint=True))

        waveform = self.amplitude * np.concatenate([rise, flat, fall])

        if self.axis_angle is not None:
            waveform = waveform * np.exp(1j * self.axis_angle)

        return waveform


if __name__ == "__main__":
    # # Example usage
    # pulse = RampPulse(length=10, start_value=0, end_value=0.5)
    # print(pulse.waveform_function())
    # print(pulse.__class__.__name__)
    
    # # Example usage for CosinePulse
    # cosine_pulse = ParaCosinePulse(length=20, amplitude=0.5, frequency=0.1, phase=0.0)
    # print(cosine_pulse.waveform_function())
    # print(cosine_pulse.__class__.__name__)
    
    # Example usage for TwoStepPulse
    two_step_pulse = TwoStepReadoutPulse(length=10, up_amp_ratio=3.0, amplitude=0.2, up_width=2)
    print(two_step_pulse.waveform_function())
    print(two_step_pulse.__class__.__name__)

    # Example usage for ThreeStepReadoutPulse
    three_step_pulse = ThreeStepReadoutPulse(length=10, up_amp_ratio=3.0, amplitude=0.2, down_amp_ratio=-3.0, up_width=2, down_width=2)
    print(three_step_pulse.waveform_function())
    print(three_step_pulse.__class__.__name__)

    # Example usage for FlatTopCosinePulse
    flat_top_cosine_pulse = FlatTopCosinePulse(length=20, amplitude=0.2, edge_width=4)
    print(flat_top_cosine_pulse.waveform_function())
    print(flat_top_cosine_pulse.__class__.__name__)
