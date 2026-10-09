from quam.core import quam_dataclass
from quam.components.macro import QubitMacro

#: QUA's floor for ``wait()``, in ns. A shorter wait is not a short wait: it is a
#: compile error naming an internal variable, so the ring-down is checked here.
MIN_WAIT_NS = 16


@quam_dataclass
class ParametricReset(QubitMacro):
    """Parametric reset on a qubit's own flux (z) line.

    Mirrors the technique in `LCH_qubit_parametric_drive_{fixed,freq}_time`: set the z-line
    IF to a calibrated sideband/drive frequency, reset the IF phase for coherence, then play
    a dedicated `parametric_reset` z pulse (its amplitude and length live in the pulse op).
    Bare-callable so it works as a named macro: `qubit.macros["reset"].apply()`.

    The dedicated `parametric_reset` operation must exist on the qubit's z line
    (`qubit.z.operations["parametric_reset"]`). Amplitude/length intentionally live in that
    pulse op, not on this macro (optional `amplitude_scale`/`duration` overrides could be
    added later without breaking the bare `.apply()` call).

    AFTER THE PULSE the z line's oscillator goes back to the line's stored
    `intermediate_frequency`, so a later flux pulse on the same line in the same program
    plays as stored instead of modulated at the drive frequency. The line therefore needs
    a stored IF (0 for a plain flux line): with None its element has no oscillator at all,
    and `apply()` refuses by name.

    `ringdown_ns` is how long the qubit then waits. The pulse hands the qubit's excitation
    to its readout resonator as a photon, which leaves at kappa; until it has left, a gate
    or a readout on this qubit sees its dispersive shift. The wait covers every channel of
    the qubit, so whatever plays on it next starts after the ring-down. 0 (the default)
    waits nothing and returns as soon as the pulse ends. `apply(ringdown_ns=...)` replaces
    the stored wait for that one call: `qubit_ramsey_reset_ringdown` plays the reset with
    0 to measure what the wait has to be.
    """

    drive_frequency: int  # parametric drive frequency in Hz (z-line IF)
    flux_pulse: str = "parametric_reset"  # z operation to play
    ringdown_ns: int = 0  # wait after the pulse (ns): 0, or a multiple of 4 from 16 up

    def apply(self, *, ringdown_ns=None):
        # Both refusals fire before a single statement is emitted.
        rest_frequency = self._rest_frequency()
        ringdown_cycles = self._ringdown_cycles(
            self.ringdown_ns if ringdown_ns is None else ringdown_ns)

        z = self.qubit.z
        z.update_frequency(self.drive_frequency)
        z.reset_if_phase()
        z.play(self.flux_pulse)
        z.update_frequency(rest_frequency)
        if ringdown_cycles:
            # Align first: the pulse moved the z line's timeline only, and a wait
            # counts from each channel's own.
            self.qubit.align()
            self.qubit.wait(ringdown_cycles)

    def to_dict(self, *args, **kwargs) -> dict:
        """QUAM's dict of this macro, WITHOUT a zero ``ringdown_ns``.

        A reset that stores no ring-down then saves exactly as before the field existed,
        so an older scqo-qm still loads it, a setup snapshot of it still parses equal to
        its state.json, and a staged save (``scqo-qm register-partial-swap``,
        ``adopt-channel``) still finds that nothing else moved. The key appears once a
        ring-down is stored."""
        contents = super().to_dict(*args, **kwargs)
        if not self.ringdown_ns:
            contents.pop("ringdown_ns", None)
        return contents

    @property
    def inferred_duration(self) -> float:
        """How long the reset holds its qubit, in seconds: the pulse plus the ring-down."""
        pulse = self.qubit.z.operations[self.flux_pulse]
        return (pulse.length + self.ringdown_ns) * 1e-9

    def _rest_frequency(self) -> int:
        """The z line's stored IF, which the oscillator returns to after the pulse."""
        z = self.qubit.z
        if z.intermediate_frequency is None:
            raise ValueError(
                f"{z.name} has no intermediate_frequency, so its element has no oscillator: "
                f"the parametric reset cannot move it to {self.drive_frequency} Hz, and has "
                f"nothing to return it to afterwards. Set {z.name}.intermediate_frequency "
                f"to 0 in the QUAM state.")
        return int(z.intermediate_frequency)

    def _ringdown_cycles(self, ringdown_ns) -> int:
        """``ringdown_ns`` in 4 ns clock cycles; refuses one QUA cannot wait."""
        if ringdown_ns == 0:
            return 0
        if ringdown_ns < MIN_WAIT_NS or ringdown_ns % 4:
            raise ValueError(
                f"{self.qubit.name} parametric reset: ringdown_ns must be 0 or a multiple "
                f"of 4 ns of at least {MIN_WAIT_NS} ns (QUA's shortest wait), got "
                f"{ringdown_ns!r}.")
        return int(ringdown_ns) // 4
