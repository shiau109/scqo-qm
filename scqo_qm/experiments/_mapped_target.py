"""A transmon-SHAPED handle for a target driven and/or read through someone else's
wiring - the probe half of SCQO docs/coupler-transmon-plan.md (section 4).

A qubit probe's ``build_program`` touches a handful of things on each target:
``name``; ``xy`` (it plays on it); ``reset(...)``; ``readout_state(...)`` or
``resonator.measure(...)``; and, through ``machine.initialize_qpu(target=...)``,
``z`` and ``align()``. :class:`MappedTarget` answers exactly those, so a carrier's
program body does not change - only how it selects its targets
(:func:`mapped_targets` instead of ``_lib.select_qubits``):

* ``xy`` is the channel that DRIVES the target: the element the QUAM state adopted
  for ``<drive_line>.<target>`` (``machine.borrowed_channels["xy2.q1_q2_c"]``);
* the READ goes through ``reader``: the target itself (``drive_line`` alone, e.g. a
  qubit driven through a neighbour's line), or - for a MAPPED readout - the pair
  member, after the map copies the coupler's state onto it: the member's SELECTIVE
  pi (its ``saturation``, scaled to the x180's rotation area and stretched to
  ``selective_pi_len_ns``; ``_selective_pi``), then the member's x180;
* ``reset`` is the reader's (thermal: the member's wait covers the coupler's far
  shorter T1; an active reset would reset the member, not the coupler, and the
  SCQO Parameters refuse it);
* ``z`` is None for a mapped target: v1 plays nothing on the target's flux line,
  and ``initialize_qpu`` then skips the settle, which the reset wait covers.

Anything else raises AttributeError naming this handle: a carrier that needs more
extends it, and finds out when its program is BUILT, never on the instrument.
"""

from __future__ import annotations

from typing import Any, Optional

from qm.qua import align

from scqo_qm.experiments._selective_pi import PI_REFERENCE, SQUARE_OPERATION

_CLOCK_NS = 4


def routes_through(params: Any) -> bool:
    """True when the Parameters drive or read the target through another route
    than its own (the ``drive_line`` / ``mapped_readout`` capabilities)."""
    return (getattr(params, "drive_line", None) is not None
            or getattr(params, "readout_member", None) is not None)


class MappedTarget:
    """One target as a qubit probe sees it (see the module docstring)."""

    def __init__(self, name: str, xy: Any, reader: Any, *,
                 map_scale: Optional[float] = None,
                 map_cycles: Optional[int] = None) -> None:
        self.name = name
        self.xy = xy
        self.reader = reader
        self._map = None if map_scale is None else (float(map_scale), int(map_cycles))
        self.z = None if self._map is not None else getattr(reader, "z", None)

    @property
    def mapped(self) -> bool:
        return self._map is not None

    @property
    def resonator(self) -> Any:
        if self.mapped:
            raise AttributeError(
                f"{self.name}: read through {self.reader.name}'s map - a mapped "
                f"target has no resonator of its own, and it is read discriminated "
                f"only (readout_state)")
        return self.reader.resonator

    def _channel_names(self) -> list[str]:
        return [self.xy.name, *(ch.name for ch in self.reader.channels.values())]

    def align(self, *_others: Any) -> None:
        """The driving element and every channel of the reader, together."""
        align(*self._channel_names())

    def reset(self, reset_type: str = "thermal", simulate: bool = False, *args: Any,
              **kwargs: Any) -> None:
        if self.mapped and reset_type != "thermal":
            raise ValueError(
                f"{self.name}: a mapped target resets thermally only - an "
                f"{reset_type!r} reset would reset {self.reader.name}, not {self.name}")
        self.reader.reset(reset_type, simulate, *args, **kwargs)

    def readout_state(self, state: Any, pulse_name: str = "readout",
                      threshold: Optional[float] = None) -> None:
        if self.mapped:
            scale, cycles = self._map
            self.align()  # the drive has finished before the map starts
            self.reader.xy.play(SQUARE_OPERATION, amplitude_scale=scale, duration=cycles)
            self.reader.xy.play(PI_REFERENCE)
            self.reader.align()  # the member's pulses, then its readout
        self.reader.readout_state(state, pulse_name=pulse_name, threshold=threshold)


def mapped_targets(experiment: Any, machine: Any):
    """The ``BatchableList`` a carrier's ``build_program`` takes when its
    Parameters name a ``drive_line`` and/or a ``readout_member``: ONE handle (the
    SCQO Parameters allow one target per run with either field)."""
    from scqo_qm._vendored.qualibration_libs import BatchableList
    from scqo_qm.experiments._coupler_tone import missing_thresholds
    from scqo_qm.experiments._selective_pi import selective_pi_scale
    from scqo_qm.experiments._vendor import vendor_borrowed, vendor_qubit

    params = experiment.params
    targets = list(params.targets)
    if len(targets) != 1:
        raise ValueError(f"a routed target runs alone; got targets={targets}")
    target = targets[0]
    line = getattr(params, "drive_line", None)
    member = getattr(params, "readout_member", None)
    if member is not None:
        reader = vendor_qubit(experiment, member, field="readout_member")
        if missing_thresholds(experiment, [member]):
            raise ValueError(
                f"{experiment.name}: the map reads {member} discriminated, and it "
                f"has no readout_threshold. Run single_shot_readout on {member} and "
                f"accept its readout_rotation_rad / readout_threshold suggestions "
                f"first.")
        length = int(params.selective_pi_len_ns)
        handle_kw = {"map_scale": selective_pi_scale(reader, length),
                     "map_cycles": length // _CLOCK_NS}
    else:
        reader = vendor_qubit(experiment, target, field="targets")
        handle_kw = {}
    if line is not None:
        xy = vendor_borrowed(experiment, f"{line}.{target}")
    else:
        xy = vendor_qubit(experiment, target, field="targets").xy
    return BatchableList([MappedTarget(target, xy, reader, **handle_kw)], [[0]])
