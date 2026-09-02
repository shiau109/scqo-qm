"""QM backend: maps the scqo abstractions onto the QUAM device + the fused experiment builders.

The QM stack (qm/quam) and the probe helpers are imported lazily (inside methods)
so that `import scqo_qm.backend.qm_backend` works without an instrument and pulls in
qualang_tools only when data is actually acquired. This module is never imported by
the qualibrate calibration nodes - scqo is its consumer.

Since the greenfield model a driver serves a view PER CHANNEL ENTITY, not per
qubit: the roster's ``q1_xy`` / ``q1_ro`` / ``q1_z`` each carry their own
function's knobs and resolve onto the matching SUBTREE of the same QUAM qubit
(``q.xy`` / ``q.resonator`` / ``q.z``), while a coupler's ``q1_q2_c_z`` lands on
the pair's ``TunableCoupler``. Composites keep a real QM surface — unlike Qblox,
QUAM has qubit_pairs with gate macros — served by :class:`QMQubitPair` through
the generic ``read_knob``/``write_knob`` contract (per-operation knob names such
as ``iswap_coupler_flux`` are roster-dependent, so they cannot be properties).
``QMDeviceModel.component`` does every one of those resolutions THROUGH THE
ROSTER, never by parsing names.

Neutral-name mapping: declared ONCE in ``scqo_qm/backend/fieldmap.py`` (the
catalog ``scqo state --fields`` renders; drift-tested per channel kind against
scqo's knob fields). The executable conversions are the channel views below (via
scqo_qm.quam_fields, shared with the qualibrate writebacks).
"""

from __future__ import annotations

import math
import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import xarray as xr
from scqo.backend import Backend, PreviewWarning
from scqo.catalog import OP_KNOBS, derived_op
from scqo.device import (
    ComponentInfo,
    CompositeView,
    DeviceModel,
    EntityView,
    make_view_base,
)
from scqo.entities import Channel, Composite
from scqo.fieldmap import Unrealized, VendorBinding, VendorOnly

from scqo_qm import quam_fields
from scqo_qm.backend.fieldmap import (
    FIELD_BINDINGS,
    OP_KNOB_BINDINGS,
    OP_KNOB_UNREALIZED,
    UNREALIZED,
    VENDOR_ONLY,
)

if TYPE_CHECKING:
    from pathlib import Path

    from scqo.experiment import Experiment
    from scqo.roster import Roster

#: A shell whose probe() EXECUTES on the instrument and returns a ready
#: Dataset (instead of a (program, axes[, module]) build) sets this class
#: attribute to a one-line WHY — the truthy value is the flag, the text is
#: the refusal reason. getattr default None = the probe builds a standalone
#: program and preview can render it (default-ALLOW: the majority build;
#: contrast ACTIVE_RESET_ATTR's default-DENY). Census in tests/test_preview.py
#: keeps the declarations exactly in step with the shells that really acquire.
SELF_ACQUIRING_ATTR = "probe_self_acquires"


def _preview_refusal(experiment) -> "str | None":
    """Why a QM preview is refused, or ``None`` when it may proceed.

    A self-acquiring shell (``SELF_ACQUIRING_ATTR`` set) builds ONE program per
    target inside ``probe()``, so it has no standalone program to render — UNLESS
    it exposes a ``preview_program()`` hook AND exactly one target is selected, in
    which case that single program is built (no acquire). Anything else that
    self-acquires is refused; a normal build-and-return shell is never refused.
    Returns the refusal DETAIL (the backend prepends the experiment name + the
    shell's own reason); ``None`` means previewable.

    ``preview_program()`` is NOT only a self-acquiring shell's escape hatch: ANY
    shell may expose it to render something cheaper than what ``probe()`` builds
    (``qubit_tomography`` drops its training shots that way). The single-target
    rule above is a constraint on SELF-ACQUIRING shells specifically — a normal
    shell's hook is used whatever the target count, because its ``probe()`` was
    a legal fallback to begin with.
    """
    if not getattr(type(experiment), SELF_ACQUIRING_ATTR, None):
        return None
    builder = getattr(experiment, "preview_program", None)
    targets = list(getattr(getattr(experiment, "params", None), "targets", None) or [])
    if builder is not None and len(targets) == 1:
        return None
    if builder is not None:
        return (f"select exactly one --target — a self-acquiring shell previews "
                f"the single program it would build for that target, and you "
                f"selected {len(targets)}")
    return ("its QUA program only exists inside a real run; run it for real or "
            "preview a different experiment")


#: preview's simulated-waveform window (ns) when --simulate-ns is not given.
#: Small on purpose: the LF-FEM samples at 2 GS/s, so the html grows 2k
#: points per ns per port. 20 us shows an active-reset shot from t=0; a
#: thermal shot starts with a ms-scale wait, which preview warns about.
_SIMULATE_DEFAULT_NS = 20_000
#: hard ceiling for --simulate-ns (400k samples/port at 2 GS/s) — refused by
#: name above it, because the html and the gateway simulation both scale
#: linearly with the window.
_SIMULATE_MAX_NS = 200_000

#: MW-FEM full-scale grid (dBm): -11..+16 in 3 dB steps (power_tools validates the
#: same values; its docstring's "[-41,10]" range is stale).
_FS_GRID_MIN, _FS_GRID_MAX, _FS_GRID_STEP = -11, 16, 3
#: The canonical digital operating point: keep the pulse amplitude <= 0.5 full scale
#: (shared by the readout AND drive chain solves).
_CANONICAL_MAX_AMP = 0.5


def _solve_full_scale(name: str, target: float) -> int:
    """Bidirectional full-scale selection for an absolute port power: the SMALLEST
    grid value that keeps the amplitude <= 0.5 (it lands in (0.354, 0.5]). Chosen
    here — not by the bare power_tools helper, whose internal loop only ever bumps
    full-scale UP and would leave a tiny amplitude after a high-power era."""
    headroom = 20.0 * math.log10(2.0)  # amp 0.5 -> -6.02 dB below full scale
    fs = _FS_GRID_MIN + _FS_GRID_STEP * math.ceil(
        (target + headroom - _FS_GRID_MIN) / _FS_GRID_STEP
    )
    fs = min(_FS_GRID_MAX, max(_FS_GRID_MIN, fs))
    amp_needed = 10.0 ** ((target - fs) / 20.0)
    if amp_needed > _CANONICAL_MAX_AMP:  # only when fs is pinned at the grid top
        warnings.warn(
            f"{name}: hitting {target} dBm at full scale {fs} dBm needs "
            f"amplitude {amp_needed:.3f} > {_CANONICAL_MAX_AMP} — above the "
            f"canonical operating point"
        )
    return int(fs)


# ------------------------------------------------------------- channel views

class _QMChannelView:
    """Shared plumbing of the three channel views.

    ``name`` is the ROSTER ENTITY name (``q1_ro``) — what scqo addresses and what
    every error message must cite; the VENDOR name (``q1``) is ``qubit.name``.
    The two were the same string in the pre-greenfield model, which is why the
    split matters.

    ``qubit`` is the QUAM qubit this channel's target resolves to (None for a
    coupler flux channel, whose vendor object hangs off a qubit_pair instead);
    ``vendor`` is the SUBTREE this kind of channel serves — ``q.xy`` for drive,
    ``q.resonator`` for readout, ``q.z`` (or the ``TunableCoupler``) for flux.
    Probes that genuinely need raw QUAM (pulse names, ports) go through those two
    properties and nothing else.
    """

    #: attribute of the QUAM qubit this kind of channel serves.
    _SUBTREE: str = ""

    def __init__(self, name: str, qubit: Any, vendor: Any = None) -> None:
        self.name = name
        self._q = qubit
        self._vendor = (vendor if vendor is not None
                        else getattr(qubit, self._SUBTREE))

    @property
    def qubit(self) -> Any:
        """The QUAM qubit behind this channel (None for a coupler flux line)."""
        return self._q

    @property
    def vendor(self) -> Any:
        """The QUAM subtree this view serves (``q.xy`` / ``q.resonator`` /
        ``q.z`` / the pair's ``TunableCoupler``) — the probes' raw-vendor door."""
        return self._vendor


class QMReadoutChannel(_QMChannelView, make_view_base("readout")):
    """The scqo READOUT channel view (``q1_ro``) over ``q.resonator``.

    Carries the dispersive-readout knobs: tone frequency, pulse amplitude/power,
    the pulse/window pair, and the full single-shot discriminator (rotation,
    threshold, RUS exit) — the trio Qblox declares Unrealized. The monitors
    (fidelity_g/e, blob positions) are never pushed and so have no vendor
    surface at all.
    """

    _SUBTREE = "resonator"

    @property
    def readout_freq_hz(self) -> float:
        return quam_fields.get_readout_freq(self._q)

    @readout_freq_hz.setter
    def readout_freq_hz(self, value: float) -> None:
        quam_fields.set_readout_freq(self._q, value)

    @property
    def readout_amp(self) -> float:
        return quam_fields.get_readout_amp(self._q)

    @readout_amp.setter
    def readout_amp(self, value: float) -> None:
        quam_fields.set_readout_amp(self._q, value)

    @property
    def readout_duration_s(self) -> float:
        return quam_fields.get_readout_duration(self._q)

    @readout_duration_s.setter
    def readout_duration_s(self, value: float) -> None:
        quam_fields.set_readout_duration(self._q, value)

    @property
    def readout_integration_s(self) -> float:
        return quam_fields.get_readout_integration(self._q)

    @readout_integration_s.setter
    def readout_integration_s(self, value: float) -> None:
        quam_fields.set_readout_integration(self._q, value)

    @property
    def readout_rotation_rad(self) -> float:
        return quam_fields.get_readout_rotation(self._q)

    @readout_rotation_rad.setter
    def readout_rotation_rad(self, value: float) -> None:
        quam_fields.set_readout_rotation(self._q, value)

    @property
    def readout_depletion_s(self) -> float:
        return quam_fields.get_readout_depletion(self._q)

    @readout_depletion_s.setter
    def readout_depletion_s(self, value: float) -> None:
        quam_fields.set_readout_depletion(self._q, value)

    @property
    def readout_threshold(self) -> float:
        return quam_fields.get_readout_threshold(self._q)

    @readout_threshold.setter
    def readout_threshold(self, value: float) -> None:
        quam_fields.set_readout_threshold(self._q, value)

    @property
    def readout_rus_threshold(self) -> float:
        return quam_fields.get_readout_rus_threshold(self._q)

    @readout_rus_threshold.setter
    def readout_rus_threshold(self, value: float) -> None:
        quam_fields.set_readout_rus_threshold(self._q, value)

    # readout_power_dbm / drive_power_dbm live HERE (not in quam_fields): the chain
    # solve needs quam_builder.tools.power_tools, whose module top imports quam —
    # quam_fields is contractually pure (stub-testable without an instrument).
    # backend.py already owns the lazy-vendor-import pattern, so the import stays
    # inside the accessors.
    @property
    def readout_power_dbm(self) -> float:
        from quam_builder.tools.power_tools import get_output_power_mw_channel

        amp = self._vendor.operations[quam_fields.READOUT_OPERATION].amplitude
        if not amp:  # None or 0 -> log10 domain error / -inf must never reach the config
            raise ValueError(
                f"{self.name}: readout amplitude is unset/zero — absolute power undefined"
            )
        return float(get_output_power_mw_channel(
            self._vendor, quam_fields.READOUT_OPERATION))

    @readout_power_dbm.setter
    def readout_power_dbm(self, value: float) -> None:
        from quam_builder.tools.power_tools import set_output_power_mw_channel

        target = float(value)
        # max_amplitude=1: the explicit full_scale_power_dbm already encodes the
        # <=0.5 policy; the helper then just sets fs + the exact amplitude.
        set_output_power_mw_channel(
            self._vendor, target, quam_fields.READOUT_OPERATION,
            full_scale_power_dbm=_solve_full_scale(self.name, target), max_amplitude=1,
        )


class QMDriveChannel(_QMChannelView, make_view_base("drive")):
    """The scqo DRIVE channel view (``q1_xy``) over ``q.xy``.

    Carries the xy knobs: drive frequency, the calibrated pi pulse (amplitude,
    length and DRAG coefficient), and the saturation-drive pair behind the
    absolute drive power.
    """

    _SUBTREE = "xy"

    @property
    def drive_freq_hz(self) -> float:
        return quam_fields.get_drive_freq(self._q)

    @drive_freq_hz.setter
    def drive_freq_hz(self, value: float) -> None:
        quam_fields.set_drive_freq(self._q, value)

    @property
    def pi_amp(self) -> float:
        return quam_fields.get_pi_amp(self._q)

    @pi_amp.setter
    def pi_amp(self, value: float) -> None:
        quam_fields.set_pi_amp(self._q, value)

    @property
    def pi_amp_x90(self) -> float:
        return quam_fields.get_pi_amp(self._q, operation="x90")

    @pi_amp_x90.setter
    def pi_amp_x90(self, value: float) -> None:
        # An INDEPENDENT knob, never pi_amp/2: qubit_deterministic_benchmarking
        # calibrates the pi/2 in its own right, and amplitude response is not
        # perfectly linear, which is the error that experiment measures.
        quam_fields.set_pi_amp(self._q, value, operation="x90")

    @property
    def pi_duration_s(self) -> float:
        return quam_fields.get_pi_duration(self._q)

    @pi_duration_s.setter
    def pi_duration_s(self, value: float) -> None:
        quam_fields.set_pi_duration(self._q, value)

    @property
    def drag_beta(self) -> float:
        return quam_fields.get_drag_beta(self._q, operation="x180")

    @drag_beta.setter
    def drag_beta(self, value: float) -> None:
        quam_fields.set_drag_beta(self._q, value, operation="x180")

    @property
    def drag_beta_x90(self) -> float:
        return quam_fields.get_drag_beta(self._q, operation="x90")

    @drag_beta_x90.setter
    def drag_beta_x90(self, value: float) -> None:
        quam_fields.set_drag_beta(self._q, value, operation="x90")

    # Lives on the QUBIT, not on q.xy: the wait is the qubit's own relaxation,
    # and QUAM hangs thermalization off the transmon. The drive channel is the
    # neutral home because scqo knobs live on channels, never on modes.
    @property
    def thermalization_time_s(self) -> float:
        return quam_fields.get_thermalization_time(self._q)

    @thermalization_time_s.setter
    def thermalization_time_s(self, value: float) -> None:
        quam_fields.set_thermalization_time(self._q, value)

    # The drive twin of the readout power, anchored to the SATURATION (spec)
    # operation. The xy full_scale_power_dbm is PORT-level and shared by every xy
    # operation: while it is off its standing value the stored pi_amp means a
    # different power (qubit_spectroscopy's run() sets and exactly reverts it; the
    # discrete grid + verbatim amplitude restore make the revert lossless).
    @property
    def drive_amp(self) -> float:
        return quam_fields.get_saturation_amp(self._q)

    @drive_amp.setter
    def drive_amp(self, value: float) -> None:
        quam_fields.set_saturation_amp(self._q, value)

    @property
    def drive_power_dbm(self) -> float:
        from quam_builder.tools.power_tools import get_output_power_mw_channel

        amp = self._vendor.operations[quam_fields.SATURATION_OPERATION].amplitude
        if not amp:  # None or 0 -> log10 domain error / -inf must never reach the config
            raise ValueError(
                f"{self.name}: saturation amplitude is unset/zero — absolute power undefined"
            )
        return float(get_output_power_mw_channel(
            self._vendor, quam_fields.SATURATION_OPERATION))

    @drive_power_dbm.setter
    def drive_power_dbm(self, value: float) -> None:
        from quam_builder.tools.power_tools import set_output_power_mw_channel

        target = float(value)
        set_output_power_mw_channel(
            self._vendor, target, quam_fields.SATURATION_OPERATION,
            full_scale_power_dbm=_solve_full_scale(self.name, target), max_amplitude=1,
        )


class QMFluxChannel(_QMChannelView, make_view_base("flux")):
    """The scqo FLUX channel view (``q1_z``, ``q1_q2_c_z``) over a flux source.

    Two vendor shapes behind one neutral knob, which is exactly the point of the
    greenfield split: a QUBIT's flux channel serves ``q.z`` (a QUAM ``FluxLine``,
    flux points joint/independent/min/arbitrary), a COUPLER's serves the pair's
    ``TunableCoupler`` (flux points off/on/arbitrary). ``idle_flux`` is the
    offset SELECTED by that object's ``flux_point``; which point is active stays
    vendor config. A coupler's decouple bias is therefore an ordinary
    ``idle_flux`` write on its own flux channel — the pair-level
    ``coupler_decouple_v`` / ``coupler_interaction_v`` fields are gone.

    The transfer-function FACTS (flux_offset, flux_per_phi0, distortion taps) are
    physical.json content and never push, so this view carries two knobs:
    ``idle_flux`` (the standing bias) and ``flux_delay_s`` (the line's output
    delay). Both resolve against ``self._vendor``, the flux-carrying channel —
    ``q.z`` for a qubit, the ``TunableCoupler`` for a coupler (both SingleChannels).
    """

    _SUBTREE = "z"

    @property
    def idle_flux(self) -> float:
        if self._q is None:
            return quam_fields.get_coupler_idle_flux(self._vendor)
        return quam_fields.get_idle_flux(self._q)

    @idle_flux.setter
    def idle_flux(self, value: float) -> None:
        if self._q is None:
            quam_fields.set_coupler_idle_flux(self._vendor, value)
        else:
            quam_fields.set_idle_flux(self._q, value)

    @property
    def flux_delay_s(self) -> float:
        # _vendor is the flux SingleChannel in BOTH shapes (q.z / TunableCoupler),
        # and the delay lives on its shared output port — no _q branch needed.
        return quam_fields.get_flux_delay(self._vendor)

    @flux_delay_s.setter
    def flux_delay_s(self, value: float) -> None:
        quam_fields.set_flux_delay(self._vendor, value)


#: channel kind -> the view class this backend serves for it. A kind absent here
#: (``pump``) and every non-channel, non-composite entity (modes, lines) is a
#: KeyError from ``component()`` — the contract scqo degrades gracefully against.
_CHANNEL_VIEWS: dict[str, type[_QMChannelView]] = {
    "drive": QMDriveChannel,
    "readout": QMReadoutChannel,
    "flux": QMFluxChannel,
}


# ------------------------------------------------------------ composite view

#: virtual-Z knob suffix -> the ROSTER role it corrects (the vendor side is
#: resolved from that role by name; see QMQubitPair._phase_attr).
_VZ_ROLE = {"vz_high_rad": "high", "vz_low_rad": "low"}


class QMQubitPair(CompositeView):
    """The scqo COMPOSITE view over a QUAM ``qubit_pair`` (QCQ architecture).

    Per-operation knob names are roster-dependent (``iswap_coupler_flux`` exists
    only because the roster declares the pair's ``iswap`` operation), so the
    surface is the generic ``read_knob``/``write_knob`` pair rather than
    properties. A field is split into ``(operation, suffix)`` against the
    catalog's OP_KNOBS family and the operations the roster DECLARES — never by
    prefix guessing — and the suffix is then mapped onto the QUAM gate macro
    (``qp.macros['CZ']``; the macro name is matched case-insensitively because
    QUAM spells the macro "CZ" while the roster spells the operation "cz").

    What is NOT here: the coupler's standing bias, which is ``idle_flux`` on the
    coupler MODE's own flux channel (:class:`QMFluxChannel`).
    """

    #: OP_KNOBS suffixes, longest first — "cz_waveform_dt_s" must not match the
    #: "waveform" suffix. The set itself comes from scqo, never re-declared here.
    _SUFFIXES: tuple[str, ...] = tuple(
        sorted(OP_KNOBS, key=len, reverse=True))

    def __init__(self, name: str, pair: Any, entity: Composite) -> None:
        self.name = name
        self.kind = entity.kind
        self._qp = pair
        self._entity = entity

    @property
    def vendor(self) -> Any:
        """The QUAM ``qubit_pair`` — the probes' raw-vendor door for gates."""
        return self._qp

    # ------------------------------------------------------------- resolution
    def _split(self, field: str) -> tuple[str, str]:
        """``'iswap_coupler_flux'`` -> ``('iswap', 'coupler_flux')``, validated
        against the operations the ROSTER declares on this composite.

        Longest suffix first (``cz_waveform_dt_s`` is dt, not a waveform), and a
        structural match on an UNDECLARED operation does not stop the scan — the
        undeclared-operation error is only raised once no suffix yields a
        declared one, so the message always names the real cause.
        """
        undeclared: str | None = None
        for suffix in self._SUFFIXES:
            if not field.endswith("_" + suffix):
                continue
            op = field[: -len(suffix) - 1]
            if op in self._entity.operations:
                return op, suffix
            if undeclared is None:
                undeclared = op
        if undeclared is not None:
            raise KeyError(
                f"{self.name}.{field}: operation {undeclared!r} is not declared "
                f"on this composite "
                f"(declared: {sorted(self._entity.operations)})")
        raise KeyError(
            f"{self.name}.{field}: not a per-operation knob — the legal suffixes "
            f"are {sorted(self._SUFFIXES)} on operations "
            f"{sorted(self._entity.operations)}")

    def _macro(self, op: str) -> Any:
        """The QUAM gate macro realizing one declared operation."""
        macros = getattr(self._qp, "macros", {}) or {}
        if op in macros:
            return macros[op]
        hits = [k for k in macros if k.lower() == op.lower()]
        if len(hits) == 1:
            return macros[hits[0]]
        raise KeyError(
            f"{self.name}: no QUAM macro for operation {op!r} on pair "
            f"{getattr(self._qp, 'id', self.name)!r} (macros: {sorted(macros)}) "
            f"— the roster declares the operation, the vendor config must "
            f"provide the gate macro that realizes it")

    def _phase_attr(self, role: str) -> str:
        """QUAM ``phase_shift_control``/``phase_shift_target`` for a ROSTER role.

        control/target is vendor gate plumbing (which side carries the flux
        pulse); high/low is roster topology. The two are joined by NAME here, so
        a pair whose QUAM members disagree with its roster roles is refused
        rather than silently writing the wrong qubit's virtual Z.
        """
        names = {r: self._entity.roles.get(r, ()) for r in ("high", "low")}
        if len(names["high"]) != 1 or len(names["low"]) != 1:
            raise KeyError(
                f"{self.name}: roles high={list(names['high'])} "
                f"low={list(names['low'])} — the virtual-Z knobs need exactly "
                f"one qubit per role")
        control = getattr(getattr(self._qp, "qubit_control", None), "name", None)
        if control == names["high"][0]:
            mapping = {"high": "phase_shift_control", "low": "phase_shift_target"}
        elif control == names["low"][0]:
            mapping = {"high": "phase_shift_target", "low": "phase_shift_control"}
        else:
            target = getattr(getattr(self._qp, "qubit_target", None), "name", None)
            raise KeyError(
                f"{self.name}: QUAM pair members (control={control!r}, "
                f"target={target!r}) match neither roster role "
                f"(high={names['high'][0]!r}, low={names['low'][0]!r}) — refusing "
                f"to guess which qubit a virtual-Z correction belongs to")
        return mapping[role]

    @staticmethod
    def _unrealized(suffix: str, field: str) -> None:
        entry = OP_KNOB_UNREALIZED.get(suffix)
        if entry is not None:
            raise NotImplementedError(f"{field}: {entry.reason}")

    # --------------------------------------------------------------- surface
    def read_knob(self, field: str) -> float | list[float] | None:
        op, suffix = self._split(field)
        self._unrealized(suffix, field)
        macro = self._macro(op)
        if suffix == "coupler_flux":
            pulse = getattr(macro, "coupler_flux_pulse", None)
            if pulse is None:  # fixed coupler: the gate plays no coupler pulse
                return None
            amp = getattr(pulse, "amplitude", None)
            return None if amp is None else float(amp)
        if suffix in _VZ_ROLE:
            turns = getattr(macro, self._phase_attr(_VZ_ROLE[suffix]), None)
            # QUAM stores frame rotations in TURNS (frame_rotation_2pi units).
            return None if turns is None else float(turns) * 2.0 * math.pi
        raise KeyError(f"{self.name}.{field}: no QM binding for suffix {suffix!r}")

    def write_knob(self, field: str, value) -> None:
        op, suffix = self._split(field)
        self._unrealized(suffix, field)
        macro = self._macro(op)
        if suffix == "coupler_flux":
            pulse = getattr(macro, "coupler_flux_pulse", None)
            if pulse is None:
                raise KeyError(
                    f"{self.name}.{field}: macro {op!r} has no coupler_flux_pulse "
                    f"(a fixed-coupler gate) — nothing realizes a coupler "
                    f"operating point for it")
            pulse.amplitude = float(value)
            return
        if suffix in _VZ_ROLE:
            setattr(macro, self._phase_attr(_VZ_ROLE[suffix]),
                    float(value) / (2.0 * math.pi))
            return
        raise KeyError(f"{self.name}.{field}: no QM binding for suffix {suffix!r}")


# ------------------------------------------------------------- device model

def _read_or_none(view: EntityView, field: str) -> float | list[float] | None:
    """Read a neutral knob, returning None when this vendor object cannot answer.

    ValueError covers readout_power_dbm on a zero/unset amplitude (the absolute
    power is undefined there, not zero); AttributeError an uncalibrated or
    partly-wired QUAM node; NotImplementedError (a RuntimeError subclass) the
    Unrealized composite knobs. Provenance must never crash a session."""
    try:
        if isinstance(view, CompositeView):
            return view.read_knob(field)
        return getattr(view, field)
    except (TypeError, AttributeError, KeyError, ValueError, RuntimeError):
        return None


def _vendor_names(obj: Any) -> set[str]:
    """The names a QUAM component may answer to (``name`` is derived from
    ``id`` but raises on a detached component — guard both)."""
    out: set[str] = set()
    for attr in ("name", "id"):
        try:
            value = getattr(obj, attr, None)
        except Exception:
            continue
        if isinstance(value, str) and value:
            out.add(value)
    return out


class QMDeviceModel(DeviceModel):
    """Wraps a QUAM machine (`Quam`), addressed by ROSTER entity name.

    ``component("q1_ro")`` resolves the channel entity through the roster, takes
    its KIND and its single TARGET, fetches the vendor subtree for that target
    and returns the matching channel view; ``component("q1_q2")`` resolves the
    composite onto the QUAM qubit_pair. The roster is therefore not optional —
    without it the driver cannot tell what ``q1_ro`` means.

    ``state_dir``: explicit save target. Set it whenever the machine was loaded from a
    non-default location (e.g. the setup's ``instrument_config`` folder) — a bare
    ``machine.save()`` writes to QUAM's configured default (the live ``quam_state/``),
    which may not be the folder this session's state actually lives in.
    """

    def __init__(self, machine: Any, roster: "Roster",
                 state_dir: str | None = None) -> None:
        self._machine = machine
        self._roster = roster
        self._state_dir = state_dir

    @property
    def roster(self) -> "Roster":
        """The device's authority on which entities exist (read-only)."""
        return self._roster

    # ------------------------------------------------------ vendor resolution
    def _pairs(self) -> dict:
        return getattr(self._machine, "qubit_pairs", {}) or {}

    def _pair_object(self, e: Composite) -> Any:
        """The QUAM qubit_pair behind a roster composite: by NAME first, else the
        unique pair whose two QUAM members are exactly the composite's high/low
        (QM names pairs after their coupler — ``coupler_q1_q2`` — so the roster
        name rarely matches, and the membership match is the honest join)."""
        pairs = self._pairs()
        if e.name in pairs:
            return pairs[e.name]
        members = {n for role in ("high", "low") for n in e.roles.get(role, ())}
        hits = [
            name for name, qp in pairs.items()
            if {getattr(getattr(qp, side, None), "name", None)
                for side in ("qubit_control", "qubit_target")} == members
        ]
        if len(hits) == 1:
            return pairs[hits[0]]
        raise KeyError(
            f"{e.name!r} is a {e.kind} of {sorted(members)}; no QUAM qubit_pair "
            f"matches it "
            + (f"(several do: {sorted(hits)})" if hits else
               f"(pairs in this state: {sorted(pairs)})"))

    def _coupler_object(self, target: str) -> Any | None:
        """The ``TunableCoupler`` a roster COUPLER MODE names, or None.

        Two joins, both roster-anchored: the coupler component answering to the
        mode's own name, and — when the vendor names it differently — the coupler
        of the pair whose roster composite declares this mode in its ``coupler``
        role."""
        for qp in self._pairs().values():
            coupler = getattr(qp, "coupler", None)
            if coupler is not None and target in _vendor_names(coupler):
                return coupler
        for comp in self._roster.composites().values():
            if target not in comp.roles.get("coupler", ()):
                continue
            try:
                coupler = getattr(self._pair_object(comp), "coupler", None)
            except KeyError:
                continue
            if coupler is not None:
                return coupler
        return None

    def _channel_view(self, name: str, e: Channel) -> EntityView:
        view_cls = _CHANNEL_VIEWS.get(e.kind)
        if view_cls is None:
            raise KeyError(
                f"{name!r} is a {e.kind} channel — the QM backend realizes "
                f"{sorted(_CHANNEL_VIEWS)} channels only")
        if len(e.target) != 1:
            raise KeyError(
                f"{name!r} is a multi-target {e.kind} channel {e.target} — the "
                f"QM backend serves one QUAM object per channel")
        target = e.target[0]
        qubit = self._machine.qubits.get(target)
        if qubit is not None:
            subtree = getattr(qubit, view_cls._SUBTREE, None)
            if subtree is None:
                raise KeyError(
                    f"{name!r} targets {target!r}, which has no "
                    f"{view_cls._SUBTREE!r} in the loaded QUAM state — this "
                    f"machine does not wire a {e.kind} line to it (a "
                    f"fixed-frequency qubit has no z)")
            return view_cls(name, qubit, subtree)
        if e.kind == "flux":
            # A coupler is an ordinary roster MODE but NOT a QUAM qubit: it hangs
            # off its pair. Only its flux line exists (couplers are not driven or
            # read), so this hop is flux-only by construction.
            coupler = self._coupler_object(target)
            if coupler is not None:
                return view_cls(name, None, coupler)
        raise KeyError(
            f"{name!r} targets {target!r}, which is neither a qubit of the "
            f"loaded QUAM state (qubits: {sorted(self._machine.qubits)}) nor a "
            f"coupler of one of its qubit_pairs")

    def component(self, name: str) -> EntityView:
        """The view for one vendor-realized entity, addressed by ROSTER name.

        KeyError for everything this backend does not realize — an unknown name,
        a mode or line (knobs live on channels), a pump or multi-target channel,
        a channel whose target has no QUAM object, and a composite with no
        qubit_pair. That is the contract: scqo degrades gracefully and the doctor
        reports the gap against ``components()``.
        """
        e = self._roster.entities.get(name)
        if e is None:
            raise KeyError(f"unknown entity {name!r} — not in this device's roster")
        if isinstance(e, Channel):
            return self._channel_view(name, e)
        if isinstance(e, Composite):
            return QMQubitPair(name, self._pair_object(e), e)
        channels = [c.name for c in self._roster.channels_of(name)]
        raise KeyError(
            f"{name!r} is a {type(e).__name__.lower()}; the QM backend serves "
            f"channel and composite entities only (knobs live there) — address "
            f"{channels or '(none wired)'}")

    def components(self) -> dict[str, ComponentInfo]:
        """Derived inventory (the doctor's WITNESS, never truth): every roster
        channel and composite this backend actually serves a view for, with the
        kind it serves it AS — ``vendor_checks`` FAILS on a kind disagreement, so
        this reports the ROSTER's kind of the entity we resolved, never a QUAM
        class name (which cannot tell a coupler from a qubit).
        """
        out: dict[str, ComponentInfo] = {}
        for name, ch in self._roster.channels().items():
            try:
                self.component(name)
            except KeyError:
                continue
            out[name] = ComponentInfo(
                kind=ch.kind, target=tuple(ch.target), line=ch.line,
                operations=self._derived_operations(ch))
        for name, comp in self._roster.composites().items():
            try:
                self.component(name)
            except KeyError:
                continue
            out[name] = ComponentInfo(
                kind=comp.kind, operations=tuple(comp.operations),
                members={role: tuple(members)
                         for role, members in comp.roles.items()})
        return out

    def _derived_operations(self, channel: Channel) -> tuple[str, ...]:
        """The single-mode operation this channel derives on its target — read
        from scqo's (channel kind x target kind) table, never declared here."""
        target = self._roster.entities.get(channel.target[0])
        if target is None:
            return ()
        try:
            op = derived_op(channel.kind, target.kind)
        except KeyError:
            return ()
        return (op,) if op else ()

    def save(self) -> None:
        if self._state_dir is not None:
            self._machine.save(path=self._state_dir)
        else:
            self._machine.save()

    def snapshot(self) -> dict:
        """``{entity: {knob: value}}`` over the entities this backend realizes.

        Channels report the knobs the fieldmap declares BOUND (an Unrealized one
        has no vendor value to seed from); composites report the per-operation
        knobs the ROSTER compiled for them (their legal names are per-device).
        None for a knob a given QUAM node cannot answer — a real lab state
        carries uncalibrated qubits (f_01=None), and provenance must never crash
        a session."""
        state: dict[str, dict] = {}
        for name, ch in self._roster.channels().items():
            try:
                view = self.component(name)
            except KeyError:
                continue  # not realized here: absent from the snapshot entirely
            state[name] = {
                field: _read_or_none(view, field)
                for field in FIELD_BINDINGS.get(ch.kind, {})
            }
        for name in self._roster.composites():
            try:
                view = self.component(name)
            except KeyError:
                continue
            state[name] = {
                field: _read_or_none(view, field)
                for field, spec in self._roster.fields_of(name).items()
                if spec.role == "knob"
            }
        return state


def _progress_shot_total(experiment: "Experiment") -> int:
    """Denominator for the acquisition progress counter.

    The averaging count or the explicit shot count when the params carry one;
    otherwise -- for a record_time_s-derived experiment (the parity-switch
    monitors) whose ``params.num_shots`` is present but ``None`` -- the resolved count the
    probe actually buffered (``experiment.resolved_num_shots()``). NEVER None:
    ``qualang_tools.progress_counter`` divides an int by this, so a None here
    is a hard ``TypeError: ... for /: 'int' and 'NoneType'`` after the whole
    program has already run (the crash is in the Python progress display, not
    the sequence). A present-but-None attr does NOT trip ``getattr``'s default,
    which is why the raw ``getattr(params, "num_shots", 1)`` returned None.
    """
    params = experiment.params
    total = getattr(params, "num_averages", None) or getattr(params, "num_shots", None)
    if total is None:
        resolver = getattr(experiment, "resolved_num_shots", None)
        total = resolver() if callable(resolver) else 1
    return int(total)


class QMBackend(Backend):
    """scqo Backend over a Quantum Machines OPX (via QUAM + the fused experiment builders)."""

    def __init__(self, machine: Any, *, roster: "Roster",
                 timeout: float = 120) -> None:
        self._machine = machine
        self._roster = roster
        self._device = QMDeviceModel(machine, roster)
        self._timeout = timeout

    @classmethod
    def load(cls, *, roster: "Roster", state_path: str | None = None,
             timeout: float = 120) -> "QMBackend":
        """Construct from a QUAM state. ``state_path`` overrides ``QUAM_STATE_PATH``;
        when omitted the env / default configuration is used."""
        import os

        from quam_config import Quam

        if state_path is not None:
            os.environ["QUAM_STATE_PATH"] = state_path
        return cls(Quam.load(), roster=roster, timeout=timeout)

    @property
    def device(self) -> QMDeviceModel:
        return self._device

    @property
    def roster(self) -> "Roster":
        """The device roster this backend was built against."""
        return self._roster

    @property
    def machine(self) -> Any:
        """The underlying QUAM machine (read by the QM experiment probes)."""
        return self._machine

    def field_bindings(self) -> dict[str, dict[str, VendorBinding]]:
        """The declared per-CHANNEL-KIND neutral-knob catalog
        (scqo_qm.backend.fieldmap) — the conversion CODE is the channel views
        above; this is its description. The composite per-operation surface is
        :meth:`op_knob_bindings` (its field names are roster-dependent, so it
        cannot be keyed by catalog field name)."""
        return {kind: dict(bindings) for kind, bindings in FIELD_BINDINGS.items()}

    def unrealized(self) -> dict[str, dict[str, Unrealized]]:
        """Knobs this backend cannot realize, per channel kind (see fieldmap) —
        empty: QM realizes every drive/readout/flux knob in the catalog."""
        return {kind: dict(entries) for kind, entries in UNREALIZED.items()}

    def op_knob_bindings(self) -> tuple[dict[str, VendorBinding],
                                        dict[str, Unrealized]]:
        """The composite (``qubit_pair``) per-OPERATION knob catalog, keyed by
        the ``scqo.catalog.OP_KNOBS`` suffix: ``(bindings, unrealized)``. The
        full field name is ``<operation>_<suffix>`` for every operation the
        roster declares on the pair — see :class:`QMQubitPair`."""
        return dict(OP_KNOB_BINDINGS), dict(OP_KNOB_UNREALIZED)

    def vendor_only(self) -> dict[str, VendorOnly]:
        """QM-unique calibration knobs, vendor-owned (see fieldmap)."""
        return dict(VENDOR_ONLY)

    def _drive_views(self, targets: list[str]) -> dict[str, EntityView]:
        """Every run target's DEFAULT drive view, keyed by channel name.

        A composite target (a pair) has no drive channel of its own, so it
        expands to its MEMBER modes' drive channels — the entities the reset
        actually happens on. Roster-resolved throughout; never string
        arithmetic on names."""
        views: dict[str, EntityView] = {}
        for target in targets:
            entity = self._roster.entities.get(target)
            modes = [target]
            if isinstance(entity, Composite):
                modes = [m for names in entity.roles.values() for m in names]
            for mode in modes:
                try:
                    name = self._roster.default_channel(mode, "drive")
                except Exception:
                    continue
                if name not in views:
                    views[name] = self._device.component(name)
        return views

    @contextmanager
    def _thermalization_override(self, experiment: "Experiment"):
        """Per-run override of the thermal-reset wait: recorded set -> build ->
        exact revert, the punchout/drive-power discipline.

        The STANDING wait already lives in QUAM (scqo pushes the neutral
        ``thermalization_time_s`` knob), so ``qubit.reset("thermal")`` needs no
        argument and none of the 16 probes changes. Only an explicit per-run
        override has anything to do, and because ``reset_qubit_thermal`` reads
        ``thermalization_time`` while the QUA program is being BUILT, bracketing
        the build is enough — the wait is baked into the compiled program.

        Writes go straight to the QUAM tree, NOT through the recording device:
        a per-run override must leave no ChangeRecord and no store row."""
        override_ns = getattr(experiment.params, "thermalization_time_ns", None)
        if override_ns is None:
            yield
            return
        views = self._drive_views(list(experiment.params.targets))
        if not views:
            raise RuntimeError(
                f"thermalization_time_ns={override_ns} was set but no drive "
                f"channel resolved for targets {list(experiment.params.targets)} "
                f"— refusing to run with the override silently ignored")
        previous = {n: v.thermalization_time_s for n, v in views.items()}
        for view in views.values():
            view.thermalization_time_s = float(override_ns) / 1e9
        try:
            yield
        finally:
            for name, view in views.items():
                before = previous[name]
                if before is not None:
                    view.thermalization_time_s = before
                else:  # never calibrated: restore "unset", not a fabricated value
                    view._q.thermalization_time_ns = None

    def _default_view(self, target: str, kind: str) -> EntityView | None:
        """The target's DEFAULT channel view of one kind, or None when the roster
        or the QUAM tree cannot serve it (provenance must never fail a run)."""
        try:
            return self._device.component(
                self._roster.default_channel(target, kind))
        except Exception:
            return None

    def distortion_apply_command(self, target: str,
                                 run_id: str | None = None) -> str:
        """The command that turns a cryoscope's measured taps into THIS
        instrument's predistortion filter — scqo's duck-typed hint hook
        (``scqo.experiments._distortion_hint``), printed by both cryoscopes right
        after they propose ``distortion_amp``/``distortion_tau_s``.

        Run-addressed whenever the run is known: ``--run`` names exactly which
        measurement feeds the filter, so ``scqo accept`` order cannot change what
        lands on the OPX (the two cryoscopes share ONE fact slot with REPLACE
        semantics — see :mod:`scqo_qm.backend.apply_distortion`). Without a run id
        the command reads the accepted facts instead, which is the same default
        the CLI has.
        """
        run = f" --run {run_id}" if run_id else ""
        return f"python -m scqo_qm.backend.apply_distortion --target {target}{run}"

    def power_context(self, qubits: list[str]) -> dict:
        """Raw readout + drive chain values per qubit (run-record provenance only).

        Addressed by MODE name (``params.targets``): each target's default readout
        and drive channels are resolved through the roster, so the two blocks stay
        independent — a qubit without a saturation op still reports its readout chain.
        """
        from quam_builder.tools.power_tools import get_output_power_mw_channel

        out: dict = {}
        for name in qubits:
            out[name] = {}
            try:
                view = self._default_view(name, "readout")
                resonator = view.vendor
                ro = resonator.operations[quam_fields.READOUT_OPERATION]
                out[name] = {
                    "full_scale_power_dbm": resonator.opx_output.full_scale_power_dbm,
                    "readout_amplitude": float(ro.amplitude),
                    "readout_power_dbm": float(get_output_power_mw_channel(
                        resonator, quam_fields.READOUT_OPERATION)),
                }
                # The readout LO the data was taken at (QUAM resolves LO_frequency
                # to the port's upconverter_frequency): a hand-edited LO is
                # otherwise invisible in provenance. Only when the channel has one
                # (an LF-FEM resonator does not).
                lo = getattr(resonator, "LO_frequency", None)
                if lo is not None:
                    out[name]["readout_lo_freq_hz"] = float(lo)
            except Exception:  # provenance must never fail a run
                out[name] = {}
            # The drive chain behind drive_power_dbm — same never-fail rule, and
            # independent of the readout block.
            try:
                view = self._default_view(name, "drive")
                xy = view.vendor
                sat = xy.operations[quam_fields.SATURATION_OPERATION]
                out[name].update({
                    "drive_full_scale_power_dbm": xy.opx_output.full_scale_power_dbm,
                    "saturation_amp": float(sat.amplitude),
                    "drive_power_dbm": float(get_output_power_mw_channel(
                        xy, quam_fields.SATURATION_OPERATION)),
                })
                lo = getattr(xy, "LO_frequency", None)
                if lo is not None:
                    out[name]["drive_lo_freq_hz"] = float(lo)
            except Exception:  # provenance must never fail a run
                pass
        return out

    def acquire(self, experiment: "Experiment") -> xr.Dataset:
        # The reset-method backstop. Every shell resolves reset_method through
        # check_reset_method in its probe(), but only if it remembers to; this
        # fires for anything carrying the neutral field, before any vendor import
        # and before _thermalization_override — which the checker protects, since
        # it refuses the thermalization_time_ns + 'active' combination by name.
        # Side-effect free, so being called twice costs nothing.
        from scqo_qm.experiments._reset import check_reset_method
        check_reset_method(experiment)
        from scqo_qm.experiments._lib import acquire as run_acquire

        # Progress denominator: per-shot experiments (single_shot_readout) declare
        # `num_shots` instead of the averaging mixin's `num_averages`; a
        # record_time_s-derived experiment (the parity-switch monitors) leaves
        # `num_shots` None and exposes the real count via resolved_num_shots().
        shots = _progress_shot_total(experiment)
        # A probe returns ONE of three shapes:
        #  - a ready-made xr.Dataset (drag_equator acquires itself with a baked config);
        #  - (program, sweep_axes, acquire): the fused module's own acquire
        #    CALLABLE fetches (tomography/t1 trackers build heterogeneous-dims
        #    datasets per-shot that XarrayDataFetcher refuses);
        #  - (program, sweep_axes): the shared _lib.acquire fetches (the common path).
        with self._thermalization_override(experiment):
            res = experiment.probe()
        if isinstance(res, xr.Dataset):
            raw = res
        else:
            if isinstance(res, tuple) and len(res) == 3:
                program, sweep_axes, acquire_fn = res
            else:
                program, sweep_axes = res
                acquire_fn = run_acquire
            raw = acquire_fn(
                self._machine, program, sweep_axes,
                num_shots=shots, timeout=self._timeout,
            )
        # Optional per-experiment raw reduction (e.g. joint two-qubit state
        # populations -> the pair experiment's joint_population variable),
        # applied BEFORE canonicalization so the contract sees final variables.
        reduce = getattr(experiment, "reduce_raw", None)
        if reduce is not None:
            raw = reduce(raw)
        # The readout schema's naming: the QUA probes' averaged discriminated
        # stream arrives as `state`, but `state` canonically means a PER-SHOT
        # outcome — when this experiment's contract accepts `population` (the
        # averaged form) and none of its forms accepts `state`, rename. An
        # experiment with a per-shot `state` form (the parity-switch monitors,
        # qc_n_swap_amp shot mode) keeps the name.
        contract = experiment.Contract
        accepted = set(contract.variables).union(*contract.alt_variables) \
            if contract.alt_variables else set(contract.variables)
        if ("state" in raw.data_vars and "state" not in accepted
                and "population" in accepted):
            raw = raw.rename({"state": "population"})
        return self._to_canonical(raw, experiment)

    def preview(self, experiment: "Experiment", out_dir: "Path", *,
                simulate_ns: "int | None" = None,
                no_simulate: bool = False) -> "list[Path]":
        """Dump the QUA script, then TRY the gateway simulator — never the qubits.

        The scqo ``Backend.preview`` hook (``Session.preview`` /
        ``scqo run <name> --preview``): the same build slice as ``acquire()``
        minus the instrument — the reset backstop and the thermalization
        bracket around ``probe()`` (the wait is baked at build time) — then
        ``generate_qua_script(prog, machine.generate_config())``, which opens
        no socket (``generate_config`` walks the QUAM tree in memory). Shells
        whose probe() itself ACQUIRES (declared via ``probe_self_acquires``,
        see :data:`SELF_ACQUIRING_ATTR`) are refused by name BEFORE probe()
        can touch the instrument — UNLESS the shell exposes a ``preview_program``
        hook and exactly one target is selected, in which case that single
        program is built (no acquire) and rendered.

        The waveform view is attempted AUTOMATICALLY: the OPX1000 gateway's
        simulator (the wiring's ``network`` host) compiles the program and
        returns the analog output samples, saved as an interactive
        ``simulated_waveforms.html`` beside the script. Simulation runs on
        the gateway server — nothing plays on the physical outputs. Any
        gateway problem (host down, wedged, refused) degrades to the script
        with a :class:`~scqo.backend.PreviewWarning`, gated by a 2 s TCP
        probe first so a dead host cannot stall the preview.
        ``no_simulate=True`` (CLI ``--no-simulate``) skips the attempt —
        guaranteed fully offline; ``simulate_ns`` (CLI ``--simulate-ns``)
        widens the window, capped at :data:`_SIMULATE_MAX_NS`.
        """
        from scqo_qm.experiments._reset import check_reset_method

        name = getattr(type(experiment), "name", type(experiment).__name__)
        if simulate_ns is not None:
            if no_simulate:
                raise ValueError(
                    "simulate_ns and no_simulate contradict each other")
            if not 0 < simulate_ns <= _SIMULATE_MAX_NS:
                raise ValueError(
                    f"simulate_ns={simulate_ns} is refused: the window must "
                    f"be in (0, {_SIMULATE_MAX_NS}] ns — the simulated html "
                    f"holds 2000 samples per ns per port and the gateway "
                    f"simulation time scales with it")
        reason = getattr(type(experiment), SELF_ACQUIRING_ATTR, None)
        refusal = _preview_refusal(experiment)
        if refusal is not None:
            raise ValueError(
                f"{name} cannot be previewed on the QM backend: {reason} — {refusal}")
        check_reset_method(experiment)  # same backstop as acquire()
        with self._thermalization_override(experiment):
            # Any shell exposing the hook renders through it — a self-acquiring one
            # because it has no other standalone program (the gate above passed), a
            # normal one because its hook is the cheaper build of the same sequence.
            if hasattr(experiment, "preview_program"):
                prog = experiment.preview_program()
            else:
                res = experiment.probe()
                if isinstance(res, xr.Dataset):  # the census missed one — refuse loud
                    raise ValueError(
                        f"{name}.probe() returned a ready Dataset — it acquires "
                        f"during probe(); declare {SELF_ACQUIRING_ATTR} on the shell")
                prog = res[0]  # covers both the 2-tuple and 3-tuple probe shapes
        from datetime import datetime, timezone

        from qm import generate_qua_script

        config = self._machine.generate_config()
        # A shell whose program only compiles against an AMENDED config (the
        # parametric-drive z oscillator; acquire() binds the amendment into its
        # own fetch callable) exposes patch_preview_config(config) so the
        # dumped script and the gateway simulation see the same config the
        # real run would execute against.
        patch = getattr(experiment, "patch_preview_config", None)
        if patch is not None:
            config = patch(config)
        script = generate_qua_script(prog, config)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "qua_script.py"
        header = (f"# scqo preview: {name}\n"
                  f"# backend: qm\n"
                  f"# generated: {datetime.now(timezone.utc).isoformat()}\n"
                  f"# params: {experiment.params.model_dump_json()}\n\n")
        path.write_text(header + script, encoding="utf-8")
        files = [path]

        if not no_simulate:
            window_ns = simulate_ns or _SIMULATE_DEFAULT_NS
            try:
                waveforms = self._simulated_waveforms(prog, config, out_dir,
                                                      window_ns)
            except Exception as err:  # degrade, never fail the preview
                warnings.warn(PreviewWarning(
                    f"simulated waveforms skipped "
                    f"({type(err).__name__}: {err}) — qua_script.py is "
                    f"complete; pass --no-simulate to silence this, or check "
                    f"the gateway if it should have answered"))
            else:
                if waveforms is not None:
                    files.append(waveforms)
        return files

    def _simulated_waveforms(self, prog: Any, config: dict,
                             out_dir: "Path", window_ns: int) -> "Path | None":
        """Compile+simulate ``window_ns`` of output on the gateway and write
        the interactive plot. Raises on any gateway problem (the caller
        degrades to script-only); returns None when the window contains no
        pulses (warned, no file). The 2 s TCP probe fails a dead host fast —
        a WEDGED gateway (accepts TCP, stalls on requests) still costs the
        vendor's own request deadline before the degrade path fires."""
        import socket

        net = getattr(self._machine, "network", None) or {}
        get = net.get if hasattr(net, "get") else lambda k, d=None: getattr(
            net, k, d)
        host, port = get("host"), get("port")
        if host:  # fail a dead host in 2 s, not a gRPC timeout
            socket.create_connection((str(host), int(port or 80)),
                                     timeout=2.0).close()

        from qm import SimulationConfig

        qmm = self._machine.connect()
        try:
            job = qmm.simulate(config, prog,
                               SimulationConfig(duration=max(
                                   1, window_ns // 4)))  # duration: 4 ns cycles
            samples = job.get_simulated_samples()
        finally:
            # qm-qua 1.2.6's QuantumMachinesManager has no close(); newer
            # versions (and the test double) do — call it when present, and
            # never mask the simulate outcome with a close() failure.
            close = getattr(qmm, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:
                    pass

        import numpy as np
        import plotly.graph_objects as go

        def port_label(port_key: str, kind: str) -> str:
            parts = str(port_key).split("-")  # "<fem>-<out>[-<upconv>]"
            if len(parts) >= 2:
                label = f"FEM{parts[0]}-{kind}O{parts[1]}"
                return label + (f"-UP{parts[2]}" if len(parts) == 3 else "")
            return str(port_key)

        fig = go.Figure()
        many = len(samples) > 1
        for controller, cs in samples.items():
            prefix = f"{controller} " if many else ""
            for port_key, values in cs.analog.items():
                arr = np.asarray(values)
                if not np.any(arr):
                    continue
                rate = cs.analog_sampling_rate[str(port_key)]
                t = np.arange(len(arr)) / rate * 1e9
                label = prefix + port_label(port_key, "A")
                if np.iscomplexobj(arr):  # MW-FEM: I/Q pair
                    fig.add_trace(go.Scatter(x=t, y=arr.real,
                                             name=f"{label} I"))
                    fig.add_trace(go.Scatter(x=t, y=arr.imag,
                                             name=f"{label} Q"))
                else:
                    fig.add_trace(go.Scatter(x=t, y=arr, name=label))
            for port_key, values in cs.digital.items():
                arr = np.asarray(values)
                if not np.any(arr):
                    continue
                t = np.arange(len(arr), dtype=float)  # digital: 1 GS/s = 1 ns steps
                fig.add_trace(go.Scatter(
                    x=t, y=arr.astype(int),
                    name=prefix + port_label(port_key, "D"),
                    line={"shape": "hv"}))
        if not fig.data:
            warnings.warn(PreviewWarning(
                f"the simulated window (first {window_ns} ns) contains no "
                f"pulses — a thermal-reset shot starts with a long wait; "
                f"try a larger --simulate-ns or reset_method='active'"))
            return None
        fig.update_layout(
            title=(f"simulated gateway output — first {window_ns} ns "
                   f"(latencies included; sample step from each port's rate)"),
            xaxis_title="Time [ns]", yaxis_title="Output")
        waveforms = out_dir / "simulated_waveforms.html"
        fig.write_html(str(waveforms))
        return waveforms

    def close_qm(self, *, qm_id: str | None = None) -> dict[str, Any]:
        """Halt running jobs and close open Quantum Machines on the cluster.

        The recovery door for a cluster whose hardware locks are held by a
        crashed or abandoned session — the state that otherwise shows up as a
        job stalling forever, or an open that never returns. Driven by the
        operator CLI :mod:`scqo_qm.backend.close_qm`, which is where the
        vendor-facing command lives (scqo's neutral core has no ``close_qm``:
        the command needs the vendor libraries, so it lives with them).

        BEST-EFFORT BY DESIGN, and the one place in this backend where blanket
        exception capture is right: every step is an independent cleanup on a
        cluster that is, by hypothesis, already misbehaving, so a QM that
        refuses to close must not stop the next one from closing. A failure is
        RECORDED in ``errors`` and the sweep continues. The report is the
        result — read ``success``, never the absence of a raise.

        Opens its own manager through ``self._machine.connect()`` and closes it
        again, exactly as :meth:`preview`'s simulation path does. Safe because
        ``QMBackend`` never caches a manager: nothing else holds this one.
        """
        qmm = self._machine.connect()
        open_qms: list[str] = []
        errors: list[str] = []

        for lister in ("list_open_qms", "list_open_quantum_machines"):
            listing = getattr(qmm, lister, None)
            if listing is None:
                continue
            try:
                open_qms = [str(q) for q in listing()]
            except Exception as err:
                errors.append(f"{lister}: {type(err).__name__}: {err}")
            break

        halted_jobs: list[str] = []
        closed_qms: list[str] = []

        for target_id in ([qm_id] if qm_id is not None else list(open_qms)):
            try:
                qm = qmm.get_qm(target_id)
            except Exception as err:
                errors.append(f"get_qm({target_id}): {type(err).__name__}: {err}")
                continue
            if qm is None:
                errors.append(f"get_qm({target_id}) returned nothing")
                continue

            job = None
            try:
                job = qm.get_running_job()
            except Exception as err:
                errors.append(
                    f"{target_id}: get_running_job: {type(err).__name__}: {err}")
            if job is not None:
                job_id = str(getattr(job, "id", job))
                stopper = getattr(job, "halt", None) or getattr(job, "cancel", None)
                if stopper is None:
                    errors.append(f"{target_id}: job {job_id} has no halt/cancel")
                else:
                    try:
                        stopper()
                        halted_jobs.append(job_id)
                    except Exception as err:
                        errors.append(f"{target_id}: halting job {job_id}: "
                                      f"{type(err).__name__}: {err}")
            try:
                qm.close()
                closed_qms.append(str(target_id))
            except Exception as err:
                errors.append(f"{target_id}: close: {type(err).__name__}: {err}")

        # The whole-cluster sweep also releases what the listing missed (a QM
        # opened by a process that died before registering, say). Skipped for a
        # targeted --qm-id, which must not touch its neighbours.
        if qm_id is None:
            for closer in ("close_all_qms", "close_all_quantum_machines"):
                close_all = getattr(qmm, closer, None)
                if close_all is None:
                    continue
                try:
                    close_all()
                except Exception as err:
                    errors.append(f"{closer}: {type(err).__name__}: {err}")
                break

        close_manager = getattr(qmm, "close", None)
        if close_manager is not None:
            try:
                close_manager()
            except Exception as err:
                errors.append(f"closing the manager: {type(err).__name__}: {err}")

        return {
            "success": not errors,
            "backend": "qm",
            "open_qms": open_qms,
            "halted_jobs": halted_jobs,
            "closed_qms": closed_qms,
            "errors": errors,
        }

    @staticmethod
    def _to_canonical(raw: xr.Dataset, experiment: "Experiment") -> xr.Dataset:
        """Relabel the raw probe dataset into scqo's convention: dims (qubit, *sweeps),
        vars I/Q. The probe already emits I/Q with a qubit dim.

        Two mapping modes for the sweep axes:

        1. Name-based (preferred, order-independent): when the raw non-qubit dim
           NAMES already equal the scqo `define_sweep` names as a set, nothing is
           renamed — only per-name sizes are asserted. This is how probes whose raw
           nesting order differs from the scqo declaration order stay correct: the
           QUA stream nesting fixes the raw dim order (e.g. flux spectroscopy is
           (detuning, flux) on the wire while scqo declares (flux, detuning));
           downstream code transposes by NAME, so order does not matter, but a
           positional rename would silently swap equal-sized axes. Wrappers opt in
           by returning `sweep_axes` keyed with the canonical names in raw nesting
           order (readout_fidelity's probe already is canonical).

        2. Positional fallback: axes are renamed positionally to the scqo names
           (e.g. idle_time -> idle_time_ns, (detuning, power) -> (detuning_hz,
           power_dbm)) — the fetcher preserves the probe's sweep_axes order, and the
           wrapper guarantees that order matches `define_sweep`; sizes are asserted
           per axis.

        On a structure mismatch the raw dataset is pickled for offline inspection —
        a bring-up run against the real QOP is never wasted (a raise here happens
        before the datastore ever sees the dataset).
        """
        # The raw target axis may be named per the probe family (single-qubit
        # probes: "qubit"; pair probes: "qubit_pair"); canonical is "target".
        raw_target = next((d for d in ("target", "qubit", "qubit_pair")
                           if d in raw.dims), None)
        if raw_target is None:
            raise ValueError(
                f"raw dataset has no target axis (expected one of "
                f"'qubit'/'qubit_pair'/'target'; dims={dict(raw.sizes)}, "
                f"data_vars={list(raw.data_vars)}); {_dump_raw(raw)}"
            )
        if raw_target != "target":
            raw = raw.rename({raw_target: "target"})
        # A probe that already emits a CONFORMING dataset passes straight through —
        # this is the path for probes whose sweep-axis names already equal the
        # experiment's (tomography's heterogeneous non-standard dims, drag_equator's
        # pre-built dataset, and the ported single-qubit probes). Experiments whose
        # probe uses non-canonical axis names (e.g. "detuning" vs "detuning_hz") do
        # NOT conform here and fall through to the positional renamer below.
        try:
            experiment.Contract.validate(raw)
            return raw
        except Exception:
            pass
        target = list(experiment.sweep_axes.keys())
        # Axis order from an actual data variable (Dataset.dims is unordered).
        ref_var = "I" if "I" in raw.data_vars else next(iter(raw.data_vars))
        raw_axes = [d for d in raw[ref_var].dims if d != "target"]
        if len(raw_axes) != len(target):
            raise NotImplementedError(
                f"sweep-axis count mismatch: raw {raw_axes} vs scqo {target}, "
                f"data_vars={list(raw.data_vars)}; {_dump_raw(raw)}"
            )
        if set(raw_axes) == set(target):
            # Name-based: already canonical; assert sizes by name, never reorder.
            for name in target:
                if raw.sizes[name] != len(experiment.sweep_axes[name]):
                    raise ValueError(
                        f"axis size mismatch: raw {name}={raw.sizes[name]} vs "
                        f"scqo {name}={len(experiment.sweep_axes[name])}; {_dump_raw(raw)}"
                    )
            return raw
        for raw_dim, name in zip(raw_axes, target):
            if raw.sizes[raw_dim] != len(experiment.sweep_axes[name]):
                raise ValueError(
                    f"axis size mismatch: raw {raw_dim}={raw.sizes[raw_dim]} vs "
                    f"scqo {name}={len(experiment.sweep_axes[name])}; {_dump_raw(raw)}"
                )
        rename = {r: t for r, t in zip(raw_axes, target) if r != t}
        return raw.rename(rename) if rename else raw


def _dump_raw(raw: xr.Dataset) -> str:
    """Pickle the raw QOP dataset for offline inspection (bring-up diagnostics)."""
    import pickle
    import tempfile
    from datetime import datetime
    from pathlib import Path

    dump = Path(tempfile.gettempdir()) / f"qm_raw_{datetime.now():%Y%m%d-%H%M%S}.pkl"
    try:
        with open(dump, "wb") as f:
            pickle.dump(raw, f)
        return f"raw dataset pickled to {dump}"
    except Exception as err:
        return f"raw dataset could not be pickled ({type(err).__name__}: {err})"
