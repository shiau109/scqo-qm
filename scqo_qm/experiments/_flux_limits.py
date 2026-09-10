"""What volts a flux port may emit — the one place that answers it.

Every failure this module prevents is SILENT on hardware: a stored waveform peak
or a standing bias at or past its port's full scale is clipped by the DAC, the
fit still converges, and the SIMULATOR SHOWS NOTHING. So the checks run at BUILD
time and refuse by name.

**The rail is a property of the PORT, not a constant.** An OPX1000 LF-FEM analog
output reaches +/-0.5 V in ``direct`` mode and +/-2.5 V in ``amplified``.
Hardcoding 0.5 refuses every legitimate amplified-mode config outright. An OPX+
analog output has no mode switch at all: a hard +/-0.5 V. Numerically that is the
LF-FEM's ``direct`` value, but it is not the same claim, and the difference is
entirely in the REMEDY — telling someone to switch an OPX+ port to 'amplified'
sends them after a setting that does not exist on their instrument.

**Two frames, two entry points.** They mirror scqo's two flux capability mixins
(``FluxSweepParameters`` / ``FluxPulseSweepParameters``) one-for-one, so the same
vocabulary describes the experiment and the check:

* :func:`check_flux_bias_absolute` — the probe SETS the line's DC offset
  (``z.set_dc_offset``). The swept value IS the line voltage; nothing rides on
  anything.
* :func:`check_flux_pulse_relative` — the probe PLAYS on top of the standing idle
  bias (``z.play("const", amplitude_scale=...)``). The DAC emits the SUM, so a
  window that is fine on its own can still clip once it rides on a bias.

Getting that distinction backwards is itself a silent bug — adding ``idle_v`` in
the absolute frame would refuse legal sweeps, and omitting it in the relative
frame would admit clipping ones.

Probes import from here directly; they must never import scqo (they are shared
with the qualibrate path, where no capability tags exist). Under scqo the
experiment adapter picks the entry point from ``flux_frame(params)``.
"""

from typing import Optional

from scqo_qm._family import FLUX_OPX_PLUS, flux_port_family
from scqo_qm.experiments._amp_limits import MAX_AMP_SCALE

#: OPX1000 LF-FEM full-scale output (V), per port OUTPUT MODE.
_FEM_FULL_SCALE_V = {"direct": 0.5, "amplified": 2.5}

#: OPX+ analog output full-scale (V). A CONSTANT here, unlike the LF-FEM: the port
#: class carries no output_mode, so there is no second rail to reach for.
_OPX_PLUS_FULL_SCALE_V = 0.5

#: What to assume when the tree does not describe the port at all (a test stub, or
#: a QUAM class that carries no opx_output): the CONSERVATIVE rail, so an unknown
#: port refuses early rather than silently permitting a clipped waveform. Equal to
#: the OPX+ rail by arithmetic, separate from it by meaning.
_DEFAULT_FULL_SCALE_V = _FEM_FULL_SCALE_V["direct"]

#: The QUA amplitude_scale bound lives in ``_amp_limits`` — it is a property of
#: ``amp()``, not of a flux port, and it also bounds the drive/readout sweeps.
#: Imported above; a volts->scale conversion landing outside it is a build-time
#: error, not a clipped waveform.

#: The flux `const` amplitude convention: HALF the port's rail. It is what makes
#: the whole rail reachable through amp() -- scale +/-2 then spans +/-rail exactly
#: -- and it holds on both live chips (5Q4C direct 0.5 -> 0.25, 10Q amplified
#: 2.5 -> 1.25). Probes VALIDATE it rather than assume it, because the failure it
#: prevents is silent: a too-small const caps the reachable window with no error,
#: and a too-large one clips inside the DAC where the simulator shows nothing.
_CONST_AMP_RAIL_FRACTION = 0.5


def _output_mode(channel) -> Optional[str]:
    """The port's declared output mode, or None when the channel exposes no port."""
    return getattr(getattr(channel, "opx_output", None), "output_mode", None)


def dac_rail_v(channel) -> float:
    """The full-scale output voltage of a flux channel's port.

    LF-FEM: per ``output_mode``. OPX+: the fixed +/-0.5 V rail. A port the tree
    does not describe -> the conservative direct-mode rail.

    The OPX+ branch is spelled out rather than left to the fallback even though
    the two agree numerically: "this is an OPX+ port, whose rail is 0.5 V" and
    "I could not identify this port, so I am assuming 0.5 V" are different
    findings, and :func:`rail_remedy` has to tell them apart to give advice that
    exists on the operator's instrument.
    """
    if flux_port_family(channel) == FLUX_OPX_PLUS:
        return _OPX_PLUS_FULL_SCALE_V
    return _FEM_FULL_SCALE_V.get(_output_mode(channel), _DEFAULT_FULL_SCALE_V)


def rail_remedy(channel, *, name: str, needed_v: float, rail: float) -> str:
    """The one sentence every over-rail refusal ends with.

    Says what to actually DO, and says something different in each of the four
    cases -- a port that can be upgraded, a port already at its best, a port whose
    instrument has no upgrade to offer, and a port the tree cannot identify.
    Telling someone to "run that port in amplified mode" when it already IS
    amplified is worse than saying nothing; telling an OPX+ operator the same
    thing is worse still, because they will go looking for a switch their
    instrument does not have.
    """
    if flux_port_family(channel) == FLUX_OPX_PLUS:
        return (
            f"{name} needs {needed_v} V, but it is on an OPX+ analog output, whose "
            f"full scale is a fixed {rail} V -- there is no 'amplified' mode on "
            f"that instrument. Narrow the window, move the idle bias, or add "
            f"external amplification on that flux line.")
    mode = _output_mode(channel)
    if mode == "amplified":
        return (
            f"{name} needs {needed_v} V, but its port is already in 'amplified' "
            f"mode and even that tops out at {rail} V. Narrow the window or move "
            f"the idle bias.")
    if mode == "direct":
        amplified = _FEM_FULL_SCALE_V["amplified"]
        return (
            f"{name} needs {needed_v} V, but its port is in 'direct' mode "
            f"({rail} V full scale). Set output_mode=\"amplified\" ({amplified} V) "
            f"on that port in wiring.json, and set "
            f"z.operations['const'].amplitude = "
            f"{amplified * _CONST_AMP_RAIL_FRACTION} to keep the rail/2 "
            f"convention.")
    return (
        f"{name} needs {needed_v} V, but its channel does not describe an output "
        f"port this driver recognizes, so the conservative 'direct' rail of "
        f"{rail} V is assumed. Give the channel a real port; if it is an LF-FEM, "
        f"'amplified' mode reaches {_FEM_FULL_SCALE_V['amplified']} V.")


#: A TunableCoupler names its flux POINTS and its offset ATTRIBUTES differently
#: ("off" -> decouple_offset, "on" -> interaction_offset), while a qubit FluxLine
#: uses <point>_offset throughout (joint/independent/min/arbitrary). Resolving by
#: the <point>_offset convention alone raises on every coupler.
_COUPLER_POINT_ATTR = {"off": "decouple_offset", "on": "interaction_offset"}


def idle_offset_v(channel, flux_point: str) -> float:
    """The standing DC offset a given named flux point applies, in volts.

    Deliberately takes the point as an ARGUMENT rather than reading
    ``channel.flux_point``: a probe that HAS its own flux_point parameter must
    validate the bias it is actually applying, and the line's declared point is
    only the same thing while the scqo factory guard holds
    (``quam_fields.flux_point_problems``). Reading the declaration there would
    re-create the very gap that guard exists to close. A probe with no such
    parameter applies the declaration itself and uses
    :func:`declared_idle_offset_v`.
    """
    if flux_point == "zero":
        return 0.0
    for attr in (f"{flux_point}_offset", _COUPLER_POINT_ATTR.get(flux_point)):
        if attr is None:
            continue
        value = getattr(channel, attr, None)
        if value is not None:
            return float(value)
    raise ValueError(
        f"flux_point={flux_point!r} selects no offset on this flux line "
        f"(expected a '{flux_point}_offset' attribute, or for a coupler one of "
        f"{sorted(_COUPLER_POINT_ATTR.values())})")


def declared_idle_offset_v(channel) -> float:
    """The standing bias for a channel whose point the PROBE does not choose.

    ``initialize_qpu`` applies whatever ``channel.flux_point`` declares, so for a
    probe that takes no flux_point parameter the declaration IS what the hardware
    holds — there is no override that could disagree with it, which is the only
    reason reading it here is safe. Missing/unknown declaration -> 0.0, the
    permissive direction, because refusing on a stub with no flux_point would
    break probes that never bias at all.
    """
    point = getattr(channel, "flux_point", None)
    if not isinstance(point, str):
        return 0.0
    try:
        return idle_offset_v(channel, point)
    except ValueError:
        return 0.0


def flux_reference_amplitude(channel, *, name: str, operation: str = "const") -> float:
    """The stored amplitude a volts->``amplitude_scale`` conversion divides by.

    Checks only what makes THIS sweep wrong: the op must exist, be nonzero (it is
    a divisor), and sit below the rail (a stored peak at full scale is clipped
    before any scaling happens).

    It deliberately does NOT enforce the ``const`` = rail/2 convention. That is
    config HYGIENE, not per-sweep correctness — an undersized ``const`` does not
    clip anything, it only caps how far the sweep can reach, and the
    ``amplitude_scale`` bound in :func:`check_flux_pulse_relative` already
    refuses the moment a probe actually asks for more than it can express.
    Enforcing it here instead would refuse real configs that work: the live 5Q4C
    couplers sit at 0.15 V, and the chevron's ``const`` is a free parameter. The
    convention is audited tree-wide, once, by
    ``quam_fields.flux_headroom_problems``.
    """
    op = getattr(channel, "operations", {}).get(operation)
    if op is None:
        raise ValueError(
            f"{name}: flux channel has no '{operation}' operation - a flux "
            f"PULSE sweep has no amplitude reference to scale against")
    amplitude = float(getattr(op, "amplitude", 0.0))
    if amplitude == 0.0:
        raise ValueError(
            f"{name}: '{operation}' amplitude is 0 V - it is the divisor of the "
            f"volts->amplitude_scale conversion and cannot be zero")

    rail = dac_rail_v(channel)
    if abs(amplitude) >= rail:
        raise ValueError(
            f"{name}: '{operation}' amplitude is {amplitude} V, at or past the "
            f"port's {rail} V full scale, so the stored waveform peak is clipped "
            f"on hardware and the simulator hides it. "
            + rail_remedy(channel, name=f"{name} '{operation}'",
                          needed_v=abs(amplitude), rail=rail))
    return amplitude


def check_flux_bias_absolute(channel, *, name: str, bias_v) -> None:
    """Validate an ABSOLUTE flux-bias sweep (``z.set_dc_offset``).

    ``bias_v`` are the swept values in volts, and they ARE the line voltage: a DC
    offset REPLACES the standing bias rather than adding to it, so there is no
    idle term here and adding one would refuse legal sweeps. No amplitude
    reference and no ``amplitude_scale`` are involved either.
    """
    peak = max(abs(float(v)) for v in bias_v)
    rail = dac_rail_v(channel)
    if peak > rail:
        raise ValueError(
            f"{name}: the flux bias sweep reaches {peak} V, past the port's "
            f"{rail} V full scale. The DAC would clip and the SIMULATOR WOULD "
            f"NOT SHOW IT. " + rail_remedy(channel, name=name, needed_v=peak, rail=rail))


def check_flux_pulse_relative(channel, *, name: str, idle_v: float, amps_v,
                              operation: str = "const") -> float:
    """Validate a RELATIVE flux-pulse sweep; return the reference amplitude.

    ``amps_v`` are the swept values in VOLTS, measured from ``idle_v`` (the
    standing DC bias the pulse rides on). Three ways this goes wrong, all
    silent on hardware:

    1. the reference op breaks its contract -> :func:`flux_reference_amplitude`;
    2. a requested excursion needs ``|amplitude_scale| >= 2``, which QUA cannot
       express;
    3. ``idle_v + excursion`` exceeds the rail. The DAC emits the SUM, so a
       window that is fine on its own can still clip once it rides on a
       standing bias -- and nothing else in this repo checks the sum.
    """
    reference = flux_reference_amplitude(channel, name=name, operation=operation)
    rail = dac_rail_v(channel)

    # The RAIL first, then expressibility. Both can fire on the same over-wide
    # window, and "you asked for more volts than the port can emit" is the more
    # useful thing to say than "the fixed-point multiplier cannot encode it".
    lo, hi = min(float(v) for v in amps_v), max(float(v) for v in amps_v)
    total = max(abs(idle_v + hi), abs(idle_v + lo))
    if total > rail:
        raise ValueError(
            f"{name}: the flux window rides on the standing bias, and "
            f"{idle_v} V idle + the window [{lo}, {hi}] V reaches {total} V, "
            f"past the port's {rail} V full scale. The DAC would clip and the "
            f"SIMULATOR WOULD NOT SHOW IT. "
            + rail_remedy(channel, name=name, needed_v=total, rail=rail))

    peak = max(abs(float(v)) for v in amps_v)
    scale = peak / abs(reference)
    if scale >= MAX_AMP_SCALE:
        raise ValueError(
            f"{name}: flux window peak {peak} V needs amplitude_scale "
            f"{scale:.3f}, outside QUA's +/-{MAX_AMP_SCALE} range "
            f"(reference {reference} V from '{operation}'). The reachable "
            f"excursion is +/-{MAX_AMP_SCALE * abs(reference)} V. "
            f"Raise the stored '{operation}' amplitude to widen it.")
    return reference
