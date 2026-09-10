"""Which QM hardware family a loaded QUAM tree runs on — the one place that answers it.

This driver serves more than one Quantum Machines product line through a single
``scqo`` backend name (``"qm"``), because they are all programmed in QUA: whatever
the chassis, ``probe()`` builds the same program. What differs is the SHAPE of the
QUAM tree — and it differs along TWO INDEPENDENT axes:

* the **RF chain**, how a drive/readout tone reaches the fridge.
  ``MWChannel`` + ``MWFEMAnalogOutputPort`` synthesizes microwave natively and
  carries ``band`` / ``upconverter_frequency`` / ``full_scale_power_dbm``.
  ``IQChannel`` + ``OctaveUpConverter`` plays a baseband IQ pair into an external
  analog upconverter and carries ``LO_frequency`` / ``gain`` / ``output_mode`` —
  plus a mixer calibration the MW-FEM does not have at all.
* the **baseband port**, what a flux (z) line's DAC is.
  ``LFFEMAnalogOutputPort`` has an ``output_mode`` (+/-0.5 V ``direct``, +/-2.5 V
  ``amplified``) and ``exponential_filter`` predistortion.
  ``OPXPlusAnalogOutputPort`` has neither: a hard +/-0.5 V rail, and predistortion
  through ``feedforward_filter`` / ``feedback_filter`` — different arithmetic, not
  a renamed field.

**The two axes are genuinely independent**, which is why nothing here answers
"OPX+ or OPX1000?". An OPX1000 can drive an Octave
(``quam_config/wiring_examples/wiring_lffem_octave.py``), so branching on the
chassis would get that tree wrong in both directions. Ask instead which RF chain a
drive/readout channel uses, and which baseband port a flux channel uses.

**Detection is DUCK-TYPED, never ``isinstance``.** Much of this repo's test suite
builds channels as ``SimpleNamespace`` stubs, and the flux guards are deliberately
pinned against a channel whose port is ``None``; an ``isinstance`` check against the
quam classes would report the wrong family for every one of them. So each answer
needs POSITIVE evidence — an attribute only that family carries — and everything
else is ``None``, which means "this tree does not say", never a family. Callers take
the conservative branch on ``None`` and say by name that they did.
"""

from __future__ import annotations

from typing import Any, Optional

#: RF-chain kinds. These strings reach the run record through
#: ``QMBackend.versions()``, so they are a stored vocabulary — keep them stable.
RF_MW_FEM = "mw_fem"
RF_OCTAVE = "octave"
RF_EXTERNAL_MIXER = "external_mixer"

#: Baseband (flux) port kinds. Same stability rule.
FLUX_LF_FEM = "lf_fem"
FLUX_OPX_PLUS = "opx_plus"

#: What a census reports for a channel whose family the tree does not declare.
UNKNOWN = "unknown"


def _has(obj: Any, name: str) -> bool:
    """True when ``obj`` exists and carries ``name``.

    A missing owner is a missing attribute, so callers never have to nest their
    own ``getattr(getattr(...))``.
    """
    return obj is not None and hasattr(obj, name)


def rf_chain(channel: Any) -> Optional[str]:
    """Which RF chain a drive or readout channel plays through.

    Returns :data:`RF_OCTAVE`, :data:`RF_MW_FEM`, :data:`RF_EXTERNAL_MIXER`, or
    ``None`` when the channel declares none of them.

    The Octave and the external-mixer converters BOTH carry ``gain``, so gain
    cannot tell them apart. ``OctaveUpConverter`` is identified by ``LO_source`` /
    ``input_attenuators`` (its own fields); the generic ``FrequencyConverter`` by
    ``local_oscillator`` / ``mixer``.

    The MW-FEM port answers to EITHER of its two exclusive fields, ``band`` and
    ``full_scale_power_dbm``. Requiring both would make the answer depend on how
    completely a caller happened to build the port -- and a partial port reading
    as "unknown" sends an absolute-power knob down the refusal path, which is
    the failure this module exists to prevent.
    """
    converter = getattr(channel, "frequency_converter_up", None)
    if converter is not None:
        if _has(converter, "LO_source") or _has(converter, "input_attenuators"):
            return RF_OCTAVE
        if _has(converter, "local_oscillator") or _has(converter, "mixer"):
            return RF_EXTERNAL_MIXER
        return None
    port = getattr(channel, "opx_output", None)
    if _has(port, "band") or _has(port, "full_scale_power_dbm"):
        return RF_MW_FEM
    return None


def flux_port_family(channel: Any) -> Optional[str]:
    """Which baseband DAC a flux (z) channel drives.

    Returns :data:`FLUX_LF_FEM`, :data:`FLUX_OPX_PLUS`, or ``None`` — the last for
    a channel with no port at all (a stub, or a QUAM class that carries none) AND
    for a port whose family the tree does not declare.

    A microwave port answers ``None`` too: it is not a baseband DAC, so naming it
    one would be a worse answer than admitting ignorance.
    """
    port = getattr(channel, "opx_output", None)
    if port is None:
        return None
    if _has(port, "band") or _has(port, "full_scale_power_dbm"):
        return None
    if _has(port, "output_mode") or _has(port, "exponential_filter"):
        return FLUX_LF_FEM
    if _has(port, "feedforward_filter"):
        return FLUX_OPX_PLUS
    return None


def tree_families(machine: Any) -> dict[str, Any]:
    """A census of the families present in a loaded tree, for the run record.

    Lists rather than single values ON PURPOSE: a tree may legitimately mix them
    (an LF-FEM flux line beside an Octave drive), and collapsing that to one label
    would hide exactly the case a reader needs to see. Sorted for stability, with
    :data:`UNKNOWN` standing in for a channel that declares nothing.

    Never raises — a census that aborts a run would be worse than a partial one.
    """
    rf: set[str] = set()
    flux: set[str] = set()
    try:
        qubits = list(getattr(machine, "qubits", {}).values())
    except Exception:
        qubits = []
    for qubit in qubits:
        for line in ("xy", "resonator"):
            channel = getattr(qubit, line, None)
            if channel is not None:
                rf.add(rf_chain(channel) or UNKNOWN)
        z = getattr(qubit, "z", None)
        if z is not None:
            flux.add(flux_port_family(z) or UNKNOWN)
    try:
        pairs = list(getattr(machine, "qubit_pairs", {}).values())
    except Exception:
        pairs = []
    for pair in pairs:
        coupler = getattr(pair, "coupler", None)
        if coupler is not None:
            flux.add(flux_port_family(coupler) or UNKNOWN)
    census: dict[str, Any] = {"rf_chain": sorted(rf), "flux_port": sorted(flux)}
    try:
        octaves = sorted(getattr(machine, "octaves", {}) or {})
    except Exception:
        octaves = []
    if octaves:
        census["octaves"] = octaves
    return census
