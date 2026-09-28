"""What an MW-FEM output port IS, for the code that retunes or extends one.

Facts only - the band table, the port-pair rule, the IF window this repo plays,
and the port's LO in either of QUAM's two spellings. Package root (like
``_octave.py``) because ``backend/`` and ``experiments/`` both need it and
experiments cannot import from backend without cycling.

TWO SPELLINGS OF ONE LO. A port declares its local oscillator either as
``upconverter_frequency`` - one upconverter, number 1 - or as an
``upconverters`` dict ``{1: {"frequency": ...}, 2: {"frequency": ...}}`` once a
second element rides the port on upconverter 2 (an adopted borrowed drive
channel: ``scqo-qm adopt-channel``, SCQO docs/coupler-transmon-plan.md). qm-qua
refuses a port that carries both, and quam's ``MWChannel`` reads the scalar
whenever it is set, ignoring its own ``upconverter``. So nothing reads or writes
the attribute directly: every caller goes through :func:`port_los` /
:func:`port_lo` / :func:`set_port_lo`.

PORT PAIRS SHARE A BAND. (2,3), (4,5), (6,7) must carry the same band (a lab
rule, not stated in qm/quam); each port keeps its own LO(s).
"""

from __future__ import annotations

from typing import Any

#: band -> (lo_min, lo_max) in Hz (qualang_tools OPX1000_MW_BANDS)
MW_FEM_BANDS: dict[int, tuple[float, float]] = {
    1: (0.05e9, 5.5e9),
    2: (4.5e9, 7.5e9),
    3: (6.5e9, 10.5e9),
}

#: the IF window one LO plays on an MW-FEM (Hz): a repo convention, not a vendor
#: bound (the Octave's hardware +/-400 MHz lives in _octave.IF_MAX_ABS_HZ)
MAX_IF_HZ = 250.0e6


def port_info(opx_out: Any) -> tuple[str, int, int] | None:
    """``(controller, fem, port_id)`` of an output port object, or None. Tries the
    attribute names several QUAM versions use."""
    ctrl = getattr(opx_out, "controller_id", None) or getattr(opx_out, "controller", None)
    fem = getattr(opx_out, "fem_id", None) or getattr(opx_out, "fem", None)
    port = getattr(opx_out, "port_id", None) or getattr(opx_out, "port", None)
    if ctrl is None or fem is None or port is None:
        return None
    return (str(ctrl), int(fem), int(port))


def partner_port_id(port_id: int) -> int:
    """The port sharing ``port_id``'s band: even ports pair with the next odd
    one, odd ports with the previous even one."""
    return port_id + 1 if port_id % 2 == 0 else port_id - 1


def band_holds(band: int, *los_hz: float) -> bool:
    """True when every LO lies inside ``band``'s range."""
    lo_min, lo_max = MW_FEM_BANDS[band]
    return all(lo_min <= float(lo) <= lo_max for lo in los_hz)


def port_los(port: Any) -> dict[int, float]:
    """``{upconverter: LO Hz}`` of one output port, whichever spelling it uses
    (empty when it declares none)."""
    ups = getattr(port, "upconverters", None)
    if ups:
        return {int(k): float(v["frequency"]) for k, v in ups.items()}
    lo = getattr(port, "upconverter_frequency", None)
    return {} if lo is None else {1: float(lo)}


def port_lo(port: Any, upconverter: int = 1) -> float | None:
    """One upconverter's LO (Hz), or None when the port does not declare it."""
    return port_los(port).get(int(upconverter))


def set_port_lo(port: Any, upconverter: int, hz: float) -> None:
    """Retune one upconverter, in the spelling the port already uses.

    A scalar port has upconverter 1 only: asking it for another is refused - a
    SECOND upconverter is created by converting the port to the dict form, which
    is ``adopt-channel``'s business, never a retune's."""
    ups = getattr(port, "upconverters", None)
    if ups:
        if int(upconverter) not in {int(k) for k in ups}:
            raise KeyError(
                f"this port declares upconverters {sorted(int(k) for k in ups)}, "
                f"not {upconverter}")
        ups[int(upconverter)]["frequency"] = float(hz)
        return
    if int(upconverter) != 1:
        raise KeyError(
            f"this port declares one upconverter (upconverter_frequency); "
            f"upconverter {upconverter} does not exist on it")
    port.upconverter_frequency = float(hz)


def second_upconverter(port: Any) -> float | None:
    """The LO of the port's upconverter 2 (an adopted channel rides it), or None."""
    return port_lo(port, 2)
