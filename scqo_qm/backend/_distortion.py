"""Neutral flux-distortion facts -> QM OPX ``exponential_filter`` config values.

Pure config-value helpers (no push, no quam import): turn the scqo flux-channel
facts ``distortion_amp[]`` (relative amplitudes ``A_i``) + ``distortion_tau_s[]``
(seconds) into the value a QM LF-FEM z port's ``opx_output.exponential_filter``
accepts. The live OPX (QOP >= 3.3) takes the SUM form directly; QOP 3.4.1 takes
the single-pole CASCADE (from scqat), with an overall ``scale`` applied to the
FIR/waveform. Amplitudes are already relative (``amp/a_dc``), so the SUM mapping
is purely ``tau: seconds -> ns``.

These are library helpers a notebook/CLI calls with the recorded facts.
``apply_exponential_filter`` additionally WRITES the value onto a loaded QUAM (still
no quam import — it duck-types ``qubits[t].z.opx_output`` — and never persists; the
caller saves). Wiring any of this into an auto-push knob is a separate decision: the
``distortion_*`` facts stay physical.json record-only today.
"""

from __future__ import annotations

from typing import Any, Sequence

from scqo_qm._family import FLUX_OPX_PLUS, flux_port_family

#: OPX sample period (1 GS/s), seconds. The same on an OPX+ analog output (where
#: it is a ClassVar) and on an LF-FEM at its 1 GS/s setting.
OPX_TS_S = 1e-9


def _exponential_filter_port(machine, target: str, *, noun: str):
    """The z-line output port for ``target``, or a refusal naming why there is none.

    The ONE door both entry points below resolve their port through, because the
    failure it prevents is otherwise INVISIBLE. ``exponential_filter`` is an
    ``LFFEMAnalogOutputPort`` field; an ``OPXPlusAnalogOutputPort`` does not have
    it, and quam does not object to inventing one — the assignment succeeds, the
    attribute lives on the instance, and then ``to_dict()`` drops it and
    ``get_port_properties()`` never looks. So without this guard an operator gets
    a printed success, a saved state.json with no filter in it, and a flux line
    that is still distorted.

    OPX+ predistortion exists, but as ``feedforward_filter`` (FIR) +
    ``feedback_filter`` (IIR) — a different decomposition, not a renamed field —
    so the remedy is a conversion this driver does not do yet, not a retry.
    """
    try:
        qubit = machine.qubits[target]
    except (KeyError, TypeError):
        try:
            have = sorted(machine.qubits)
        except Exception:
            have = "?"
        raise ValueError(
            f"{target!r} is not a qubit in this machine (have {have})") from None
    z = getattr(qubit, "z", None)
    if z is None:
        raise ValueError(
            f"{target!r} has no flux (z) line — no exponential_filter to {noun}")
    if flux_port_family(z) == FLUX_OPX_PLUS:
        raise ValueError(
            f"{target}.z is on an OPX+ analog output, which has no "
            f"exponential_filter — flux predistortion there is feedforward_filter "
            f"(FIR) + feedback_filter (IIR), a different decomposition this driver "
            f"does not write yet. Refusing rather than {noun}ing: quam would ACCEPT "
            f"the assignment and then drop it on save, so the filter would silently "
            f"never reach the port.")
    return z.opx_output


def to_exponential_filter(
    amps: Sequence[float], taus_s: Sequence[float]
) -> list[list[float]]:
    """The QOP >= 3.3 SUM value for ``z.opx_output.exponential_filter``:
    ``[[A_i, tau_i_ns], ...]`` — amplitudes verbatim (already relative), tau s->ns.
    """
    if len(amps) != len(taus_s):
        raise ValueError(
            f"amps ({len(amps)}) and taus_s ({len(taus_s)}) must be equal length")
    return [[float(a), float(t) * 1e9] for a, t in zip(amps, taus_s)]


def to_exponential_filter_cascade(
    amps: Sequence[float], taus_s: Sequence[float], *, ts_s: float = OPX_TS_S
) -> dict[str, Any]:
    """The QOP 3.4.1 single-pole CASCADE value + scale:
    ``{"exponential_filter": [[A_c, tau_c_ns], ...], "scale": float}``.

    Apply ``scale`` to the FIR coefficients or the flux-waveform amplitude per your
    QOP. The facts are relative (a_dc factored out), so ``a_dc=1.0`` is passed to
    the decomposition (which also satisfies its ``a_dc > MIN_A_DC`` guard).
    """
    from scqat.tools.flux_predistortion import exp_sum_to_cascade

    casc = exp_sum_to_cascade(amps, taus_s, a_dc=1.0, ts_s=ts_s)
    return {
        "exponential_filter": [
            [a, t * 1e9] for a, t in zip(casc["amps_c"], casc["taus_c_s"])
        ],
        "scale": casc["scale"],
    }


def clear_exponential_filter(machine, target: str) -> dict[str, Any]:
    """Remove ALL taps from ``machine.qubits[target].z.opx_output.exponential_filter``
    — the fresh-line reset before a clean-slate cryoscope characterization (a
    full correction must be MEASURED on a cleared line). Returns
    ``{"removed": [the taps that were set]}``; does NOT persist (caller saves).
    Same target/z guards as :func:`apply_exponential_filter`.
    """
    port = _exponential_filter_port(machine, target, noun="clear")
    removed = [list(pair) for pair in (port.exponential_filter or [])]
    port.exponential_filter = []
    return {"removed": removed}


def apply_exponential_filter(
    machine, target: str, amps: Sequence[float], taus_s: Sequence[float], *,
    replace: bool = True, form: str = "sum", ts_s: float = OPX_TS_S,
) -> dict[str, Any]:
    """Write the cryoscope distortion taps to a qubit's OPX z-output filter.

    Sets ``machine.qubits[target].z.opx_output.exponential_filter`` from the
    recorded flux facts and returns ``{"exponential_filter": <written value>,
    "scale": float}``. The taps are fed as MEASURED (relative ``A_i``, ``tau`` in
    seconds) — the OPX firmware builds the inverse, so do NOT pre-invert them; this
    mirrors ``calibrations/18_cryoscope.py``.

    Args:
        machine: the loaded QUAM (``Quam.load(<setup backend_config dir>)`` — the
            same state.json the scqo session loads; ``state_sync="pull"`` makes it
            authoritative).
        target: qubit name, e.g. ``"q1"`` (the flux channel is ``<target>.z``).
        amps: ``distortion_amp`` — relative amplitudes ``A_i = amp/a_dc``.
        taus_s: ``distortion_tau_s`` — time constants in SECONDS.
        replace: ``True`` OVERWRITES the port filter (scqo REPLACE fact semantics —
            for a fresh full correction measured on a cleared line); ``False``
            EXTENDS it (iterative refinement of a residual measured with the current
            filter active).
        form: ``"sum"`` (QOP >= 3.3, the direct sum value; ``scale`` is ``1.0``) or
            ``"cascade"`` (QOP 3.4.1 single-pole cascade — ALSO apply the returned
            ``scale`` to the flux-waveform amplitude yourself).
        ts_s: OPX sample period, only used by the cascade discretization.

    Does NOT persist — call ``machine.save()`` afterwards (batch several targets
    first if you like). Raises ``ValueError`` for an unknown ``target``, a target
    with no ``z`` line, an unknown ``form``, or ``form="cascade"`` with
    ``replace=False`` (a cascade is a whole-response decomposition, not a stack).
    """
    port = _exponential_filter_port(machine, target, noun="set")

    if form == "sum":
        value: list = to_exponential_filter(amps, taus_s)
        scale = 1.0
    elif form == "cascade":
        if not replace:
            raise ValueError(
                "form='cascade' is a whole-response decomposition and cannot be "
                "EXTENDED onto existing taps — use replace=True")
        casc = to_exponential_filter_cascade(amps, taus_s, ts_s=ts_s)
        value, scale = casc["exponential_filter"], casc["scale"]
    else:
        raise ValueError(f"unknown form {form!r} (use 'sum' or 'cascade')")

    if replace or not port.exponential_filter:
        # wholesale (re)assignment builds a fresh list of plain [A, tau] pairs.
        port.exponential_filter = list(value)
    else:
        # EXTEND in place: reassigning old+new re-parents the existing QuamList
        # children and QUAM refuses that ("Cannot overwrite parent attribute").
        # Appending fresh pairs is how 18_cryoscope grows the filter.
        port.exponential_filter.extend(value)
    return {"exponential_filter": port.exponential_filter, "scale": scale}
