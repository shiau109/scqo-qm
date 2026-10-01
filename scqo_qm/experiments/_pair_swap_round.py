"""The repeated-swap round shared by ``qc_n_stark_amp`` and ``qc_n_swap_tomography``.

Per shot: reset every involved qubit, excite the swap pair's CONTROL with
``x180`` at its resonant IF, detune the control's xy to the stark IF, then ``N``
times: the bare swap macro (its baked amplitudes), an idle of
``operation_gap_ns`` on the pair, and the stark tone on the control's xy at the
swept ``amplitude_scale``; restore the resonant IF. What follows (a readout, or
tomography pre-rotations and then a readout) is the caller's.

``update_frequency`` keeps its default ``keep_phase=False`` - the COHERENT mode:
switching back to the base IF gives the phase the base IF would have had all
along, so the control's idle frame is intact for whatever plays next. The
tomography relies on that.

The two experiments must play the SAME round (their compensations are compared
directly), which is why it lives once, here. The QUA it emits is the one
``qc_n_stark_amp`` emitted before the split, statement for statement.
"""

from __future__ import annotations

from typing import List

from qm.qua import align, for_

from scqo_qm.experiments._amp_limits import check_amp_scale_window


def dedup_involved(measure_qubits, swap_pair) -> List:
    """The unique (by ``.name``) qubit elements to initialize every shot: the
    measured qubits plus the swap pair's control and target."""
    involved = []
    seen = set()
    for q in list(measure_qubits) + [swap_pair.qubit_control, swap_pair.qubit_target]:
        if q.name not in seen:
            seen.add(q.name)
            involved.append(q)
    return involved


def gap_cycles(operation_gap_ns: int) -> int:
    """The gap in 4 ns clock cycles; refuses a negative or off-grid gap."""
    if operation_gap_ns < 0 or operation_gap_ns % 4 != 0:
        raise ValueError(f"operation_gap_ns must be a non-negative multiple of 4 ns, got {operation_gap_ns}.")
    return operation_gap_ns // 4


def check_swap_round(swap_pair, *, swap_operation: str, stark_operation: str, stark_amps) -> None:
    """Refuse, before any QUA is built, a round the pair cannot play: the swap
    macro and its control-z flux pulse must exist, the stark tone must exist on
    the control's xy line, and its amplitude factors must be expressible."""
    ctrl = swap_pair.qubit_control
    if swap_operation not in swap_pair.macros:
        raise ValueError(f"Pair {swap_pair.name} has no macro {swap_operation!r}; available: {list(swap_pair.macros)}.")
    flux_pulse_name = getattr(swap_pair.macros[swap_operation], "flux_pulse", None)
    if not isinstance(flux_pulse_name, str) or flux_pulse_name not in ctrl.z.operations:
        raise ValueError(
            f"Macro {swap_operation!r} on {swap_pair.name} has no z flux_pulse playable on "
            f"{ctrl.name}.z (flux_pulse={flux_pulse_name!r})."
        )
    if stark_operation not in ctrl.xy.operations:
        raise ValueError(
            f"Qubit {ctrl.name} has no xy operation {stark_operation!r}; available: "
            f"{list(ctrl.xy.operations)}. Register it first (quam_config/register_stark.py)."
        )
    check_amp_scale_window(stark_amps, name=f"{ctrl.name}.xy {stark_operation!r}",
                           knob="max_stark_amp")


def stark_ifs(swap_pair, stark_detuning_hz: float) -> tuple[int, int]:
    """``(base_if, stark_if)`` of the control's xy: resonant, and detuned."""
    base_if = swap_pair.qubit_control.xy.intermediate_frequency
    return base_if, int(stark_detuning_hz) + base_if


def play_swap_round(involved, swap_pair, *, swap_operation: str, stark_operation: str,
                    stark_amp, swap_count, counter, gap: int, base_if: int, stark_if: int,
                    reset_type: str, simulate: bool) -> None:
    """Emit one shot's prefix: reset, prep, ``swap_count`` rounds, IF restored.

    ``stark_amp`` / ``swap_count`` are QUA variables (or values) of the caller's
    loops; ``counter`` is a declared QUA int for the inner round loop; ``gap`` is
    in clock cycles (``gap_cycles``). Ends with an ``align()``."""
    ctrl = swap_pair.qubit_control

    # Initialization: thermalize / actively reset every involved qubit.
    for q in involved:
        q.reset(reset_type, simulate)
    align()

    # State prep: excite the swap pair's control qubit to |1> (resonant IF).
    ctrl.xy.play("x180")
    align()

    # Detune the control's xy so the stark tone is off-resonant (a Stark
    # SHIFT, not a rotation). Hoisted outside the swap loop: the control's
    # xy frequency only matters for the stark play, and x180 (above) and
    # whatever follows the round run at the resonant IF.
    ctrl.xy.update_frequency(stark_if)

    # N swaps on the pair, each bare (fixed baked amplitude), each followed
    # SEQUENTIALLY by the off-resonant stark tone on the control xy. The swap
    # and the stark do NOT overlap: apply() ends with the pair's align()
    # (FluxTunableTransmonPair.align aligns every channel of both qubits,
    # INCLUDING control.xy). Dynamic loop bound -> N=0 skips the body
    # entirely. `gap` idles the pair BETWEEN the swap and the stark tone, so
    # the swap's flux pulse settles before the stark tone plays.
    with for_(counter, 0, counter < swap_count, counter + 1):
        swap_pair.macros[swap_operation].apply()
        if gap > 0:
            swap_pair.wait(gap)
        ctrl.xy.play(stark_operation, amplitude_scale=stark_amp)
        align()

    # Restore the resonant IF.
    ctrl.xy.update_frequency(base_if)
    align()
