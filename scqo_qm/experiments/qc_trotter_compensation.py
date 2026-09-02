"""Trotter-chain compensation scan: vendor code only (qm/quam/qualang_tools)
- no qualibrate, no scqo, no scqat in the builder half.

``qc_unidirectional_trotter`` with ONE extra swept axis: the AC-Stark
compensation amplitude on a single chosen chain qubit. The round is otherwise
identical, and deliberately so -- this scan exists to calibrate that exact
sequence, so any divergence would calibrate something else.

Circuit per shot (for a swept compensation amplitude a and step count N):
  1. Initialize every involved qubit with ``q.reset(reset_type, simulate)``.
  2. State prep: ``prep_qubit.xy.play(prep_operation)``, ONCE, at the resonant IF.
  3. Detune every compensated qubit's xy by ``stark_detuning_hz``
     (``update_frequency``), hoisted out of the loops.
  4. Repeat N times: bare ``first_pair`` swap, bare ``second_pair`` swap, bare
     ``reset_qubit`` reset, then the Stark tones -- the FIXED ones at their
     Python-float factors and the swept one at the QUA variable ``a``.
  5. Restore the resonant IFs, then read out every measured qubit.

THE SWEPT TONE IS A QUA VARIABLE, the fixed ones are compile-time constants.
``xy.play(op, amplitude_scale=...)`` accepts either, so the swept qubit's tone
costs no extra program structure -- the amplitude loop simply wraps the round
loop, outside the shot loop's body and inside the shot loop itself, so every
amplitude sees the same thermal history.

THE SWEPT QUBIT ALWAYS PLAYS, including at amplitude 0. A qubit absent from the
tone list would shorten the round and change the very phase being measured, so
the baseline point is a zero-amplitude PLAY, not a skipped one -- the same reason
``qc_unidirectional_trotter`` documents for a factor of 0.0.

READOUT. The probe always discriminates on the FPGA. ``keep_shots`` chooses what
the stream does with them: True keeps every shot (``state`` over
``(qubit, shot_idx, compensation_amp, round_count)``); False averages on the FPGA
into one population per (amplitude, round). Unlike the chain probe there is no
joint-distribution path downstream -- see the scqo module docstring.

QM Trotter-chain compensation scan for scqo -- supplies ``probe()``.

Parameters, the record-only summary and the (absent) writeback are inherited from
``scqo.experiments.QcTrotterCompensation``. This adapter resolves the chain roles
through scqo's own ``chain_roles`` (so the neutral layer and the probe can never
disagree about which member is the relay), joins every roster name to its QUAM
object through the ``_vendor`` door, and renames the raw axes onto the readout
schema in ``reduce_raw``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import xarray as xr
from qm.qua import *

from qualang_tools.loops import from_array

from scqo_qm.experiments._lib import acquire as _acquire
from scqo_qm.experiments._amp_limits import check_amp_scale_window
from scqo_qm.experiments._coupler_knob import guard_coupler_amplitudes


def _dedup_involved(measure_qubits, pairs, extra) -> List:
    """The unique (by ``.name``) qubit elements that must be initialized.

    The measured qubits need not include the pairs' members, the reset qubit or
    the prep qubit, but every one of them is pulsed and so must be
    flux-initialized and reset at the start of each shot."""
    involved: List = []
    seen = set()
    candidates = list(measure_qubits)
    for pair in pairs:
        candidates.extend([pair.qubit_control, pair.qubit_target])
    candidates.extend(extra)
    for qubit in candidates:
        if qubit.name not in seen:
            seen.add(qubit.name)
            involved.append(qubit)
    return involved


def _check_macro(holder, operation: str, what: str) -> None:
    """Refuse a missing bare-callable macro BY NAME, before any QUA is built."""
    macros = getattr(holder, "macros", {}) or {}
    if operation not in macros:
        raise ValueError(
            f"{what} {holder.name} has no macro {operation!r}; available: "
            f"{sorted(macros)}. Register it first (quam_config/"
            f"register_swap_macro.py for a pair swap, register_reset_macro.py "
            f"for a parametric reset).")


def build_program(
    machine,
    measure_qubits,
    first_pair,
    second_pair,
    reset_qubit,
    *,
    prep_qubit,
    prep_operation: str,
    swap_operation: str,
    reset_operation: str,
    stark_operation: str,
    stark_detuning_hz: float,
    compensation: Sequence[tuple],
    swept_qubit,
    compensation_amps,
    rounds_array,
    num_shots: int,
    reset_type: str,
    keep_shots: bool,
    operation_gap_ns: int = 0,
    first_coupler_amp: Optional[float] = None,
    second_coupler_amp: Optional[float] = None,
    simulate: bool = False,
):
    """Build the Trotter-chain compensation-scan QUA program.

    Returns ``(program, sweep_axes)``.

    ``measure_qubits`` is a plain list of qubit objects read out at the end of
    the circuit, IN CHAIN ORDER. ``compensation`` is a sequence of
    ``(qubit object, amplitude factor)`` pairs held FIXED; ``swept_qubit`` is the
    one whose tone carries the swept ``compensation_amps`` factors instead, and
    it must NOT also appear in ``compensation`` (scqo refuses that upstream, and
    it is asserted here because a doubled tone would be silent).
    """
    measure_qubits = list(measure_qubits)
    num_qubits = len(measure_qubits)
    rounds_array = np.asarray(rounds_array).astype(int)
    compensation_amps = np.asarray(compensation_amps, dtype=float)
    compensation = list(compensation)

    if operation_gap_ns < 0 or operation_gap_ns % 4 != 0:
        raise ValueError(
            f"operation_gap_ns must be a non-negative multiple of 4 ns, got "
            f"{operation_gap_ns}.")
    gap_cycles = operation_gap_ns // 4

    # Everything that can refuse, refuses HERE - before a single QUA statement,
    # so a mis-registered chip costs no instrument time.
    _check_macro(first_pair, swap_operation, "first_pair")
    _check_macro(second_pair, swap_operation, "second_pair")
    _check_macro(reset_qubit, reset_operation, "reset_qubit")
    if prep_operation not in prep_qubit.xy.operations:
        raise ValueError(
            f"prep_qubit {prep_qubit.name} has no xy operation "
            f"{prep_operation!r}; available: {list(prep_qubit.xy.operations)}.")
    fixed_names = {qubit.name for qubit, _factor in compensation}
    if swept_qubit.name in fixed_names:
        raise ValueError(
            f"{swept_qubit.name} is both the swept compensation target and a "
            f"fixed tone - it would play TWICE per round, which is silent in the "
            f"data and doubles the phase. Remove it from compensation_amps.")
    for qubit in [q for q, _f in compensation] + [swept_qubit]:
        if stark_operation not in qubit.xy.operations:
            raise ValueError(
                f"Qubit {qubit.name} has no xy operation {stark_operation!r}; "
                f"available: {list(qubit.xy.operations)}. Register it first "
                f"(quam_config/register_stark.py).")
    check_amp_scale_window([factor for _q, factor in compensation],
                           name="stark compensation",
                           knob="compensation_amps")
    check_amp_scale_window(compensation_amps,
                           name=f"swept stark compensation on {swept_qubit.name}",
                           knob="max_compensation_amp")
    # The swap ANGLE knob, when a run sets it (see _coupler_knob). A pair left at
    # None plays its baked coupler amplitude and is not checked.
    for pair, amp in ((first_pair, first_coupler_amp),
                      (second_pair, second_coupler_amp)):
        if amp is not None:
            guard_coupler_amplitudes(
                pair, swap_operation, [float(amp)],
                why="swap_coupler_flux sets the swap angle through the coupler.",
                label=f"{pair.name} swap_coupler_flux")

    involved = _dedup_involved(measure_qubits, [first_pair, second_pair],
                               [reset_qubit, prep_qubit])
    # Every qubit that plays a tone needs its IF detuned, the swept one included.
    toned = [qubit for qubit, _factor in compensation] + [swept_qubit]
    base_if = {qubit.name: qubit.xy.intermediate_frequency for qubit in toned}
    stark_if = {name: int(stark_detuning_hz) + value
                for name, value in base_if.items()}

    first_swap_kwargs = ({} if first_coupler_amp is None
                         else {"cplr_amp": float(first_coupler_amp)})
    second_swap_kwargs = ({} if second_coupler_amp is None
                          else {"cplr_amp": float(second_coupler_amp)})

    axes: Dict[str, xr.DataArray] = {
        "qubit": xr.DataArray([q.name for q in measure_qubits]),
    }
    if keep_shots:
        axes["shot_idx"] = xr.DataArray(np.arange(num_shots))
    # Outer sweep -> compensation amplitude; inner -> the Trotter-step count.
    axes["compensation_amp"] = xr.DataArray(
        compensation_amps,
        attrs={"long_name": "stark compensation amplitude", "units": ""})
    axes["round_count"] = xr.DataArray(
        rounds_array, attrs={"long_name": "Trotter steps", "units": ""})
    sweep_axes = axes

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        a = declare(fixed)  # swept compensation amplitude factor
        r = declare(int)    # swept Trotter-step count (current value)
        rr = declare(int)   # inner round counter
        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]

        for qubit in involved:
            machine.initialize_qpu(target=qubit)
        align()

        with for_(n, 0, n < num_shots, n + 1):
            save(n, n_st)
            # Compensation-amplitude loop (outer) INSIDE the shot loop, so every
            # amplitude is sampled once per shot and shares the drift epoch: an
            # outermost amplitude loop would let the chip walk between the ends
            # of the sweep and read as a phase optimum that is really drift.
            with for_(*from_array(a, compensation_amps)):
                with for_(*from_array(r, rounds_array)):
                    for qubit in involved:
                        qubit.reset(reset_type, simulate)
                    align()

                    # State prep: ONE excitation on the chain source, resonant IF.
                    prep_qubit.xy.play(prep_operation)
                    align()

                    # Detune the compensated qubits so their tones shift rather
                    # than rotate. Nothing between prep and readout needs the
                    # resonant IF.
                    for qubit in toned:
                        qubit.xy.update_frequency(stark_if[qubit.name])

                    # The Trotter step, N times. A dynamic loop bound on r means
                    # N=0 skips the body entirely (the prep-only baseline).
                    with for_(rr, 0, rr < r, rr + 1):
                        first_pair.macros[swap_operation].apply(**first_swap_kwargs)
                        if gap_cycles > 0:
                            first_pair.wait(gap_cycles)
                        align()
                        second_pair.macros[swap_operation].apply(**second_swap_kwargs)
                        if gap_cycles > 0:
                            second_pair.wait(gap_cycles)
                        align()
                        reset_qubit.macros[reset_operation].apply()
                        if gap_cycles > 0:
                            reset_qubit.wait(gap_cycles)
                        align()
                        # Concurrent by design: one compensation instant per
                        # round. The swept tone always plays, at amplitude 0 too,
                        # so the round keeps its duration at the baseline point.
                        for qubit, factor in compensation:
                            qubit.xy.play(stark_operation, amplitude_scale=factor)
                        swept_qubit.xy.play(stark_operation, amplitude_scale=a)
                        align()

                    # Restore the resonant IFs before readout.
                    for qubit in toned:
                        qubit.xy.update_frequency(base_if[qubit.name])
                    align()

                    # Joint (multiplexed) readout of every measured qubit.
                    for i, qubit in enumerate(measure_qubits):
                        qubit.readout_state(state[i])
                        save(state[i], state_st[i])
                    align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                # Inner buffer = the round axis, then the amplitude axis; shot
                # mode groups num_shots of them, average mode collapses them on
                # the FPGA into one population per (amplitude, round).
                stream = (state_st[i]
                          .buffer(len(rounds_array))
                          .buffer(len(compensation_amps)))
                if keep_shots:
                    stream = stream.buffer(num_shots)
                else:
                    stream = stream.average()
                stream.save(f"state{i + 1}")

    return prog, sweep_axes


def acquire(
    machine,
    prog,
    sweep_axes,
    *,
    num_shots: int,
    timeout: float,
    log: Optional[Callable] = None,
) -> xr.Dataset:
    """Connect to the QOP, execute the program and fetch the raw xr.Dataset."""
    return _acquire(machine, prog, sweep_axes, num_shots=num_shots,
                    timeout=timeout, log=log)


from scqo import register
from scqo.experiments import QcTrotterCompensation
from scqo.experiments.qc_unidirectional_trotter import chain_roles


@register
class QMQcTrotterCompensation(QcTrotterCompensation):
    """Build, run and fetch the Trotter-chain compensation scan on the QM OPX."""

    def probe(self) -> Any:
        from ._reset import check_reset_method
        from ._vendor import vendor_pair, vendor_qubit

        # One door for the reset method: this shell does not opt into active
        # reset (default DENY). Note this is the BETWEEN-SHOTS reset, not the
        # mid-circuit parametric one the round plays.
        reset = check_reset_method(self)
        # The chain topology comes from scqo's own resolver, so the neutral layer
        # and the probe can never disagree about which member is the relay.
        _source, _relay, _sink, prep = chain_roles(self.device.roster, self.params)

        machine = self.backend.machine  # type: ignore[attr-defined]
        # Chain order is the TARGET order, and it survives into the readout list.
        measure_qubits = [vendor_qubit(self, name, field="targets")
                          for name in self.params.targets]
        compensation = [
            (vendor_qubit(self, name, field="compensation_amps"), float(factor))
            for name, factor in self.params.compensation_amps.items()
        ]
        # keyed by PAIR name in scqo's surface; scqo's chain_roles has already
        # refused any name that is not one of the two declared pairs.
        coupler_flux = dict(self.params.swap_coupler_flux or {})
        return build_program(
            machine,
            measure_qubits,
            vendor_pair(self, self.params.first_pair),
            vendor_pair(self, self.params.second_pair),
            # the parametric reset is a FLUX-line technique, so the roster join
            # goes through the z channel
            vendor_qubit(self, self.params.reset_qubit, field="reset_qubit",
                         kind="flux"),
            prep_qubit=vendor_qubit(self, prep, field="prep_qubit"),
            prep_operation=self.params.prep_operation,
            swap_operation=self.params.swap_operation,
            reset_operation=self.params.reset_operation,
            stark_operation=self.params.stark_operation,
            stark_detuning_hz=self.params.stark_detuning_hz,
            compensation=compensation,
            swept_qubit=vendor_qubit(self, self.params.compensation_target,
                                     field="compensation_target"),
            compensation_amps=np.asarray(
                self.sweep_axes["compensation_amp"], dtype=float),
            rounds_array=np.asarray(self.sweep_axes["round_count"]).astype(int),
            num_shots=int(self.params.num_averages),
            reset_type=reset,
            keep_shots=self.params.readout_mode == "shot",
            operation_gap_ns=int(self.params.operation_gap_ns),
            first_coupler_amp=coupler_flux.get(self.params.first_pair),
            second_coupler_amp=coupler_flux.get(self.params.second_pair),
        )

    def reduce_raw(self, raw: xr.Dataset) -> xr.Dataset:
        """The readout schema's naming for the two modes.

        Shot mode keeps ``state`` -- per-shot integer LEVELS, which is what that
        name means. Average mode's stream is already the FPGA mean of those
        levels, i.e. the marginal ``population``; the backend's own
        state->population rename cannot do it here because this contract also
        ACCEPTS ``state`` (the shot form), so the rename is explicit."""
        if self.params.readout_mode == "shot":
            return raw
        return raw.rename({"state": "population"})
