"""Unidirectional-coupling Trotter chain: vendor code only (qm/quam/qualang_tools)
- no qualibrate, no scqo, no scqat in the builder half.

The two-pair descendant of the retired ``qc_swap_paramreset`` probe (one pair +
one reset qubit, R rounds), extended with a SECOND swap pair and a per-qubit
AC-Stark phase compensation borrowed from ``qc_n_stark_amp``.

Circuit per shot (for a swept Trotter-step count N):
  1. Initialize every involved qubit with ``q.reset(reset_type, simulate)``
     (involved = the measured qubits + both pairs' members + the reset qubit +
     the prep qubit).
  2. State prep: ``prep_qubit.xy.play(prep_operation)``, ONCE, at the resonant
     IF - the chain is watched from a single excitation, not pumped.
  3. Detune every compensated qubit's xy by ``stark_detuning_hz``
     (``update_frequency``), so its tone SHIFTS the qubit rather than rotating
     it. Hoisted out of the loop: prep (above) and readout (below) both need the
     resonant IF, and nothing between them does.
  4. Repeat N times: bare ``first_pair.macros[swap_operation].apply()``, bare
     ``second_pair.macros[swap_operation].apply()``, bare
     ``reset_qubit.macros[reset_operation].apply()``, then one
     ``xy.play(stark_operation, amplitude_scale=a)`` per compensated qubit.
  5. Restore the resonant IFs, then read out every measured qubit.

The macros are invoked BARE (``.apply()`` with no arguments), so each must be
callable that way - amplitudes and pulses baked into the QUAM macro definition.
``quam_config/register_swap_macro.py`` and ``register_reset_macro.py`` write
exactly such macros; a macro whose ``apply()`` has required positional arguments
needs a default-carrying variant before it can be named here.

WHY A GLOBAL ``align()`` BETWEEN EVERY OPERATION. The two pairs SHARE their
relay member, so their own ``apply()`` aligns overlap only partially: the first
swap's align covers source+relay, the second's covers relay+sink, and the sink's
timeline is not in the first. Relying on the pair aligns to serialize the round
would make correctness depend on which channels each macro happens to touch.
The round is a sequential circuit, not a latency-critical one, so it is aligned
explicitly at every step and the question does not arise. The per-qubit Stark
tones, by contrast, are deliberately CONCURRENT with each other (different xy
lines, one shared compensation instant), bracketed by aligns on both sides.

READOUT. The probe always discriminates on the FPGA - this experiment has no
I/Q form. ``keep_shots`` chooses what the stream does with them: True keeps
every shot (``state`` over ``(qubit, shot_idx, round_count)``), which is what
makes the JOINT chain distribution reconstructable downstream; False averages on
the FPGA into one population per round (``(qubit, round_count)``), which is
cheaper on the wire and cannot answer joint questions.

QM unidirectional-coupling Trotter chain for scqo -- supplies ``probe()``.

Parameters, the record-only summary and the (absent) writeback are inherited
from ``scqo.experiments.QcUnidirectionalTrotter``. This adapter resolves the
chain roles through scqo's own ``chain_roles`` (so the neutral layer and the
probe can never disagree about which member is the relay), joins every roster
name to its QUAM object through the ``_vendor`` door, and renames the raw axes
onto the readout schema in ``reduce_raw``.
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
    rounds_array,
    num_shots: int,
    reset_type: str,
    keep_shots: bool,
    operation_gap_ns: int = 0,
    first_coupler_amp: Optional[float] = None,
    second_coupler_amp: Optional[float] = None,
    simulate: bool = False,
):
    """Build the unidirectional-Trotter-chain QUA program.

    Returns ``(program, sweep_axes)``.

    ``measure_qubits`` is a plain list of qubit objects read out at the end of
    the circuit, IN CHAIN ORDER (the order fixes the joint-state digit order
    downstream). ``first_pair`` / ``second_pair`` are qubit-pair objects whose
    ``macros[swap_operation]`` is applied bare, in that sequence, each round;
    ``reset_qubit``'s ``macros[reset_operation]`` fires after them.
    ``compensation`` is a sequence of ``(qubit object, amplitude factor)`` pairs
    - one Stark tone each, played concurrently at the end of a round. A qubit
    absent from it gets no tone at all; a qubit present with factor 0.0 still
    plays, which keeps the round's duration the same as a compensated run's.
    ``rounds_array`` is the integer sweep over the number of Trotter steps
    (N=0 allowed, giving just the prep). All measured qubits are read out within
    the same shot (joint / multiplexed readout), since they share one circuit.

    ``operation_gap_ns`` (multiple of 4, default 0) idles after each of the
    round's three flux operations, so a pulse settles before the next fires.
    """
    measure_qubits = list(measure_qubits)
    num_qubits = len(measure_qubits)
    rounds_array = np.asarray(rounds_array).astype(int)
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
    for qubit, _factor in compensation:
        if stark_operation not in qubit.xy.operations:
            raise ValueError(
                f"Qubit {qubit.name} has no xy operation {stark_operation!r}; "
                f"available: {list(qubit.xy.operations)}. Register it first "
                f"(quam_config/register_stark.py).")
    check_amp_scale_window([factor for _q, factor in compensation],
                           name="stark compensation",
                           knob="compensation_amps")
    # The swap ANGLE knob, when a run sets it: the coupler pulse must exist and
    # be turnable (baked at zero it is not), and the port must be able to emit
    # the volts. A pair left at None plays its baked coupler amplitude and is
    # not checked, because nothing is being asked of it.
    for pair, amp in ((first_pair, first_coupler_amp),
                      (second_pair, second_coupler_amp)):
        if amp is not None:
            guard_coupler_amplitudes(
                pair, swap_operation, [float(amp)],
                why="swap_coupler_flux sets the swap angle through the coupler.",
                label=f"{pair.name} swap_coupler_flux")

    involved = _dedup_involved(measure_qubits, [first_pair, second_pair],
                               [reset_qubit, prep_qubit])
    # The off-resonant IF each compensated qubit's tone plays at, and the
    # resonant one it is restored to for prep and readout.
    base_if = {qubit.name: qubit.xy.intermediate_frequency
               for qubit, _factor in compensation}
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
    axes["round_count"] = xr.DataArray(
        rounds_array, attrs={"long_name": "Trotter steps", "units": ""})
    # Canonical scqo axis names in raw NESTING order, so _to_canonical takes its
    # name-based path (only `qubit` -> `target` is renamed) instead of matching
    # equal-sized axes positionally.
    sweep_axes = axes

    with program() as prog:
        # Macro to declare I, Q, n and their respective streams for the
        # measured qubits (n / n_st drive the shot loop and progress counter;
        # I/Q go unused - this experiment is discriminated by construction).
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        r = declare(int)   # swept Trotter-step count (current value)
        rr = declare(int)  # inner round counter
        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]

        # Initialize the QPU in terms of flux points for every involved element.
        for qubit in involved:
            machine.initialize_qpu(target=qubit)
        align()

        with for_(n, 0, n < num_shots, n + 1):
            save(n, n_st)
            with for_(*from_array(r, rounds_array)):
                # Initialization: thermalize / actively reset every involved qubit.
                for qubit in involved:
                    qubit.reset(reset_type, simulate)
                align()

                # State prep: ONE excitation on the chain source, resonant IF.
                prep_qubit.xy.play(prep_operation)
                align()

                # Detune the compensated qubits so their tones shift rather than
                # rotate. Outside the loop: nothing between prep and readout
                # needs their resonant IF.
                for qubit, _factor in compensation:
                    qubit.xy.update_frequency(stark_if[qubit.name])

                # The Trotter step, N times. A dynamic loop bound on r means
                # N=0 skips the body entirely (the prep-only baseline).
                with for_(rr, 0, rr < r, rr + 1):
                    # A pair named in swap_coupler_flux is driven at that coupler
                    # amplitude (the angle knob); one that is not plays bare, so
                    # the default {} is byte-for-byte the pre-existing sequence.
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
                    # Concurrent by design: one compensation instant per round.
                    for qubit, factor in compensation:
                        qubit.xy.play(stark_operation, amplitude_scale=factor)
                    align()

                # Restore the resonant IFs before readout.
                for qubit, _factor in compensation:
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
                # Inner buffer = the round axis; shot mode groups num_shots of
                # them so the joint distribution stays reconstructable, average
                # mode collapses them on the FPGA into one population per round.
                stream = state_st[i].buffer(len(rounds_array))
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
from scqo.experiments import QcUnidirectionalTrotter
from scqo.experiments.qc_unidirectional_trotter import chain_roles


@register
class QMQcUnidirectionalTrotter(QcUnidirectionalTrotter):
    """Build, run and fetch the unidirectional Trotter chain on the QM OPX."""

    def probe(self) -> Any:
        from ._reset import check_reset_method
        from ._vendor import vendor_pair, vendor_qubit

        # One door for the reset method: this shell does not opt into active
        # reset (default DENY), so 'active' refuses by name until the sequence
        # has hardware evidence. Note this is the BETWEEN-SHOTS reset, not the
        # mid-circuit parametric one the round plays.
        reset = check_reset_method(self)
        # The chain topology comes from scqo's own resolver, so the neutral
        # layer and the probe can never disagree about which member is the relay.
        _source, _relay, _sink, prep = chain_roles(self.device.roster, self.params)

        machine = self.backend.machine  # type: ignore[attr-defined]
        # Chain order is the TARGET order: it fixes the joint-state digit order
        # that estimate() reads back, so it must survive into the readout list.
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
            # goes through the z channel — a reset qubit with no flux wire is
            # refused there by name
            vendor_qubit(self, self.params.reset_qubit, field="reset_qubit",
                         kind="flux"),
            prep_qubit=vendor_qubit(self, prep, field="prep_qubit"),
            prep_operation=self.params.prep_operation,
            swap_operation=self.params.swap_operation,
            reset_operation=self.params.reset_operation,
            stark_operation=self.params.stark_operation,
            stark_detuning_hz=self.params.stark_detuning_hz,
            compensation=compensation,
            first_coupler_amp=coupler_flux.get(self.params.first_pair),
            second_coupler_amp=coupler_flux.get(self.params.second_pair),
            rounds_array=np.asarray(self.sweep_axes["round_count"]).astype(int),
            num_shots=int(self.params.num_averages),
            reset_type=reset,
            keep_shots=self.params.readout_mode == "shot",
            operation_gap_ns=int(self.params.operation_gap_ns),
        )

    def reduce_raw(self, raw: xr.Dataset) -> xr.Dataset:
        """The readout schema's naming for the two modes.

        Shot mode keeps ``state`` — per-shot integer LEVELS, which is what that
        name means. Average mode's stream is already the FPGA mean of those
        levels, i.e. the marginal ``population``; the backend's own
        state->population rename cannot do it here because this contract also
        ACCEPTS ``state`` (the shot form), so the rename is explicit."""
        if self.params.readout_mode == "shot":
            return raw
        return raw.rename({"state": "population"})
