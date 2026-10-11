"""Trotter-chain compensation scan: vendor code only (qm/quam/qualang_tools)
- no qualibrate, no scqo, no scqat in the builder half.

``qc_unidirectional_trotter`` with ONE extra swept axis: the AC-Stark
compensation amplitude on a single chosen chain qubit. The round is otherwise
identical, and deliberately so -- this scan exists to calibrate that exact
sequence, so any divergence would calibrate something else.

Circuit per shot (for a swept compensation amplitude a and step count N):
  1. Initialize every involved qubit with ``q.reset(reset_type, simulate,
     max_attempts=reset_max_attempts)`` - the between-shots thermal or active
     reset, the same as ``qc_unidirectional_trotter``'s.
  2. State prep: ``qubit.xy.play(operation)`` for each ``(qubit, operation)`` in
     ``prep``, ONCE, at the resonant IF. scqo lets this scan prepare exactly one
     qubit; the builder takes the chain probe's shape all the same.
  3. Detune every compensated qubit's xy by ``stark_detuning_hz``
     (``update_frequency``), hoisted out of the loops.
  4. Repeat N times: bare ``first_pair`` swap, bare ``second_pair`` swap, bare
     ``reset_qubit`` reset, then the Stark tones -- the FIXED ones at their
     Python-float factors and the swept one at the QUA variable ``a``. Each pair
     names its own operation, and either may be IDLE (``operation=None``), which
     waits the swap's length instead of playing it: scanning the tone with one
     step idled is the background arm. With ``stark_with_reset`` every tone,
     the swept one included, starts together with the reset instead of after it.
     The round body is the chain probe's own (``_chain_round.play_chain_round``).
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

from typing import Any, Callable, ClassVar, Dict, List, Optional, Sequence

import numpy as np
import xarray as xr
from qm.qua import *

from qualang_tools.loops import from_array

from scqo_qm.experiments._lib import acquire as _acquire
from scqo_qm.experiments._amp_limits import check_amp_scale_window
from scqo_qm.experiments._chain_round import (
    chain_reset_qubits,
    check_init_resets,
    check_tones_clear_of_reset,
    idle_wait_cycles,
    play_chain_round,
    play_init_resets,
    reset_gap_cycles,
)
from scqo_qm.experiments._coupler_knob import guard_coupler_amplitudes


def _dedup_involved(measure_qubits, pairs, extra) -> List:
    """The unique (by ``.name``) qubit elements that must be initialized.

    The measured qubits need not include the pairs' members, the reset qubit or
    the prep qubits, but every one of them is pulsed and so must be
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
    prep: Sequence[tuple],
    # None = an IDLE step: play nothing, wait the swap's length instead. The
    # neutral layer spells that "idle"; the word is translated in probe() so this
    # builder stays vendor-only, and None cannot collide with a macro key.
    first_operation: Optional[str],
    second_operation: Optional[str],
    idle_reference_operation: str,
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
    reset_max_attempts: int = 15,
    operation_gap_ns: int = 0,
    # The idle after the relay's reset alone, where a run sets it apart (scqo's
    # reset_gap_ns): 0 = none. None = operation_gap_ns there too.
    reset_gap_ns: Optional[int] = None,
    first_coupler_amp: Optional[float] = None,
    second_coupler_amp: Optional[float] = None,
    # False = the tones play after the relay's reset; True = with it. The
    # neutral layer spells the two "after_reset" / "with_reset" (stark_timing).
    stark_with_reset: bool = False,
    # Qubit objects whose reset macro also plays ONCE at the start of every shot,
    # and the ring-down each then waits before the prep (scqo's init_reset_qubits
    # / init_reset_ringdown_ns). Empty = the shot as it always was.
    init_resets: Sequence = (),
    init_reset_ringdown_ns: int = 0,
    simulate: bool = False,
):
    """Build the Trotter-chain compensation-scan QUA program.

    Returns ``(program, sweep_axes)``.

    ``measure_qubits`` is a plain list of qubit objects read out at the end of
    the circuit, IN CHAIN ORDER. ``prep`` is a sequence of ``(qubit object, xy
    operation)`` pairs played once before the rounds. ``compensation`` is a sequence of
    ``(qubit object, amplitude factor)`` pairs held FIXED; ``swept_qubit`` is the
    one whose tone carries the swept ``compensation_amps`` factors instead, and
    it must NOT also appear in ``compensation`` (scqo refuses that upstream, and
    it is asserted here because a doubled tone would be silent).

    Each pair carries its OWN operation, and an operation of ``None`` is an IDLE
    step: it plays nothing and waits ``idle_wait_cycles(pair,
    idle_reference_operation)`` instead, so the round keeps its duration.
    Scanning the tone with one step idled is the background arm — what the Stark
    tone does on its own, with no transport for it to act on.

    ``stark_with_reset`` starts every tone together with the relay's reset
    instead of after it, exactly as the chain probe does; a tone on
    ``reset_qubit``, fixed or swept, is refused in that mode.

    ``init_resets`` / ``init_reset_ringdown_ns`` are the chain probe's reset at
    the start of a shot, played by the same ``_chain_round.play_init_resets``.

    ``reset_gap_ns`` is the chain probe's too: the idle after the relay's reset
    alone, None leaving it at ``operation_gap_ns``.
    """
    measure_qubits = list(measure_qubits)
    num_qubits = len(measure_qubits)
    rounds_array = np.asarray(rounds_array).astype(int)
    compensation_amps = np.asarray(compensation_amps, dtype=float)
    prep = list(prep)
    compensation = list(compensation)

    if operation_gap_ns < 0 or operation_gap_ns % 4 != 0:
        raise ValueError(
            f"operation_gap_ns must be a non-negative multiple of 4 ns, got "
            f"{operation_gap_ns}.")
    gap_cycles = operation_gap_ns // 4
    after_reset_cycles = reset_gap_cycles(reset_gap_ns, gap_cycles)

    # Everything that can refuse, refuses HERE - before a single QUA statement,
    # so a mis-registered chip costs no instrument time. An idle step is resolved
    # in the same pass: its wait has to be a real number before the round is
    # built, and a pair that cannot supply one refuses by name like any other.
    steps = [(first_pair, first_operation, "first_pair"),
             (second_pair, second_operation, "second_pair")]
    idle_cycles = {}
    for pair, operation, what in steps:
        if operation is None:
            idle_cycles[what] = idle_wait_cycles(pair, idle_reference_operation)
        else:
            _check_macro(pair, operation, what)
    _check_macro(reset_qubit, reset_operation, "reset_qubit")
    init_resets = list(init_resets)
    check_init_resets(init_resets, reset_operation, init_reset_ringdown_ns)
    for qubit, operation in prep:
        if operation not in qubit.xy.operations:
            raise ValueError(
                f"prep qubit {qubit.name} has no xy operation {operation!r}; "
                f"available: {list(qubit.xy.operations)}.")
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
    if stark_with_reset:
        check_tones_clear_of_reset(
            reset_qubit, [qubit for qubit, _factor in compensation] + [swept_qubit])
    # The swap ANGLE knob, when a run sets it (see _coupler_knob). A pair left at
    # None plays its baked coupler amplitude and is not checked. Each pair is
    # guarded against ITS OWN operation - the coupler pulse belongs to the macro.
    for (pair, operation, _what), amp in zip(steps, (first_coupler_amp,
                                                     second_coupler_amp)):
        if amp is None:
            continue
        if operation is None:
            raise ValueError(
                f"Pair {pair.name} is an idle step but was given a coupler "
                f"amplitude ({amp}) - an idle plays no pulse, so there is "
                f"nothing to set it on. Drop it from swap_coupler_flux, or give "
                f"this pair a real swap operation.")
        guard_coupler_amplitudes(
            pair, operation, [float(amp)],
            why="swap_coupler_flux sets the swap angle through the coupler.",
            label=f"{pair.name} swap_coupler_flux")

    involved = _dedup_involved(measure_qubits, [first_pair, second_pair],
                               [reset_qubit, *(qubit for qubit, _op in prep),
                                *init_resets])
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
                        qubit.reset(reset_type, simulate,
                                    max_attempts=reset_max_attempts)
                    align()

                    # The chain probe's reset at the start of the shot, where a
                    # run asks for one; nothing when no qubit is named.
                    play_init_resets(init_resets, reset_operation,
                                     init_reset_ringdown_ns)

                    # State prep, once, resonant IF (one qubit: scqo refuses more).
                    for qubit, operation in prep:
                        qubit.xy.play(operation)
                    align()

                    # Detune the compensated qubits so their tones shift rather
                    # than rotate. Nothing between prep and readout needs the
                    # resonant IF.
                    for qubit in toned:
                        qubit.xy.update_frequency(stark_if[qubit.name])

                    # The Trotter step, N times. A dynamic loop bound on r means
                    # N=0 skips the body entirely (the prep-only baseline).
                    with for_(rr, 0, rr < r, rr + 1):
                        # The chain probe's own round. The swept tone is one
                        # more entry of its tone list, at the QUA variable `a`:
                        # it always plays, at amplitude 0 too, so the round
                        # keeps its duration at the baseline point.
                        play_chain_round(
                            first_pair, second_pair, reset_qubit,
                            first_operation=first_operation,
                            second_operation=second_operation,
                            idle_cycles=idle_cycles,
                            first_swap_kwargs=first_swap_kwargs,
                            second_swap_kwargs=second_swap_kwargs,
                            reset_operation=reset_operation,
                            stark_operation=stark_operation,
                            tones=[*compensation, (swept_qubit, a)],
                            gap_cycles=gap_cycles,
                            stark_with_reset=stark_with_reset,
                            after_reset_cycles=after_reset_cycles)

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
from scqo.experiments.qc_unidirectional_trotter import chain_roles, pair_specs
from scqo_qm.experiments.qc_unidirectional_trotter import (
    _stark_with_reset,
    _vendor_operation,
)


@register
class QMQcTrotterCompensation(QcTrotterCompensation):
    """Build, run and fetch the Trotter-chain compensation scan on the QM OPX."""

    #: true of THIS probe only; scqo shows them in
    #: `scqo run qc_trotter_compensation --help`
    backend_notes: ClassVar[tuple[str, ...]] = (
        "the swaps, the relay's reset and the stark tone are operations stored in the "
        "QUAM tree and played as stored; a missing one is refused by name, with the "
        "quam_config script that registers it",
        "with stark_timing=after_reset every operation of a round is followed by a "
        "global align, so the round is strictly sequential; the stark tones of one "
        "round play together",
        "with stark_timing=with_reset the align between the relay's reset and the stark "
        "tones is left out: the tones start when the reset macro does, and the round "
        "waits for the later of the reset with its gap and the tones. The macro sets its "
        "flux line's frequency before it plays and returns it after; how far the first "
        "puts its pulse behind the tones has not been measured",
        "the stark tones are made off-resonant by moving each toned qubit's intermediate "
        "frequency by stark_detuning_hz for the rounds and back before the readout; a "
        "factor of 2 or more is refused",
        "a stark_timing=after_reset round lasts its operations, the ring-down stored on "
        "the reset macro and its gaps plus program overhead: about 20 ns on the gateway "
        "simulator, measured before the reset macro returned its flux line's frequency "
        "after the pulse and not since (a with_reset round has not been measured); "
        "operation_gap_ns has to be a multiple of 4 ns",
        "reset_gap_ns replaces operation_gap_ns after the relay's reset alone and has "
        "to be 0 or a multiple of 4 ns from 16 up (QUA's shortest wait); at 0 the next "
        "round's first swap follows the reset (and the tones) directly, apart from the "
        "program overhead above",
        "an idle step waits the longer of the control's and the coupler's copy of "
        "idle_reference_operation's pulse",
        "swap_coupler_flux is divided by the stored amplitude of the operation's coupler "
        "pulse, so a coupler pulse stored at zero is refused by name",
        "with reset_method=active every qubit a shot resets needs its discriminator, "
        "also one that is not a target",
        "init_reset_qubits plays each named qubit's stored reset macro once per shot, "
        "after the between-shots reset and before the prep, all starting together; "
        "each waits init_reset_ringdown_ns in place of the ring-down stored on the "
        "macro. A qubit without that macro, or whose macro is not a parametric reset, "
        "is refused by name",
        "readout_mode=average averages on the instrument; shot keeps every shot",
    )

    #: Active reset is valid here for the same reason as in
    #: qc_unidirectional_trotter, whose parameter file this scan shares: a fixed
    #: readout condition, and a genuine between-shots state reset. The relay's
    #: mid-circuit parametric reset is untouched.
    supports_active_reset: ClassVar[bool] = True

    def probe(self) -> Any:
        from ._reset import check_reset_method, reset_max_attempts
        from ._vendor import vendor_pair, vendor_qubit

        # The chain topology comes from scqo's own resolver, so the neutral layer
        # and the probe can never disagree about which member is the relay.
        source, relay, sink, prep = chain_roles(self.device.roster, self.params)
        # One door for the reset method, over EVERY qubit a shot resets.
        reset = check_reset_method(
            self, qubits=chain_reset_qubits(self.params, source, relay, sink, prep))
        (first_name, first_op), (second_name, second_op) = pair_specs(self.params)

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
            vendor_pair(self, first_name),
            vendor_pair(self, second_name),
            # the parametric reset is a FLUX-line technique, so the roster join
            # goes through the z channel
            vendor_qubit(self, self.params.reset_qubit, field="reset_qubit",
                         kind="flux"),
            # the RESOLVED prep: None has already become {source: "x180"}
            prep=[(vendor_qubit(self, name, field="prep_operations"), operation)
                  for name, operation in prep.items()],
            first_operation=_vendor_operation(first_op),
            second_operation=_vendor_operation(second_op),
            idle_reference_operation=self.params.idle_reference_operation,
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
            reset_max_attempts=reset_max_attempts(self),
            keep_shots=self.params.readout_mode == "shot",
            operation_gap_ns=int(self.params.operation_gap_ns),
            reset_gap_ns=self.params.reset_gap_ns,
            first_coupler_amp=coupler_flux.get(first_name),
            second_coupler_amp=coupler_flux.get(second_name),
            stark_with_reset=_stark_with_reset(self.params.stark_timing),
            init_resets=[vendor_qubit(self, name, field="init_reset_qubits",
                                      kind="flux")
                         for name in self.params.init_reset_qubits],
            init_reset_ringdown_ns=int(self.params.init_reset_ringdown_ns),
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
