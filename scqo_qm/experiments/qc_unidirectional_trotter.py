"""Unidirectional-coupling Trotter chain: vendor code only (qm/quam/qualang_tools)
- no qualibrate, no scqo, no scqat in the builder half.

The two-pair descendant of the retired ``qc_swap_paramreset`` probe (one pair +
one reset qubit, R rounds), extended with a SECOND swap pair and a per-qubit
AC-Stark phase compensation borrowed from ``qc_n_stark_amp``.

Circuit per shot (for a swept Trotter-step count N):
  1. Initialize every involved qubit with ``q.reset(reset_type, simulate,
     max_attempts=reset_max_attempts)`` (involved = the measured qubits + both
     pairs' members + the reset qubit + the prep qubit) - a thermal wait, or
     QUAM's active reset (measure, conditional pi). This is the BETWEEN-SHOTS
     initialization; the relay's mid-circuit reset in step 4 is a separate,
     parametric one and does not change with it.
     Where a run names ``init_resets``, each of those qubits then plays its own
     parametric-reset macro once, all together, and waits out a ring-down: the
     thermal wait ends at the thermal population, and this empties what is left.
  2. State prep: ``qubit.xy.play(operation)`` for every ``(qubit, operation)``
     in ``prep``, ONCE and concurrently, at the resonant IF - by default a single
     excitation on the chain source; the chain is watched, not pumped.
  3. Detune every compensated qubit's xy by ``stark_detuning_hz``
     (``update_frequency``), so its tone SHIFTS the qubit rather than rotating
     it. Hoisted out of the loop: prep (above) and readout (below) both need the
     resonant IF, and nothing between them does.
  4. Repeat N times: bare ``first_pair.macros[first_operation].apply()``, bare
     ``second_pair.macros[second_operation].apply()``, bare
     ``reset_qubit.macros[reset_operation].apply()``, then one
     ``xy.play(stark_operation, amplitude_scale=a)`` per compensated qubit.
     Each pair names its own operation - a chip does not carry the same gate on
     every couple - and either step may instead be IDLE (``operation=None``
     here), which plays nothing and waits the length that pair's swap would have
     taken, leaving the round's timing intact for the comparison. With
     ``stark_with_reset`` the tones start together with the reset instead of
     after it (scqo's ``stark_timing="with_reset"``): a shorter round of the
     same circuit.
  5. Restore the resonant IFs, then read out every measured qubit.

The macros are invoked BARE (``.apply()`` with no arguments), so each must be
callable that way - amplitudes and pulses baked into the QUAM macro definition.
``quam_config/register_swap_macro.py`` and ``register_reset_macro.py`` write
exactly such macros; a macro whose ``apply()`` has required positional arguments
needs a default-carrying variant before it can be named here.

THE ROUND BODY IS SHARED with ``qc_trotter_compensation``, which calibrates this
exact sequence: ``_chain_round.play_chain_round`` emits it for both, and its
module docstring says why every swap is followed by a global ``align()`` and
where the Stark tones go in each ``stark_timing``. The per-qubit tones are
CONCURRENT with each other in both (different xy lines, one shared compensation
instant).

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
    """Build the unidirectional-Trotter-chain QUA program.

    Returns ``(program, sweep_axes)``.

    ``measure_qubits`` is a plain list of qubit objects read out at the end of
    the circuit, IN CHAIN ORDER (the order fixes the joint-state digit order
    downstream). ``first_pair`` / ``second_pair`` are qubit-pair objects whose
    ``macros[<its own operation>]`` is applied bare, in that sequence, each
    round; ``reset_qubit``'s ``macros[reset_operation]`` fires after them.

    Each pair carries its OWN operation because a chip does not register the same
    gate on every couple. An operation of ``None`` is an IDLE step: it plays
    nothing and instead waits ``idle_wait_cycles(pair, idle_reference_operation)``
    — the length that pair's swap would have occupied — so the round keeps its
    duration and an idle run differs from a swapping one by the exchange alone.

    ``prep`` is a sequence of ``(qubit object, xy operation)`` pairs, all played
    concurrently once before the rounds; empty prepares nothing.

    ``compensation`` is a sequence of ``(qubit object, amplitude factor)`` pairs
    - one Stark tone each, played concurrently at the end of a round. A qubit
    absent from it gets no tone at all; a qubit present with factor 0.0 still
    plays, which keeps the round's duration the same as a compensated run's.
    ``rounds_array`` is the integer sweep over the number of Trotter steps
    (N=0 allowed, giving just the prep). All measured qubits are read out within
    the same shot (joint / multiplexed readout), since they share one circuit.

    ``operation_gap_ns`` (multiple of 4, default 0) idles after each of the
    round's three flux operations, so a pulse settles before the next fires.
    ``reset_gap_ns`` sets the one after the relay's reset on its own (0, or a
    multiple of 4 from 16 up); None leaves it at ``operation_gap_ns``.

    ``stark_with_reset`` starts the tones together with the relay's reset
    instead of after it; the round then waits for the later of the reset with
    its gap and the tones. A tone on ``reset_qubit`` is refused in that mode.

    ``reset_type`` / ``reset_max_attempts`` are the between-shots initialization
    of every involved qubit (QUAM ``reset(..., max_attempts=)``; an upper bound
    on QM's repeat-until-success loop, unused by a thermal reset).

    ``init_resets`` are qubit objects whose ``macros[reset_operation]`` also
    plays once per shot, right after that initialization and before the prep,
    all starting together; each waits ``init_reset_ringdown_ns`` in place of its
    stored ring-down (``_chain_round.play_init_resets``). A qubit without the
    macro, or with a macro that is not a parametric reset, is refused by name.
    """
    measure_qubits = list(measure_qubits)
    num_qubits = len(measure_qubits)
    rounds_array = np.asarray(rounds_array).astype(int)
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
    for qubit, _factor in compensation:
        if stark_operation not in qubit.xy.operations:
            raise ValueError(
                f"Qubit {qubit.name} has no xy operation {stark_operation!r}; "
                f"available: {list(qubit.xy.operations)}. Register it first "
                f"(quam_config/register_stark.py).")
    check_amp_scale_window([factor for _q, factor in compensation],
                           name="stark compensation",
                           knob="compensation_amps")
    if stark_with_reset:
        check_tones_clear_of_reset(reset_qubit,
                                   [qubit for qubit, _factor in compensation])
    # The swap ANGLE knob, when a run sets it: the coupler pulse must exist and
    # be turnable (baked at zero it is not), and the port must be able to emit
    # the volts. A pair left at None plays its baked coupler amplitude and is
    # not checked, because nothing is being asked of it. Each pair is guarded
    # against ITS OWN operation - the coupler pulse belongs to the macro, so
    # checking one pair's amplitude against the other's gate would be nonsense.
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
                    qubit.reset(reset_type, simulate,
                                max_attempts=reset_max_attempts)
                align()

                # The parametric reset at the start of the shot, where a run asks
                # for one: it empties what the reset above left. Nothing at all
                # when no qubit is named.
                play_init_resets(init_resets, reset_operation,
                                 init_reset_ringdown_ns)

                # State prep, once, resonant IF: concurrent across the prep
                # qubits (each on its own xy line), then one align.
                for qubit, operation in prep:
                    qubit.xy.play(operation)
                align()

                # Detune the compensated qubits so their tones shift rather than
                # rotate. Outside the loop: nothing between prep and readout
                # needs their resonant IF.
                for qubit, _factor in compensation:
                    qubit.xy.update_frequency(stark_if[qubit.name])

                # The Trotter step, N times. A dynamic loop bound on r means
                # N=0 skips the body entirely (the prep-only baseline).
                with for_(rr, 0, rr < r, rr + 1):
                    # One round, shared with the compensation scan. A pair named
                    # in swap_coupler_flux is driven at that coupler amplitude
                    # (the angle knob); one that is not plays bare.
                    play_chain_round(
                        first_pair, second_pair, reset_qubit,
                        first_operation=first_operation,
                        second_operation=second_operation,
                        idle_cycles=idle_cycles,
                        first_swap_kwargs=first_swap_kwargs,
                        second_swap_kwargs=second_swap_kwargs,
                        reset_operation=reset_operation,
                        stark_operation=stark_operation,
                        tones=compensation,
                        gap_cycles=gap_cycles,
                        stark_with_reset=stark_with_reset,
                        after_reset_cycles=after_reset_cycles)

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
from scqo.experiments.qc_unidirectional_trotter import (
    IDLE,
    STARK_WITH_RESET,
    chain_roles,
    pair_specs,
)


def _vendor_operation(operation: str) -> Optional[str]:
    """scqo's reserved ``IDLE`` -> the builder's ``None``.

    The word lives in the neutral layer (``simulate()`` needs it too) and the
    builder stays vendor-only, so the translation happens exactly here — one
    line, in the adapter whose whole job is mapping the two vocabularies.
    """
    return None if operation == IDLE else operation


def _stark_with_reset(stark_timing: str) -> bool:
    """scqo's ``stark_timing`` -> the builders' ``stark_with_reset`` flag, for
    both chain probes; the same one-line translation as ``_vendor_operation``."""
    return stark_timing == STARK_WITH_RESET


@register
class QMQcUnidirectionalTrotter(QcUnidirectionalTrotter):
    """Build, run and fetch the unidirectional Trotter chain on the QM OPX."""

    #: true of THIS probe only; scqo shows them in
    #: `scqo run qc_unidirectional_trotter --help`
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

    #: Active reset is valid here: the readout condition is fixed for the whole
    #: run (one joint readout at the end of every shot), and the initialization
    #: is a genuine state reset of every qubit before the prep. It is the
    #: BETWEEN-SHOTS reset only - the relay's mid-circuit parametric reset is a
    #: macro of the round and never becomes a measurement.
    supports_active_reset: ClassVar[bool] = True

    def probe(self) -> Any:
        from ._reset import check_reset_method, reset_max_attempts
        from ._vendor import vendor_pair, vendor_qubit

        # The chain topology comes from scqo's own resolver, so the neutral
        # layer and the probe can never disagree about which member is the relay.
        source, relay, sink, prep = chain_roles(self.device.roster, self.params)
        # One door for the reset method, over EVERY qubit a shot resets - an
        # active reset thresholds each against its own discriminator, read out
        # at the end or not.
        reset = check_reset_method(
            self, qubits=chain_reset_qubits(self.params, source, relay, sink, prep))
        (first_name, first_op), (second_name, second_op) = pair_specs(self.params)

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
            vendor_pair(self, first_name),
            vendor_pair(self, second_name),
            # the parametric reset is a FLUX-line technique, so the roster join
            # goes through the z channel — a reset qubit with no flux wire is
            # refused there by name
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
            first_coupler_amp=coupler_flux.get(first_name),
            second_coupler_amp=coupler_flux.get(second_name),
            rounds_array=np.asarray(self.sweep_axes["round_count"]).astype(int),
            num_shots=int(self.params.num_averages),
            reset_type=reset,
            reset_max_attempts=reset_max_attempts(self),
            keep_shots=self.params.readout_mode == "shot",
            operation_gap_ns=int(self.params.operation_gap_ns),
            reset_gap_ns=self.params.reset_gap_ns,
            stark_with_reset=_stark_with_reset(self.params.stark_timing),
            # the same flux-line join as the reset qubit's: scqo's chain_roles
            # has already refused a name that is not on the chain
            init_resets=[vendor_qubit(self, name, field="init_reset_qubits",
                                      kind="flux")
                         for name in self.params.init_reset_qubits],
            init_reset_ringdown_ns=int(self.params.init_reset_ringdown_ns),
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
