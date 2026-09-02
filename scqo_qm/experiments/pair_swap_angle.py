"""Partial-swap ANGLE calibration probe: vendor code only (qm/quam/qualang_tools)
- no qualibrate, no scqo, no scqat in the builder half.

The angle-knob sibling of ``qc_n_swap_amp``. That probe sweeps the swap macro's
``ctrl_amp`` - the CONTROL QUBIT's flux amplitude, which is the RESONANCE knob -
with the coupler playing bare. This one does the mirror: it sweeps ``cplr_amp``,
the COUPLER's flux amplitude, with the control playing bare at its calibrated
baked amplitude. ``J_eff(Phi_c)`` is what the coupler tunes, so at a fixed pulse
duration that sweep IS the angle sweep (TUTORIAL section 12).

Circuit per shot (for a swept coupler amplitude c and swap count N):
  1. Initialize every involved qubit with ``q.reset(reset_type, simulate)``
     (involved = measured qubits + the swap pair's control/target).
  2. State prep: ``swap_pair.qubit_control.xy.play("x180")``.
  3. Repeat N times: ``swap_pair.macros[swap_operation].apply(cplr_amp=c)``, then
     idle the pair's flux lines for ``operation_gap_ns`` (if nonzero).
  4. Read out every measured qubit (always state-discriminated - this experiment
     has no I/Q form).

Reading the transfer against N at each coupler amplitude gives an oscillation of
period ``pi/theta``, so the map answers "what angle does this coupler setting
give?" rather than "where is resonance?".

THE ZERO-AMPLITUDE TRAP, and why it is refused BY NAME here. The macro converts
``cplr_amp`` to a QUA ``amplitude_scale`` by dividing by its stored coupler pulse
amplitude (``ISwapImplementation.apply``). A pair whose coupler pulse is baked at
0.0 V - which is the state a chip is in whenever the swap has only ever been
driven by detuning the control qubit - makes that a division by zero, and the
failure would surface as a QUA build error naming an internal variable rather
than the un-registered pulse. So the amplitude is checked before any QUA is
built, and the message names ``register_flattop_cosine.py``.

The chosen macro must expose a string ``flux_pulse`` playable on the COUPLER and
accept ``apply(cplr_amp=...)`` (the lab ``ISwapImplementation`` does both).

QM partial-swap angle calibration for scqo -- supplies ``probe()``.

Parameters, the angle summary and the (absent) writeback are inherited from
``scqo.experiments.PairSwapAngle``. This adapter refuses unless
``drive_side``/``flux_side`` resolve to the vendor CONTROL member (the probe has
no target-side mode; a silent role mismatch would mislabel the
prepared/transfer panels), orders the per-shot states onto the readout schema's
(high, low) ``member`` axis, and in ``readout_mode="average"`` reduces them to
``joint_population`` with scqo's shared ``states_to_joint_population``.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import numpy as np
import xarray as xr
from qm.qua import *

from qualang_tools.loops import from_array

from scqo_qm.experiments._lib import acquire as _acquire
from scqo_qm.experiments._coupler_knob import guard_coupler_amplitudes
from scqo_qm.experiments._amp_limits import check_amp_scale_window


def _dedup_involved(measure_qubits, swap_pair) -> List:
    """The unique (by ``.name``) qubit elements that must be initialized.

    The measured qubits need not include the swap pair's control/target, but all
    of them must be flux-initialized and reset at the start of each shot. The
    COUPLER is not a qubit and carries no reset - it is driven, not prepared.
    """
    involved = []
    seen = set()
    for q in list(measure_qubits) + [swap_pair.qubit_control, swap_pair.qubit_target]:
        if q.name not in seen:
            seen.add(q.name)
            involved.append(q)
    return involved


def build_program(
    machine,
    measure_qubits,
    swap_pair,
    *,
    swap_operation: str,
    rounds_array,
    coupler_amplitudes,
    num_shots: int,
    reset_type: str,
    operation_gap_ns: int = 0,
    compensation: Sequence[tuple] = (),
    stark_operation: str = "stark",
    stark_detuning_hz: float = 50e6,
    simulate: bool = False,
):
    """Build the N-swap x coupler-flux-amplitude QUA program.

    Returns ``(program, sweep_axes)``.

    ``measure_qubits`` is a plain list of qubit objects read out at the end of
    the circuit; ``swap_pair`` is a qubit-pair object whose
    ``macros[swap_operation]`` is applied each swap. ``rounds_array`` is the
    integer sweep over the number of swaps (N=0 allowed, giving just the x180
    prep, inner axis); ``coupler_amplitudes`` is the COUPLER flux amplitude sweep
    in absolute volts (outer axis), passed to each swap as the macro's
    ``cplr_amp`` while ``ctrl_amp`` is left None so the control qubit's own flux
    pulse plays at its calibrated baked amplitude.

    ``operation_gap_ns`` (multiple of 4, default 0) idles the pair's flux lines
    after each swap, so the flux pulse can settle before the next swap fires.
    """
    measure_qubits = list(measure_qubits)
    num_qubits = len(measure_qubits)
    rounds_array = np.asarray(rounds_array).astype(int)
    coupler_amplitudes = np.asarray(coupler_amplitudes, dtype=float)

    if operation_gap_ns < 0 or operation_gap_ns % 4 != 0:
        raise ValueError(
            f"operation_gap_ns must be a non-negative multiple of 4 ns, got "
            f"{operation_gap_ns}.")
    gap_cycles = operation_gap_ns // 4

    involved = _dedup_involved(measure_qubits, swap_pair)
    # The angle knob has to EXIST and be turnable (a coupler pulse baked at zero
    # is neither), and the port has to be able to emit the swept volts. Both
    # refuse before a single QUA statement -- see _coupler_knob.
    guard_coupler_amplitudes(
        swap_pair, swap_operation, coupler_amplitudes,
        why="pair_swap_angle sweeps the coupler flux; qc_n_swap_amp sweeps the "
            "control qubit's flux and needs no coupler.")

    # The phase-compensation tones. They are what turn the fitted per-round
    # COMPOSITE angle back into the exchange angle (module docstring), so a
    # missing stark op is refused by name rather than silently skipped.
    compensation = list(compensation)
    for qubit, _factor in compensation:
        if stark_operation not in qubit.xy.operations:
            raise ValueError(
                f"Qubit {qubit.name} has no xy operation {stark_operation!r}; "
                f"available: {list(qubit.xy.operations)}. Register it first "
                f"(quam_config/register_stark.py).")
    check_amp_scale_window([factor for _q, factor in compensation],
                           name="stark compensation",
                           knob="compensation_amps")
    # The off-resonant IF each tone plays at, and the resonant one it is
    # restored to for the prep pulse and the readout.
    base_if = {qubit.name: qubit.xy.intermediate_frequency
               for qubit, _factor in compensation}
    stark_if = {name: int(stark_detuning_hz) + value
                for name, value in base_if.items()}

    sweep_axes = {
        "qubit": xr.DataArray([q.name for q in measure_qubits]),
        "shot": xr.DataArray(np.arange(num_shots)),
        # Outer loop -> y axis.
        "coupler_amplitude": xr.DataArray(
            coupler_amplitudes,
            attrs={"long_name": "coupler flux amplitude", "units": "V"},
        ),
        # Inner loop -> x axis.
        "round": xr.DataArray(rounds_array,
                              attrs={"long_name": "number of swaps"}),
    }

    with program() as prog:
        # Macro to declare I, Q, n and their respective streams for the measured
        # qubits (n / n_st drive the shot loop and progress counter; I/Q go
        # unused - this experiment is discriminated by construction).
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        c_a = declare(fixed)  # swept coupler flux amplitude (absolute volts)
        r = declare(int)      # swept swap count (current value)
        rr = declare(int)     # inner swap counter
        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]

        # Initialize the QPU in terms of flux points for every involved element.
        for q in involved:
            machine.initialize_qpu(target=q)
        align()

        with for_(n, 0, n < num_shots, n + 1):
            save(n, n_st)
            # Coupler-flux amplitude loop (outer -> y axis)
            with for_(*from_array(c_a, coupler_amplitudes)):
                # Swap-count loop (inner -> x axis)
                with for_(*from_array(r, rounds_array)):
                    # Initialization: thermalize / actively reset every involved qubit.
                    for q in involved:
                        q.reset(reset_type, simulate)
                    align()

                    # State prep: excite the swap pair's control qubit to |1>.
                    swap_pair.qubit_control.xy.play("x180")
                    align()

                    # Detune the compensated members so their tones SHIFT rather
                    # than rotate. Hoisted out of the round loop: the prep above
                    # and the readout below both need the resonant IF, and
                    # nothing between them does.
                    for qubit, _factor in compensation:
                        qubit.xy.update_frequency(stark_if[qubit.name])

                    # Circuit body: N swaps on the pair, each at the swept COUPLER
                    # amplitude. ctrl_amp is left None so the control qubit's flux
                    # pulse plays bare at its calibrated resonance amplitude - the
                    # exact mirror of qc_n_swap_amp, which sweeps that and leaves
                    # the coupler bare. A dynamic loop bound on r means N=0 skips
                    # the body entirely (the prep-only baseline).
                    with for_(rr, 0, rr < r, rr + 1):
                        swap_pair.macros[swap_operation].apply(cplr_amp=c_a)
                        if gap_cycles > 0:
                            swap_pair.wait(gap_cycles)
                        align()
                        # The phase compensation, one instant per round, after
                        # the gap so it cannot overlap the flux pulse. Every
                        # phase source in the round is a Z rotation and they
                        # commute with each other, so only their placement
                        # relative to the EXCHANGE matters - and this is between
                        # exchanges, which is the whole point.
                        if compensation:
                            for qubit, factor in compensation:
                                qubit.xy.play(stark_operation,
                                              amplitude_scale=factor)
                            align()

                    # Restore the resonant IFs before readout.
                    for qubit, _factor in compensation:
                        qubit.xy.update_frequency(base_if[qubit.name])
                    if compensation:
                        align()

                    # Joint (multiplexed) readout of all measured qubits.
                    for i, qubit in enumerate(measure_qubits):
                        qubit.readout_state(state[i])
                        save(state[i], state_st[i])
                    align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                # Keep every shot (no average) so the joint populations stay
                # reconstructable: round buffer, then coupler_amplitude, then
                # group num_shots -> (shot, coupler_amplitude, round).
                (state_st[i]
                 .buffer(len(rounds_array))
                 .buffer(len(coupler_amplitudes))
                 .buffer(num_shots)
                 .save(f"state{i + 1}"))

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


from typing import Any

from scqo import register
from scqo.experiments import PairSwapAngle, states_to_joint_population


def _member_states(state: xr.DataArray, high_side: str) -> xr.DataArray:
    """The probe's per-shot states, reordered onto the schema's axes.

    ``state`` arrives ``(qubit, shot, coupler_amplitude, round)`` with the qubit
    axis in READOUT order -- ``[control, target]``, fixed by the adapter's
    measure list. The member axis is ROLE-ordered (high, low), so the vendor
    order flips when the roster's high member is the vendor target."""
    order = [0, 1] if high_side == "control" else [1, 0]
    da = state.isel(qubit=order).rename({
        "qubit": "member", "shot": "shot_idx",
        "coupler_amplitude": "coupler_flux_v", "round": "swap_count",
    })
    return da.assign_coords(member=["high", "low"])


@register
class QMPairSwapAngle(PairSwapAngle):
    """Build, run and fetch the partial-swap angle calibration on the QM OPX."""

    # preview opt-out (backend.SELF_ACQUIRING_ATTR): truthy reason = refuse
    probe_self_acquires = ("it executes one program per swap pair in a "
                           "Python loop inside probe()")

    def _build_pair_program(self, pair):
        """Build ONE target pair's QUA program -- the build half of ``probe()``,
        no acquire. Shared by ``probe()`` (looped over targets + acquired) and
        ``preview_program()`` (a single pair, dumped for ``--preview``)."""
        from ._reset import check_reset_method
        from ._vendor import role_side, vendor_pair, vendor_qubit

        # Resolved BEFORE any QUA is built, so a roster/params mismatch refuses
        # without costing instrument time.
        drive = role_side(self, self.params.drive_side, field="drive_side")
        flux = role_side(self, self.params.flux_side, field="flux_side",
                         needs_flux=True)
        if drive != "control" or flux != "control":
            raise ValueError(
                f"pair_swap_angle on QM excites the vendor pair's CONTROL member "
                f"and plays the swap's control-side flux pulse on it (only the "
                f"COUPLER amplitude is swept), but drive_side="
                f"{self.params.drive_side!r} / flux_side={self.params.flux_side!r} "
                f"resolve to (drive={drive}, flux={flux}). Select the control-side "
                f"role for both -- or run pair_swap_chevron, which honors either "
                f"side.")

        # One door for the reset method: this shell does not opt into active
        # reset (default DENY), so 'active' refuses by name until the sequence
        # has hardware evidence.
        reset = check_reset_method(self)
        qp = vendor_pair(self, pair)
        # scqo has already refused a name that is not a member of this pair.
        compensation = [
            (vendor_qubit(self, name, field="compensation_amps"), float(factor))
            for name, factor in self.params.compensation_amps.items()
        ]
        return build_program(
            self.backend.machine, [qp.qubit_control, qp.qubit_target], qp,
            compensation=compensation,
            stark_operation=self.params.stark_operation,
            stark_detuning_hz=self.params.stark_detuning_hz,
            swap_operation=self.params.swap_operation,
            rounds_array=np.asarray(self.sweep_axes["swap_count"]).astype(int),
            coupler_amplitudes=np.asarray(self.sweep_axes["coupler_flux_v"],
                                          dtype=float),
            num_shots=int(self.params.num_averages),
            reset_type=reset,
            operation_gap_ns=int(self.params.operation_gap_ns),
        )

    def preview_program(self) -> Any:
        """The single-target ``--preview`` build (QMBackend's single-pair preview
        path). The backend gates on exactly one target before calling this, so a
        self-acquiring shell can still be inspected without touching the QPU."""
        prog, _sweep_axes = self._build_pair_program(self.params.targets[0])
        return prog

    def probe(self) -> Any:
        from ._vendor import role_side

        machine = self.backend.machine  # type: ignore[attr-defined]
        # `high` fixes the member ordering for the readout-schema reshape;
        # resolved once (the per-pair build re-checks drive/flux and reset).
        high_side = role_side(self, "high", field="targets")
        shots = int(self.params.num_averages)

        # One program per pair (the probe takes a single swap pair); the two
        # measured qubits are exactly the pair's members, control first.
        per_pair = []
        for pair in self.params.targets:
            prog, sweep_axes = self._build_pair_program(pair)
            raw = acquire(machine, prog, sweep_axes,
                          num_shots=shots,
                          timeout=self.backend._timeout)
            per_pair.append(_member_states(raw["state"], high_side))

        state = xr.concat(per_pair, dim="qubit_pair").assign_coords(
            qubit_pair=list(self.params.targets))
        if self.params.readout_mode == "shot":
            return state.to_dataset(name="state")
        jp = states_to_joint_population(state, member_dim="member",
                                        shot_dim="shot_idx")
        return jp.to_dataset()
