"""N-swap (swap-chain) x AC-Stark-amplitude acquisition probe: vendor code only
(qm/quam/qualang_tools) - no qualibrate, no scqo, no scqat.

The AC-Stark sibling of the qc_n_swap_amp probe: instead of sweeping the swap's
own flux amplitude, the swap runs at its FIXED baked amplitude (bare
``macros[swap_operation].apply()``) and after every swap a separate off-resonant
RF (XY) tone -- a new named ``stark_operation`` played on the control qubit's xy
line -- is played at the swept amplitude factor (outer axis). The tone is made
off-resonant by ``update_frequency`` (a `SquarePulse` carries no per-pulse
detuning), so it AC-Stark-shifts the qubit rather than rotating it; the frequency
is restored before readout.

Circuit per shot (for a swept stark amplitude factor a and swap count N):
  1. Initialize every involved qubit with ``q.reset(reset_type, simulate)``
     (involved = measured qubits + the swap pair's control/target).
  2. State prep: ``swap_pair.qubit_control.xy.play("x180")`` (at the RESONANT IF).
  3. Detune the control xy by ``stark_detuning_hz`` (``update_frequency``).
  4. Repeat N times: bare ``macros[swap_operation].apply()``, an idle of the
     pair's flux lines for ``operation_gap_ns`` (if nonzero), then
     ``qubit_control.xy.play(stark_operation, amplitude_scale=a)``.
     (Steps 1-4 are ``_pair_swap_round.play_swap_round``, shared with
     qc_n_swap_tomography.)
  5. Restore the control xy IF, then read out every measured qubit.

Reading the measured qubits versus (stark amplitude, N) gives a 2D population map
per joint state (a Stark-amplitude fine-tuning map by error amplification: more
swaps amplify a small residual detuning the Stark tone compensates). With state
discrimination the probe saves the **per-shot** discriminated states (var
``state``, dims ``(qubit, shot, stark_amplitude, round)``) so the node can render
the joint multi-qubit populations; without it the shot-averaged raw I/Q is saved
instead. There is no fit and no state writeback.

QM N-swap x AC-Stark-amplitude error-amplification map for scqo -- supplies ``probe()``.

Parameters, the record-only map summary and the (absent) writeback are inherited
from ``scqo.experiments.QcNStarkAmp``. This adapter:

* refuses unless ``drive_side``/``flux_side`` resolve to the vendor CONTROL
  member (the probe excites the control, plays the Stark tone on its xy line, and
  rides the swap on its flux line; a silent role mismatch would mislabel the
  prepared/transfer panels);
* orders the per-shot states into the readout schema's (high, low) ``member``
  axis and, in ``readout_mode="average"``, reduces them to ``joint_population``
  via scqo's shared ``states_to_joint_population`` -- ``"shot"`` returns the
  per-member ``state`` form as-is (the full-information trade).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import xarray as xr
from qm.qua import *

from qualang_tools.loops import from_array

from scqo_qm.experiments._lib import acquire as _acquire
from scqo_qm.experiments._pair_swap_round import (
    check_swap_round,
    dedup_involved,
    gap_cycles as _gap_cycles,
    play_swap_round,
    stark_ifs,
)


def build_program(
    machine,
    measure_qubits,
    swap_pair,
    *,
    swap_operation: str,
    stark_operation: str,
    stark_detuning_hz: float,
    rounds_array,
    stark_amps,
    num_shots: int,
    reset_type: str,
    use_state_discrimination: bool,
    operation_gap_ns: int = 0,
    simulate: bool = False,
):
    """Build the N-swap x AC-Stark-amplitude QUA program.

    Returns ``(program, sweep_axes)``.

    `measure_qubits` is a plain list of qubit objects read out at the end of the circuit;
    `swap_pair` is a qubit-pair object whose `macros[swap_operation]` is applied (bare, at
    its baked amplitude) each swap. `rounds_array` is the integer sweep over the number of
    swaps (N=0 allowed, giving just the x180 prep, inner axis); `stark_amps` is the swept
    stark amplitude FACTOR (dimensionless amplitude_scale, outer axis) applied to the
    control qubit's `stark_operation` xy tone. `stark_detuning_hz` is the FIXED off-resonant
    detuning of that tone. All measured qubits are read out within the same shot (joint /
    multiplexed readout), since they share one circuit.

    `operation_gap_ns` (multiple of 4, default 0) idles the swap pair's flux lines between
    each swap and its stark tone (swap -> gap -> stark), so the swap's flux pulse settles
    before the tone plays. The round itself is `_pair_swap_round.play_swap_round`.
    """
    measure_qubits = list(measure_qubits)
    num_qubits = len(measure_qubits)
    rounds_array = np.asarray(rounds_array).astype(int)
    stark_amps = np.asarray(stark_amps, dtype=float)

    gap = _gap_cycles(operation_gap_ns)
    involved = dedup_involved(measure_qubits, swap_pair)
    # The swap plays BARE at its baked amplitude (no rail-sweep check), so only the
    # macro, its z flux pulse and the swept stark tone are validated.
    check_swap_round(swap_pair, swap_operation=swap_operation,
                     stark_operation=stark_operation, stark_amps=stark_amps)
    base_if, stark_if = stark_ifs(swap_pair, stark_detuning_hz)

    # With state discrimination we save the per-shot discriminated states (so the joint
    # multi-qubit populations can be reconstructed downstream), hence the extra `shot` axis.
    # Without it we keep the shot-averaged raw I/Q schema (no `shot` axis).
    if use_state_discrimination:
        sweep_axes = {
            "qubit": xr.DataArray([q.name for q in measure_qubits]),
            "shot": xr.DataArray(np.arange(num_shots)),
            # Outer loop -> y axis.
            "stark_amplitude": xr.DataArray(
                stark_amps, attrs={"long_name": "AC-Stark amplitude factor", "units": ""}
            ),
            # Inner loop -> x axis.
            "round": xr.DataArray(rounds_array, attrs={"long_name": "number of swaps"}),
        }
    else:
        sweep_axes = {
            "qubit": xr.DataArray([q.name for q in measure_qubits]),
            # Outer loop -> y axis.
            "stark_amplitude": xr.DataArray(
                stark_amps, attrs={"long_name": "AC-Stark amplitude factor", "units": ""}
            ),
            # Inner loop -> x axis.
            "round": xr.DataArray(rounds_array, attrs={"long_name": "number of swaps"}),
        }

    with program() as prog:
        # Macro to declare I, Q, n and their respective streams for the measured qubits.
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        q_a = declare(fixed)  # swept stark amplitude factor (amplitude_scale)
        r = declare(int)  # swept swap count (current value from rounds_array)
        rr = declare(int)  # inner swap counter
        if use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        # Initialize the QPU in terms of flux points for every involved element.
        for q in involved:
            machine.initialize_qpu(target=q)
        align()

        with for_(n, 0, n < num_shots, n + 1):
            save(n, n_st)
            # Stark-amplitude loop (outer -> y axis)
            with for_(*from_array(q_a, stark_amps)):
                # Swap-count loop (inner -> x axis)
                with for_(*from_array(r, rounds_array)):
                    play_swap_round(
                        involved, swap_pair, swap_operation=swap_operation,
                        stark_operation=stark_operation, stark_amp=q_a, swap_count=r,
                        counter=rr, gap=gap, base_if=base_if, stark_if=stark_if,
                        reset_type=reset_type, simulate=simulate)

                    # Joint (multiplexed) readout of all measured qubits.
                    for i, qubit in enumerate(measure_qubits):
                        if use_state_discrimination:
                            qubit.readout_state(state[i])
                            save(state[i], state_st[i])
                        else:
                            qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])
                    align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                # Inner buffer = swap count (x), next buffer = stark amplitude (y).
                if use_state_discrimination:
                    # Keep every shot (no average) so the joint populations stay reconstructable:
                    # round buffer, then stark_amplitude, then group num_shots -> (shot, stark_amplitude, round).
                    state_st[i].buffer(len(rounds_array)).buffer(len(stark_amps)).buffer(num_shots).save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(rounds_array)).buffer(len(stark_amps)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(rounds_array)).buffer(len(stark_amps)).average().save(f"Q{i + 1}")

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
    return _acquire(machine, prog, sweep_axes, num_shots=num_shots, timeout=timeout, log=log)


from typing import Any, ClassVar

import numpy as np
import xarray as xr
from scqo import register
from scqo.experiments import QcNStarkAmp, states_to_joint_population


def _member_states(state: xr.DataArray, high_side: str) -> xr.DataArray:
    """The probe's per-shot states, reordered onto the schema's axes.

    ``state`` arrives ``(qubit, shot, stark_amplitude, round)`` with the qubit
    axis in READOUT order — ``[control, target]``, fixed by the adapter's
    measure list. The member axis is ROLE-ordered (high, low), so the vendor
    order flips when the roster's high member is the vendor target."""
    order = [0, 1] if high_side == "control" else [1, 0]
    da = state.isel(qubit=order).rename({
        "qubit": "member", "shot": "shot_idx",
        "stark_amplitude": "stark_amp", "round": "swap_count",
    })
    return da.assign_coords(member=["high", "low"])


@register
class QMQcNStarkAmp(QcNStarkAmp):
    """Build, run and fetch the N-swap AC-Stark amplitude map on the QM OPX."""

    #: true of THIS probe only; scqo shows them in
    #: `scqo run qc_n_stark_amp --help`
    backend_notes: ClassVar[tuple[str, ...]] = (
        "drive_side and flux_side both have to name the pair's control member: the swap "
        "operation plays its member pulse on the control's z line only",
        "the tone is made off-resonant by moving the control's intermediate frequency by "
        "stark_detuning_hz for the rounds and back before the readout; a stark amplitude "
        "factor of 2 or more is refused",
        "a round lasts the swap, the gap and the tone plus about 8 ns of program overhead "
        "(measured on the gateway simulator); operation_gap_ns has to be a multiple of 4 ns",
        "one program per pair, one after another, run inside probe(); --preview builds "
        "the program of a single pair; every shot is fetched, in both readout modes",
    )

    # preview opt-out (backend.SELF_ACQUIRING_ATTR): truthy reason = refuse
    probe_self_acquires = ("it executes one program per swap pair in a "
                           "Python loop inside probe()")

    def _build_pair_program(self, pair):
        """Build ONE target pair's QUA program — the build half of ``probe()``,
        no acquire. Shared by ``probe()`` (looped over targets + acquired) and
        ``preview_program()`` (a single pair, dumped for ``--preview``).

        The role/reset validation reads ``self.params.targets`` (it refuses a
        role that maps onto different vendor sides across the run), so it is the
        same whichever pair is passed — cheap to re-run per call for a handful
        of targets, and it keeps ``preview_program`` self-contained."""
        from ._reset import check_reset_method
        from ._vendor import role_side, vendor_pair

        # Resolved BEFORE any QUA is built, so a roster/params mismatch refuses
        # without costing instrument time.
        drive = role_side(self, self.params.drive_side, field="drive_side")
        flux = role_side(self, self.params.flux_side, field="flux_side",
                         needs_flux=True)
        if drive != "control" or flux != "control":
            raise ValueError(
                f"qc_n_stark_amp on QM excites the vendor pair's CONTROL member, "
                f"plays the AC-Stark tone on its xy line, and rides the swap on its "
                f"flux line, but drive_side={self.params.drive_side!r} / flux_side="
                f"{self.params.flux_side!r} resolve to (drive={drive}, flux={flux}). "
                f"Select the control-side role for both — or run pair_swap_chevron, "
                f"which honors either side.")

        # One door for the reset method: refuses what QM cannot honour (this
        # shell does not opt into active reset — default DENY — so 'active'
        # refuses by name until it has hardware evidence for this sequence).
        reset = check_reset_method(self)
        qp = vendor_pair(self, pair)
        return build_program(
            self.backend.machine, [qp.qubit_control, qp.qubit_target], qp,
            swap_operation=self.params.swap_operation,
            stark_operation=self.params.stark_operation,
            stark_detuning_hz=self.params.stark_detuning_hz,
            rounds_array=np.asarray(self.sweep_axes["swap_count"]).astype(int),
            stark_amps=np.asarray(self.sweep_axes["stark_amp"], dtype=float),
            num_shots=int(self.params.num_averages),
            reset_type=reset,
            use_state_discrimination=True,
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
