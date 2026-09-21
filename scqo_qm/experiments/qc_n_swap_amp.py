"""N-swap (swap-chain) x qubit-flux-amplitude acquisition probe: vendor code only
(qm/quam/qualang_tools) - no qualibrate, no scqo, no scqat.

A 2D variant of the retired qc_N_swap probe (git history): on top of the swap-count sweep (N, inner
axis) the **control-qubit flux amplitude of the swap macro is swept** (outer axis), the
same knob `pair_swap_flux_map` sweeps in its `swap_via_macro` mode.
The swept amplitude is in **absolute volts**; this builder divides it by the macro z
pulse's stored amplitude in PYTHON and the QUA loop iterates the resulting
amplitude_scale, which each swap receives as `apply(ctrl_scale=...)`, while the coupler
plays bare at its baked amplitude. Every swap in a chain uses the same swept amplitude.
Dividing in QUA instead (`apply(ctrl_amp=<QUA variable>)`) would put the arithmetic
inside every round, between its align() and its play, so the rounds would no longer be
the bare gate's rounds -- the macro refuses it for that reason.

Circuit per shot (for a swept amplitude a and swap count N):
  1. Initialize every involved qubit with `q.reset(reset_type, simulate)`
     (involved = measured qubits + the swap pair's control/target).
  2. State prep: `swap_pair.qubit_control.xy.play("x180")`.
  3. Repeat N times: `swap_pair.macros[swap_operation].apply(ctrl_scale=a/ref)`, then idle
     the pair's flux lines for `operation_gap_ns` (if nonzero).
  4. Read out every measured qubit (state discrimination -> discriminated state, else raw I/Q).

Reading the measured qubits versus (amplitude, N) gives a 2D population map per joint state
(a swap-amplitude fine-tuning map by error amplification: more swaps amplify a small
amplitude miscalibration). With state discrimination the probe saves the **per-shot**
discriminated states (var `state`, dims `(qubit, shot, qubit_amplitude, round)`) so the node
can render the joint multi-qubit populations (one 2D map per state) or the per-qubit
marginals; without it the shot-averaged raw I/Q is saved instead. There is no fit and no
state writeback.

The chosen macro must expose a string `flux_pulse` playable on the control qubit's z
line and accept `apply(ctrl_scale=...)` (e.g. the lab `ISwapImplementation`); its stored
z-pulse amplitude is the rescaling reference and must be nonzero.

QM N-swap x flux-amplitude error-amplification map for scqo — supplies ``probe()``.

Parameters, the record-only map summary and the (absent) writeback are inherited
from ``scqo.experiments.QcNSwapAmp``. The vendor probe
(``build_program`` below) excites the pair's CONTROL qubit and plays
every swap through ``pair.macros[swap_operation].apply(ctrl_scale=...)`` — the
control-side flux amplitude, swept in absolute volts (exactly scqo's
``flux_amp_v`` axis) and converted to the macro's scale before the program is
built — and keeps EVERY shot's per-qubit discriminated state. This adapter:

* refuses unless ``drive_side``/``flux_side`` resolve to the vendor CONTROL
  member (the probe has no target-side mode; a silent role mismatch would
  mislabel the prepared/transfer panels);
* orders the per-shot states into the readout schema's (high, low) ``member``
  axis and, in ``readout_mode="average"``, reduces them to ``joint_population``
  via scqo's shared ``states_to_joint_population`` — ``"shot"`` returns the
  per-member ``state`` form as-is (the full-information trade).

Unlike the FPGA-averaged pair adapters this one does not use
``JointPopulationMixin``: the reduction happens here from per-shot data, with
the SAME helper the simulated backend uses.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
import xarray as xr
from qm.qua import *

from qualang_tools.loops import from_array

from scqo_qm.experiments._lib import acquire as _acquire
from scqo_qm.experiments._flux_limits import check_flux_pulse_relative, declared_idle_offset_v


def _dedup_involved(measure_qubits, swap_pair) -> List:
    """Return the unique (by `.name`) set of qubit elements that must be initialized.

    The measured qubits need not include the swap pair's control/target, but all of them
    must be flux-initialized and reset at the start of each shot.
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
    qubit_amplitudes,
    num_shots: int,
    reset_type: str,
    use_state_discrimination: bool,
    operation_gap_ns: int = 0,
    simulate: bool = False,
):
    """Build the N-swap x qubit-flux-amplitude QUA program.

    Returns ``(program, sweep_axes)``.

    `measure_qubits` is a plain list of qubit objects read out at the end of the circuit;
    `swap_pair` is a qubit-pair object whose `macros[swap_operation]` is applied each swap.
    `rounds_array` is the integer sweep over the number of swaps (N=0 allowed, giving just
    the x180 prep, inner axis); `qubit_amplitudes` is the control-qubit flux amplitude
    sweep in absolute volts (outer axis), converted here to the macro's `ctrl_scale`
    (volts / the pulse's stored amplitude, in Python, so no division runs inside a round).
    All measured qubits are read out within the same shot (joint / multiplexed readout),
    since they share one circuit.

    `operation_gap_ns` (multiple of 4, default 0) idles the swap pair's flux lines after
    each swap, so its flux pulse can settle before the next swap fires (the gap also
    separates the last swap from readout).
    """
    measure_qubits = list(measure_qubits)
    num_qubits = len(measure_qubits)
    rounds_array = np.asarray(rounds_array).astype(int)
    qubit_amplitudes = np.asarray(qubit_amplitudes, dtype=float)

    if operation_gap_ns < 0 or operation_gap_ns % 4 != 0:
        raise ValueError(f"operation_gap_ns must be a non-negative multiple of 4 ns, got {operation_gap_ns}.")
    gap_cycles = operation_gap_ns // 4

    involved = _dedup_involved(measure_qubits, swap_pair)

    # Validate the macro's swept control-amplitude path (the pair_qcq_fixed_time swap_via_macro
    # contract): the macro must exist, its z flux pulse must be playable and its stored
    # amplitude is the rescaling reference for the absolute-volt sweep.
    if swap_operation not in swap_pair.macros:
        raise ValueError(f"Pair {swap_pair.name} has no macro {swap_operation!r}; available: {list(swap_pair.macros)}.")
    flux_pulse_name = getattr(swap_pair.macros[swap_operation], "flux_pulse", None)
    ops = swap_pair.qubit_control.z.operations
    if not isinstance(flux_pulse_name, str) or flux_pulse_name not in ops:
        raise ValueError(
            f"Macro {swap_operation!r} on {swap_pair.name} has no z flux_pulse playable at a swept amplitude "
            f"(flux_pulse={flux_pulse_name!r})."
        )
    # Rail + amplitude_scale + idle-sum guard, shared with every other flux probe.
    # The macro's z pulse is the amplitude_scale REFERENCE (not a `const`, so the
    # rail/2 convention deliberately does not apply to it), and the swept volts are
    # an excursion on top of whatever standing bias initialize_qpu applied — this
    # probe takes no flux_point argument, so the declaration is what runs. The
    # reference it returns is what the volts are divided by, here and not in QUA.
    z = swap_pair.qubit_control.z
    ctrl_ref = check_flux_pulse_relative(
        z,
        name=f"{swap_pair.name} macro {swap_operation!r} on {swap_pair.qubit_control.name}.z",
        idle_v=declared_idle_offset_v(z),
        amps_v=qubit_amplitudes,
        operation=flux_pulse_name,
    )
    ctrl_scales = qubit_amplitudes / ctrl_ref

    # With state discrimination we save the per-shot discriminated states (so the joint
    # multi-qubit populations can be reconstructed downstream), hence the extra `shot` axis.
    # Without it we keep the shot-averaged raw I/Q schema (no `shot` axis).
    if use_state_discrimination:
        sweep_axes = {
            "qubit": xr.DataArray([q.name for q in measure_qubits]),
            "shot": xr.DataArray(np.arange(num_shots)),
            # Outer loop -> y axis.
            "qubit_amplitude": xr.DataArray(
                qubit_amplitudes, attrs={"long_name": "control qubit flux amplitude", "units": "V"}
            ),
            # Inner loop -> x axis.
            "round": xr.DataArray(rounds_array, attrs={"long_name": "number of swaps"}),
        }
    else:
        sweep_axes = {
            "qubit": xr.DataArray([q.name for q in measure_qubits]),
            # Outer loop -> y axis.
            "qubit_amplitude": xr.DataArray(
                qubit_amplitudes, attrs={"long_name": "control qubit flux amplitude", "units": "V"}
            ),
            # Inner loop -> x axis.
            "round": xr.DataArray(rounds_array, attrs={"long_name": "number of swaps"}),
        }

    with program() as prog:
        # Macro to declare I, Q, n and their respective streams for the measured qubits.
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        q_a = declare(fixed)  # swept ctrl flux amplitude, as the macro's amplitude_scale
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
            # Qubit-flux amplitude loop (outer -> y axis). It iterates the SCALES
            # (volts / reference, computed above), in the same order as the volts
            # axis the data is labelled with.
            with for_(*from_array(q_a, ctrl_scales)):
                # Swap-count loop (inner -> x axis)
                with for_(*from_array(r, rounds_array)):
                    # Initialization: thermalize / actively reset every involved qubit.
                    for q in involved:
                        q.reset(reset_type, simulate)
                    align()

                    # State prep: excite the swap pair's control qubit to |1>.
                    swap_pair.qubit_control.xy.play("x180")
                    align()

                    # Circuit body: N swaps on the pair, each at the swept ctrl amplitude
                    # (the coupler plays bare at its baked amplitude). The amplitude
                    # reaches the macro as a ready scale, so nothing is computed
                    # between a round's align() and its play.
                    # Dynamic loop bound on r -> N=0 skips the body entirely (baseline).
                    # `gap_cycles` idles the pair's flux lines between gate operations so
                    # each swap's flux pulse can settle before the next one fires.
                    with for_(rr, 0, rr < r, rr + 1):
                        swap_pair.macros[swap_operation].apply(ctrl_scale=q_a)
                        if gap_cycles > 0:
                            swap_pair.wait(gap_cycles)
                        align()

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
                # Inner buffer = swap count (x), next buffer = qubit amplitude (y).
                if use_state_discrimination:
                    # Keep every shot (no average) so the joint populations stay reconstructable:
                    # round buffer, then qubit_amplitude, then group num_shots -> (shot, qubit_amplitude, round).
                    state_st[i].buffer(len(rounds_array)).buffer(len(qubit_amplitudes)).buffer(num_shots).save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(rounds_array)).buffer(len(qubit_amplitudes)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(rounds_array)).buffer(len(qubit_amplitudes)).average().save(f"Q{i + 1}")

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


from typing import Any

import numpy as np
import xarray as xr
from scqo import register
from scqo.experiments import QcNSwapAmp, states_to_joint_population


def _member_states(state: xr.DataArray, high_side: str) -> xr.DataArray:
    """The probe's per-shot states, reordered onto the schema's axes.

    ``state`` arrives ``(qubit, shot, qubit_amplitude, round)`` with the qubit
    axis in READOUT order — ``[control, target]``, fixed by the adapter's
    measure list. The member axis is ROLE-ordered (high, low), so the vendor
    order flips when the roster's high member is the vendor target."""
    order = [0, 1] if high_side == "control" else [1, 0]
    da = state.isel(qubit=order).rename({
        "qubit": "member", "shot": "shot_idx",
        "qubit_amplitude": "flux_amp_v", "round": "swap_count",
    })
    return da.assign_coords(member=["high", "low"])


@register
class QMQcNSwapAmp(QcNSwapAmp):
    """Build, run and fetch the N-swap amplitude map on the QM OPX."""

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
                f"qc_n_swap_amp on QM excites AND flux-drives the vendor pair's "
                f"CONTROL member (the swap macro's control-side amplitude is the only swept "
                f"knob), but drive_side={self.params.drive_side!r} / flux_side="
                f"{self.params.flux_side!r} resolve to (drive={drive}, "
                f"flux={flux}). Select the control-side role for both — or run "
                f"pair_swap_chevron, which honors either side.")

        # One door for the reset method: refuses what QM cannot honour (this
        # shell does not opt into active reset — default DENY — so 'active'
        # refuses by name until it has hardware evidence for this sequence).
        reset = check_reset_method(self)
        qp = vendor_pair(self, pair)
        return build_program(
            self.backend.machine, [qp.qubit_control, qp.qubit_target], qp,
            swap_operation=self.params.swap_operation,
            rounds_array=np.asarray(self.sweep_axes["swap_count"]).astype(int),
            qubit_amplitudes=np.asarray(self.sweep_axes["flux_amp_v"], dtype=float),
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
