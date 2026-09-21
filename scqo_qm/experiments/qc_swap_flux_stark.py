"""Fixed-N swap chain x (qubit-flux-amplitude x AC-Stark-amplitude) acquisition
probe: vendor code only (qm/quam/qualang_tools) - no qualibrate, no scqo, no scqat.

The two-amplitude cross of the qc_n_swap_amp and qc_n_stark_amp probes. The swap
count is a FIXED Python int (no count axis at all); what is swept is a 2D grid of
the swap macro's control-qubit flux amplitude (outer axis, absolute volts) against
the amplitude factor of a separate off-resonant RF (XY) tone -- the named
`stark_operation` on the control qubit's xy line, played after every swap (inner
axis). The tone is made off-resonant by `update_frequency` (a `SquarePulse` carries
no per-pulse detuning), so it AC-Stark-shifts the qubit rather than rotating it;
the frequency is restored before readout.

THE FLUX AMPLITUDE IS SWEPT AS A SCALE. The volts are divided by the macro pulse's
stored amplitude HERE, in Python, and the QUA loop iterates the resulting
amplitude_scale, handed to the macro as `ctrl_scale`. Passing the volts as
`ctrl_amp` instead made the macro divide on the FPGA inside every round, between
the round's align() and its play -- and while it did, this map's per-round phase
came out about 0.40 turn away from `qc_n_stark_amp`'s bare gate at the same flux
on 5Q4C q1_q2 (2026-09-21), the size of two clock cycles of round time at that
pair's 301 MHz detuning. With the scale precomputed, every round plays what the
bare gate plays, with no arithmetic in front of it.

Circuit per shot (for a swept flux amplitude a_f and stark factor a_s):
  1. Initialize every involved qubit with `q.reset(reset_type, simulate)`
     (involved = measured qubits + the swap pair's control/target).
  2. State prep: `swap_pair.qubit_control.xy.play("x180")` (at the RESONANT IF).
  3. Detune the control xy by `stark_detuning_hz` (`update_frequency`).
  4. Repeat `swap_count` times: `macros[swap_operation].apply(ctrl_scale=a_f/ref)`,
     idle the pair's flux lines for `operation_gap_ns` (if nonzero), then
     `qubit_control.xy.play(stark_operation, amplitude_scale=a_s)`.
  5. Restore the control xy IF, then read out every measured qubit.

The gap sits BETWEEN the swap and the stark tone (as in qc_n_stark_amp since the
pulse-ordering fix, not at the end of the round as in qc_n_swap_amp), so the swap's
flux pulse settles before the off-resonant tone plays.

`swap_count` is a compile-time constant, so the repeat is a QUA `for_` with a
LITERAL bound rather than an unrolled Python loop: the program stays small and a
count of 0 would simply skip the body (scqo's Parameters floor it at 1 anyway,
because a single swap leaves the stark axis inert -- its one tone plays after the
only swap, where it can do nothing but imprint a phase).

Reading the measured qubits versus (flux amplitude, stark amplitude) gives a 2D
population map per joint state: the flux brings the members onto resonance while
the Stark tone nulls the phase they accumulate between swaps, and because the
detuning winds its own between-round phase the two knobs are coupled. With state
discrimination the probe saves the **per-shot** discriminated states (var `state`,
dims `(qubit, shot, qubit_amplitude, stark_amplitude)`) so the node can render the
joint multi-qubit populations; without it the shot-averaged raw I/Q is saved
instead. There is no fit and no state writeback.

The chosen macro must expose a string `flux_pulse` playable on the control qubit's
z line and accept `apply(ctrl_scale=...)` (e.g. the lab `ISwapImplementation`); its
stored z-pulse amplitude is the rescaling reference and must be nonzero. The
coupler plays bare at its baked amplitude.

QM fixed-N flux x AC-Stark amplitude map for scqo -- supplies ``probe()``.

Parameters, the record-only map summary and the (absent) writeback are inherited
from ``scqo.experiments.QcSwapFluxStark``. This adapter:

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

from typing import Callable, List, Optional

import numpy as np
import xarray as xr
from qm.qua import *

from qualang_tools.loops import from_array

from scqo_qm.experiments._lib import acquire as _acquire
from scqo_qm.experiments._amp_limits import check_amp_scale_window
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
    stark_operation: str,
    stark_detuning_hz: float,
    swap_count: int,
    qubit_amplitudes,
    stark_amps,
    num_shots: int,
    reset_type: str,
    use_state_discrimination: bool,
    operation_gap_ns: int = 0,
    simulate: bool = False,
):
    """Build the fixed-N flux-amplitude x AC-Stark-amplitude QUA program.

    Returns ``(program, sweep_axes)``.

    `measure_qubits` is a plain list of qubit objects read out at the end of the circuit;
    `swap_pair` is a qubit-pair object whose `macros[swap_operation]` is applied each swap.
    `swap_count` is the FIXED number of swaps (not an axis). `qubit_amplitudes` is the
    control-qubit flux amplitude sweep in absolute volts (outer axis), converted here to
    the macro's `ctrl_scale` (volts / the pulse's stored amplitude, in Python, so no
    division runs inside a round); `stark_amps` is the swept stark amplitude FACTOR
    (dimensionless amplitude_scale, inner axis) applied to the control qubit's
    `stark_operation` xy tone. `stark_detuning_hz` is the FIXED off-resonant detuning of
    that tone. All measured qubits are read out within the same shot (joint / multiplexed
    readout), since they share one circuit.

    `operation_gap_ns` (multiple of 4, default 0) idles the swap pair's flux lines between
    each swap and the stark tone that follows it, so the flux pulse can settle before the
    off-resonant tone plays.
    """
    measure_qubits = list(measure_qubits)
    num_qubits = len(measure_qubits)
    qubit_amplitudes = np.asarray(qubit_amplitudes, dtype=float)
    stark_amps = np.asarray(stark_amps, dtype=float)
    swap_count = int(swap_count)

    if swap_count < 0:
        raise ValueError(f"swap_count must be non-negative, got {swap_count}.")
    if operation_gap_ns < 0 or operation_gap_ns % 4 != 0:
        raise ValueError(f"operation_gap_ns must be a non-negative multiple of 4 ns, got {operation_gap_ns}.")
    gap_cycles = operation_gap_ns // 4

    involved = _dedup_involved(measure_qubits, swap_pair)
    ctrl = swap_pair.qubit_control

    # Validate the macro's swept control-amplitude path (the pair_qcq_fixed_time swap_via_macro
    # contract): the macro must exist, its z flux pulse must be playable and its stored
    # amplitude is the rescaling reference for the absolute-volt sweep.
    if swap_operation not in swap_pair.macros:
        raise ValueError(f"Pair {swap_pair.name} has no macro {swap_operation!r}; available: {list(swap_pair.macros)}.")
    flux_pulse_name = getattr(swap_pair.macros[swap_operation], "flux_pulse", None)
    ops = ctrl.z.operations
    if not isinstance(flux_pulse_name, str) or flux_pulse_name not in ops:
        raise ValueError(
            f"Macro {swap_operation!r} on {swap_pair.name} has no z flux_pulse playable at a swept amplitude "
            f"(flux_pulse={flux_pulse_name!r})."
        )
    # Rail + amplitude_scale + idle-sum guard, shared with every other flux probe.
    # The macro's z pulse is the amplitude_scale REFERENCE (not a `const`, so the
    # rail/2 convention deliberately does not apply to it), and the swept volts are
    # an excursion on top of whatever standing bias initialize_qpu applied. The
    # reference it returns is what the volts are divided by, here and not in QUA.
    z = ctrl.z
    ctrl_ref = check_flux_pulse_relative(
        z,
        name=f"{swap_pair.name} macro {swap_operation!r} on {ctrl.name}.z",
        idle_v=declared_idle_offset_v(z),
        amps_v=qubit_amplitudes,
        operation=flux_pulse_name,
    )
    ctrl_scales = qubit_amplitudes / ctrl_ref
    # Validate the swept AC-Stark tone: the operation must exist on the control's xy line,
    # and the amplitude-factor window must be expressible by QUA's dynamic amplitude_scale.
    # BOTH guards are needed here — qc_n_swap_amp sweeps only the flux and qc_n_stark_amp
    # only the tone, but this probe sweeps each of them against the other.
    if stark_operation not in ctrl.xy.operations:
        raise ValueError(
            f"Qubit {ctrl.name} has no xy operation {stark_operation!r}; available: "
            f"{list(ctrl.xy.operations)}. Register it first (quam_config/register_stark.py)."
        )
    check_amp_scale_window(stark_amps, name=f"{ctrl.name}.xy {stark_operation!r}",
                           knob="max_stark_amp")

    base_if = ctrl.xy.intermediate_frequency          # resonant IF (x180 + readout)
    stark_if = int(stark_detuning_hz) + base_if        # off-resonant IF for the stark tone

    # With state discrimination we save the per-shot discriminated states (so the joint
    # multi-qubit populations can be reconstructed downstream), hence the extra `shot` axis.
    # Without it we keep the shot-averaged raw I/Q schema (no `shot` axis).
    axes = {
        # Outer loop -> y axis.
        "qubit_amplitude": xr.DataArray(
            qubit_amplitudes, attrs={"long_name": "control qubit flux amplitude", "units": "V"}
        ),
        # Inner loop -> x axis.
        "stark_amplitude": xr.DataArray(
            stark_amps, attrs={"long_name": "AC-Stark amplitude factor", "units": ""}
        ),
    }
    if use_state_discrimination:
        sweep_axes = {
            "qubit": xr.DataArray([q.name for q in measure_qubits]),
            "shot": xr.DataArray(np.arange(num_shots)),
            **axes,
        }
    else:
        sweep_axes = {"qubit": xr.DataArray([q.name for q in measure_qubits]), **axes}

    with program() as prog:
        # Macro to declare I, Q, n and their respective streams for the measured qubits.
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        q_f = declare(fixed)  # swept ctrl flux amplitude, as the macro's amplitude_scale
        q_s = declare(fixed)  # swept stark amplitude factor (amplitude_scale)
        rr = declare(int)  # swap counter
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
            with for_(*from_array(q_f, ctrl_scales)):
                # Stark-amplitude loop (inner -> x axis)
                with for_(*from_array(q_s, stark_amps)):
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
                    # readout (below) run at the resonant IF.
                    ctrl.xy.update_frequency(stark_if)

                    # Circuit body: swap_count swaps on the pair, each at the swept ctrl
                    # amplitude (the coupler plays bare at its baked amplitude), each
                    # followed SEQUENTIALLY by the swept off-resonant stark tone on the
                    # control xy. The amplitude reaches the macro as a ready scale, so
                    # nothing is computed between the round's align() and its play. The swap and the stark do NOT overlap:
                    # apply() ends with the pair's align() (FluxTunableTransmonPair.align
                    # aligns every channel of both qubits, INCLUDING control.xy), so the
                    # control's xy timeline is synced to the end of the swap before the
                    # stark plays. `gap_cycles` idles the pair BETWEEN the swap and the
                    # stark tone, so the swap's flux pulse settles first.
                    # The bound is a Python int, so this is a fixed-length loop, not a
                    # swept axis.
                    with for_(rr, 0, rr < swap_count, rr + 1):
                        swap_pair.macros[swap_operation].apply(ctrl_scale=q_f)
                        if gap_cycles > 0:
                            swap_pair.wait(gap_cycles)
                        ctrl.xy.play(stark_operation, amplitude_scale=q_s)
                        align()

                    # Restore the resonant IF before readout.
                    ctrl.xy.update_frequency(base_if)
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
                # Inner buffer = stark amplitude (x), next buffer = qubit amplitude (y).
                if use_state_discrimination:
                    # Keep every shot (no average) so the joint populations stay reconstructable:
                    # stark_amplitude buffer, then qubit_amplitude, then group num_shots ->
                    # (shot, qubit_amplitude, stark_amplitude).
                    state_st[i].buffer(len(stark_amps)).buffer(len(qubit_amplitudes)).buffer(num_shots).save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(stark_amps)).buffer(len(qubit_amplitudes)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(stark_amps)).buffer(len(qubit_amplitudes)).average().save(f"Q{i + 1}")

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
from scqo.experiments import QcSwapFluxStark, states_to_joint_population


def _member_states(state: xr.DataArray, high_side: str) -> xr.DataArray:
    """The probe's per-shot states, reordered onto the schema's axes.

    ``state`` arrives ``(qubit, shot, qubit_amplitude, stark_amplitude)`` with
    the qubit axis in READOUT order — ``[control, target]``, fixed by the
    adapter's measure list. The member axis is ROLE-ordered (high, low), so the
    vendor order flips when the roster's high member is the vendor target."""
    order = [0, 1] if high_side == "control" else [1, 0]
    da = state.isel(qubit=order).rename({
        "qubit": "member", "shot": "shot_idx",
        "qubit_amplitude": "flux_amp_v", "stark_amplitude": "stark_amp",
    })
    return da.assign_coords(member=["high", "low"])


@register
class QMQcSwapFluxStark(QcSwapFluxStark):
    """Build, run and fetch the fixed-N flux x AC-Stark amplitude map on the QM OPX."""

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
                f"qc_swap_flux_stark on QM excites the vendor pair's CONTROL member, "
                f"plays the AC-Stark tone on its xy line, and sweeps the swap on its "
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
            swap_count=int(self.params.swap_count),
            qubit_amplitudes=np.asarray(self.sweep_axes["flux_amp_v"], dtype=float),
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
