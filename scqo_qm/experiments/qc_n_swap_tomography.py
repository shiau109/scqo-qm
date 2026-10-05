"""N repeated partial swaps read by two-qubit tomography: QM probe.

Circuit per shot (stark amplitude factor a, swap count N, basis b):

  1-4. The round of qc_n_stark_amp (``_pair_swap_round``) in its COHERENCE
       flavour: reset every involved qubit, re-reference both members' xy frames,
       IF phases and the global phase (upconverters included), ``x180`` on the
       pair's CONTROL, then N x [bare swap macro, ``operation_gap_ns`` idle, stark
       tone at ``amplitude_scale=a``]. The stark operation carries its detuning IN
       its waveform (``stark_detuned``, quam_config/register_stark_detuned.py), so
       nothing switches a frequency: the per-shot phase reference and an
       ``update_frequency`` round trip do not mix (SCQO BACKLOG I35). Physically
       the same tone as qc_n_stark_amp's ``stark`` + ``update_frequency``.
  5.   Tomography pre-rotation on BOTH members at once, from the basis label
       (first character = the roster's HIGH member): ``z`` nothing, ``x``
       ``-y90``, ``y`` ``x90`` - the convention of qubit_tomography.
  6.   Joint (multiplexed) discriminated readout of both members.

Loop order: shots -> stark amplitude -> swap count -> basis (``switch_`` over
the nine labels). Before the main loop, a READOUT CALIBRATION block prepares
each basis state 00, 01, 10, 11 (``x180`` on the members whose digit is 1;
digits high, low) ``readout_calibration_shots`` times and reads it the same
way; the analysis inverts that 4x4 confusion.

The streams are heterogeneous (the tomography and the calibration block), so
this module carries its own ``acquire``. Like qc_n_stark_amp the shell
self-acquires: one program per target pair in a Python loop inside ``probe()``.
"""

from __future__ import annotations

from typing import Any, Callable, ClassVar, Optional

import numpy as np
import xarray as xr
from qm.qua import *

from qualang_tools.loops import from_array

from scqo_qm.experiments._pair_swap_round import (
    check_swap_round,
    dedup_involved,
    gap_cycles,
    play_swap_round,
    stark_ifs,
)

#: the pre-rotation that brings each measured axis onto Z (None = nothing)
PRE_ROTATION = {"z": None, "x": "-y90", "y": "x90"}
#: the single-qubit operations the program plays on both members
REQUIRED_XY = ("x180", "x90", "-y90")


def _check_members(members) -> None:
    for q in members:
        missing = [op for op in REQUIRED_XY if op not in q.xy.operations]
        if missing:
            raise ValueError(
                f"Qubit {q.name} has no xy operation(s) {missing}; the tomography pre-rotations "
                f"and the readout calibration need {list(REQUIRED_XY)} on both pair members.")


def _check_baked_detuning(ctrl, stark_operation: str, stark_detuning_hz: float) -> None:
    """The stark operation must be off-resonant BY ITS OWN WAVEFORM, by exactly
    ``stark_detuning_hz``: this round switches no frequency."""
    op = ctrl.xy.operations.get(stark_operation)
    detuning = getattr(op, "detuning", None)
    if detuning is None or abs(float(detuning) - float(stark_detuning_hz)) > 1.0:
        raise ValueError(
            f"{ctrl.name}.xy {stark_operation!r} must carry its own detuning of "
            f"{stark_detuning_hz:.0f} Hz in its waveform (found detuning={detuning!r}): the "
            f"tomography round re-references the frames every shot and switches no frequency. "
            f"Register it with quam_config/register_stark_detuned.py, or pass the "
            f"stark_operation that carries it.")


def build_program(
    machine,
    swap_pair,
    *,
    high_is_control: bool,
    swap_operation: str,
    stark_operation: str,
    stark_detuning_hz: float,
    rounds_array,
    stark_amps,
    bases,
    prepared_states,
    num_shots: int,
    calibration_shots: int,
    reset_type: str,
    operation_gap_ns: int = 0,
    simulate: bool = False,
):
    """Build the repeated-swap tomography program for ONE pair.

    Returns ``(program, layout)``: ``layout`` names the measured vendor qubits
    (control, target) and the stream shapes ``acquire`` reshapes against.
    ``bases`` / ``prepared_states`` are two-character labels, first character =
    the roster's HIGH member, which is the vendor control when
    ``high_is_control``."""
    rounds_array = np.asarray(rounds_array).astype(int)
    stark_amps = np.asarray(stark_amps, dtype=float)
    bases = [str(b) for b in bases]
    prepared_states = [str(p) for p in prepared_states]
    ctrl, tgt = swap_pair.qubit_control, swap_pair.qubit_target
    measure = [ctrl, tgt]
    high, low = (ctrl, tgt) if high_is_control else (tgt, ctrl)

    gap = gap_cycles(operation_gap_ns)
    involved = dedup_involved(measure, swap_pair)
    check_swap_round(swap_pair, swap_operation=swap_operation,
                     stark_operation=stark_operation, stark_amps=stark_amps)
    _check_members(measure)
    _check_baked_detuning(swap_pair.qubit_control, stark_operation, stark_detuning_hz)
    for label in bases:
        if len(label) != 2 or any(c not in PRE_ROTATION for c in label):
            raise ValueError(f"basis label {label!r} is not two of {sorted(PRE_ROTATION)}")
    base_if, stark_if = stark_ifs(swap_pair, stark_detuning_hz)

    with program() as prog:
        n = declare(int)
        n_st = declare_stream()
        q_a = declare(fixed)        # stark amplitude factor (amplitude_scale)
        r = declare(int)            # swap count
        rr = declare(int)           # inner round counter
        b = declare(int)            # basis index
        c = declare(int)            # calibration shot
        p = declare(int)            # prepared calibration state
        state = [declare(int) for _ in measure]
        state_st = [declare_stream() for _ in measure]
        cal_st = [declare_stream() for _ in measure]

        for q in involved:
            machine.initialize_qpu(target=q)
        align()

        # Readout calibration: each basis state, read exactly like the data.
        if calibration_shots > 0:
            with for_(c, 0, c < calibration_shots, c + 1):
                with for_(p, 0, p < len(prepared_states), p + 1):
                    for q in involved:
                        q.reset(reset_type, simulate)
                    align()
                    with switch_(p):
                        for k, label in enumerate(prepared_states):
                            with case_(k):
                                if label[0] == "1":
                                    high.xy.play("x180")
                                if label[1] == "1":
                                    low.xy.play("x180")
                    align()
                    for i, qubit in enumerate(measure):
                        qubit.readout_state(state[i])
                        save(state[i], cal_st[i])
                    align()

        with for_(n, 0, n < num_shots, n + 1):
            save(n, n_st)
            with for_(*from_array(q_a, stark_amps)):
                with for_(*from_array(r, rounds_array)):
                    with for_(b, 0, b < len(bases), b + 1):
                        # The coherence between the members' frames is read, so
                        # they are re-referenced every shot and no frequency is
                        # switched (the stark tone's detuning is in its waveform).
                        play_swap_round(
                            involved, swap_pair, swap_operation=swap_operation,
                            stark_operation=stark_operation, stark_amp=q_a, swap_count=r,
                            counter=rr, gap=gap, base_if=base_if, stark_if=stark_if,
                            reset_type=reset_type, simulate=simulate, switch_if=False,
                            reset_phase_of=[q.xy.name for q in measure])
                        with switch_(b):
                            for k, label in enumerate(bases):
                                with case_(k):
                                    for member, axis in ((high, label[0]), (low, label[1])):
                                        if PRE_ROTATION[axis] is not None:
                                            member.xy.play(PRE_ROTATION[axis])
                        align()
                        for i, qubit in enumerate(measure):
                            qubit.readout_state(state[i])
                            save(state[i], state_st[i])
                        align()

        with stream_processing():
            n_st.save("n")
            for i in range(len(measure)):
                (state_st[i].buffer(len(bases)).buffer(len(rounds_array))
                 .buffer(len(stark_amps)).buffer(num_shots).save(f"state{i + 1}"))
                if calibration_shots > 0:
                    (cal_st[i].buffer(len(prepared_states)).buffer(calibration_shots)
                     .save(f"cal_state{i + 1}"))

    layout = {"qubits": [q.name for q in measure], "num_shots": int(num_shots),
              "calibration_shots": int(calibration_shots)}
    return prog, layout


def acquire(machine, prog, layout, *, timeout: float,
            log: Optional[Callable] = None) -> dict[str, list[np.ndarray]]:
    """Execute and fetch: ``{"state": [control, target], "cal": [...]}`` raw
    arrays, ``state`` shaped (shot, stark_amp, swap_count, basis) and ``cal``
    (calibration shot, prepared state)."""
    from qualang_tools.multi_user import qm_session
    from qualang_tools.results import fetching_tool, progress_counter

    qmm = machine.connect()
    config = machine.generate_config()
    with qm_session(qmm, config, timeout=timeout) as qm:
        job = qm.execute(prog)
        fetcher = fetching_tool(job, ["n"], mode="live")
        while fetcher.is_processing():
            progress_counter(fetcher.fetch_all()[0], layout["num_shots"],
                             start_time=fetcher.start_time)
        results = job.result_handles
        results.wait_for_all_values()
        out: dict[str, list[np.ndarray]] = {"state": [], "cal": []}
        for i in range(len(layout["qubits"])):
            names = [f"state{i + 1}"] + ([f"cal_state{i + 1}"]
                                          if layout["calibration_shots"] > 0 else [])
            handles = {name: results.get(name) for name in names}
            missing = [name for name, h in handles.items() if h is None]
            if missing:
                raise RuntimeError(f"result handles missing {missing}")
            out["state"].append(np.asarray(handles[f"state{i + 1}"].fetch_all()))
            if layout["calibration_shots"] > 0:
                out["cal"].append(np.asarray(handles[f"cal_state{i + 1}"].fetch_all()))
        if log:
            rep = getattr(job, "execution_report", None)
            log(rep() if callable(rep) else rep)
    return out


def calibration_population(cal_high: np.ndarray, cal_low: np.ndarray,
                           prepared_states, joint_labels) -> np.ndarray:
    """``(prepared, joint)`` measured distributions from per-shot member levels
    shaped (calibration shot, prepared state); levels above 1 count as 1."""
    code = 2 * np.clip(cal_high, 0, 1).astype(int) + np.clip(cal_low, 0, 1).astype(int)
    out = np.zeros((len(prepared_states), len(joint_labels)))
    for k in range(len(prepared_states)):
        counts = np.bincount(code[:, k], minlength=4)[:4]
        out[k] = counts / max(code.shape[0], 1)
    return out


from scqo import register
from scqo.experiments import QcNSwapTomography, states_to_joint_population
from scqo.experiments.qc_n_swap_tomography import BASIS_LABELS, PREPARED_STATES
from scqo.requirements import Requirement

from ._requires import X90_AMPLITUDE


@register
class QMQcNSwapTomography(QcNSwapTomography):
    """Build, run and fetch the repeated-swap tomography on the QM OPX."""

    #: what only this backend consumes, on top of the neutral requirements
    requires: ClassVar[tuple[Requirement, ...]] = (
        *QcNSwapTomography.requires, X90_AMPLITUDE)
    #: true of THIS probe only; scqo shows them in
    #: `scqo run qc_n_swap_tomography --help`
    backend_notes: ClassVar[tuple[str, ...]] = (
        "the rotations are the -y90 and x90 operations, whose amplitude is pi_amp_x90; "
        "both members need x180, x90 and -y90",
        "drive_side and flux_side both have to name the pair's control member: the swap "
        "operation plays its member pulse on the control's z line only",
        "both members' drive frames, their intermediate-frequency phases and the global "
        "phase are reset at the start of every shot, and no frequency is switched: the "
        "stark operation has to carry stark_detuning_hz in its waveform "
        "(quam_config/register_stark_detuned.py makes it) and is refused otherwise",
        "a round lasts the swap, the gap and the tone plus about 8 ns of program overhead "
        "(measured on the gateway simulator); operation_gap_ns has to be a multiple of 4 ns",
        "the readout calibration block runs once, before the sweep",
        "one program per pair, one after another, run inside probe(); --preview builds "
        "the program of a single pair; every shot is fetched, in both readout modes",
    )

    # preview opt-out (backend.SELF_ACQUIRING_ATTR): truthy reason = refuse
    probe_self_acquires = ("it executes one program per swap pair in a "
                           "Python loop inside probe()")

    def _high_is_control(self) -> bool:
        from ._vendor import role_side

        return role_side(self, "high", field="targets") == "control"

    def _build_pair_program(self, pair):
        """Build ONE pair's program (the build half of ``probe()``); shared by
        ``probe()`` and ``preview_program()``. Refuses a non-control drive or
        flux side and an unsupported reset method before any QUA is built."""
        from ._reset import check_reset_method
        from ._vendor import role_side, vendor_pair

        drive = role_side(self, self.params.drive_side, field="drive_side")
        flux = role_side(self, self.params.flux_side, field="flux_side", needs_flux=True)
        if drive != "control" or flux != "control":
            raise ValueError(
                f"qc_n_swap_tomography on QM excites the vendor pair's CONTROL member, plays "
                f"the AC-Stark tone on its xy line, and rides the swap on its flux line, but "
                f"drive_side={self.params.drive_side!r} / flux_side={self.params.flux_side!r} "
                f"resolve to (drive={drive}, flux={flux}). Select the control-side role for "
                f"both.")
        reset = check_reset_method(self)
        qp = vendor_pair(self, pair)
        return build_program(
            self.backend.machine, qp,
            high_is_control=self._high_is_control(),
            swap_operation=self.params.swap_operation,
            stark_operation=self.params.stark_operation,
            stark_detuning_hz=self.params.stark_detuning_hz,
            rounds_array=np.asarray(self.sweep_axes["swap_count"]).astype(int),
            stark_amps=np.asarray(self.sweep_axes["stark_amp"], dtype=float),
            bases=list(BASIS_LABELS),
            prepared_states=list(PREPARED_STATES),
            num_shots=int(self.params.num_averages),
            calibration_shots=int(self.params.readout_calibration_shots),
            reset_type=reset,
            operation_gap_ns=int(self.params.operation_gap_ns),
        )

    def preview_program(self) -> Any:
        """The single-target ``--preview`` build (no acquire)."""
        prog, _layout = self._build_pair_program(self.params.targets[0])
        return prog

    def probe(self) -> Any:
        machine = self.backend.machine  # type: ignore[attr-defined]
        high_is_control = self._high_is_control()
        amps = np.asarray(self.sweep_axes["stark_amp"], dtype=float)
        counts = np.asarray(self.sweep_axes["swap_count"]).astype(int)
        joint = ["00", "01", "10", "11"]
        states, cals = [], []
        for pair in self.params.targets:
            prog, layout = self._build_pair_program(pair)
            raw = acquire(machine, prog, layout, timeout=self.backend._timeout)
            ctrl_s, tgt_s = raw["state"]
            high_s, low_s = (ctrl_s, tgt_s) if high_is_control else (tgt_s, ctrl_s)
            states.append(xr.DataArray(
                np.stack([high_s, low_s]).astype(np.int64),
                dims=("member", "shot_idx", "stark_amp", "swap_count", "basis"),
                coords={"member": ["high", "low"], "shot_idx": np.arange(high_s.shape[0]),
                        "stark_amp": amps, "swap_count": counts,
                        "basis": list(BASIS_LABELS)}))
            if raw["cal"]:
                ctrl_c, tgt_c = raw["cal"]
                high_c, low_c = (ctrl_c, tgt_c) if high_is_control else (tgt_c, ctrl_c)
                cals.append(calibration_population(high_c, low_c, PREPARED_STATES, joint))
        targets = list(self.params.targets)
        state = xr.concat(states, dim="qubit_pair").assign_coords(qubit_pair=targets)
        if self.params.readout_mode == "shot":
            ds = state.to_dataset(name="state")
        else:
            ds = states_to_joint_population(state, member_dim="member",
                                            shot_dim="shot_idx").to_dataset()
        if cals:
            ds["calibration_population"] = xr.DataArray(
                np.stack(cals), dims=("qubit_pair", "prepared_state", "joint_state"),
                coords={"qubit_pair": targets, "prepared_state": list(PREPARED_STATES),
                        "joint_state": joint})
        return ds
