"""Crosstalk Compensated Deterministic Benchmark probe (QM/QUAM).

Executes deterministic gate trains on drive qubit with synchronized active
cancellation on probe port at f_probe:
1. 'calibrate':
   Probe qubit remains idle in |0>. Drive qubit plays N = cal_repetitions
   pulses of target_gate. Cancel element plays N pulses with amplitude
   cancel_amp and initial phase init_phase. Probe excitation is measured
   to locate the cancellation minimum.
2. 'benchmark':
   Probe qubit plays N repetitions of probe_gate concurrently with drive
   qubit over three conditions: isolated, simultaneous, and compensated.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Optional, List
import numpy as np
import xarray as xr
from qm.qua import *

from scqo_qm.experiments._lib import acquire as _acquire


def build_program(
    machine,
    probe_qubit,
    drive_qubit,
    *,
    mode: str,
    target_gate: str = "x180",
    probe_gate: str = "x180",
    cal_repetitions: int = 20,
    repetitions: list[int],
    num_shots: int,
    cancel_amp: float = 0.0,
    init_phase: float = 0.0,
    phase_rate_turns: float | None = None,
    sub_phase_rate_turns: float | None = None,
    cancel_amps: list[float] | None = None,
    init_phases: list[float] | None = None,
    reset_type: str = "thermal",
    use_state_discrimination: bool = False,
    seed: int | None = None,
    simulate: bool = False,
    conditions: list[str] | None = None,
    alternate_probe: bool = False,
    alternate_target: bool = False,
):
    """Build the QUA program for Crosstalk Compensated Deterministic Benchmark."""
    cancel_elem = f"{probe_qubit.xy.name}_cancel"

    try:
        x180_len = probe_qubit.xy.operations[probe_gate].length
        pulse_cycles = max(x180_len // 4, 1)
    except Exception:
        pulse_cycles = 4

    if sub_phase_rate_turns is not None:
        sub_p = float(sub_phase_rate_turns)
    elif phase_rate_turns is not None:
        sub_p = float(phase_rate_turns)
    else:
        sub_p = 0.0

    rep_arr = np.asarray(repetitions, dtype=int)
    cal_reps = [int(cal_repetitions)] if isinstance(cal_repetitions, (int, np.integer)) else [int(r) for r in cal_repetitions]

    cond_name_map = {"isolated": 0, "simultaneous": 1, "compensated": 2}
    if conditions is not None:
        benchmark_conditions = [c for c in conditions if c in cond_name_map]
    else:
        benchmark_conditions = ["isolated", "simultaneous", "compensated"]
    cond_indices = [cond_name_map[c] for c in benchmark_conditions]

    if mode == "calibrate":
        sweep_axes = {
            "qubit": xr.DataArray([probe_qubit.name]),
            "cancel_amp": xr.DataArray(np.asarray(cancel_amps, dtype=float)),
            "init_phase": xr.DataArray(np.asarray(init_phases, dtype=float)),
        }
    else:
        sweep_axes = {
            "qubit": xr.DataArray([probe_qubit.name]),
            "condition": xr.DataArray(np.asarray(benchmark_conditions, dtype=str)),
            "repetitions": xr.DataArray(rep_arr),
        }

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()

        if use_state_discrimination:
            state = [declare(int)]
            state_st = [declare_stream()]

        k = declare(int)
        amp_fixed = declare(fixed)
        amp_neg = declare(fixed)
        init_phase_fixed = declare(fixed)
        cal_rep_var = declare(int)
        rep_var = declare(int)
        cond_idx = declare(int)

        machine.initialize_qpu(target=probe_qubit)
        machine.initialize_qpu(target=drive_qubit)
        align()

        if mode == "calibrate":
            # Calibration mode: sweep cancel_amp, init_phase, and average over cal_reps with probe idle in |0>
            with for_each_(amp_fixed, [float(a) for a in cancel_amps]):
                with for_each_(init_phase_fixed, [float(p) / (2 * np.pi) for p in init_phases]):
                    with for_each_(cal_rep_var, cal_reps):
                        with for_(n, 0, n < num_shots, n + 1):
                            probe_qubit.reset(reset_type, simulate)
                            drive_qubit.reset(reset_type, simulate)
                            align(probe_qubit.xy.name, drive_qubit.xy.name, cancel_elem)
                            reset_frame(probe_qubit.xy.name)
                            reset_frame(drive_qubit.xy.name)
                            reset_frame(cancel_elem)
                            reset_if_phase(probe_qubit.xy.name)
                            reset_if_phase(drive_qubit.xy.name)
                            reset_if_phase(cancel_elem)
                            reset_global_phase()

                            # Apply initial phase to cancel element
                            frame_rotation_2pi(init_phase_fixed, cancel_elem)

                            # Repeat target_gate N times on drive qubit and cancel element without inter-gate delay
                            # In calibrate mode: rotate drive qubit phase by -sub_p to freeze crosstalk axis
                            # in probe frame, while cancel element remains fixed.
                            with switch_(cal_rep_var):
                                for r_val in cal_reps:
                                    r_int = int(r_val)
                                    with case_(r_int):
                                        for _ in range(r_int):
                                            drive_qubit.xy.play(target_gate)
                                            play(target_gate * amp(amp_fixed), cancel_elem)
                                            if sub_p != 0.0:
                                                frame_rotation_2pi(-sub_p, drive_qubit.xy.name)

                            align()
                            if use_state_discrimination:
                                probe_qubit.readout_state(state[0])
                                save(state[0], state_st[0])
                            else:
                                probe_qubit.resonator.measure("readout", qua_vars=(I[0], Q[0]))
                                save(I[0], I_st[0])
                                save(Q[0], Q_st[0])
                            align()

        def _play_condition(c_idx: int, r_count: int):
            if c_idx == 0:
                if r_count > 0:
                    wait(r_count * pulse_cycles, drive_qubit.xy.name, cancel_elem)
                    for k_idx in range(r_count):
                        if alternate_probe and (k_idx % 2 == 1):
                            probe_qubit.xy.play(probe_gate, amplitude_scale=-1.0)
                        else:
                            probe_qubit.xy.play(probe_gate)
                else:
                    wait(4, probe_qubit.xy.name)
            elif c_idx == 1:
                if r_count > 0:
                    for k_idx in range(r_count):
                        if alternate_probe and (k_idx % 2 == 1):
                            probe_qubit.xy.play(probe_gate, amplitude_scale=-1.0)
                        else:
                            probe_qubit.xy.play(probe_gate)

                        if alternate_target and (k_idx % 2 == 1):
                            drive_qubit.xy.play(target_gate, amplitude_scale=-1.0)
                        else:
                            drive_qubit.xy.play(target_gate)
                else:
                    wait(4, probe_qubit.xy.name)
            elif c_idx == 2:
                if r_count > 0:
                    for k_idx in range(r_count):
                        if alternate_probe and (k_idx % 2 == 1):
                            probe_qubit.xy.play(probe_gate, amplitude_scale=-1.0)
                        else:
                            probe_qubit.xy.play(probe_gate)

                        if alternate_target and (k_idx % 2 == 1):
                            drive_qubit.xy.play(target_gate, amplitude_scale=-1.0)
                            play(target_gate * amp(amp_neg), cancel_elem)
                        else:
                            drive_qubit.xy.play(target_gate)
                            play(target_gate * amp(amp_fixed), cancel_elem)

                        if sub_p != 0.0:
                            frame_rotation_2pi(sub_p, cancel_elem)
                else:
                    wait(4, probe_qubit.xy.name)

        if mode == "calibrate":
            pass
        else:
            # Benchmark mode: sweep repetitions across selected conditions
            assign(amp_fixed, cancel_amp)
            assign(amp_neg, -amp_fixed)
            assign(init_phase_fixed, init_phase / (2 * np.pi))

            if len(cond_indices) == 1:
                target_cond_idx = cond_indices[0]
                if len(rep_arr) == 1:
                    r_single = int(rep_arr[0])
                    with for_(n, 0, n < num_shots, n + 1):
                        probe_qubit.reset(reset_type, simulate)
                        drive_qubit.reset(reset_type, simulate)
                        align(probe_qubit.xy.name, drive_qubit.xy.name, cancel_elem)
                        reset_frame(probe_qubit.xy.name)
                        reset_frame(drive_qubit.xy.name)
                        reset_frame(cancel_elem)
                        reset_if_phase(probe_qubit.xy.name)
                        reset_if_phase(drive_qubit.xy.name)
                        reset_if_phase(cancel_elem)
                        reset_global_phase()

                        # Apply initial phase to cancel element
                        frame_rotation_2pi(init_phase_fixed, cancel_elem)

                        _play_condition(target_cond_idx, r_single)

                        align()
                        if use_state_discrimination:
                            probe_qubit.readout_state(state[0])
                            save(state[0], state_st[0])
                        else:
                            probe_qubit.resonator.measure("readout", qua_vars=(I[0], Q[0]))
                            save(I[0], I_st[0])
                            save(Q[0], Q_st[0])
                        align()
                else:
                    with for_each_(rep_var, [int(r) for r in rep_arr]):
                        with for_(n, 0, n < num_shots, n + 1):
                            probe_qubit.reset(reset_type, simulate)
                            drive_qubit.reset(reset_type, simulate)
                            align(probe_qubit.xy.name, drive_qubit.xy.name, cancel_elem)
                            reset_frame(probe_qubit.xy.name)
                            reset_frame(drive_qubit.xy.name)
                            reset_frame(cancel_elem)
                            reset_if_phase(probe_qubit.xy.name)
                            reset_if_phase(drive_qubit.xy.name)
                            reset_if_phase(cancel_elem)
                            reset_global_phase()

                            # Apply initial phase to cancel element
                            frame_rotation_2pi(init_phase_fixed, cancel_elem)

                            with switch_(rep_var):
                                for r in rep_arr:
                                    r_int = int(r)
                                    with case_(r_int):
                                        _play_condition(target_cond_idx, r_int)

                            align()
                            if use_state_discrimination:
                                probe_qubit.readout_state(state[0])
                                save(state[0], state_st[0])
                            else:
                                probe_qubit.resonator.measure("readout", qua_vars=(I[0], Q[0]))
                                save(I[0], I_st[0])
                                save(Q[0], Q_st[0])
                            align()
            elif len(rep_arr) == 1:
                r_single = int(rep_arr[0])
                with for_each_(cond_idx, cond_indices):
                    with for_(n, 0, n < num_shots, n + 1):
                        probe_qubit.reset(reset_type, simulate)
                        drive_qubit.reset(reset_type, simulate)
                        align(probe_qubit.xy.name, drive_qubit.xy.name, cancel_elem)
                        reset_frame(probe_qubit.xy.name)
                        reset_frame(drive_qubit.xy.name)
                        reset_frame(cancel_elem)
                        reset_if_phase(probe_qubit.xy.name)
                        reset_if_phase(drive_qubit.xy.name)
                        reset_if_phase(cancel_elem)
                        reset_global_phase()

                        # Apply initial phase to cancel element
                        frame_rotation_2pi(init_phase_fixed, cancel_elem)

                        with switch_(cond_idx):
                            for c_idx in cond_indices:
                                with case_(c_idx):
                                    _play_condition(c_idx, r_single)

                        align()
                        if use_state_discrimination:
                            probe_qubit.readout_state(state[0])
                            save(state[0], state_st[0])
                        else:
                            probe_qubit.resonator.measure("readout", qua_vars=(I[0], Q[0]))
                            save(I[0], I_st[0])
                            save(Q[0], Q_st[0])
                        align()
            else:
                with for_each_(cond_idx, cond_indices):
                    with for_each_(rep_var, [int(r) for r in rep_arr]):
                        with for_(n, 0, n < num_shots, n + 1):
                            probe_qubit.reset(reset_type, simulate)
                            drive_qubit.reset(reset_type, simulate)
                            align(probe_qubit.xy.name, drive_qubit.xy.name, cancel_elem)
                            reset_frame(probe_qubit.xy.name)
                            reset_frame(drive_qubit.xy.name)
                            reset_frame(cancel_elem)
                            reset_if_phase(probe_qubit.xy.name)
                            reset_if_phase(drive_qubit.xy.name)
                            reset_if_phase(cancel_elem)
                            reset_global_phase()

                            # Apply initial phase to cancel element
                            frame_rotation_2pi(init_phase_fixed, cancel_elem)

                            # Repeat gates unrolled with switch_ to avoid inter-gate delay
                            with switch_(cond_idx):
                                for c_idx in cond_indices:
                                    with case_(c_idx):
                                        with switch_(rep_var):
                                            for r in rep_arr:
                                                r_int = int(r)
                                                with case_(r_int):
                                                    _play_condition(c_idx, r_int)

                            align()
                            if use_state_discrimination:
                                probe_qubit.readout_state(state[0])
                                save(state[0], state_st[0])
                            else:
                                probe_qubit.resonator.measure("readout", qua_vars=(I[0], Q[0]))
                                save(I[0], I_st[0])
                                save(Q[0], Q_st[0])
                            align()

        with stream_processing():
            if mode == "calibrate":
                if use_state_discrimination:
                    state_st[0].buffer(num_shots).map(FUNCTIONS.average()).buffer(len(cal_reps)).map(FUNCTIONS.average()).buffer(len(init_phases)).buffer(len(cancel_amps)).save("state1")
                else:
                    I_st[0].buffer(num_shots).map(FUNCTIONS.average()).buffer(len(cal_reps)).map(FUNCTIONS.average()).buffer(len(init_phases)).buffer(len(cancel_amps)).save("I1")
                    Q_st[0].buffer(num_shots).map(FUNCTIONS.average()).buffer(len(cal_reps)).map(FUNCTIONS.average()).buffer(len(init_phases)).buffer(len(cancel_amps)).save("Q1")
            else:
                num_conds = len(cond_indices)
                if use_state_discrimination:
                    state_st[0].buffer(num_shots).map(FUNCTIONS.average()).buffer(len(rep_arr)).buffer(num_conds).save("state1")
                else:
                    I_st[0].buffer(num_shots).map(FUNCTIONS.average()).buffer(len(rep_arr)).buffer(num_conds).save("I1")
                    Q_st[0].buffer(num_shots).map(FUNCTIONS.average()).buffer(len(rep_arr)).buffer(num_conds).save("Q1")

    return prog, sweep_axes


def acquire(
    machine,
    prog,
    sweep_axes,
    *,
    num_shots: int,
    timeout: float = 120.0,
    log: bool = False,
    config: Optional[dict] = None,
) -> xr.Dataset:
    """Acquire CrosstalkCompensatedBenchmark dataset."""
    return _acquire(
        machine,
        prog,
        sweep_axes,
        num_shots=num_shots,
        timeout=timeout,
        log=log,
        config=config,
    )


def _patch_cancel_element(config: dict, probe_qubit: Any, drive_qubit: Any, target_gate: str) -> dict:
    """Add twin cancellation element to config with an independent hardware core.

    On OPX1000 architectures, elements bound to the same core cannot execute pulses
    concurrently (they are serialized). Assigning cancel_elem an unused core on the
    same FEM allows cancel_elem and probe_qubit.xy to output simultaneously.
    """
    cancel_elem_name = f"{probe_qubit.xy.name}_cancel"
    if cancel_elem_name in config.get("elements", {}):
        return config

    base_elem = config["elements"].get(probe_qubit.xy.name)
    if base_elem is None:
        alt = probe_qubit.xy.name.replace("_", ".") if "_" in probe_qubit.xy.name else probe_qubit.xy.name.replace(".", "_")
        base_elem = config["elements"].get(alt, {})
    elem_dict = copy.deepcopy(base_elem)

    # Enforce target_gate operation on cancel_elem inherits pulse definition from drive_qubit
    drive_name = drive_qubit.name if hasattr(drive_qubit, "name") else str(drive_qubit)
    drive_elem_name = drive_qubit.xy.name if hasattr(drive_qubit, "xy") else f"{drive_name}.xy"
    drive_elem = config["elements"].get(drive_elem_name)
    if drive_elem is None:
        alt_drive = drive_elem_name.replace("_", ".") if "_" in drive_elem_name else drive_elem_name.replace(".", "_")
        drive_elem = config["elements"].get(alt_drive, {})
    if drive_elem and "operations" in drive_elem and target_gate in drive_elem["operations"]:
        elem_dict.setdefault("operations", {})[target_gate] = copy.deepcopy(
            drive_elem["operations"][target_gate]
        )

    # Assign an independent hardware core on the same FEM
    fem_port = None
    if "MWInput" in elem_dict and isinstance(elem_dict["MWInput"], dict):
        fem_port = elem_dict["MWInput"].get("port")
    elif "mixInputs" in elem_dict and isinstance(elem_dict["mixInputs"], dict):
        fem_port = elem_dict["mixInputs"].get("I")

    if fem_port and len(fem_port) >= 2:
        fem_id = fem_port[:2]
        used_cores = set()
        for el in config["elements"].values():
            if not isinstance(el, dict):
                continue
            core = el.get("core") or el.get("thread")
            el_port = None
            if "MWInput" in el and isinstance(el["MWInput"], dict):
                el_port = el["MWInput"].get("port")
            elif "mixInputs" in el and isinstance(el["mixInputs"], dict):
                el_port = el["mixInputs"].get("I")
            if core and el_port and len(el_port) >= 2 and el_port[:2] == fem_id:
                used_cores.add(core)

        # Standard core candidates on OPX1000 FEMs: 'a' through 'h'
        core_candidates = ["a", "b", "c", "d", "e", "f", "g", "h"]
        available_cores = [c for c in core_candidates if c not in used_cores]
        if available_cores:
            new_core = available_cores[0]
        else:
            new_core = "f" if elem_dict.get("core") != "f" else "g"

        if "core" in elem_dict:
            elem_dict["core"] = new_core
        if "thread" in elem_dict:
            elem_dict["thread"] = new_core

    config["elements"][cancel_elem_name] = elem_dict
    return config


from scqo import register
from scqo.experiments.crosstalk_compensated_benchmark import CrosstalkCompensatedBenchmark


@register
class QMCrosstalkCompensatedBenchmark(CrosstalkCompensatedBenchmark):
    """Build Crosstalk Compensated Deterministic Benchmark QUA program on QM OPX."""

    def patch_preview_config(self, config: dict) -> dict:
        """Patch the twin cancellation element into the generated config for preview."""
        config = copy.deepcopy(config)
        machine = self.backend.machine
        from scqo_qm.experiments._lib import select_qubits

        probe_name = self.params.probe_qubit
        drive_name = self.params.drive_qubit
        qubits = select_qubits(machine, [probe_name, drive_name], multiplexed=True)
        q_dict = {q.name: q for q in qubits}
        probe_qubit = q_dict.get(probe_name, qubits[0])
        drive_qubit = q_dict.get(drive_name, qubits[1])

        return _patch_cancel_element(config, probe_qubit, drive_qubit, self.params.target_gate)

    def probe(self) -> Any:
        from ._reset import check_reset_method
        from scqo_qm.experiments._lib import select_qubits

        machine = self.backend.machine
        probe_name = self.params.probe_qubit
        drive_name = self.params.drive_qubit

        qubits = select_qubits(machine, [probe_name, drive_name], multiplexed=True)
        q_dict = {q.name: q for q in qubits}
        probe_qubit = q_dict.get(probe_name, qubits[0])
        drive_qubit = q_dict.get(drive_name, qubits[1])

        # Detect atomic pulse duration from calibrated probe gate length
        try:
            op_obj = drive_qubit.xy.operations[self.params.target_gate]
            pulse_duration_ns = float(getattr(op_obj, "length", self.params.pulse_duration_ns))
        except Exception:
            pulse_duration_ns = float(self.params.pulse_duration_ns)

        # Determine theoretical subslot phase rate using full RF frequencies and pulse duration
        if self.params.phase_rate is None:
            try:
                probe_rf_hz = float(probe_qubit.xy.RF_frequency)
                drive_rf_hz = float(drive_qubit.xy.RF_frequency)
                delta_f_hz = drive_rf_hz - probe_rf_hz
                sub_phase_rate_turns = float(delta_f_hz * (pulse_duration_ns * 1e-9))
                theoretical_rate_rad = float(2.0 * np.pi * delta_f_hz * (pulse_duration_ns * 1e-9))
            except Exception:
                sub_phase_rate_turns = 0.0
                theoretical_rate_rad = 0.0
        else:
            sub_phase_rate_turns = float(self.params.phase_rate) / (2 * np.pi)
            theoretical_rate_rad = float(self.params.phase_rate)

        config = machine.generate_config()
        config = _patch_cancel_element(config, probe_qubit, drive_qubit, self.params.target_gate)

        stage_reps = list(self.params.get_cal_stage_repetitions()) if self.params.mode == "calibrate" else [20]
        initial_cal_reps = [stage_reps[0]] if self.params.mode == "calibrate" else list(self.params.get_cal_repetitions())

        all_benchmark_reps = list(self.params.get_repetitions())
        batch_size = max(1, getattr(self.params, "benchmark_batch_size", 1))
        initial_benchmark_reps = all_benchmark_reps[:batch_size] if self.params.mode == "benchmark" else all_benchmark_reps

        prog, sweep_axes = build_program(
            machine,
            probe_qubit,
            drive_qubit,
            mode=self.params.mode,
            target_gate=self.params.target_gate,
            probe_gate=self.params.probe_gate,
            cal_repetitions=initial_cal_reps,
            repetitions=initial_benchmark_reps,
            num_shots=int(self.params.num_averages),
            cancel_amp=float(self.params.cancel_amp),
            init_phase=float(self.params.init_phase),
            phase_rate_turns=sub_phase_rate_turns,
            sub_phase_rate_turns=sub_phase_rate_turns,
            cancel_amps=list(self.params.get_cancel_amps()),
            init_phases=list(self.params.get_init_phases()),
            reset_type=check_reset_method(self),
            use_state_discrimination=bool(self.params.use_state_discrimination),
            simulate=getattr(self.backend, "simulate", False),
            alternate_probe=bool(getattr(self.params, "alternate_probe", False)),
            alternate_target=bool(getattr(self.params, "alternate_target", False)),
        )

        if self.params.mode == "calibrate":
            return (
                prog,
                sweep_axes,
                lambda m, p, s, **kw: self._acquire_calibrate(
                    m,
                    p,
                    s,
                    config=config,
                    probe_qubit=probe_qubit,
                    drive_qubit=drive_qubit,
                    reset_type=check_reset_method(self),
                    sub_phase_rate_turns=sub_phase_rate_turns,
                    **kw,
                ),
            )

        return (
            prog,
            sweep_axes,
            lambda m, p, s, **kw: self._acquire_benchmark(
                m,
                p,
                s,
                config=config,
                probe_qubit=probe_qubit,
                drive_qubit=drive_qubit,
                reset_type=check_reset_method(self),
                sub_phase_rate_turns=sub_phase_rate_turns,
                **kw,
            ),
        )

    def preview_program(self):
        """Render a single representative program for preview without memory bloat."""
        from scqo_qm.experiments._lib import select_qubits
        from ._reset import check_reset_method

        machine = self.backend.machine
        probe_name = self.params.probe_qubit
        drive_name = self.params.drive_qubit
        qubits = select_qubits(machine, [probe_name, drive_name], multiplexed=True)
        q_dict = {q.name: q for q in qubits}
        probe_qubit = q_dict.get(probe_name, qubits[0])
        drive_qubit = q_dict.get(drive_name, qubits[1])

        # Detect atomic pulse duration from calibrated probe gate length
        try:
            op_obj = drive_qubit.xy.operations[self.params.target_gate]
            pulse_duration_ns = float(getattr(op_obj, "length", self.params.pulse_duration_ns))
        except Exception:
            pulse_duration_ns = float(self.params.pulse_duration_ns)

        if self.params.phase_rate is None:
            try:
                probe_rf_hz = float(probe_qubit.xy.RF_frequency)
                drive_rf_hz = float(drive_qubit.xy.RF_frequency)
                delta_f_hz = drive_rf_hz - probe_rf_hz
                sub_phase_rate_turns = float(delta_f_hz * (pulse_duration_ns * 1e-9))
            except Exception:
                sub_phase_rate_turns = 0.0
        else:
            sub_phase_rate_turns = float(self.params.phase_rate) / (2 * np.pi)

        preview_conds = ["compensated"] if self.params.mode == "benchmark" else None
        preview_reps = [2] if self.params.mode == "benchmark" else [20]
        prog, _ = build_program(
            machine,
            probe_qubit,
            drive_qubit,
            mode=self.params.mode,
            target_gate=self.params.target_gate,
            probe_gate=self.params.probe_gate,
            conditions=preview_conds,
            cal_repetitions=preview_reps,
            repetitions=preview_reps,
            num_shots=min(int(self.params.num_averages), 10),
            cancel_amp=float(self.params.cancel_amp),
            init_phase=float(self.params.init_phase),
            sub_phase_rate_turns=sub_phase_rate_turns,
            cancel_amps=list(self.params.get_cancel_amps())[:3],
            init_phases=list(self.params.get_init_phases())[:3],
            reset_type=check_reset_method(self),
            use_state_discrimination=bool(self.params.use_state_discrimination),
            simulate=getattr(self.backend, "simulate", False),
            alternate_probe=bool(getattr(self.params, "alternate_probe", False)),
            alternate_target=bool(getattr(self.params, "alternate_target", False)),
        )
        return prog

    def _acquire_benchmark(
        self,
        machine,
        prog,
        sweep_axes,
        *,
        num_shots: int,
        timeout: float = 120.0,
        log: bool = False,
        config: Optional[dict] = None,
        probe_qubit=None,
        drive_qubit=None,
        reset_type: str = "thermal",
        sub_phase_rate_turns: float = 0.0,
    ) -> xr.Dataset:
        """Execute batched benchmark mode on Quantum Machine to prevent OPX memory exhaustion."""
        all_reps = list(self.params.get_repetitions())
        batch_size = max(1, getattr(self.params, "benchmark_batch_size", 1))

        if batch_size >= len(all_reps):
            return _acquire(
                machine,
                prog,
                sweep_axes,
                num_shots=num_shots,
                timeout=timeout,
                log=log,
                config=config,
            )

        batches = [all_reps[i : i + batch_size] for i in range(0, len(all_reps), batch_size)]
        num_batches = len(batches)
        batch_datasets: list[xr.Dataset] = []

        print(
            f"# [Benchmark] Executing {len(all_reps)} repetition points in {num_batches} "
            f"batches (batch_size={batch_size})..."
        )

        for b_idx, batch in enumerate(batches):
            if (
                b_idx == 0
                and sweep_axes is not None
                and "repetitions" in sweep_axes
                and np.array_equal(batch, sweep_axes["repetitions"].values)
            ):
                cur_prog = prog
                cur_axes = sweep_axes
            else:
                cur_prog, cur_axes = build_program(
                    machine,
                    probe_qubit,
                    drive_qubit,
                    mode="benchmark",
                    target_gate=self.params.target_gate,
                    probe_gate=self.params.probe_gate,
                    repetitions=batch,
                    num_shots=num_shots,
                    cancel_amp=float(self.params.cancel_amp),
                    init_phase=float(self.params.init_phase),
                    sub_phase_rate_turns=sub_phase_rate_turns,
                    reset_type=reset_type,
                    use_state_discrimination=bool(self.params.use_state_discrimination),
                    simulate=getattr(self.backend, "simulate", False),
                    alternate_probe=bool(getattr(self.params, "alternate_probe", False)),
                    alternate_target=bool(getattr(self.params, "alternate_target", False)),
                )

            ds_b = _acquire(
                machine,
                cur_prog,
                cur_axes,
                num_shots=num_shots,
                timeout=timeout,
                log=log,
                config=config,
            )
            batch_datasets.append(ds_b)
            print(f"# [Benchmark Batch {b_idx + 1}/{num_batches}] Repetitions {batch} completed.")

        total_ds = xr.concat(batch_datasets, dim="repetitions")
        return total_ds

    def _acquire_calibrate(
        self,
        machine,
        prog,
        sweep_axes,
        *,
        num_shots: int,
        timeout: float = 120.0,
        log: bool = False,
        config: Optional[dict] = None,
        probe_qubit=None,
        drive_qubit=None,
        reset_type: str = "thermal",
        sub_phase_rate_turns: float = 0.0,
    ) -> xr.Dataset:
        """Execute iterative multi-stage zoom-in calibration on Quantum Machine."""
        stage_reps = list(self.params.get_cal_stage_repetitions())
        num_stages = len(stage_reps)

        if num_stages <= 1:
            return _acquire(
                machine,
                prog,
                sweep_axes,
                num_shots=num_shots,
                timeout=timeout,
                log=log,
                config=config,
            )

        cur_min_amp = float(self.params.min_cancel_amp)
        cur_max_amp = float(self.params.max_cancel_amp)
        cur_min_phase = float(self.params.min_init_phase)
        cur_max_phase = float(self.params.max_init_phase)
        num_amps = int(self.params.num_cancel_amps)
        num_phases = int(self.params.num_init_phases)
        zoom = float(self.params.zoom_factor)

        last_ds: xr.Dataset | None = None
        stage_history: list[dict[str, Any]] = []

        for stage_idx, r in enumerate(stage_reps):
            amps_stage = list(np.linspace(cur_min_amp, cur_max_amp, num_amps))
            phases_stage = list(np.linspace(cur_min_phase, cur_max_phase, num_phases))

            if stage_idx == 0:
                cur_prog = prog
                cur_axes = sweep_axes
            else:
                cur_prog, cur_axes = build_program(
                    machine,
                    probe_qubit,
                    drive_qubit,
                    mode="calibrate",
                    target_gate=self.params.target_gate,
                    probe_gate=self.params.probe_gate,
                    cal_repetitions=[r],
                    repetitions=[0],
                    num_shots=num_shots,
                    cancel_amp=float(self.params.cancel_amp),
                    init_phase=float(self.params.init_phase),
                    sub_phase_rate_turns=sub_phase_rate_turns,
                    cancel_amps=amps_stage,
                    init_phases=phases_stage,
                    reset_type=reset_type,
                    use_state_discrimination=bool(self.params.use_state_discrimination),
                    simulate=getattr(self.backend, "simulate", False),
                )

            ds_stage = _acquire(
                machine,
                cur_prog,
                cur_axes,
                num_shots=num_shots,
                timeout=timeout,
                log=log,
                config=config,
            )
            last_ds = ds_stage

            var_name = "state" if "state" in ds_stage.data_vars else ("population" if "population" in ds_stage.data_vars else "I")
            p_vals = np.squeeze(ds_stage[var_name].values)

            # Robust coarse minimum with light smoothing to reject single-shot outlier spikes
            try:
                from scipy.ndimage import gaussian_filter
                p_smooth = gaussian_filter(p_vals, sigma=0.8)
            except Exception:
                pad = np.pad(p_vals, 1, mode="edge")
                p_smooth = (
                    pad[:-2, :-2] + pad[:-2, 1:-1] + pad[:-2, 2:] +
                    pad[1:-1, :-2] + pad[1:-1, 1:-1] + pad[1:-1, 2:] +
                    pad[2:, :-2] + pad[2:, 1:-1] + pad[2:, 2:]
                ) / 9.0

            min_idx = np.unravel_index(np.argmin(p_smooth), p_smooth.shape)
            i0, j0 = int(min_idx[0]), int(min_idx[1])
            best_amp = float(ds_stage.coords["cancel_amp"].values[i0])
            best_phase = float(ds_stage.coords["init_phase"].values[j0])
            min_p = float(p_vals[i0, j0])

            # Local 2D quadratic least-squares refinement to guide zoom window centering
            Nx, Ny = p_vals.shape
            rx = min(2, i0, Nx - 1 - i0)
            ry = min(2, j0, Ny - 1 - j0)
            if rx >= 1 and ry >= 1:
                amps_vals = np.asarray(ds_stage.coords["cancel_amp"].values, dtype=float)
                phases_vals = np.asarray(ds_stage.coords["init_phase"].values, dtype=float)
                d_amp = float(amps_vals[1] - amps_vals[0]) if Nx > 1 else 1.0
                d_phase = float(phases_vals[1] - phases_vals[0]) if Ny > 1 else 1.0
                A_rows = []
                b_vals = []
                for di in range(-rx, rx + 1):
                    for dj in range(-ry, ry + 1):
                        dx = float(amps_vals[i0 + di] - best_amp)
                        dy = float(phases_vals[j0 + dj] - best_phase)
                        A_rows.append([1.0, dx, dy, dx**2, dy**2, dx * dy])
                        b_vals.append(float(p_vals[i0 + di, j0 + dj]))
                c, _, _, _ = np.linalg.lstsq(A_rows, b_vals, rcond=None)
                H = np.array([[2.0 * c[3], c[5]], [c[5], 2.0 * c[4]]], dtype=float)
                g = np.array([c[1], c[2]], dtype=float)
                if np.all(np.linalg.eigvalsh(H) > 1e-9):
                    delta = -np.linalg.solve(H, g)
                    delta_a = float(np.clip(delta[0], -abs(d_amp), abs(d_amp)))
                    delta_p = float(np.clip(delta[1], -abs(d_phase), abs(d_phase)))
                    best_amp = float(best_amp + delta_a)
                    best_phase = float(best_phase + delta_p)

            stage_info = {
                "stage": stage_idx + 1,
                "repetitions": int(r),
                "cancel_amps": [float(a) for a in amps_stage],
                "init_phases": [float(p) for p in phases_stage],
                "p_vals": p_vals.tolist(),
                "best_cancel_amp": best_amp,
                "best_init_phase": best_phase,
                "min_signal": min_p,
            }
            stage_history.append(stage_info)
            print(
                f"# [Calibrate Stage {stage_idx + 1}/{num_stages}] Reps={r}: "
                f"Best cancel_amp={best_amp:.4f}, init_phase={best_phase:.4f} rad, min response={min_p:.4f}"
            )

            # Zoom in search window for next stage
            if stage_idx < num_stages - 1:
                span_amp = (cur_max_amp - cur_min_amp) / zoom
                span_phase = (cur_max_phase - cur_min_phase) / zoom
                cur_min_amp = max(0.0, best_amp - span_amp / 2.0)
                cur_max_amp = best_amp + span_amp / 2.0
                cur_min_phase = best_phase - span_phase / 2.0
                cur_max_phase = best_phase + span_phase / 2.0

        if last_ds is not None:
            import json
            last_ds.attrs["stage_history"] = json.dumps(stage_history)
            return last_ds
        return _acquire(machine, prog, sweep_axes, num_shots=num_shots, timeout=timeout, log=log, config=config)

