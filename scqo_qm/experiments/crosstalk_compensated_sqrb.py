"""Crosstalk Compensated Simultaneous SQRB acquisition probe (QM/QUAM).

Supplies QUA program generation and data acquisition for CrosstalkCompensatedSQRB.
Compares isolated SQRB, simultaneous SQRB, and active crosstalk-compensated simultaneous SQRB.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import xarray as xr
from qm.qua import *
from qualang_tools.bakery.randomized_benchmark_c1 import c1_table
from qualang_tools.loops import from_array

from scqo import register
from scqo.experiments.crosstalk_compensated_sqrb import (
    CrosstalkCompensatedSQRB,
    compute_theoretical_phase_rate,
)
from scqo_qm.experiments._lib import acquire as _acquire

# Recovery gates lookup: inv_gates[i] is the Clifford gate index that resets state i to 0
inv_gates = [int(np.where(c1_table[i, :] == 0)[0][0]) for i in range(24)]


def play_single_clifford(gate_idx, qubit, idle_len: int):
    """Play single Clifford gate on a qubit."""
    with switch_(gate_idx, unsafe=True):
        with case_(0):
            qubit.xy.wait(idle_len)
        with case_(1):
            qubit.xy.play("x180")
        with case_(2):
            qubit.xy.play("y180")
        with case_(3):
            qubit.xy.play("y180")
            qubit.xy.play("x180")
        with case_(4):
            qubit.xy.play("x90")
            qubit.xy.play("y90")
        with case_(5):
            qubit.xy.play("x90")
            qubit.xy.play("-y90")
        with case_(6):
            qubit.xy.play("-x90")
            qubit.xy.play("y90")
        with case_(7):
            qubit.xy.play("-x90")
            qubit.xy.play("-y90")
        with case_(8):
            qubit.xy.play("y90")
            qubit.xy.play("x90")
        with case_(9):
            qubit.xy.play("y90")
            qubit.xy.play("-x90")
        with case_(10):
            qubit.xy.play("-y90")
            qubit.xy.play("x90")
        with case_(11):
            qubit.xy.play("-y90")
            qubit.xy.play("-x90")
        with case_(12):
            qubit.xy.play("x90")
        with case_(13):
            qubit.xy.play("-x90")
        with case_(14):
            qubit.xy.play("y90")
        with case_(15):
            qubit.xy.play("-y90")
        with case_(16):
            qubit.xy.play("-x90")
            qubit.xy.play("y90")
            qubit.xy.play("x90")
        with case_(17):
            qubit.xy.play("-x90")
            qubit.xy.play("-y90")
            qubit.xy.play("x90")
        with case_(18):
            qubit.xy.play("x180")
            qubit.xy.play("y90")
        with case_(19):
            qubit.xy.play("x180")
            qubit.xy.play("-y90")
        with case_(20):
            qubit.xy.play("y180")
            qubit.xy.play("x90")
        with case_(21):
            qubit.xy.play("y180")
            qubit.xy.play("-x90")
        with case_(22):
            qubit.xy.play("x90")
            qubit.xy.play("y90")
            qubit.xy.play("x90")
        with case_(23):
            qubit.xy.play("-x90")
            qubit.xy.play("y90")
            qubit.xy.play("-x90")


def play_compensation_tone(drive_gate_idx, cancel_elem_name: str, amp_val, base_phase_turns, pulse_cycles: int):
    """Play compensation pulse with phase offset derived from drive gate axis."""
    with switch_(drive_gate_idx, unsafe=True):
        with case_(0):
            # Idle on drive qubit -> no crosstalk compensation pulse
            wait(pulse_cycles, cancel_elem_name)
        with case_(1):
            # x180 (axis theta = 0)
            frame_rotation_2pi(base_phase_turns, cancel_elem_name)
            play("x180" * amp(amp_val), cancel_elem_name)
            frame_rotation_2pi(-base_phase_turns, cancel_elem_name)
        with case_(2):
            # y180 (axis theta = 0.25 turns)
            frame_rotation_2pi(base_phase_turns + 0.25, cancel_elem_name)
            play("x180" * amp(amp_val), cancel_elem_name)
            frame_rotation_2pi(-(base_phase_turns + 0.25), cancel_elem_name)
        with case_(12):
            # x90 (axis theta = 0)
            frame_rotation_2pi(base_phase_turns, cancel_elem_name)
            play("x90" * amp(amp_val), cancel_elem_name)
            frame_rotation_2pi(-base_phase_turns, cancel_elem_name)
        with case_(13):
            # -x90 (axis theta = 0.5 turns)
            frame_rotation_2pi(base_phase_turns + 0.5, cancel_elem_name)
            play("x90" * amp(amp_val), cancel_elem_name)
            frame_rotation_2pi(-(base_phase_turns + 0.5), cancel_elem_name)
        with case_(14):
            # y90 (axis theta = 0.25 turns)
            frame_rotation_2pi(base_phase_turns + 0.25, cancel_elem_name)
            play("x90" * amp(amp_val), cancel_elem_name)
            frame_rotation_2pi(-(base_phase_turns + 0.25), cancel_elem_name)
        with case_(15):
            # -y90 (axis theta = 0.75 turns)
            frame_rotation_2pi(base_phase_turns + 0.75, cancel_elem_name)
            play("x90" * amp(amp_val), cancel_elem_name)
            frame_rotation_2pi(-(base_phase_turns + 0.75), cancel_elem_name)
        for c in (3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21, 22, 23):
            with case_(c):
                # Composite gates: play standard scaled cancellation pulse along base phase
                frame_rotation_2pi(base_phase_turns, cancel_elem_name)
                play("x180" * amp(amp_val), cancel_elem_name)
                frame_rotation_2pi(-base_phase_turns, cancel_elem_name)


def build_program(
    machine,
    probe_qubit,
    drive_qubit,
    *,
    mode: str,
    depths: list[int],
    num_sequences: int,
    num_shots: int,
    cancel_amp: float,
    init_phase: float,
    phase_rate_turns: float,
    cancel_amps: list[float],
    init_phases: list[float],
    cal_depth: int,
    reset_type: str,
    use_state_discrimination: bool = False,
    seed: int | None = None,
    simulate: bool = False,
):
    """Build the QUA program for Crosstalk Compensated Simultaneous SQRB."""
    depth_arr = np.asarray(depths, dtype=int)
    max_circuit_depth = int(depth_arr.max()) if mode == "benchmark" else int(cal_depth)
    seq_arr = np.arange(num_sequences, dtype=int)
    cancel_elem = f"{probe_qubit.xy.name}_cancel"

    try:
        x180_len = probe_qubit.xy.operations["x180"].length
        idle_len = max(x180_len // 4, 1)
    except Exception:
        idle_len = 10

    if mode == "calibrate":
        sweep_axes = {
            "qubit": xr.DataArray([probe_qubit.name]),
            "cancel_amp": xr.DataArray(np.asarray(cancel_amps, dtype=float)),
            "init_phase": xr.DataArray(np.asarray(init_phases, dtype=float)),
            "sequence_idx": xr.DataArray(seq_arr),
        }
    else:
        sweep_axes = {
            "qubit": xr.DataArray([probe_qubit.name]),
            "condition": xr.DataArray(["isolated", "simultaneous", "compensated"]),
            "sequence_idx": xr.DataArray(seq_arr),
            "depth": xr.DataArray(depth_arr),
        }

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        state = [declare(int)]
        state_st = [declare_stream()]

        depth_var = declare(int)
        cond_idx = declare(int)
        m = declare(int)
        m_st = declare_stream()

        # Random generation arrays
        cayley = declare(int, value=c1_table.flatten().tolist())
        inv_list = declare(int, value=inv_gates)
        curr_state_p = declare(int)
        curr_state_d = declare(int)
        step_p = declare(int)
        step_d = declare(int)

        seq_p = declare(int, size=max_circuit_depth + 1)
        seq_d = declare(int, size=max_circuit_depth + 1)
        inv_p = declare(int, size=max_circuit_depth + 1)
        inv_d = declare(int, size=max_circuit_depth + 1)
        saved_gate_p = declare(int)
        saved_gate_d = declare(int)

        i_gen = declare(int)
        gate_k = declare(int)
        curr_phase_turns = declare(fixed)
        amp_fixed = declare(fixed)
        init_phase_fixed = declare(fixed)

        rand_p = Random(seed=seed if seed is not None else 42)
        rand_d = Random(seed=(seed + 1000) if seed is not None else 1042)

        machine.initialize_qpu(target=probe_qubit)
        machine.initialize_qpu(target=drive_qubit)
        align()

        if mode == "calibrate":
            # Calibrate mode: sweep 2D (cancel_amp, init_phase) at fixed cal_depth
            with for_each_(amp_fixed, [float(a) for a in cancel_amps]):
                with for_each_(init_phase_fixed, [float(p) / (2 * np.pi) for p in init_phases]):
                    with for_(m, 0, m < num_sequences, m + 1):
                        save(m, m_st)

                        # Generate random sequence for probe and drive
                        assign(curr_state_p, 0)
                        assign(curr_state_d, 0)
                        with for_(i_gen, 0, i_gen < cal_depth, i_gen + 1):
                            assign(step_p, rand_p.rand_int(24))
                            assign(curr_state_p, cayley[curr_state_p * 24 + step_p])
                            assign(seq_p[i_gen], step_p)
                            assign(inv_p[i_gen], inv_list[curr_state_p])

                            assign(step_d, rand_d.rand_int(24))
                            assign(curr_state_d, cayley[curr_state_d * 24 + step_d])
                            assign(seq_d[i_gen], step_d)
                            assign(inv_d[i_gen], inv_list[curr_state_d])

                        assign(saved_gate_p, seq_p[cal_depth])
                        assign(seq_p[cal_depth], inv_p[cal_depth - 1])
                        assign(saved_gate_d, seq_d[cal_depth])
                        assign(seq_d[cal_depth], inv_d[cal_depth - 1])

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

                            assign(curr_phase_turns, init_phase_fixed)
                            with for_(gate_k, 0, gate_k <= cal_depth, gate_k + 1):
                                play_single_clifford(seq_p[gate_k], probe_qubit, idle_len)
                                play_single_clifford(seq_d[gate_k], drive_qubit, idle_len)
                                play_compensation_tone(seq_d[gate_k], cancel_elem, amp_fixed, curr_phase_turns, idle_len)
                                assign(curr_phase_turns, curr_phase_turns + phase_rate_turns)
                                with if_(curr_phase_turns >= 1.0):
                                    assign(curr_phase_turns, curr_phase_turns - 1.0)
                                with if_(curr_phase_turns <= -1.0):
                                    assign(curr_phase_turns, curr_phase_turns + 1.0)
                                align(probe_qubit.xy.name, drive_qubit.xy.name, cancel_elem)

                            align()
                            if use_state_discrimination:
                                probe_qubit.readout_state(state[0])
                                save(state[0], state_st[0])
                            else:
                                probe_qubit.resonator.measure("readout", qua_vars=(I[0], Q[0]))
                                save(I[0], I_st[0])
                                save(Q[0], Q_st[0])
                            align()

                        assign(seq_p[cal_depth], saved_gate_p)
                        assign(seq_d[cal_depth], saved_gate_d)

        else:
            # Benchmark mode: 3 conditions over depths
            assign(amp_fixed, cancel_amp)
            assign(init_phase_fixed, init_phase / (2 * np.pi))

            with for_each_(cond_idx, [0, 1, 2]):
                with for_(m, 0, m < num_sequences, m + 1):
                    save(m, m_st)

                    # Generate random Clifford sequence for probe and drive
                    assign(curr_state_p, 0)
                    assign(curr_state_d, 0)
                    with for_(i_gen, 0, i_gen < max_circuit_depth, i_gen + 1):
                        assign(step_p, rand_p.rand_int(24))
                        assign(curr_state_p, cayley[curr_state_p * 24 + step_p])
                        assign(seq_p[i_gen], step_p)
                        assign(inv_p[i_gen], inv_list[curr_state_p])

                        assign(step_d, rand_d.rand_int(24))
                        assign(curr_state_d, cayley[curr_state_d * 24 + step_d])
                        assign(seq_d[i_gen], step_d)
                        assign(inv_d[i_gen], inv_list[curr_state_d])

                    with for_each_(depth_var, [int(d) for d in depth_arr]):
                        assign(saved_gate_p, seq_p[depth_var])
                        assign(seq_p[depth_var], inv_p[depth_var - 1])
                        assign(saved_gate_d, seq_d[depth_var])
                        assign(seq_d[depth_var], inv_d[depth_var - 1])

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

                            assign(curr_phase_turns, init_phase_fixed)
                            with for_(gate_k, 0, gate_k <= depth_var, gate_k + 1):
                                # 1. Probe qubit always plays SQRB sequence
                                play_single_clifford(seq_p[gate_k], probe_qubit, idle_len)

                                # 2. Drive qubit: idle in isolated condition, active in simultaneous & compensated
                                with if_(cond_idx == 0):
                                    drive_qubit.xy.wait(idle_len)
                                with if_(cond_idx > 0):
                                    play_single_clifford(seq_d[gate_k], drive_qubit, idle_len)

                                # 3. Active crosstalk cancellation tone (only in compensated condition)
                                with if_(cond_idx == 2):
                                    play_compensation_tone(seq_d[gate_k], cancel_elem, amp_fixed, curr_phase_turns, idle_len)

                                # 4. Iteratively advance phase for next gate slot with modulo wrapping
                                assign(curr_phase_turns, curr_phase_turns + phase_rate_turns)
                                with if_(curr_phase_turns >= 1.0):
                                    assign(curr_phase_turns, curr_phase_turns - 1.0)
                                with if_(curr_phase_turns <= -1.0):
                                    assign(curr_phase_turns, curr_phase_turns + 1.0)

                                align(probe_qubit.xy.name, drive_qubit.xy.name, cancel_elem)

                            align()
                            if use_state_discrimination:
                                probe_qubit.readout_state(state[0])
                                save(state[0], state_st[0])
                            else:
                                probe_qubit.resonator.measure("readout", qua_vars=(I[0], Q[0]))
                                save(I[0], I_st[0])
                                save(Q[0], Q_st[0])
                            align()

                        assign(seq_p[depth_var], saved_gate_p)
                        assign(seq_d[depth_var], saved_gate_d)

        with stream_processing():
            m_st.save("n")
            if mode == "calibrate":
                if use_state_discrimination:
                    state_st[0].buffer(num_shots).map(FUNCTIONS.average()).buffer(num_sequences).buffer(len(init_phases)).buffer(len(cancel_amps)).save("state1")
                else:
                    I_st[0].buffer(num_shots).map(FUNCTIONS.average()).buffer(num_sequences).buffer(len(init_phases)).buffer(len(cancel_amps)).save("I1")
                    Q_st[0].buffer(num_shots).map(FUNCTIONS.average()).buffer(num_sequences).buffer(len(init_phases)).buffer(len(cancel_amps)).save("Q1")
            else:
                if use_state_discrimination:
                    state_st[0].buffer(num_shots).map(FUNCTIONS.average()).buffer(len(depths)).buffer(num_sequences).buffer(3).save("state1")
                else:
                    I_st[0].buffer(num_shots).map(FUNCTIONS.average()).buffer(len(depths)).buffer(num_sequences).buffer(3).save("I1")
                    Q_st[0].buffer(num_shots).map(FUNCTIONS.average()).buffer(len(depths)).buffer(num_sequences).buffer(3).save("Q1")

    return prog, sweep_axes


def acquire(
    machine,
    prog,
    sweep_axes,
    *,
    num_shots: int,
    timeout: float,
    log: Optional[Callable] = None,
    config: Optional[dict] = None,
) -> xr.Dataset:
    """Acquire CrosstalkCompensatedSQRB dataset."""
    return _acquire(
        machine,
        prog,
        sweep_axes,
        num_shots=num_shots,
        timeout=timeout,
        log=log,
        config=config,
    )


@register
class QMCrosstalkCompensatedSQRB(CrosstalkCompensatedSQRB):
    """Build Crosstalk Compensated SQRB QUA program on QM OPX."""

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

        cancel_elem_name = f"{probe_qubit.xy.name}_cancel"
        if cancel_elem_name not in config["elements"]:
            elem_dict = copy.deepcopy(config["elements"][probe_qubit.xy.name])
            if "core" in elem_dict:
                elem_dict["core"] = "b" if elem_dict.get("core") == "a" else "a"
            config["elements"][cancel_elem_name] = elem_dict
        return config

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

        # Detect Clifford gate duration from calibrated probe x180 length
        try:
            x180_op = probe_qubit.xy.operations["x180"]
            duration_ns = float(getattr(x180_op, "length", self.params.clifford_duration_ns))
        except Exception:
            duration_ns = float(self.params.clifford_duration_ns)

        # Determine theoretical phase rate using full RF frequencies
        if self.params.phase_rate is None:
            f_d = getattr(drive_qubit.xy, "RF_frequency", None) or getattr(drive_qubit, "f_01", None) or getattr(drive_qubit.xy, "intermediate_frequency", 50e6)
            f_p = getattr(probe_qubit.xy, "RF_frequency", None) or getattr(probe_qubit, "f_01", None) or getattr(probe_qubit.xy, "intermediate_frequency", 50e6)
            phase_rate_rad = compute_theoretical_phase_rate(
                float(f_d), float(f_p), duration_ns
            )
        else:
            phase_rate_rad = float(self.params.phase_rate)

        phase_rate_turns = float((phase_rate_rad / (2 * np.pi)) % 1.0)
        if phase_rate_turns > 0.5:
            phase_rate_turns -= 1.0

        # Generate config and patch twin cancellation element
        config = machine.generate_config()
        cancel_elem_name = f"{probe_qubit.xy.name}_cancel"
        if cancel_elem_name not in config["elements"]:
            elem_dict = copy.deepcopy(config["elements"][probe_qubit.xy.name])
            if "core" in elem_dict:
                elem_dict["core"] = "b" if elem_dict.get("core") == "a" else "a"
            config["elements"][cancel_elem_name] = elem_dict

        prog, sweep_axes = build_program(
            machine,
            probe_qubit,
            drive_qubit,
            mode=self.params.mode,
            depths=list(self.params.get_depths()),
            num_sequences=int(self.params.num_random_sequences),
            num_shots=int(self.params.num_averages),
            cancel_amp=float(self.params.cancel_amp),
            init_phase=float(self.params.init_phase),
            phase_rate_turns=phase_rate_turns,
            cancel_amps=list(self.params.get_cancel_amps()) if hasattr(self.params, "get_cancel_amps") else list(self.params.cancel_amps),
            init_phases=list(self.params.get_init_phases()) if hasattr(self.params, "get_init_phases") else list(self.params.init_phases),
            cal_depth=int(self.params.cal_depth),
            reset_type=check_reset_method(self),
            use_state_discrimination=bool(self.params.use_state_discrimination),
            seed=self.params.seed,
        )

        return prog, sweep_axes, lambda m, p, s, **kw: acquire(m, p, s, config=config, **kw)
