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
# Recovery gates lookup: inv_gates[i] is the Clifford gate index that resets state i to 0
inv_gates = [int(np.where(c1_table[i, :] == 0)[0][0]) for i in range(24)]

# 3-subslot decomposition for all 24 Clifford gates: (slot0, slot1, slot2)
# Each entry is None (idle wait) or (op_name, pulse_kind, axis_phase_turns)
CLIFFORD_3SUBSLOTS = [
    # Gate 0: I
    (None, None, None),
    # Gate 1: x180
    (("x180", "x180", 0.0), None, None),
    # Gate 2: y180
    (("y180", "x180", 0.25), None, None),
    # Gate 3: y180, x180
    (("y180", "x180", 0.25), ("x180", "x180", 0.0), None),
    # Gate 4: x90, y90
    (("x90", "x90", 0.0), ("y90", "x90", 0.25), None),
    # Gate 5: x90, -y90
    (("x90", "x90", 0.0), ("-y90", "x90", 0.75), None),
    # Gate 6: -x90, y90
    (("-x90", "x90", 0.5), ("y90", "x90", 0.25), None),
    # Gate 7: -x90, -y90
    (("-x90", "x90", 0.5), ("-y90", "x90", 0.75), None),
    # Gate 8: y90, x90
    (("y90", "x90", 0.25), ("x90", "x90", 0.0), None),
    # Gate 9: y90, -x90
    (("y90", "x90", 0.25), ("-x90", "x90", 0.5), None),
    # Gate 10: -y90, x90
    (("-y90", "x90", 0.75), ("x90", "x90", 0.0), None),
    # Gate 11: -y90, -x90
    (("-y90", "x90", 0.75), ("-x90", "x90", 0.5), None),
    # Gate 12: x90
    (("x90", "x90", 0.0), None, None),
    # Gate 13: -x90
    (("-x90", "x90", 0.5), None, None),
    # Gate 14: y90
    (("y90", "x90", 0.25), None, None),
    # Gate 15: -y90
    (("-y90", "x90", 0.75), None, None),
    # Gate 16: -x90, y90, x90
    (("-x90", "x90", 0.5), ("y90", "x90", 0.25), ("x90", "x90", 0.0)),
    # Gate 17: -x90, -y90, x90
    (("-x90", "x90", 0.5), ("-y90", "x90", 0.75), ("x90", "x90", 0.0)),
    # Gate 18: x180, y90
    (("x180", "x180", 0.0), ("y90", "x90", 0.25), None),
    # Gate 19: x180, -y90
    (("x180", "x180", 0.0), ("-y90", "x90", 0.75), None),
    # Gate 20: y180, x90
    (("y180", "x180", 0.25), ("x90", "x90", 0.0), None),
    # Gate 21: y180, -x90
    (("y180", "x180", 0.25), ("-x90", "x90", 0.5), None),
    # Gate 22: x90, y90, x90
    (("x90", "x90", 0.0), ("y90", "x90", 0.25), ("x90", "x90", 0.0)),
    # Gate 23: -x90, y90, -x90
    (("-x90", "x90", 0.5), ("y90", "x90", 0.25), ("-x90", "x90", 0.5)),
]


def play_fixed_clifford(gate_idx, qubit, pulse_cycles: int):
    """Play single Clifford gate decomposed into 3 fixed-duration subslots."""
    with switch_(gate_idx, unsafe=True):
        for idx, subslots in enumerate(CLIFFORD_3SUBSLOTS):
            with case_(idx):
                for slot in subslots:
                    if slot is None:
                        qubit.xy.wait(pulse_cycles)
                    else:
                        qubit.xy.play(slot[0])


def play_fixed_compensation_clifford(
    drive_gate_idx,
    cancel_elem_name: str,
    amp_val,
    sub_phase_rate_turns: float,
    pulse_cycles: int,
):
    """Play compensation pulses for the 3 subslots mirroring the drive qubit gate."""
    theta = -2.0 * np.pi * float(sub_phase_rate_turns)
    a = float(amp_val)
    v00 = float(a * np.cos(theta))
    v01 = float(-a * np.sin(theta))
    v10 = float(a * np.sin(theta))
    v11 = float(a * np.cos(theta))

    with switch_(drive_gate_idx, unsafe=True):
        for idx, subslots in enumerate(CLIFFORD_3SUBSLOTS):
            with case_(idx):
                slots = [s for s in subslots if s is not None]
                n_pulses = len(slots)
                if n_pulses == 3:
                    # 3-pulse gates: advance frame by 2*sub_p after pulse 1 and use
                    # matrix amp rotation for pulse 2 to keep subslot 2 free of frame rotations.
                    play(slots[0][0] * amp(a), cancel_elem_name)
                    if sub_phase_rate_turns != 0.0:
                        frame_rotation_2pi(sub_phase_rate_turns, cancel_elem_name)
                    play(slots[1][0] * amp(a), cancel_elem_name)
                    if sub_phase_rate_turns != 0.0:
                        frame_rotation_2pi(2.0 * sub_phase_rate_turns, cancel_elem_name)
                    if sub_phase_rate_turns != 0.0:
                        play(slots[2][0] * amp(v00, v01, v10, v11), cancel_elem_name)
                    else:
                        play(slots[2][0] * amp(a), cancel_elem_name)
                else:
                    # 0, 1, or 2 pulse gates: idle subslots rotate frame before wait
                    for slot in subslots:
                        if slot is None:
                            if sub_phase_rate_turns != 0.0:
                                frame_rotation_2pi(sub_phase_rate_turns, cancel_elem_name)
                            wait(pulse_cycles, cancel_elem_name)
                        else:
                            play(slot[0] * amp(a), cancel_elem_name)
                            if sub_phase_rate_turns != 0.0:
                                frame_rotation_2pi(sub_phase_rate_turns, cancel_elem_name)


def play_simultaneous_clifford_subslots(
    gate_p: int,
    gate_d: int,
    *,
    cond: str,
    probe_qubit: Any,
    drive_qubit: Any,
    cancel_elem_name: str,
    amp_val: float,
    sub_phase_rate_turns: float,
    pulse_cycles: int,
):
    """Play single Clifford gate decomposed into 3 fixed-duration subslots synchronously across channels.

    Subslots are played simultaneously for probe_qubit, drive_qubit, and cancel_elem
    to ensure perfect pulse alignment and zero inter-gate timing gaps under strict_timing_().
    """
    subslots_p = CLIFFORD_3SUBSLOTS[gate_p]
    subslots_d = CLIFFORD_3SUBSLOTS[gate_d]

    theta = -2.0 * np.pi * float(sub_phase_rate_turns)
    a = float(amp_val)
    v00 = float(a * np.cos(theta))
    v01 = float(-a * np.sin(theta))
    v10 = float(a * np.sin(theta))
    v11 = float(a * np.cos(theta))

    # Subslot 0
    if subslots_p[0] is None:
        probe_qubit.xy.wait(pulse_cycles)
    else:
        probe_qubit.xy.play(subslots_p[0][0])

    if cond in ("simultaneous", "compensated"):
        if subslots_d[0] is None:
            drive_qubit.xy.wait(pulse_cycles)
        else:
            drive_qubit.xy.play(subslots_d[0][0])

    if cond == "compensated":
        if subslots_d[0] is None:
            if sub_phase_rate_turns != 0.0:
                frame_rotation_2pi(sub_phase_rate_turns, cancel_elem_name)
            wait(pulse_cycles, cancel_elem_name)
        else:
            play(subslots_d[0][0] * amp(a), cancel_elem_name)
            if sub_phase_rate_turns != 0.0:
                frame_rotation_2pi(sub_phase_rate_turns, cancel_elem_name)

    # Subslot 1
    if subslots_p[1] is None:
        probe_qubit.xy.wait(pulse_cycles)
    else:
        probe_qubit.xy.play(subslots_p[1][0])

    if cond in ("simultaneous", "compensated"):
        if subslots_d[1] is None:
            drive_qubit.xy.wait(pulse_cycles)
        else:
            drive_qubit.xy.play(subslots_d[1][0])

    if cond == "compensated":
        slots_d = [s for s in subslots_d if s is not None]
        if len(slots_d) == 3:
            play(slots_d[1][0] * amp(a), cancel_elem_name)
            if sub_phase_rate_turns != 0.0:
                frame_rotation_2pi(2.0 * sub_phase_rate_turns, cancel_elem_name)
        else:
            if subslots_d[1] is None:
                if sub_phase_rate_turns != 0.0:
                    frame_rotation_2pi(sub_phase_rate_turns, cancel_elem_name)
                wait(pulse_cycles, cancel_elem_name)
            else:
                play(subslots_d[1][0] * amp(a), cancel_elem_name)
                if sub_phase_rate_turns != 0.0:
                    frame_rotation_2pi(sub_phase_rate_turns, cancel_elem_name)

    # Subslot 2
    if subslots_p[2] is None:
        probe_qubit.xy.wait(pulse_cycles)
    else:
        probe_qubit.xy.play(subslots_p[2][0])

    if cond in ("simultaneous", "compensated"):
        if subslots_d[2] is None:
            drive_qubit.xy.wait(pulse_cycles)
        else:
            drive_qubit.xy.play(subslots_d[2][0])

    if cond == "compensated":
        slots_d = [s for s in subslots_d if s is not None]
        if len(slots_d) == 3:
            if sub_phase_rate_turns != 0.0:
                play(slots_d[2][0] * amp(v00, v01, v10, v11), cancel_elem_name)
            else:
                play(slots_d[2][0] * amp(a), cancel_elem_name)
        else:
            if subslots_d[2] is None:
                if sub_phase_rate_turns != 0.0:
                    frame_rotation_2pi(sub_phase_rate_turns, cancel_elem_name)
                wait(pulse_cycles, cancel_elem_name)
            else:
                play(subslots_d[2][0] * amp(a), cancel_elem_name)
                if sub_phase_rate_turns != 0.0:
                    frame_rotation_2pi(sub_phase_rate_turns, cancel_elem_name)


CLIFFORD_PULSES: list[list[str]] = [
    [],                                      # 0: I
    ["x180"],                                # 1: X
    ["y180"],                                # 2: Y
    ["y180", "x180"],                        # 3: Y, X
    ["x90", "y90"],                          # 4: X/2, Y/2
    ["x90", "-y90"],                         # 5: X/2, -Y/2
    ["-x90", "y90"],                         # 6: -X/2, Y/2
    ["-x90", "-y90"],                        # 7: -X/2, -Y/2
    ["y90", "x90"],                          # 8: Y/2, X/2
    ["y90", "-x90"],                         # 9: Y/2, -X/2
    ["-y90", "x90"],                         # 10: -Y/2, X/2
    ["-y90", "-x90"],                        # 11: -Y/2, -X/2
    ["x90"],                                 # 12: X/2
    ["-x90"],                                # 13: -X/2
    ["y90"],                                 # 14: Y/2
    ["-y90"],                                # 15: -Y/2
    ["-x90", "y90", "x90"],                  # 16: -X/2, Y/2, X/2
    ["-x90", "-y90", "x90"],                 # 17: -X/2, -Y/2, X/2
    ["x180", "y90"],                         # 18: X, Y/2
    ["x180", "-y90"],                        # 19: X, -Y/2
    ["y180", "x90"],                         # 20: Y, X/2
    ["y180", "-x90"],                        # 21: Y, -X/2
    ["x90", "y90", "x90"],                   # 22: X/2, Y/2, X/2
    ["-x90", "y90", "-x90"],                 # 23: -X/2, Y/2, -X/2
]


def generate_clifford_sequences(
    depth: int,
    num_sequences: int,
    seed: int | None = None,
) -> list[list[int]]:
    """Precompute random Clifford sequences and their recovery gate at compile time."""
    rng = np.random.default_rng(seed)
    sequences = []
    for _ in range(num_sequences):
        gates = rng.integers(0, 24, size=depth).tolist()
        curr = 0
        for g in gates:
            curr = int(c1_table[curr, g])
        inv_gate = int(inv_gates[curr])
        sequences.append(gates + [inv_gate])
    return sequences


def play_compact_clifford(gate_idx, qubit, pulse_cycles: int):
    """Play single Clifford gate compactly matching qubit_sqrb without subslot padding."""
    with switch_(gate_idx, unsafe=True):
        for idx, ops in enumerate(CLIFFORD_PULSES):
            with case_(idx):
                if len(ops) == 0:
                    qubit.xy.wait(pulse_cycles)
                else:
                    for op in ops:
                        qubit.xy.play(op)


def play_compact_compensation_clifford(
    drive_gate_idx,
    cancel_elem_name: str,
    amp_val,
    sub_phase_rate_turns: float,
    pulse_cycles: int,
):
    """Play compensation pulses compactly mirroring the drive qubit gate without subslot padding."""
    with switch_(drive_gate_idx, unsafe=True):
        for idx, ops in enumerate(CLIFFORD_PULSES):
            with case_(idx):
                if len(ops) == 0:
                    wait(pulse_cycles, cancel_elem_name)
                    if sub_phase_rate_turns != 0.0:
                        frame_rotation_2pi(sub_phase_rate_turns, cancel_elem_name)
                else:
                    for op in ops:
                        play(op * amp(amp_val), cancel_elem_name)
                        if sub_phase_rate_turns != 0.0:
                            frame_rotation_2pi(sub_phase_rate_turns, cancel_elem_name)


def build_program(
    machine,
    probe_qubit,
    drive_qubit,
    *,
    mode: str,
    target_gate: str = "x180",
    depths: list[int] | None = None,
    num_sequences: int = 30,
    sequence_indices: list[int] | np.ndarray | None = None,
    num_shots: int = 1000,
    cancel_amp: float = 0.0,
    init_phase: float = 0.0,
    phase_rate_turns: float | None = None,
    sub_phase_rate_turns: float | None = None,
    cancel_amps: list[float] | None = None,
    init_phases: list[float] | None = None,
    cal_stage_repetitions: list[int] | int | None = None,
    cal_repetitions: list[int] | int | None = None,
    cal_depth: int = 20,
    reset_type: str = "thermal",
    use_state_discrimination: bool = False,
    seed: int | None = None,
    simulate: bool = False,
    conditions: list[str] | None = None,
    strict_timing: bool = True,
):
    """Build the QUA program for Crosstalk Compensated Simultaneous SQRB."""
    depth_arr = np.asarray(depths, dtype=int) if depths is not None else np.array([4])
    if sequence_indices is not None:
        seq_arr = np.asarray(sequence_indices, dtype=int)
    else:
        seq_arr = np.arange(num_sequences, dtype=int)
    cancel_elem = f"{probe_qubit.xy.name}_cancel"

    if cal_stage_repetitions is not None:
        cal_reps = [int(cal_stage_repetitions)] if isinstance(cal_stage_repetitions, (int, np.integer)) else [int(r) for r in cal_stage_repetitions]
    elif cal_repetitions is not None:
        cal_reps = [int(cal_repetitions)] if isinstance(cal_repetitions, (int, np.integer)) else [int(r) for r in cal_repetitions]
    else:
        cal_reps = [int(cal_depth)]

    max_circuit_depth = int(depth_arr.max()) if mode == "benchmark" else int(cal_reps[0])

    try:
        x180_len = probe_qubit.xy.operations["x180"].length
        pulse_cycles = max(x180_len // 4, 1)
    except Exception:
        pulse_cycles = 4

    if sub_phase_rate_turns is not None:
        sub_p = float(sub_phase_rate_turns)
    elif phase_rate_turns is not None:
        sub_p = float(phase_rate_turns)
    else:
        sub_p = 0.0

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
            "sequence_idx": xr.DataArray(seq_arr),
            "depth": xr.DataArray(depth_arr),
        }

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        if use_state_discrimination:
            state = [declare(int)]
            state_st = [declare_stream()]

        m = declare(int)
        m_st = declare_stream()

        amp_fixed = declare(fixed)
        init_phase_fixed = declare(fixed)

        if mode == "benchmark":
            total_seqs = max(int(num_sequences), int(seq_arr.max()) + 1)
            seqs_p = {
                int(d): generate_clifford_sequences(int(d), total_seqs, seed=seed if seed is not None else 42)
                for d in depth_arr
            }
            seqs_d = {
                int(d): generate_clifford_sequences(
                    int(d), total_seqs, seed=(seed + 1000) if seed is not None else 1042
                )
                for d in depth_arr
            }

        machine.initialize_qpu(target=probe_qubit)
        machine.initialize_qpu(target=drive_qubit)
        align()

        if mode == "calibrate":
            # Calibrate mode: sweep cancel_amp, init_phase, and average over cal_reps with probe idle in |0>
            with for_each_(amp_fixed, [float(a) for a in cancel_amps]):
                with for_each_(init_phase_fixed, [float(p) / (2 * np.pi) for p in init_phases]):
                    for r_val in cal_reps:
                        r_int = int(r_val)
                        with for_(n, 0, n < num_shots, n + 1):
                            probe_qubit.reset(reset_type, simulate)
                            drive_qubit.reset(reset_type, simulate)
                            if strict_timing:
                                with strict_timing_():
                                    align(probe_qubit.xy.name, drive_qubit.xy.name, cancel_elem)
                                    reset_frame(probe_qubit.xy.name)
                                    reset_frame(drive_qubit.xy.name)
                                    reset_frame(cancel_elem)
                                    reset_if_phase(probe_qubit.xy.name)
                                    reset_if_phase(drive_qubit.xy.name)
                                    reset_if_phase(cancel_elem)
                                    reset_global_phase()
                                    frame_rotation_2pi(init_phase_fixed, cancel_elem)

                                    for _ in range(r_int):
                                        drive_qubit.xy.play(target_gate)
                                        play(target_gate * amp(amp_fixed), cancel_elem)
                                        if sub_p != 0.0:
                                            frame_rotation_2pi(-sub_p, drive_qubit.xy.name)
                            else:
                                align(probe_qubit.xy.name, drive_qubit.xy.name, cancel_elem)
                                reset_frame(probe_qubit.xy.name)
                                reset_frame(drive_qubit.xy.name)
                                reset_frame(cancel_elem)
                                reset_if_phase(probe_qubit.xy.name)
                                reset_if_phase(drive_qubit.xy.name)
                                reset_if_phase(cancel_elem)
                                reset_global_phase()
                                frame_rotation_2pi(init_phase_fixed, cancel_elem)

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

        else:
            # Benchmark mode: sweep depths across selected conditions with precomputed sequences
            assign(init_phase_fixed, float(init_phase / (2 * np.pi)))

            for cond in benchmark_conditions:
                for m_idx in seq_arr:
                    m_int = int(m_idx)
                    assign(m, m_int)
                    save(m, m_st)

                    for d_val in depth_arr:
                        d_int = int(d_val)
                        gate_list_p = seqs_p[d_int][m_int]
                        gate_list_d = seqs_d[d_int][m_int]

                        with for_(n, 0, n < num_shots, n + 1):
                            probe_qubit.reset(reset_type, simulate)
                            drive_qubit.reset(reset_type, simulate)
                            if strict_timing:
                                with strict_timing_():
                                    align(probe_qubit.xy.name, drive_qubit.xy.name, cancel_elem)
                                    reset_frame(probe_qubit.xy.name)
                                    reset_frame(drive_qubit.xy.name)
                                    reset_frame(cancel_elem)
                                    reset_if_phase(probe_qubit.xy.name)
                                    reset_if_phase(drive_qubit.xy.name)
                                    reset_if_phase(cancel_elem)
                                    reset_global_phase()
                                    frame_rotation_2pi(init_phase_fixed, cancel_elem)

                                    for k in range(d_int + 1):
                                        play_simultaneous_clifford_subslots(
                                            gate_list_p[k],
                                            gate_list_d[k],
                                            cond=cond,
                                            probe_qubit=probe_qubit,
                                            drive_qubit=drive_qubit,
                                            cancel_elem_name=cancel_elem,
                                            amp_val=float(cancel_amp),
                                            sub_phase_rate_turns=sub_p,
                                            pulse_cycles=pulse_cycles,
                                        )
                            else:
                                align(probe_qubit.xy.name, drive_qubit.xy.name, cancel_elem)
                                reset_frame(probe_qubit.xy.name)
                                reset_frame(drive_qubit.xy.name)
                                reset_frame(cancel_elem)
                                reset_if_phase(probe_qubit.xy.name)
                                reset_if_phase(drive_qubit.xy.name)
                                reset_if_phase(cancel_elem)
                                reset_global_phase()
                                frame_rotation_2pi(init_phase_fixed, cancel_elem)

                                for k in range(d_int + 1):
                                    play_simultaneous_clifford_subslots(
                                        gate_list_p[k],
                                        gate_list_d[k],
                                        cond=cond,
                                        probe_qubit=probe_qubit,
                                        drive_qubit=drive_qubit,
                                        cancel_elem_name=cancel_elem,
                                        amp_val=float(cancel_amp),
                                        sub_phase_rate_turns=sub_p,
                                        pulse_cycles=pulse_cycles,
                                    )

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
                m_st.save("n")
                num_conds = len(cond_indices)
                num_seqs = len(seq_arr)
                num_depths = len(depth_arr)
                if use_state_discrimination:
                    state_st[0].buffer(num_shots).map(FUNCTIONS.average()).buffer(num_depths).buffer(num_seqs).buffer(num_conds).save("state1")
                else:
                    I_st[0].buffer(num_shots).map(FUNCTIONS.average()).buffer(num_depths).buffer(num_seqs).buffer(num_conds).save("I1")
                    Q_st[0].buffer(num_shots).map(FUNCTIONS.average()).buffer(num_depths).buffer(num_seqs).buffer(num_conds).save("Q1")

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


def _patch_cancel_element(config: dict, probe_qubit: Any, drive_qubit: Any, target_gate: str = "x180") -> dict:
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
        drive_qubit = q_dict.get(drive_name, qubits[1])

        target_gate = getattr(self.params, "target_gate", "x180")
        return _patch_cancel_element(config, probe_qubit, drive_qubit, target_gate)

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

        # Detect atomic pulse duration from calibrated probe x180 length
        try:
            x180_op = probe_qubit.xy.operations["x180"]
            pulse_duration_ns = float(getattr(x180_op, "length", self.params.clifford_duration_ns))
        except Exception:
            pulse_duration_ns = float(self.params.clifford_duration_ns)

        # Determine theoretical subslot phase rate using full RF frequencies and pulse duration
        if self.params.phase_rate is None:
            f_d = getattr(drive_qubit.xy, "RF_frequency", None) or getattr(drive_qubit, "f_01", None) or getattr(drive_qubit.xy, "intermediate_frequency", 50e6)
            f_p = getattr(probe_qubit.xy, "RF_frequency", None) or getattr(probe_qubit, "f_01", None) or getattr(probe_qubit.xy, "intermediate_frequency", 50e6)
            sub_phase_rate_rad = compute_theoretical_phase_rate(
                float(f_d), float(f_p), pulse_duration_ns
            )
        else:
            sub_phase_rate_rad = float(self.params.phase_rate)

        sub_phase_rate_turns = float((sub_phase_rate_rad / (2 * np.pi)) % 1.0)
        if sub_phase_rate_turns > 0.5:
            sub_phase_rate_turns -= 1.0

        target_gate = getattr(self.params, "target_gate", "x180")
        config = machine.generate_config()
        config = _patch_cancel_element(config, probe_qubit, drive_qubit, target_gate)

        stage_reps = list(self.params.get_cal_stage_repetitions()) if self.params.mode == "calibrate" else [20]
        initial_cal_reps = [stage_reps[0]] if self.params.mode == "calibrate" else [int(getattr(self.params, "cal_depth", 20))]

        all_depths = list(self.params.get_depths())
        batch_size = max(1, getattr(self.params, "benchmark_batch_size", 1))
        initial_depths = all_depths[:batch_size] if self.params.mode == "benchmark" else all_depths

        prog, sweep_axes = build_program(
            machine,
            probe_qubit,
            drive_qubit,
            mode=self.params.mode,
            target_gate=target_gate,
            depths=initial_depths,
            num_sequences=int(self.params.num_random_sequences),
            num_shots=int(self.params.num_averages),
            cancel_amp=float(self.params.cancel_amp),
            init_phase=float(self.params.init_phase),
            phase_rate_turns=sub_phase_rate_turns,
            sub_phase_rate_turns=sub_phase_rate_turns,
            cancel_amps=list(self.params.get_cancel_amps()),
            init_phases=list(self.params.get_init_phases()),
            cal_stage_repetitions=initial_cal_reps,
            reset_type=check_reset_method(self),
            use_state_discrimination=bool(self.params.use_state_discrimination),
            seed=self.params.seed,
            simulate=getattr(self.backend, "simulate", False),
            strict_timing=bool(getattr(self.params, "strict_timing", True)),
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

        try:
            x180_op = probe_qubit.xy.operations["x180"]
            pulse_duration_ns = float(getattr(x180_op, "length", self.params.clifford_duration_ns))
        except Exception:
            pulse_duration_ns = float(self.params.clifford_duration_ns)

        if self.params.phase_rate is None:
            f_d = getattr(drive_qubit.xy, "RF_frequency", None) or getattr(drive_qubit, "f_01", None) or getattr(drive_qubit.xy, "intermediate_frequency", 50e6)
            f_p = getattr(probe_qubit.xy, "RF_frequency", None) or getattr(probe_qubit, "f_01", None) or getattr(probe_qubit.xy, "intermediate_frequency", 50e6)
            sub_phase_rate_rad = compute_theoretical_phase_rate(
                float(f_d), float(f_p), pulse_duration_ns
            )
        else:
            sub_phase_rate_rad = float(self.params.phase_rate)

        sub_phase_rate_turns = float((sub_phase_rate_rad / (2 * np.pi)) % 1.0)
        if sub_phase_rate_turns > 0.5:
            sub_phase_rate_turns -= 1.0

        target_gate = getattr(self.params, "target_gate", "x180")
        preview_conds = ["compensated"] if self.params.mode == "benchmark" else None
        preview_depths = [4] if self.params.mode == "benchmark" else [20]
        preview_num_seq = 1 if self.params.mode == "benchmark" else int(self.params.num_random_sequences)
        preview_cal_reps = [20] if self.params.mode == "calibrate" else None

        prog, _ = build_program(
            machine,
            probe_qubit,
            drive_qubit,
            mode=self.params.mode,
            target_gate=target_gate,
            conditions=preview_conds,
            depths=preview_depths,
            num_sequences=preview_num_seq,
            cal_stage_repetitions=preview_cal_reps,
            num_shots=1,
            cancel_amp=float(self.params.cancel_amp),
            init_phase=float(self.params.init_phase),
            sub_phase_rate_turns=sub_phase_rate_turns,
            cancel_amps=[float(self.params.cancel_amp)],
            init_phases=[float(self.params.init_phase)],
            reset_type=check_reset_method(self),
            use_state_discrimination=bool(self.params.use_state_discrimination),
            simulate=getattr(self.backend, "simulate", False),
            strict_timing=bool(getattr(self.params, "strict_timing", True)),
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
        """Execute fine-grained batched benchmark mode on Quantum Machine.

        Chunks the workload across condition, depth, and sequence dimensions to ensure
        fast compilation and stay well within OPX memory and timeout limits.
        """
        cond_map = ["isolated", "simultaneous", "compensated"]
        param_conds = getattr(self.params, "conditions", None)
        benchmark_conditions = [c for c in param_conds if c in cond_map] if param_conds else cond_map

        all_depths = list(self.params.get_depths())
        batch_size = max(1, getattr(self.params, "benchmark_batch_size", 1))
        depth_batches = [all_depths[i : i + batch_size] for i in range(0, len(all_depths), batch_size)]

        total_sequences = int(self.params.num_random_sequences)
        max_gates = int(getattr(self.params, "max_gates_per_batch", 120))
        fixed_seq_batch_size = getattr(self.params, "benchmark_sequence_batch_size", None)
        target_gate = getattr(self.params, "target_gate", "x180")

        total_steps = len(benchmark_conditions) * len(depth_batches)
        current_step = 0
        cond_datasets: list[xr.Dataset] = []

        for c_idx, cond in enumerate(benchmark_conditions, 1):
            depth_datasets: list[xr.Dataset] = []
            for d_idx, d_batch in enumerate(depth_batches, 1):
                current_step += 1
                max_d = max(int(d) for d in d_batch)
                gates_per_seq = max_d + 1

                if fixed_seq_batch_size is not None and int(fixed_seq_batch_size) > 0:
                    seq_chunk_size = int(fixed_seq_batch_size)
                else:
                    seq_chunk_size = max(1, min(total_sequences, max_gates // gates_per_seq))

                num_chunks = int(np.ceil(total_sequences / seq_chunk_size))
                seq_chunks = [
                    list(chunk)
                    for chunk in np.array_split(np.arange(total_sequences, dtype=int), num_chunks)
                ]

                seq_datasets: list[xr.Dataset] = []
                for sc_idx, s_chunk in enumerate(seq_chunks, 1):
                    cur_prog, cur_axes = build_program(
                        machine,
                        probe_qubit,
                        drive_qubit,
                        mode="benchmark",
                        target_gate=target_gate,
                        conditions=[cond],
                        depths=d_batch,
                        num_sequences=total_sequences,
                        sequence_indices=s_chunk,
                        num_shots=num_shots,
                        cancel_amp=float(self.params.cancel_amp),
                        init_phase=float(self.params.init_phase),
                        sub_phase_rate_turns=sub_phase_rate_turns,
                        reset_type=reset_type,
                        use_state_discrimination=bool(self.params.use_state_discrimination),
                        seed=self.params.seed,
                        simulate=getattr(self.backend, "simulate", False),
                        strict_timing=bool(getattr(self.params, "strict_timing", True)),
                    )

                    ds_chunk = _acquire(
                        machine,
                        cur_prog,
                        cur_axes,
                        num_shots=num_shots,
                        timeout=timeout,
                        log=log,
                        config=config,
                    )
                    seq_datasets.append(ds_chunk)

                ds_depth = xr.concat(seq_datasets, dim="sequence_idx")
                depth_datasets.append(ds_depth)
                print(
                    f"# [Benchmark {current_step}/{total_steps}] "
                    f"Cond: '{cond}' ({c_idx}/{len(benchmark_conditions)}), "
                    f"Depth: {d_batch} ({d_idx}/{len(depth_batches)}), "
                    f"Sequences: {total_sequences} in {len(seq_chunks)} chunk(s) completed."
                )

            ds_cond = xr.concat(depth_datasets, dim="depth")
            cond_datasets.append(ds_cond)

        total_ds = xr.concat(cond_datasets, dim="condition")
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
        target_gate = getattr(self.params, "target_gate", "x180")

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
                    target_gate=target_gate,
                    cal_stage_repetitions=[r],
                    num_shots=num_shots,
                    cancel_amp=float(self.params.cancel_amp),
                    init_phase=float(self.params.init_phase),
                    sub_phase_rate_turns=sub_phase_rate_turns,
                    cancel_amps=amps_stage,
                    init_phases=phases_stage,
                    reset_type=reset_type,
                    use_state_discrimination=bool(self.params.use_state_discrimination),
                    simulate=getattr(self.backend, "simulate", False),
                    strict_timing=bool(getattr(self.params, "strict_timing", True)),
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
            min_idx = np.unravel_index(np.argmin(p_vals), p_vals.shape)
            best_amp = float(ds_stage.coords["cancel_amp"].values[min_idx[0]])
            best_phase = float(ds_stage.coords["init_phase"].values[min_idx[1]])
            min_p = float(p_vals[min_idx])

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

