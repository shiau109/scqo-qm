"""Qubit Tomography acquisition probe: vendor code only (qm/quam).

QM qubit tomography for scqo - supplies only ``probe()``.

Parameters, fit, and reporting are inherited from ``scqo.experiments.QubitTomography``.
"""

from __future__ import annotations

from typing import Callable, Optional, Dict, List, Any
import numpy as np
import xarray as xr
from qm.qua import *

from scqo_qm.experiments._lib import acquire as _acquire


def play_init_state(qubit, state_str: str):
    """Play state preparation pulses."""
    st = str(state_str).strip().lower()
    if st in ("0", "g"):
        pass
    elif st in ("1", "e"):
        play("x180", qubit.xy.name)
    elif st in ("+", "+x"):
        play("y90", qubit.xy.name)
    elif st in ("-", "-x"):
        play("-y90", qubit.xy.name)
    elif st in ("+i", "+y"):
        play("-x90", qubit.xy.name)
    elif st in ("-i", "-y"):
        play("x90", qubit.xy.name)
    else:
        raise ValueError(
            f"Unsupported initial state '{state_str}'. "
            "Supported states are: '0', 'g', '1', 'e', '+', '+x', '-', '-x', '+i', '+y', '-i', '-y'."
        )


def get_op_cycles(qubit, op_name: str = "x180", default_cycles: int = 10) -> int:
    """Get QUAM operation duration in QUA clock cycles (1 cycle = 4 ns)."""
    gt = str(op_name).strip().lower()
    if gt in ("i", "id", "x", "x180"):
        gt = "x180"
    elif gt in ("x90", "x/2"):
        gt = "x90"
    elif gt in ("y", "y180"):
        gt = "y180"
    elif gt in ("y90", "y/2"):
        gt = "y90"
    elif gt in ("-x90", "-x/2"):
        gt = "-x90"
    elif gt in ("-y90", "-y/2"):
        gt = "-y90"
    else:
        raise ValueError(
            f"Unsupported operation '{op_name}'. "
            "Supported operations are: 'i', 'id', 'x', 'x180', 'x90', 'x/2', 'y', 'y180', 'y90', 'y/2', '-x90', '-y90'."
        )

    try:
        if hasattr(qubit, "xy") and hasattr(qubit.xy, "operations") and gt in qubit.xy.operations:
            length_ns = qubit.xy.operations[gt].length
            return max(1, int(length_ns) // 4)
    except Exception:
        pass
    return default_cycles


def play_target_gate(qubit, gate_str: str, amp_scale: float = 1.0):
    """Play target gate pulse once with optional amplitude scaling."""
    gt = str(gate_str).strip().lower()
    if gt in ("i", "id"):
        cycles = get_op_cycles(qubit, "x180")
        wait(cycles, qubit.xy.name)
        return

    if gt in ("x", "x180"):
        op = "x180"
    elif gt in ("x90", "x/2"):
        op = "x90"
    elif gt in ("y", "y180"):
        op = "y180"
    elif gt in ("y90", "y/2"):
        op = "y90"
    elif gt in ("-x90", "-x/2"):
        op = "-x90"
    elif gt in ("-y90", "-y/2"):
        op = "-y90"
    else:
        raise ValueError(
            f"Unsupported target gate '{gate_str}'. "
            "Supported gates are: 'i', 'id', 'x', 'x180', 'x90', 'x/2', 'y', 'y180', 'y90', 'y/2', '-x90', '-y90'."
        )

    if amp_scale is not None and abs(float(amp_scale) - 1.0) > 1e-6:
        play(op * amp(float(amp_scale)), qubit.xy.name)
    else:
        play(op, qubit.xy.name)


def get_qubit_freq_tuning(qubit, q_cfg: dict) -> tuple[int | None, int | None]:
    """Returns (target_if, base_if) if frequency tuning is requested, else (None, None)."""
    detuning = q_cfg.get("detuning", q_cfg.get("frequency_detuning", q_cfg.get("frequency_shift", None)))
    direct_freq = q_cfg.get("intermediate_frequency", q_cfg.get("frequency", None))
    base_if = getattr(getattr(qubit, "xy", None), "intermediate_frequency", None)
    if base_if is not None:
        base_if = int(base_if)

    if detuning is not None and float(detuning) != 0.0:
        if base_if is not None:
            return int(base_if + float(detuning)), base_if
    elif direct_freq is not None:
        return int(float(direct_freq)), base_if
    return None, base_if


def play_basis_rotation(qubit, basis_str: str):
    """Play basis measurement rotation pulse."""
    b = str(basis_str).strip().lower()
    if b == "z":
        pass
    elif b == "x":
        play("-y90", qubit.xy.name)
    elif b == "y":
        play("x90", qubit.xy.name)
    else:
        raise ValueError(
            f"Unsupported measurement basis '{basis_str}'. Supported bases are: 'z', 'x', 'y'."
        )


def build_program(
    machine,
    qubits,
    *,
    qubit_configs: Dict[str, Dict[str, str]],
    gate_counts: List[int],
    num_shots: int,
    num_training_shots: int = 2000,
    interleave_noise: bool = True,
    symmetrized_readout: bool = True,
    reset_type: str = "thermal",
    simulate: bool = False,
    include_training: bool = True,
    log: Optional[Callable] = None,
):
    """Build the Qubit Tomography QUA program.

    A qubit whose config sets ``noise_mode: True`` is a spectator noise
    source: it plays its target gates in phase 2 (driving it injects
    crosstalk while a neighbour runs tomography) but skips init, basis
    rotation and measurement; its streams carry dummy zeros so the dataset
    shape stays uniform and the estimator marks it success=0.

    When ``interleave_noise: True`` and noise qubits are present, the sequence
    interleaves ``noise_condition: ["off", "on"]`` per shot. Under ``off``,
    the spectator executes an idle wait matching its gate length, guaranteeing
    drift-free common-mode baseline comparisons.
    """
    num_qubits = len(qubits)
    qubit_names = qubits.get_names()
    bases = ["z", "x", "y"]
    sym_names = ["reg", "inv"] if symmetrized_readout else ["reg"]
    sym_indices = list(range(len(sym_names)))

    has_noise = any(
        bool(cfg.get("noise_mode", False))
        for cfg in qubit_configs.values()
    )
    if has_noise and interleave_noise:
        noise_conditions = ["off", "on"]
    else:
        noise_conditions = ["off"]
    nc_indices = list(range(len(noise_conditions)))

    sweep_axes = {
        "target": xr.DataArray(qubit_names),
        "noise_condition": xr.DataArray(noise_conditions),
        "basis": xr.DataArray(bases),
        "sym": xr.DataArray(sym_names),
        "gate_count": xr.DataArray(gate_counts),
        "shot_idx": xr.DataArray(np.arange(num_shots)),
        "prepared_state": xr.DataArray([0, 1]),
        "train_shot_idx": xr.DataArray(np.arange(num_training_shots)),
    }

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        if include_training and num_training_shots > 0:
            I_tr, I_tr_st, Q_tr, Q_tr_st, n_tr, n_tr_st = machine.declare_qua_variables()

        nc_idx = declare(int)
        b_idx = declare(int)
        s_idx = declare(int)
        gc_idx = declare(int)
        rep = declare(int)
        if include_training and num_training_shots > 0:
            ps_idx = declare(int)

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                machine.initialize_qpu(target=qubit)

            all_elem_names = [q.xy.name for q in multiplexed_qubits.values()] + [
                q.resonator.name for q in multiplexed_qubits.values()
            ]

            # 1. Training Shots
            if include_training and num_training_shots > 0:
                with for_(n_tr, 0, n_tr < num_training_shots, n_tr + 1):
                    save(n_tr, n_tr_st)
                    with for_each_(ps_idx, [0, 1]):
                        for i_q, qubit in multiplexed_qubits.items():
                            qubit.reset(reset_type, simulate, log_callable=log)
                        align(*all_elem_names)
                        for i_q, qubit in multiplexed_qubits.items():
                            reset_frame(qubit.xy.name)
                            reset_if_phase(qubit.xy.name)
                        try:
                            reset_global_phase()
                        except Exception:
                            pass
                        align(*all_elem_names)

                        for i_q, qubit in multiplexed_qubits.items():
                            q_name = qubit_names[i_q]
                            q_cfg = qubit_configs.get(q_name, {})
                            noise_mode = bool(q_cfg.get("noise_mode", False))
                            if not noise_mode:
                                with if_(ps_idx == 1):
                                    play("x180", qubit.xy.name)
                        align(*all_elem_names)

                        for i_q, qubit in multiplexed_qubits.items():
                            q_name = qubit_names[i_q]
                            q_cfg = qubit_configs.get(q_name, {})
                            noise_mode = bool(q_cfg.get("noise_mode", False))

                            if not noise_mode:
                                qubit.resonator.measure("readout", qua_vars=(I_tr[i_q], Q_tr[i_q]))
                                save(I_tr[i_q], I_tr_st[i_q])
                                save(Q_tr[i_q], Q_tr_st[i_q])
                            else:
                                assign(I_tr[i_q], 0.0)
                                assign(Q_tr[i_q], 0.0)
                                save(I_tr[i_q], I_tr_st[i_q])
                                save(Q_tr[i_q], Q_tr_st[i_q])
                        align(*all_elem_names)

            # 2. Tomography Shots (Interleaved Noise and Baseline)
            with for_(n, 0, n < num_shots, n + 1):
                save(n, n_st)
                with for_each_(nc_idx, nc_indices):
                    with for_each_(b_idx, [0, 1, 2]):
                        with for_each_(s_idx, sym_indices):
                            with for_each_(gc_idx, gate_counts):
                                # Reset
                                for i_q, qubit in multiplexed_qubits.items():
                                    qubit.reset(reset_type, simulate, log_callable=log)
                                align(*all_elem_names)
                                for i_q, qubit in multiplexed_qubits.items():
                                    reset_frame(qubit.xy.name)
                                    reset_if_phase(qubit.xy.name)
                                try:
                                    reset_global_phase()
                                except Exception:
                                    pass
                                align(*all_elem_names)

                                # Phase 1: Init State (all qubits simultaneously, noise_mode qubits skipped)
                                for i_q, qubit in multiplexed_qubits.items():
                                    q_name = qubit_names[i_q]
                                    q_cfg = qubit_configs.get(q_name, {})
                                    noise_mode = bool(q_cfg.get("noise_mode", False))
                                    if not noise_mode:
                                        init_st = q_cfg.get("init_state", "0")
                                        play_init_state(qubit, init_st)
                                align(*all_elem_names)

                                # Phase 2: Target Gate (unrolled via switch_ and aligned across all qubits)
                                # Apply frequency tuning before target gates if specified
                                for i_q, qubit in multiplexed_qubits.items():
                                    q_name = qubit_names[i_q]
                                    q_cfg = qubit_configs.get(q_name, {})
                                    tgt_if, _ = get_qubit_freq_tuning(qubit, q_cfg)
                                    if tgt_if is not None:
                                        update_frequency(qubit.xy.name, tgt_if)

                                with switch_(gc_idx):
                                    for gc in gate_counts:
                                        gc_int = int(gc)
                                        with case_(gc_int):
                                            for i_q, qubit in multiplexed_qubits.items():
                                                q_name = qubit_names[i_q]
                                                q_cfg = qubit_configs.get(q_name, {})
                                                tgt_gt = q_cfg.get("target_gate", "X180")
                                                amp_sc = float(q_cfg.get("amp", q_cfg.get("amp_scale", 1.0)))
                                                noise_mode = bool(q_cfg.get("noise_mode", False))

                                                if not noise_mode:
                                                    for _ in range(gc_int):
                                                        play_target_gate(qubit, tgt_gt, amp_scale=amp_sc)
                                                else:
                                                    if len(noise_conditions) > 1 and gc_int > 0:
                                                        with if_(nc_idx == 1):
                                                            for _ in range(gc_int):
                                                                play_target_gate(qubit, tgt_gt, amp_scale=amp_sc)
                                                        with else_():
                                                            cycles = get_op_cycles(qubit, tgt_gt) * gc_int
                                                            if cycles > 0:
                                                                wait(cycles, qubit.xy.name)
                                                    elif gc_int > 0:
                                                        for _ in range(gc_int):
                                                            play_target_gate(qubit, tgt_gt, amp_scale=amp_sc)

                                # Restore base frequency if frequency tuning was applied
                                for i_q, qubit in multiplexed_qubits.items():
                                    q_name = qubit_names[i_q]
                                    q_cfg = qubit_configs.get(q_name, {})
                                    tgt_if, base_if = get_qubit_freq_tuning(qubit, q_cfg)
                                    if tgt_if is not None and base_if is not None:
                                        update_frequency(qubit.xy.name, base_if)

                                # Phase 3: Basis Rotation (all active qubits played concurrently per basis)
                                with switch_(b_idx):
                                    with case_(0):
                                        pass
                                    with case_(1):
                                        for i_q, qubit in multiplexed_qubits.items():
                                            if not bool(qubit_configs.get(qubit_names[i_q], {}).get("noise_mode", False)):
                                                play_basis_rotation(qubit, "x")
                                    with case_(2):
                                        for i_q, qubit in multiplexed_qubits.items():
                                            if not bool(qubit_configs.get(qubit_names[i_q], {}).get("noise_mode", False)):
                                                play_basis_rotation(qubit, "y")

                                if symmetrized_readout:
                                    with if_(s_idx == 1):
                                        for i_q, qubit in multiplexed_qubits.items():
                                            if not bool(qubit_configs.get(qubit_names[i_q], {}).get("noise_mode", False)):
                                                play("x180", qubit.xy.name)

                                align(*all_elem_names)

                                # Measurement (noise_mode qubits save dummy zeros)
                                for i_q, qubit in multiplexed_qubits.items():
                                    q_name = qubit_names[i_q]
                                    q_cfg = qubit_configs.get(q_name, {})
                                    noise_mode = bool(q_cfg.get("noise_mode", False))

                                    if not noise_mode:
                                        qubit.resonator.measure("readout", qua_vars=(I[i_q], Q[i_q]))
                                        save(I[i_q], I_st[i_q])
                                        save(Q[i_q], Q_st[i_q])
                                    else:
                                        assign(I[i_q], 0.0)
                                        assign(Q[i_q], 0.0)
                                        save(I[i_q], I_st[i_q])
                                        save(Q[i_q], Q_st[i_q])
                                align(*all_elem_names)

        with stream_processing():
            n_st.save("n")
            if include_training and num_training_shots > 0:
                n_tr_st.save("n_tr")
                for i_q in range(num_qubits):
                    I_tr_st[i_q].buffer(2).buffer(num_training_shots).save(f"I_train{i_q + 1}")
                    Q_tr_st[i_q].buffer(2).buffer(num_training_shots).save(f"Q_train{i_q + 1}")

            for i_q in range(num_qubits):
                I_st[i_q].buffer(len(gate_counts)).buffer(len(sym_indices)).buffer(3).buffer(len(noise_conditions)).buffer(num_shots).save(f"I_tomo{i_q + 1}")
                Q_st[i_q].buffer(len(gate_counts)).buffer(len(sym_indices)).buffer(3).buffer(len(noise_conditions)).buffer(num_shots).save(f"Q_tomo{i_q + 1}")

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
    from qualang_tools.multi_user import qm_session
    from qualang_tools.results import fetching_tool, progress_counter

    qmm = machine.connect()
    config = config if config is not None else machine.generate_config()

    qubit_names = list(sweep_axes["target"].values if "target" in sweep_axes else sweep_axes["qubit"].values)
    num_qubits = len(qubit_names)
    noise_conditions = list(sweep_axes["noise_condition"].values)
    bases = list(sweep_axes["basis"].values)
    sym_indices = list(sweep_axes["sym"].values)
    gate_counts = list(sweep_axes["gate_count"].values)
    shot_idx = list(sweep_axes["shot_idx"].values)
    prepared_states = list(sweep_axes["prepared_state"].values)
    train_shot_idx = list(sweep_axes["train_shot_idx"].values)

    with qm_session(qmm, config, timeout=timeout) as qm:
        job = qm.execute(prog)

        # Display progress bar for tomography shots
        fetcher = fetching_tool(job, ["n"], mode="live")
        while fetcher.is_processing():
            n_val = fetcher.fetch_all()[0]
            progress_counter(n_val, num_shots, start_time=fetcher.start_time)

        results = job.result_handles
        results.wait_for_all_values()

        I_train_list = []
        Q_train_list = []
        I_tomo_list = []
        Q_tomo_list = []

        for i_q in range(num_qubits):
            h_i_tr = results.get(f"I_train{i_q + 1}")
            h_q_tr = results.get(f"Q_train{i_q + 1}")
            h_i_to = results.get(f"I_tomo{i_q + 1}")
            h_q_to = results.get(f"Q_tomo{i_q + 1}")

            missing = []
            if h_i_tr is None: missing.append(f"I_train{i_q + 1}")
            if h_q_tr is None: missing.append(f"Q_train{i_q + 1}")
            if h_i_to is None: missing.append(f"I_tomo{i_q + 1}")
            if h_q_to is None: missing.append(f"Q_tomo{i_q + 1}")

            if missing:
                try:
                    avail = list(results.iter_all())
                except Exception:
                    avail = "unknown"
                raise RuntimeError(f"Tomography result handles missing {missing}. Available handles: {avail}")

            i_tr = h_i_tr.fetch_all()
            q_tr = h_q_tr.fetch_all()
            i_to = h_i_to.fetch_all()
            q_to = h_q_to.fetch_all()

            I_train_list.append(np.transpose(i_tr, (1, 0)))
            Q_train_list.append(np.transpose(q_tr, (1, 0)))

            I_tomo_list.append(np.transpose(i_to, (1, 2, 3, 4, 0)))
            Q_tomo_list.append(np.transpose(q_to, (1, 2, 3, 4, 0)))

        I_train_arr = np.stack(I_train_list, axis=0)
        Q_train_arr = np.stack(Q_train_list, axis=0)
        I_tomo_arr = np.stack(I_tomo_list, axis=0)
        Q_tomo_arr = np.stack(Q_tomo_list, axis=0)

        if log:
            try:
                rep = getattr(job, "execution_report", None)
                if callable(rep):
                    log(rep())
                elif rep is not None:
                    log(rep)
            except Exception:
                pass

    ds = xr.Dataset(
        data_vars={
            "I_tomo": (("target", "noise_condition", "basis", "sym", "gate_count", "shot_idx"), I_tomo_arr),
            "Q_tomo": (("target", "noise_condition", "basis", "sym", "gate_count", "shot_idx"), Q_tomo_arr),
            "I_train": (("target", "prepared_state", "train_shot_idx"), I_train_arr),
            "Q_train": (("target", "prepared_state", "train_shot_idx"), Q_train_arr),
        },
        coords={
            "target": qubit_names,
            "noise_condition": noise_conditions,
            "basis": bases,
            "sym": sym_indices,
            "gate_count": gate_counts,
            "shot_idx": shot_idx,
            "prepared_state": prepared_states,
            "train_shot_idx": train_shot_idx,
        },
    )
    return ds


from typing import Any

from scqo import register
from scqo.experiments import QubitTomography


@register
class QMQubitTomography(QubitTomography):
    """Build a multiplexed qubit tomography QUA program on the QM OPX."""

    def preview_program(self) -> Any:
        """The ``--preview`` build: omits training shots and skips thermal wait."""
        from ._reset import check_reset_method
        from scqo_qm.experiments._lib import select_qubits

        machine = self.backend.machine
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)

        prog, _sweep_axes = build_program(
            machine,
            qubits,
            qubit_configs=self.params.qubit_configs,
            gate_counts=list(self.params.gate_counts),
            num_shots=int(self.params.num_averages),
            num_training_shots=0,
            interleave_noise=getattr(self.params, "interleave_noise", True),
            symmetrized_readout=self.params.symmetrized_readout,
            reset_type=check_reset_method(self),
            simulate=True,
            include_training=False,
        )
        return prog

    def probe(self) -> Any:
        from ._reset import check_reset_method
        from scqo_qm.experiments._lib import select_qubits

        machine = self.backend.machine
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)

        prog, sweep_axes = build_program(
            machine,
            qubits,
            qubit_configs=self.params.qubit_configs,
            gate_counts=list(self.params.gate_counts),
            num_shots=int(self.params.num_averages),
            num_training_shots=int(self.params.num_training_shots),
            interleave_noise=getattr(self.params, "interleave_noise", True),
            symmetrized_readout=self.params.symmetrized_readout,
            reset_type=check_reset_method(self),
        )
        return prog, sweep_axes, acquire
