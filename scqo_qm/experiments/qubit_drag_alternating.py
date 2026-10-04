"""DRAG Alternating (180/-180) calibration acquisition probe: vendor code only (qm/quam).

QM DRAG alternating calibration for scqo - supplies only ``probe()``.

Parameters, fit, and reporting are inherited from ``scqo.experiments.QubitDragAlternating``.
"""

from __future__ import annotations

from typing import Callable, Optional, List
import numpy as np
import xarray as xr
from qm.qua import *

from scqo_qm.experiments._lib import acquire as _acquire


def build_program(
    machine,
    qubits,
    *,
    num_shots: int,
    beta_array: List[float],
    nb_pulses_array: List[int],
    reset_type: str,
    use_state_discrimination: bool,
    target_gate: str = "x180",
    simulate: bool = False,
    log: Optional[Callable] = None,
):
    from scqo_qm import quam_fields

    num_qubits = len(qubits)
    alpha_array = np.asarray(beta_array, dtype=float)
    nb_pulses_array = np.asarray(nb_pulses_array)

    # same normalization as scqo's _gate_target (probes must not import scqo)
    op_name = "x90" if str(target_gate).strip().lower() == "x90" else "x180"

    first_q_name = qubits.get_names()[0]
    first_q = machine.qubits[first_q_name]
    alpha_base = quam_fields.get_drag_beta(first_q, operation=op_name)
    if abs(alpha_base) < 1e-6:
        alpha_base = 1.0

    scale_array = alpha_array / alpha_base

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "nb_of_pulses": xr.DataArray(nb_pulses_array, attrs={"long_name": "number of pulses"}),
        "beta": xr.DataArray(alpha_array, attrs={"long_name": "DRAG alpha coefficient", "units": ""}),
    }

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        a = declare(fixed)  # DRAG alpha scale factor
        npi = declare(int)  # Number of alternating pairs
        count = declare(int)

        if use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < num_shots, n + 1):
                save(n, n_st)
                with for_each_(npi, nb_pulses_array):
                    with for_each_(a, scale_array):
                        # Qubit initialization
                        for i_q, qubit in multiplexed_qubits.items():
                            qubit.reset(reset_type, simulate, log_callable=log)
                        align()

                        # Play alternating pulses
                        for i_q, qubit in multiplexed_qubits.items():
                            qubit.align()
                            with for_(count, 0, count < npi, count + 1):
                                if op_name == "x90":
                                    # Each x180 is replaced by two x90 pulses
                                    play("x90" * amp(1, 0, 0, a), qubit.xy.name)
                                    play("x90" * amp(1, 0, 0, a), qubit.xy.name)
                                    # Each -x180 is replaced by two -x90 pulses
                                    play("x90" * amp(-1, 0, 0, -a), qubit.xy.name)
                                    play("x90" * amp(-1, 0, 0, -a), qubit.xy.name)
                                else:
                                    play("x180" * amp(1, 0, 0, a), qubit.xy.name)
                                    play("x180" * amp(-1, 0, 0, -a), qubit.xy.name)
                            qubit.align()


                        # Measurement
                        for i_q, qubit in multiplexed_qubits.items():
                            if use_state_discrimination:
                                qubit.readout_state(state[i_q])
                                save(state[i_q], state_st[i_q])
                            else:
                                qubit.resonator.measure("readout", qua_vars=(I[i_q], Q[i_q]))
                                save(I[i_q], I_st[i_q])
                                save(Q[i_q], Q_st[i_q])
                        align()

        with stream_processing():
            n_st.save("n")
            for i_q in range(num_qubits):
                if use_state_discrimination:
                    state_st[i_q].buffer(len(beta_array)).buffer(len(nb_pulses_array)).average().save(f"state{i_q + 1}")
                else:
                    I_st[i_q].buffer(len(beta_array)).buffer(len(nb_pulses_array)).average().save(f"I{i_q + 1}")
                    Q_st[i_q].buffer(len(beta_array)).buffer(len(nb_pulses_array)).average().save(f"Q{i_q + 1}")

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
    return _acquire(machine, prog, sweep_axes, num_shots=num_shots, timeout=timeout, log=log)


from typing import Any, ClassVar

from scqo import register
from scqo.experiments import QubitDragAlternating
from scqo.requirements import Requirement


@register
class QMQubitDragAlternating(QubitDragAlternating):
    """Build a multiplexed DRAG alternating QUA program on the QM OPX."""

    #: what only this backend consumes, on top of the neutral requirements: the
    #: sweep scales the STORED pulse, so the stored coefficient is an input here
    requires: ClassVar[tuple[Requirement, ...]] = (
        *QubitDragAlternating.requires,
        Requirement("drag_beta",
                    "the swept coefficient is the stored pulse's DRAG quadrature, "
                    "rescaled: a stored zero leaves nothing to scale",
                    when=("target_gate", "x180")),
        Requirement("drag_beta_x90",
                    "the swept coefficient is the stored pulse's DRAG quadrature, "
                    "rescaled: a stored zero leaves nothing to scale",
                    when=("target_gate", "x90")),
    )
    #: true of THIS probe only; scqo shows them in
    #: `scqo run qubit_drag_alternating --help`
    backend_notes: ClassVar[tuple[str, ...]] = (
        "each coefficient is realized by scaling the DRAG quadrature of the stored pulse "
        "by beta / stored coefficient: with a stored coefficient of zero that quadrature "
        "is zero and the sweep changes nothing",
        "the scale is computed from the FIRST target's stored coefficient and applied to "
        "every target: with several targets the beta axis is exact for the first only",
        "--preview is refused: the probe executes and fetches itself",
    )

    # preview opt-out (backend.SELF_ACQUIRING_ATTR): truthy reason = refuse
    probe_self_acquires = "it executes and fetches inside probe()"

    def probe(self) -> Any:
        from ._reset import check_reset_method
        from scqo_qm.experiments._lib import select_qubits

        machine = self.backend.machine
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)
        
        sweeps = self.define_sweep()
        beta_array = list(sweeps["beta"])
        nb_pulses_array = [int(x) for x in sweeps["nb_of_pulses"]]

        prog, sweep_axes = build_program(
            machine,
            qubits,
            num_shots=int(self.params.num_averages),
            beta_array=beta_array,
            nb_pulses_array=nb_pulses_array,
            reset_type=check_reset_method(self),
            use_state_discrimination=False,
            target_gate=getattr(self.params, "target_gate", "x180"),
        )

        params = self.params
        shots = getattr(params, "num_averages", None) or getattr(params, "num_shots", 1)

        return acquire(
            machine,
            prog,
            sweep_axes,
            num_shots=int(shots),
            timeout=self.backend._timeout,
        )

