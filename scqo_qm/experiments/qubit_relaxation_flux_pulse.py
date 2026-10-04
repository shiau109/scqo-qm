"""T1 vs Flux spectrum acquisition probe: vendor code only (qm/quam).

PULSE CONTRACT: `flux_amps_v` are VOLTS measured from the standing DC bias that
`initialize_qpu` applies -- the z `const` pulse rides on top of that offset, so
the axis is idle-relative and 0 V means "stay parked". `flux_point` selects which
named offset stands, and it is passed EXPLICITLY (rather than leaning on
quam_builder's "joint" default) because the same value has to reach the rail
validation below: the DAC emits idle + excursion, so the number that was checked
must be the number that plays.

QM qubit relaxation vs flux PULSE (T1 spectrum) for scqo - supplies only ``probe()``.

Parameters, fit, and reporting are inherited from
``scqo.experiments.QubitRelaxationFluxPulse``.

PULSE CONTRACT: this probe conforms to the ``_pulse`` name by construction - the
z bias is a ``const`` PULSE played during the idle delay
(``build_program`` below), which the DAC adds to the
standing offset ``initialize_qpu`` applies, so ``flux_bias_v`` is an excursion
FROM ``idle_flux`` and 0 V means "stay parked". ``flux_point`` is passed
explicitly for the reason the probe documents: the bias that was rail-validated
must be the bias that plays. The probe MODULE keeps its historical name (it is
shared with the qualibrate ``LCH_T1_spectrum`` shell).
"""

from __future__ import annotations

from typing import Callable, Optional, List
import numpy as np
import xarray as xr
from qm.qua import *

from scqo_qm.experiments._lib import acquire as _acquire
from scqo_qm.experiments._flux_limits import check_flux_pulse_relative, idle_offset_v


def build_program(
    machine,
    qubits,
    *,
    wait_times_cycles,
    flux_amps_v: List[float],
    num_shots: int,
    reset_type: str = "thermal",
    use_state_discrimination: bool = False,
    simulate: bool = False,
    flux_point: str = "joint",
    log: Optional[Callable] = None,
):
    """Build the T1 vs Flux QUA program. Returns (program, sweep_axes).

    `flux_amps_v` is the flux-PULSE window in volts RELATIVE to `flux_point`'s
    standing offset (see the module docstring).
    """
    wait_times_cycles = np.unique(np.asarray(wait_times_cycles))
    flux_amps_v = np.asarray(flux_amps_v)
    num_qubits = len(qubits)

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "flux_bias_v": xr.DataArray(
            flux_amps_v,
            attrs={"long_name": "flux pulse amplitude relative to the idle bias", "units": "V"},
        ),
        "wait_time_ns": xr.DataArray(4 * wait_times_cycles, attrs={"long_name": "wait after pi pulse", "units": "ns"}),
    }

    # Volts -> amplitude_scale reference per qubit, validated against the rail
    # BEFORE any QUA is emitted: the sum idle + excursion is what the DAC puts
    # out, and clipping there is invisible to the simulator.
    amp_ref = {}
    for qubit in qubits:
        z = getattr(qubit, "z", None)
        if z is None:
            raise ValueError(
                f"{qubit.name}: no flux line, but this probe sweeps a z pulse "
                f"on every measured qubit - it cannot silently skip one and "
                f"still report a flux axis in volts")
        amp_ref[qubit.name] = check_flux_pulse_relative(
            z, name=qubit.name, idle_v=idle_offset_v(z, flux_point), amps_v=flux_amps_v)

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        t = declare(int)
        v = declare(fixed)

        if use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                machine.initialize_qpu(target=qubit, flux_point=flux_point)
            align()

            with for_(n, 0, n < num_shots, n + 1):
                save(n, n_st)

                with for_each_(v, flux_amps_v):
                    with for_each_(t, wait_times_cycles):
                        # Qubit reset
                        for i, qubit in multiplexed_qubits.items():
                            qubit.reset(reset_type, simulate, log_callable=log)
                        align()

                        # Excite and apply flux pulse during wait
                        for i, qubit in multiplexed_qubits.items():
                            qubit.align()
                            play("x180", qubit.xy.name)
                            qubit.align()

                            # v is VOLTS; amp() takes a scale factor, so divide by
                            # the stored const amplitude. Without this the emitted
                            # excursion was const_amplitude x the requested volts
                            # (0.25x on 5Q4C) while the axis still claimed V.
                            play("const" * amp(v / amp_ref[qubit.name]),
                                 qubit.z.name, duration=t)
                            wait(t, qubit.xy.name)
                            qubit.align()

                        # Readout
                        for i, qubit in multiplexed_qubits.items():
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
                if use_state_discrimination:
                    state_st[i].buffer(len(wait_times_cycles)).buffer(len(flux_amps_v)).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(wait_times_cycles)).buffer(len(flux_amps_v)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(wait_times_cycles)).buffer(len(flux_amps_v)).average().save(f"Q{i + 1}")

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

import numpy as np

from scqo import register
from scqo.experiments import QubitRelaxationFluxPulse


@register
class QMQubitRelaxationFluxPulse(QubitRelaxationFluxPulse):
    """Build a multiplexed T1 vs flux-PULSE QUA program on the QM OPX."""

    #: true of THIS probe only; scqo shows them in
    #: `scqo run qubit_relaxation_flux_pulse --help`
    backend_notes: ClassVar[tuple[str, ...]] = (
        "prepare_state is not realized: the x180 is played whatever its value "
        "(SCQO BACKLOG I19)",
    )

    def probe(self) -> Any:
        from ._reset import check_reset_method
        from scqo_qm.quam_fields import GOVERNED_FLUX_POINT
        from scqo_qm.experiments._lib import select_qubits

        machine = self.backend.machine
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)

        sweeps = self.define_sweep()
        flux_bias = list(sweeps["flux_bias_v"])
        wait_ns = sweeps["wait_time_ns"]
        wait_cycles = np.maximum(1, np.round(wait_ns / 4)).astype(int)

        return build_program(
            machine,
            qubits,
            wait_times_cycles=wait_cycles,
            flux_amps_v=flux_bias,
            num_shots=int(self.params.num_averages),
            reset_type=check_reset_method(self),
            use_state_discrimination=bool(self.params.use_state_discrimination),
            flux_point=GOVERNED_FLUX_POINT,
        )
