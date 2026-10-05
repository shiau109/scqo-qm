"""T2 Echo vs Flux spectrum acquisition probe: vendor code only (qm/quam).

PULSE CONTRACT: identical to `qubit_relaxation_flux` -- `flux_amps_v` are VOLTS
measured from the standing DC bias `initialize_qpu` applies, the z `const` pulse
rides on top, and `flux_point` is passed explicitly so the bias that was rail-
validated is the bias that plays. Both echo arms share one reference.

QM qubit echo vs flux PULSE (T2 echo spectrum) for scqo - supplies only ``probe()``.

Parameters, fit, and reporting are inherited from
``scqo.experiments.QubitEchoFluxPulse``.

PULSE CONTRACT: as in ``qubit_relaxation_flux_pulse`` - the z bias is a ``const``
PULSE (played in BOTH echo arms) riding on the standing offset, so
``flux_bias_v`` is an excursion FROM ``idle_flux``. The probe MODULE keeps its
historical name (shared with the qualibrate shell path).
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
    """Build the T2 Echo vs Flux QUA program. Returns (program, sweep_axes).

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
        "wait_time_ns": xr.DataArray(4 * wait_times_cycles, attrs={"long_name": "total wait time", "units": "ns"}),
    }

    # Volts -> amplitude_scale reference per qubit, rail-validated against
    # idle + excursion before any QUA is emitted (see _flux_limits).
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

                        # Echo sequence: x90 -> [wait(t) + Z(v)] -> x180 -> [wait(t) + Z(v)] -> x90
                        for i, qubit in multiplexed_qubits.items():
                            qubit.align()
                            play("x90", qubit.xy.name)
                            qubit.align()

                            # v is VOLTS; amp() takes a scale factor (see the
                            # relaxation probe). Both arms share one reference.
                            play("const" * amp(v / amp_ref[qubit.name]),
                                 qubit.z.name, duration=t)
                            wait(t, qubit.xy.name)
                            qubit.align()

                            play("x180", qubit.xy.name)
                            qubit.align()

                            play("const" * amp(v / amp_ref[qubit.name]),
                                 qubit.z.name, duration=t)
                            wait(t, qubit.xy.name)
                            qubit.align()

                            play("x90", qubit.xy.name)
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
from scqo.experiments import QubitEchoFluxPulse
from scqo.requirements import Requirement

from ._requires import X90_AMPLITUDE


@register
class QMQubitEchoFluxPulse(QubitEchoFluxPulse):
    """Build a multiplexed T2 Echo vs flux-PULSE QUA program on the QM OPX."""

    #: what only this backend consumes, on top of the neutral requirements
    requires: ClassVar[tuple[Requirement, ...]] = (
        *QubitEchoFluxPulse.requires, X90_AMPLITUDE)
    #: true of THIS probe only; scqo shows them in
    #: `scqo run qubit_echo_flux_pulse --help`
    backend_notes: ClassVar[tuple[str, ...]] = (
        "the two pi/2 pulses are the x90 operation, whose amplitude is pi_amp_x90",
        "the stored wait_time_ns is the length of ONE arm, half of the total idle the "
        "experiment declares, so the fitted T2 echo comes out at half its value "
        "(SCQO BACKLOG I39)",
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
        wait_cycles = np.maximum(1, np.round((wait_ns / 2) / 4)).astype(int)

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
