"""1->2 power-Rabi acquisition probe: vendor code only (qm/quam) - no qualibrate, no scqat.

Each shot: thermal reset -> x180 (the qubit in |1>) -> the qubit's own xy
element retuned to the 1->2 frequency -> the EF_x180 operation at
``amplitude_scale`` = the swept factor -> the 0->1 IF restored -> ``align()`` ->
readout. The factor is the scqo ``amp_prefactor`` axis, a multiple of the stored
``pi_amp_ef`` (the EF_x180 amplitude), exactly QUA's ``amplitude_scale``.

THE 1->2 FREQUENCY comes from the drive channel's ``drive_freq_ef_hz`` (QUAM
``q.f_12``) through the neutral view: an unmeasured one is refused by name
there, before anything is built. It is played as the IF ``f_12 - LO`` on the
element that plays the x180, and the 0->1 IF is restored right after the pulse
so the next shot's x180 plays at its own frequency. One upconverter plays only
``quam_fields.EF_IF_MAX_HZ`` either side of its LO.

QM 1->2 power Rabi for scqo - supplies only ``probe()``. Parameters, the
estimator binding and the ``pi_amp_ef`` writeback are inherited from
``scqo.experiments.QubitPowerRabiEf``.
"""

from __future__ import annotations

from typing import Callable, Optional

import xarray as xr
from qm.qua import *
from qualang_tools.loops import from_array

from scqo_qm.experiments._amp_limits import check_amp_scale_window
from scqo_qm.experiments._lib import acquire as _acquire


def ef_intermediate_frequency(qubit, f_12: float) -> int:
    """The IF that plays ``f_12`` on the qubit's xy element, refused by name past
    what one upconverter plays."""
    from scqo_qm.quam_fields import EF_IF_MAX_HZ

    lo = float(qubit.xy.LO_frequency)
    if_ef = float(f_12) - lo
    if abs(if_ef) > EF_IF_MAX_HZ:
        raise ValueError(
            f"{qubit.name}: drive_freq_ef_hz={float(f_12) / 1e9:.6g} GHz lies "
            f"{if_ef / 1e6:+.1f} MHz from the xy LO ({lo / 1e9:.6g} GHz), past the "
            f"+-{EF_IF_MAX_HZ / 1e6:.0f} MHz one upconverter plays")
    return int(round(if_ef))


def build_program(
    machine,
    qubits,
    *,
    amps,
    ef_ifs: dict,
    operation: str,
    num_shots: int,
    reset_type: str,
    simulate: bool = False,
):
    """Build the 1->2 power-Rabi QUA program. Returns (program, sweep_axes).

    ``amps`` is the factor sweep (refused by name outside QUA's (-2, 2));
    ``ef_ifs`` maps a qubit name to the IF of its 1->2 frequency; ``qubits`` is a
    BatchableList (see ``_lib.select_qubits``).
    """
    check_amp_scale_window(amps, name=", ".join(qubits.get_names()))
    num_qubits = len(qubits)
    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "amp_prefactor": xr.DataArray(amps, attrs={"long_name": "1->2 pulse amplitude prefactor"}),
    }

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        a = declare(fixed)  # the amplitude pre-factor

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < num_shots, n + 1):
                save(n, n_st)
                with for_(*from_array(a, amps)):
                    for i, qubit in multiplexed_qubits.items():
                        qubit.reset(reset_type, simulate)
                    align()

                    for i, qubit in multiplexed_qubits.items():
                        # |1>, at the 0->1 frequency the element is configured for
                        qubit.xy.play("x180")
                        # the 1->2 pulse on the same element, retuned
                        qubit.xy.update_frequency(ef_ifs[qubit.name])
                        qubit.xy.play(operation, amplitude_scale=a)
                        # back to the 0->1 IF before the next shot's x180
                        qubit.xy.update_frequency(qubit.xy.intermediate_frequency)
                    align()

                    for i, qubit in multiplexed_qubits.items():
                        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
                    align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_st[i].buffer(len(amps)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(amps)).average().save(f"Q{i + 1}")

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


from typing import Any, ClassVar

from scqo import register
from scqo.experiments import QubitPowerRabiEf


@register
class QMQubitPowerRabiEf(QubitPowerRabiEf):
    """Build the 1->2 power-Rabi QUA program on the QM OPX (thermal reset only;
    the core Parameters refuse an active one)."""

    backend_notes: ClassVar[tuple[str, ...]] = (
        "the 1->2 pulse is the qubit's EF_x180 operation (its amplitude is pi_amp_ef, its "
        "length pi_duration_ef_s), played on the qubit's own xy element retuned with "
        "update_frequency; the 0->1 IF is restored right after it",
        "drive_freq_ef_hz must lie within +-400 MHz of the xy LO (one upconverter's band)",
    )

    def probe(self) -> Any:
        from ._reset import check_reset_method
        from scqo_qm.experiments._lib import select_qubits
        from scqo_qm.quam_fields import EF_OPERATION

        machine = self.backend.machine  # type: ignore[attr-defined]
        targets = list(self.params.targets)
        qubits = select_qubits(machine, targets, multiplexed=True)
        reset_type = check_reset_method(self)
        ef_ifs = {q: ef_intermediate_frequency(
                      machine.qubits[q], self.device.channel(q, "drive").drive_freq_ef_hz)
                  for q in targets}
        return build_program(
            machine,
            qubits,
            amps=self.sweep_axes["amp_prefactor"],
            ef_ifs=ef_ifs,
            operation=EF_OPERATION,
            num_shots=self.params.num_averages,
            reset_type=reset_type,
        )
