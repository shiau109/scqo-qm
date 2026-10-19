"""1->2 spectroscopy acquisition probe: vendor code only (qm/quam) - no qualibrate, no scqat.

Each shot: thermal reset -> x180 (the qubit in |1>) -> a saturation pulse on the
same xy element, retuned to ``drive_freq_hz + detuning`` -> the 0->1 IF restored
-> ``align()`` -> readout. The detuning is swept below the 0->1 line; the 1->2
line moves the response away from the |1> point.

THE 1->2 TONE IS THE QUBIT'S OWN XY ELEMENT, RETUNED. ``update_frequency`` sets
the element's IF for every later pulse on it, so the 0->1 IF is restored right
after the saturation pulse: the next shot's x180 must play at its own frequency.
The IF the tone needs is the 0->1 IF plus the detuning, and one upconverter plays
only ``quam_fields.EF_IF_MAX_HZ`` either side of its LO - a window past that is
refused by name with the edge that IS reachable, before anything is built.

THE SATURATION POWER HOLDS THE PORT. scqo's ``qubit_spectroscopy`` solves the
drive chain for its power (the port's full scale moves); here that would detune
the x180 in the same shot. So the saturation operation is played at
``amplitude_scale = 10 ** ((drive_power_dbm - standing) / 20)``, where
``standing`` is the drive channel's stored ``drive_power_dbm`` of that same
operation: power goes as amplitude squared on both RF chains, so the ratio holds
whichever chain it is. Refused by name when QUA's ``amp()`` cannot express the
scale or the chain cannot emit the amplitude.

QM 1->2 spectroscopy for scqo - supplies only ``probe()``. Parameters, the
estimator binding and the ``drive_freq_ef_hz`` / ``anharmonicity_hz`` writeback
are inherited from ``scqo.experiments.QubitSpectroscopyEf``.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import xarray as xr
from qm.qua import *
from qualang_tools.loops import from_array

from scqo_qm.experiments._lib import acquire as _acquire


def _cycles(name: str, value_ns) -> int:
    """ns -> QUA clock cycles, refusing anything off the 4 ns grid."""
    value_ns = int(round(float(value_ns)))
    if value_ns % 4:
        raise ValueError(f"{name}={value_ns} ns is not a multiple of the 4 ns QUA clock cycle")
    return value_ns // 4


def check_ef_window(qubit, dfs) -> None:
    """Refuse a detuning window the qubit's xy element cannot play, BY NAME.

    The tone sits at the 0->1 IF plus the detuning, and one upconverter plays
    only EF_IF_MAX_HZ either side of its LO. The message names the reachable
    edges in the experiment's own frame (detuning from drive_freq_hz)."""
    from scqo_qm.quam_fields import EF_IF_MAX_HZ

    if_01 = float(qubit.xy.intermediate_frequency)
    dfs = np.asarray(dfs, dtype=float)
    ifs = if_01 + dfs
    if np.max(np.abs(ifs)) > EF_IF_MAX_HZ:
        lo_edge = -EF_IF_MAX_HZ - if_01
        hi_edge = EF_IF_MAX_HZ - if_01
        raise ValueError(
            f"{qubit.name}: the detuning window reaches an IF of "
            f"{ifs[np.argmax(np.abs(ifs))] / 1e6:+.1f} MHz (the 0->1 drive sits at "
            f"{if_01 / 1e6:+.1f} MHz from the xy LO), past the +-{EF_IF_MAX_HZ / 1e6:.0f} "
            f"MHz one upconverter plays. For {qubit.name} keep start_drive_detuning_hz / "
            f"end_drive_detuning_hz within [{lo_edge:.4g}, {hi_edge:.4g}] Hz.")


def build_program(
    machine,
    qubits,
    *,
    dfs,
    sat_scales: dict,
    drive_len_ns,
    num_shots: int,
    reset_type: str,
    simulate: bool = False,
    log: Optional[Callable] = None,
):
    """Build the 1->2 spectroscopy QUA program. Returns (program, sweep_axes).

    ``dfs`` is the detuning from each qubit's 0->1 drive (Hz); ``sat_scales`` maps
    a qubit name to the amplitude_scale of its saturation pulse; ``qubits`` is a
    BatchableList (see ``_lib.select_qubits``).
    """
    num_qubits = len(qubits)
    drive_cycles = _cycles("drive_len_ns", drive_len_ns)
    for qubit in qubits:
        check_ef_window(qubit, dfs)

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "detuning from the 0->1 drive",
                                             "units": "Hz"}),
    }

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        df = declare(int)  # QUA variable for the detuning from the 0->1 drive

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < num_shots, n + 1):
                save(n, n_st)
                with for_(*from_array(df, dfs)):
                    for i, qubit in multiplexed_qubits.items():
                        qubit.reset(reset_type, simulate, log_callable=log)
                    align()

                    for i, qubit in multiplexed_qubits.items():
                        # |1>, at the 0->1 frequency the element is configured for
                        qubit.xy.play("x180")
                        # the 1->2 search tone on the same element, retuned
                        qubit.xy.update_frequency(df + qubit.xy.intermediate_frequency)
                        qubit.xy.play(
                            "saturation",
                            amplitude_scale=sat_scales[qubit.name],
                            duration=drive_cycles,
                        )
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
                I_st[i].buffer(len(dfs)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(dfs)).average().save(f"Q{i + 1}")

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
from scqo.experiments import QubitSpectroscopyEf


def saturation_scale(experiment, target: str, qubit) -> float:
    """The amplitude_scale that brings ``target``'s saturation pulse to the run's
    ``drive_power_dbm`` with the port's level held, refused by name when QUA or
    the RF chain cannot play it."""
    from scqo_qm._family import RF_OCTAVE, rf_chain
    from scqo_qm._octave import MAX_IF_AMP_V
    from scqo_qm.experiments._amp_limits import MAX_AMP_SCALE

    view = experiment.device.channel(target, "drive")
    standing = view.drive_power_dbm
    amp = view.drive_amp
    if standing is None or amp is None or not amp:
        raise ValueError(
            f"{target}: the drive channel has no standing drive_power_dbm / drive_amp to "
            f"scale the saturation pulse from - set drive_power_dbm first")
    wanted = float(experiment.params.drive_power_dbm)
    scale = 10.0 ** ((wanted - float(standing)) / 20.0)
    ceiling, unit = ((MAX_IF_AMP_V, " V") if rf_chain(qubit.xy) == RF_OCTAVE else (1.0, ""))
    if scale >= MAX_AMP_SCALE or float(amp) * scale >= ceiling:
        reachable = float(standing) + 20.0 * np.log10(
            min(MAX_AMP_SCALE * 0.999, 0.999 * ceiling / float(amp)))
        raise ValueError(
            f"{target}: drive_power_dbm={wanted:g} dBm is out of reach with the port held "
            f"at its standing level (standing saturation power {float(standing):.2f} dBm at "
            f"amplitude {float(amp):.4g}{unit}): it needs amplitude_scale {scale:.3f}. "
            f"Lower drive_power_dbm to {reachable:.1f} dBm or less - the port's level is "
            f"not moved here, because the x180 in the same shot has to stay calibrated.")
    return scale


@register
class QMQubitSpectroscopyEf(QubitSpectroscopyEf):
    """Build the 1->2 spectroscopy QUA program on the QM OPX (thermal reset only;
    the core Parameters refuse an active one)."""

    backend_notes: ClassVar[tuple[str, ...]] = (
        "the 1->2 tone is the qubit's own xy element retuned with update_frequency; the 0->1 IF "
        "is restored right after it, so every x180 plays at the 0->1 frequency",
        "the window must stay within +-400 MHz of the xy LO (one upconverter's band, counted "
        "from the 0->1 IF); a wider one is refused by name with the reachable edges",
        "the saturation power is the saturation operation scaled from the standing "
        "drive_power_dbm; the port's full scale (or the Octave gain) is not moved",
    )

    def probe(self) -> Any:
        from ._reset import check_reset_method
        from scqo_qm.experiments._lib import select_qubits

        machine = self.backend.machine  # type: ignore[attr-defined]
        targets = list(self.params.targets)
        qubits = select_qubits(machine, targets, multiplexed=True)
        reset_type = check_reset_method(self)
        scales = {q: saturation_scale(self, q, machine.qubits[q]) for q in targets}

        return build_program(
            machine,
            qubits,
            dfs=self.sweep_axes["detuning_hz"],
            sat_scales=scales,
            drive_len_ns=int(self.params.drive_len_ns),
            num_shots=self.params.num_averages,
            reset_type=reset_type,
        )
