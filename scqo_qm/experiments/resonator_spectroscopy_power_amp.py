"""Resonator-spectroscopy-vs-power acquisition probe: vendor code only (qm/quam) -
no qualibrate, no scqo, no scqat.

2-D resonator spectroscopy vs readout power: per readout-amplitude point, repeat a
fast sweep of the readout intermediate frequency around each resonator's current IF.
The |IQ| dip locates the resonance at every power. No qubit reset - the resonator is
measured directly.

Loop order (2026-07-14, user-decided; both scqo backends match): amplitude (outer)
-> averages (middle) -> frequency (INNER = fastest) — each power point repeats the
frequency sweep ``num_shots`` times, so the resonator only jumps power between the
slow outer steps. The acquired axis order is therefore (power, detuning). The caller
must have set each resonator's base output power to the sweep's max power *before*
generating the config (the qualibrate shell does this via ``tracked_updates``); this
probe only builds the amplitude/frequency sweep, keeping it framework-free.

``depletion_time_ns`` optionally overrides every resonator's configured
``depletion_time`` for the between-readout ring-down wait (None keeps the per-qubit
QUAM values) — it also covers the wrap-around high->low amplitude jump between
repetitions of the inner sweep.

QM resonator spectroscopy vs ABSOLUTE power (amplitude sweep) for scqo - supplies
only ``probe()``.

Parameters, the punchout analysis and the readout_power_dbm/readout_freq_hz writeback
are inherited from ``scqo.experiments.ResonatorSpectroscopyPowerAmp``. scqo sweeps
``(power_dbm, detuning_hz)``; the QM builder builds the same sweep on coords
``(power, detuning)``, which the backend's ``_to_canonical`` renames positionally.

Power convention: the core ``run()`` already solved the chain for the window top
(``readout_power_dbm = max_power_dbm``), so the probe's ``amps`` prefactors are
relative to THAT top — ``10**((power_dbm - max_power_dbm)/20)``, top point exactly
1.0 (``amplitude_scale`` scales the readout pulse the setter just parked at ~0.5 of
full scale; every prefactor is <= 1, well inside QUA's range). Same realization as
the qualibrate ``LCH_resonator_spectroscopy_power`` node (set-top -> sweep down ->
revert), whose shared probe module is reused unchanged. Loop order is amplitude ->
averages -> frequency (the probe's contract); the ring-down wait is the governed
depletion time — the per-run ``readout_depletion_ns`` override, else the readout
channel's calibrated ``readout_depletion_s`` knob, which on this backend IS the
resonators' ``depletion_time`` (None = never calibrated, keep the QUAM values).
"""

from __future__ import annotations

from typing import Callable, Optional

import xarray as xr
from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.units import unit

from scqo_qm.experiments._lib import acquire as _acquire


def build_program(
    machine,
    qubits,
    *,
    dfs,
    amps,
    power_dbm,
    num_shots: int,
    depletion_time_ns: Optional[float] = None,
):
    """Build the resonator-spectroscopy-vs-power QUA program. Returns (program, sweep_axes).

    `dfs` is the readout-frequency detuning sweep in Hz (relative to each
    resonator's current IF); `amps` is the readout-amplitude pre-factor sweep
    (dimensionless, within [-2, 2)); `power_dbm` is the matching readout-power axis
    in dB (same length as `amps`); `qubits` is a BatchableList (see
    `_lib.select_qubits`); `depletion_time_ns` overrides the resonators' configured
    depletion wait (None = per-qubit QUAM `depletion_time`).
    """
    u = unit(coerce_to_integer=True)
    num_qubits = len(qubits)
    n_avg = num_shots

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "power": xr.DataArray(power_dbm, attrs={"long_name": "readout power", "units": "dBm"}),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "readout frequency", "units": "Hz"}),
    }

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        a = declare(fixed)  # QUA variable for the readout amplitude pre-factor
        df = declare(int)  # QUA variable for the readout frequency detuning
        idx = declare(int)  # progress index over the outer amplitude loop

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                machine.initialize_qpu(target=qubit)
            align()

            assign(idx, 0)
            with for_each_(a, amps):  # amplitude (outer, slow)
                # Save the amplitude-point counter for the progress bar
                save(idx, n_st)
                assign(idx, idx + 1)
                with for_(n, 0, n < n_avg, n + 1):  # averages (middle)
                    with for_(*from_array(df, dfs)):  # frequency (INNER = fastest sweep)
                        for i, qubit in multiplexed_qubits.items():
                            rr = qubit.resonator
                            # set the readout IF for this detuning point
                            rr.update_frequency(df + rr.intermediate_frequency)
                            # readout the resonator at the swept amplitude
                            rr.measure("readout", qua_vars=(I[i], Q[i]), amplitude_scale=a)
                            # wait for the resonator to deplete (ring-down)
                            wait_ns = (
                                int(depletion_time_ns)
                                if depletion_time_ns is not None
                                else rr.depletion_time
                            )
                            rr.wait(wait_ns * u.ns)
                            # save data
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])
                align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                # Saves arrive frequency-fastest: group the inner freq sweep, stack the
                # n_avg repeats, average over that (middle) axis, then stack powers ->
                # final shape (power, detuning).
                I_st[i].buffer(len(dfs)).buffer(n_avg).map(FUNCTIONS.average(0)).buffer(
                    len(amps)
                ).save(f"I{i + 1}")
                Q_st[i].buffer(len(dfs)).buffer(n_avg).map(FUNCTIONS.average(0)).buffer(
                    len(amps)
                ).save(f"Q{i + 1}")

    return prog, sweep_axes


def acquire(
    machine,
    prog,
    sweep_axes,
    *,
    num_amp_points: int,
    timeout: float,
    log: Optional[Callable] = None,
) -> xr.Dataset:
    """Connect to the QOP, execute the program and fetch the raw xr.Dataset.

    The progress counter tracks the OUTER loop, which here is the amplitude sweep —
    the argument was called ``num_detuning_points`` for the retired qualibrate
    shell's call site, which is the axis it is not.
    """
    return _acquire(machine, prog, sweep_axes, num_shots=num_amp_points, timeout=timeout, log=log)


from typing import Any

import numpy as np

from scqo import register
from scqo.experiments import ResonatorSpectroscopyPowerAmp
from scqo.experiments._depletion import depletion_wait_ns


@register
class QMResonatorSpectroscopyPowerAmp(ResonatorSpectroscopyPowerAmp):
    """Build a multiplexed punchout QUA program on the QM OPX."""

    def probe(self) -> Any:
        from scqo_qm.experiments._lib import select_qubits

        machine = self.backend.machine  # type: ignore[attr-defined]
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)

        power_dbm = np.asarray(self.sweep_axes["power_dbm"])
        # prefactors relative to the window top run() solved the chain for
        amps = 10.0 ** ((power_dbm - self.params.max_power_dbm) / 20.0)
        return build_program(
            machine,
            qubits,
            dfs=self.sweep_axes["detuning_hz"],
            amps=amps,
            power_dbm=power_dbm,  # absolute dBm axis
            num_shots=self.params.num_averages,
            # Per-run override, else the calibrated readout_depletion_s knob —
            # resolved by scqo's ONE precedence helper, so this probe and the
            # Qblox one cannot come to disagree about what the wait is. None
            # (never calibrated) keeps the probe's per-qubit QUAM values, which
            # on this backend IS the same field the knob writes.
            depletion_time_ns=depletion_wait_ns(self, self.params.targets[0]),
        )
