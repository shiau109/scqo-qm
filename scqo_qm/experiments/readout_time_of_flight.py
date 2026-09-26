"""QM readout time of flight for scqo — the probe and the ADC reduction.

Parameters, the edge fit and the writeback hint are inherited from
``scqo.experiments.ReadoutTimeOfFlight``. What this file owns is the one QUA
shape no other probe here uses: a RAW ADC trace.

``declare_stream(adc_trace=True)`` records the digitizer itself rather than a
demodulated result, so the trace shows the readout pulse arriving. Two details
carry the measurement:

* ``reset_if_phase`` before every shot. The trace is AVERAGED, and without a
  deterministic oscillator phase the cosine averages toward zero — the step
  vanishes into the noise and the fit reports ``arrival_unresolved`` on a setup
  that is perfectly fine. The retired official node did the same thing for the
  same reason.
* TWO config values are AMENDED for this run: the element's
  ``time_of_flight``, because it IS the acquisition window's origin, and the
  readout pulse's ``length``, because an ADC trace lasts exactly as long as the
  measurement that produced it (the live 5Q4C tree stores an 800 ns readout
  while this experiment's axis defaults to 1000 samples, so leaving it alone
  would declare an axis the returned trace cannot match).

  THE AMENDMENT IS ON THE GENERATED CONFIG, NOT ON THE QUAM TREE. Writing the
  tree and restoring it afterwards is the obvious approach and it is wrong
  twice over. First, both values are read by ``generate_config()``, which the
  backend calls AFTER ``probe()`` returns, so a borrow released at the end of
  ``probe()`` never reaches the instrument at all — the program would open its
  window at the very setting under test, which is the one failure mode this
  experiment is designed around, and it assembles cleanly either way.
  (Measured: ``scqo run ... --preview`` against the real 5Q4C tree on
  2026-09-26 still showed ``time_of_flight: 384``.) Second, a tree mutation
  that outlives a crash leaves the setup carrying a delay deliberately chosen
  to be WRONG — worse than the mis-set value the operator ran this to repair.

  So nothing here touches vendor state. ``probe()`` returns the 3-tuple form
  with its own acquire callable carrying the amended config, and
  ``patch_preview_config`` gives ``--preview`` the same one — the pattern the
  parametric-drive shells established (``_parametric.py``), and the reason the
  preview and the run cannot diverge.

Adapted from the official node ``01b_time_of_flight_mw_fem``, vendored in this
repo until v3.13.0. What changed: the node reported ``tof_to_add`` against a
window it hard-coded to 28 ns, and the state update added the two back together
at writeback time; here the frame is resolved neutrally, travels with the
dataset, and the estimator returns the absolute delay — so a re-fit of a saved
run cannot silently use a different origin than the one it was measured in.

ONE TARGET AT A TIME (``multiplexed=False``). Two resonators on one feedline
share an input port: a multiplexed raw capture records their pulses superposed
on the same trace, and while the arrival edge would still be the cable, the
amplitude and the rise would be the sum of two pulses. The delay is a property
of the WIRING and is measured rarely, so there is nothing to win by batching.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr
from qm.qua import *

from scqo import register
from scqo.experiments import ReadoutTimeOfFlight
from scqo.experiments.readout_time_of_flight import TIME_AXIS


#: OPX ADC full scale, volts. The raw stream is 12-bit signed counts over this
#: range, which is what ``_ADC_TO_VOLTS`` converts and what the estimator's
#: saturation check compares against.
ADC_FULL_SCALE_V = 0.5

#: normalized full scale for an MW-FEM waveform (an Octave IQ channel is
#: bounded in VOLTS instead - see _octave.MAX_IF_AMP_V; the two families do not
#: share the unit, which is why this is a local constant and not that one).
_MAX_WAVEFORM = 1.0

#: the readout operation this probe plays and resizes.
_OPERATION = "readout"

#: raw ADC counts -> volts, sign included. The OPX input is inverting; the
#: official node applied exactly ``-adc / 2**12``.
_ADC_TO_VOLTS = -1.0 / 2 ** 12


def build_program(machine, qubits, *, num_shots: int, times_ns,
                  operation: str = "readout"):
    """Build the raw-ADC-trace QUA program. Returns (program, sweep_axes)."""
    num_qubits = len(qubits)

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        TIME_AXIS: xr.DataArray(
            np.asarray(times_ns, dtype=float),
            attrs={"long_name": "time from window origin", "units": "ns"}),
    }

    with program() as prog:
        n = declare(int)
        n_st = declare_stream()
        adc_st = [declare_stream(adc_trace=True) for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            with for_(n, 0, n < num_shots, n + 1):
                save(n, n_st)
                for i, qubit in multiplexed_qubits.items():
                    # The averaged trace needs a deterministic phase per shot,
                    # or the cosine averages away and the step with it.
                    reset_if_phase(qubit.resonator.name)
                    qubit.resonator.measure(operation, stream=adc_st[i])
                    qubit.resonator.wait(machine.depletion_time // 4)
            align()

        with stream_processing():
            n_st.save("n")
            for i, qubit in enumerate(qubits):
                port = getattr(getattr(qubit.resonator, "opx_input", None),
                               "port_id", None)
                stream = adc_st[i].input2() if port == 2 else adc_st[i].input1()
                stream.real().average().save(f"adcI{i + 1}")
                stream.image().average().save(f"adcQ{i + 1}")

    return prog, sweep_axes


@register
class QMReadoutTimeOfFlight(ReadoutTimeOfFlight):
    """Record the raw ADC trace of the readout pulse on the QM OPX."""

    def _amend(self, config: dict) -> dict:
        """Open the window at the resolved frame and stretch the readout to the
        trace length — on the CONFIG, in place, for this run only."""
        from scqo_qm.experiments._lib import select_qubits

        machine = self.backend.machine  # type: ignore[attr-defined]
        frame = self.resolved_frame()
        window_ns = int(round(frame["window_start_ns"]))
        trace_ns = int(round(float(self.params.readout_len_ns)))

        factor = float(self.params.readout_amp_factor)

        for qubit in select_qubits(machine, self.params.targets,
                                   multiplexed=False):
            element = config["elements"][qubit.resonator.name]
            element["time_of_flight"] = window_ns
            pulse = config["pulses"][element["operations"][_OPERATION]]
            pulse["length"] = trace_ns
            for name in pulse.get("waveforms", {}).values():
                waveform = config["waveforms"][name]
                if waveform.get("type") == "arbitrary":
                    samples = [s * factor for s in waveform["samples"]]
                    # hold the last sample: the plateau has to outlast the
                    # window, and a shorter stored envelope would end inside it
                    waveform["samples"] = (
                        samples + [samples[-1]] * (trace_ns - len(samples))
                        if len(samples) < trace_ns else samples[:trace_ns])
                    peak = max(abs(s) for s in waveform["samples"])
                else:
                    waveform["sample"] = waveform.get("sample", 0.0) * factor
                    peak = abs(waveform["sample"])
                if peak > _MAX_WAVEFORM:
                    raise ValueError(
                        f"readout_amp_factor={factor:g} puts {qubit.name}'s "
                        f"readout waveform {name!r} at {peak:.3f}, past the "
                        f"{_MAX_WAVEFORM} full-scale rail. The DAC would clip "
                        f"and the SIMULATOR WOULD NOT SHOW IT - and a clipped "
                        f"pulse flattens the very edge this experiment measures. "
                        f"Lower the factor, or raise the channel's "
                        f"full_scale_power_dbm and re-calibrate the readout.")
        return config

    def patch_preview_config(self, config: dict) -> dict:
        """``--preview`` must show the config the run executes against."""
        return self._amend(config)

    def probe(self) -> Any:
        from functools import partial

        from scqo_qm.experiments._lib import acquire as _lib_acquire
        from scqo_qm.experiments._lib import select_qubits

        machine = self.backend.machine  # type: ignore[attr-defined]
        qubits = select_qubits(machine, self.params.targets, multiplexed=False)
        prog, sweep_axes = build_program(
            machine, qubits,
            num_shots=self.params.num_averages,
            times_ns=self.sweep_axes[TIME_AXIS],
            operation=_OPERATION,
        )
        config = self._amend(machine.generate_config())
        return prog, sweep_axes, partial(_lib_acquire, config=config)

    def reduce_raw(self, raw: xr.Dataset) -> xr.Dataset:
        """Raw ADC counts -> volts, under the contract's variable names.

        Applied before canonicalization, so the contract sees ``I``/``Q`` in
        volts rather than the ``adcI``/``adcQ`` counts the QUA streams carry.
        """
        out = raw
        renames = {}
        for source, target in (("adcI", "I"), ("adcQ", "Q")):
            if source in out.data_vars:
                out = out.assign({source: out[source] * _ADC_TO_VOLTS})
                renames[source] = target
        if renames:
            out = out.rename(renames)
        for name in ("I", "Q"):
            if name in out.data_vars:
                out[name].attrs = {"long_name": f"ADC {name}", "units": "V"}
        return out
