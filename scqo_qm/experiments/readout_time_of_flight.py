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
* TWO vendor fields are borrowed for the run and restored in a ``finally``:
  ``resonator.time_of_flight``, because it IS the acquisition window's origin,
  and the readout operation's ``length``, because an ADC trace is exactly as
  long as the measurement that produced it. The second one is not cosmetic —
  the live 5Q4C tree stores an 800 ns readout while this experiment's axis
  defaults to 1000 samples, so leaving it alone hands the backend a trace that
  does not match the declared axis. (The retired node set both for the same
  reason.) These are run-scoped vendor mutations of the kind
  ``setup_snapshot.drift`` exists to catch, which is why the restore is a
  ``finally`` and is tested on the error path.

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

    def probe(self) -> Any:
        from scqo_qm.experiments._lib import select_qubits

        machine = self.backend.machine  # type: ignore[attr-defined]
        frame = self.resolved_frame()
        window_ns = int(round(frame["window_start_ns"]))
        qubits = select_qubits(machine, self.params.targets, multiplexed=False)
        times = self.sweep_axes[TIME_AXIS]

        # Run-scoped vendor mutation: open the acquisition window at the frame
        # the experiment resolved, and put it back. A miss here would leave the
        # setup with a deliberately-wrong delay, which is the exact failure this
        # experiment exists to repair.
        trace_ns = int(round(float(times.size) * float(frame["sample_ns"])))
        previous = {
            q.name: (q.resonator.time_of_flight,
                     q.resonator.operations[_OPERATION].length)
            for q in qubits
        }
        try:
            for qubit in qubits:
                qubit.resonator.time_of_flight = window_ns
                # the trace lasts as long as the measurement: a stored readout
                # shorter than the requested axis returns fewer samples than the
                # contract declares, and a longer one keeps the instrument busy
                # past the plateau for nothing
                qubit.resonator.operations[_OPERATION].length = trace_ns
            return build_program(
                machine, qubits,
                num_shots=self.params.num_averages,
                times_ns=times,
                operation=_OPERATION,
            )
        finally:
            for qubit in qubits:
                tof, length = previous[qubit.name]
                qubit.resonator.time_of_flight = tof
                qubit.resonator.operations[_OPERATION].length = length

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
