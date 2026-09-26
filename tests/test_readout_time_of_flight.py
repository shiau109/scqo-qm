"""The QM raw-ADC-trace time-of-flight probe.

Three things are worth pinning offline, and none of them is the edge fit (that
is scqat's, and it has its own tests):

* the program ASSEMBLES against the live config - the raw-ADC stream is a QUA
  shape no other probe here builds, and ``adc_trace=True`` streams have their
  own stream-processing rules;
* the window is opened by TEMPORARILY writing ``resonator.time_of_flight`` and
  is put back afterwards, on the error path too - a miss leaves the setup with
  a deliberately-wrong delay, which is the exact damage this experiment exists
  to repair;
* the ADC reduction converts counts to volts under the contract's names.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from conftest import make_experiment
# the live-state fixtures live in test_qm_backend, as the other probe
# tests that need a real tree also take them
from test_qm_backend import live_roster, machine  # noqa: F401

from scqo_qm.experiments.readout_time_of_flight import (
    ADC_FULL_SCALE_V,
    QMReadoutTimeOfFlight,
)


def _experiment(backend, roster, **params):
    return make_experiment(
        QMReadoutTimeOfFlight, backend, roster,
        QMReadoutTimeOfFlight.Parameters(targets=["q1"], num_averages=10,
                                         **params))


# ------------------------------------------------------- the vendor mutation

def test_the_probe_restores_both_fields_it_borrowed(backend, roster, monkeypatch):
    """TWO fields are borrowed and both must come back.

    ``time_of_flight`` is the window origin - leaving it written would hand the
    setup a delay chosen to be WRONG (as early as the hardware allows), which is
    worse than the mis-set value the operator ran this to fix.

    The readout operation's LENGTH matters for a less obvious reason: an ADC
    trace lasts exactly as long as the measurement that produced it. The live
    5Q4C tree stores an 800 ns readout while this experiment's axis defaults to
    1000 samples, so a probe that left the op alone would declare an axis the
    returned trace does not match."""
    resonator = backend.machine.qubits["q1"].resonator
    before = (resonator.time_of_flight, resonator.operations["readout"].length)

    seen = {}
    monkeypatch.setattr(
        "scqo_qm.experiments.readout_time_of_flight.build_program",
        lambda machine, qubits, **kw: seen.update(
            kw,
            borrowed_tof=qubits[0].resonator.time_of_flight,
            borrowed_len=qubits[0].resonator.operations["readout"].length,
        ) or ("prog", {}))

    exp = _experiment(backend, roster, readout_len_ns=500)
    exp.sweep_axes = exp.define_sweep()
    exp.probe()

    assert seen["borrowed_tof"] == 28       # the floor, while the program built
    assert seen["borrowed_len"] == 500      # == the declared axis, not the stored op
    assert (resonator.time_of_flight,
            resonator.operations["readout"].length) == before


def test_the_time_of_flight_is_restored_even_when_the_build_raises(
        backend, roster, monkeypatch):
    """The restore is in a finally, not after the return."""
    resonator = backend.machine.qubits["q1"].resonator
    before = (resonator.time_of_flight, resonator.operations["readout"].length)

    def boom(machine, qubits, **kw):
        raise RuntimeError("build failed")

    monkeypatch.setattr(
        "scqo_qm.experiments.readout_time_of_flight.build_program", boom)
    exp = _experiment(backend, roster, readout_len_ns=500)
    exp.sweep_axes = exp.define_sweep()

    with pytest.raises(RuntimeError, match="build failed"):
        exp.probe()
    assert (resonator.time_of_flight,
            resonator.operations["readout"].length) == before


def test_the_probe_runs_one_target_at_a_time(backend, roster, monkeypatch):
    """Two resonators on one feedline share an input port: a multiplexed raw
    capture records both pulses superposed on one trace."""
    seen = {}
    monkeypatch.setattr(
        "scqo_qm.experiments.readout_time_of_flight.build_program",
        lambda machine, qubits, **kw: seen.update(
            batches=[list(b) for b in qubits.batch()]) or ("prog", {}))

    exp = _experiment(backend, roster, readout_len_ns=500)
    exp.sweep_axes = exp.define_sweep()
    exp.probe()

    assert all(len(batch) == 1 for batch in seen["batches"])


# ------------------------------------------------------------ the reduction

def test_raw_adc_counts_become_volts_under_the_contract_names(backend, roster):
    """The QUA streams are adcI/adcQ in 12-bit counts on an inverting input;
    the contract wants I/Q in volts."""
    exp = _experiment(backend, roster, readout_len_ns=4)
    raw = xr.Dataset(
        {"adcI": (("readout_time_ns",), np.array([0.0, 2048.0, -4096.0, 4096.0])),
         "adcQ": (("readout_time_ns",), np.zeros(4))},
        coords={"readout_time_ns": np.arange(4.0)})

    out = exp.reduce_raw(raw)

    assert set(out.data_vars) == {"I", "Q"}
    assert out["I"].attrs["units"] == "V"
    # -count / 2**12: the sign is the inverting input, the scale is the ADC word
    np.testing.assert_allclose(out["I"].values, [0.0, -0.5, 1.0, -1.0])
    # ...and full scale is half the word, so +-1.0 above is past the rail - which
    # is what the estimator's saturation flag is compared against
    assert ADC_FULL_SCALE_V == 0.5


def test_the_reduction_is_a_no_op_without_the_adc_variables(backend, roster):
    """A dataset that already carries I/Q (a reloaded run) passes through."""
    exp = _experiment(backend, roster, readout_len_ns=4)
    raw = xr.Dataset({"I": (("readout_time_ns",), np.zeros(4)),
                      "Q": (("readout_time_ns",), np.zeros(4))},
                     coords={"readout_time_ns": np.arange(4.0)})
    assert set(exp.reduce_raw(raw).data_vars) == {"I", "Q"}


# ------------------------------------------------------------- the QUA build

def test_the_raw_trace_program_assembles_against_the_live_config(machine,
                                                                 live_roster):
    """``declare_stream(adc_trace=True)`` plus ``reset_if_phase`` is a shape no
    other probe here builds, and its stream processing has its own rules."""
    from qm import generate_qua_script

    from scqo_qm.experiments._lib import select_qubits
    from scqo_qm.experiments.readout_time_of_flight import build_program

    qubits = select_qubits(machine, ["q1"], multiplexed=False)
    prog, axes = build_program(machine, qubits, num_shots=10,
                               times_ns=np.arange(500.0))
    script = generate_qua_script(prog, machine.generate_config())

    assert "reset_if_phase" in script       # else the average kills the step
    assert "adcI1" in script and "adcQ1" in script
    assert list(axes) == ["qubit", "readout_time_ns"]
    assert axes["readout_time_ns"].size == 500


# --------------------------------------------------------------- the context

def test_the_backend_names_its_own_vendor_field_and_floor(backend):
    """The neutral layer learns no vendor spelling: the backend says WHICH
    VendorOnly entry holds the answer and what the instrument can do."""
    context = backend.readout_delay_context("q1")

    assert context["field"] == "time_of_flight"
    assert context["field"] in backend.vendor_only()
    assert context["floor_ns"] == 28.0      # what the official nodes used
    assert context["grid_ns"] == 4.0        # QUA takes multiples of 4
    assert context["full_scale_v"] == ADC_FULL_SCALE_V


def test_an_unserved_target_reports_nothing_rather_than_a_default(backend):
    """No readout channel means no acquisition path, so there is nothing to
    say - not a floor that reads as a real answer."""
    assert backend.readout_delay_context("not_a_target") == {}
