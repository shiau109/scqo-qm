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

def _config_with(resonator_name="q1.resonator", tof=384, length=800):
    """The two config entries this experiment amends, in QM's own shape."""
    return {
        "elements": {resonator_name: {
            "time_of_flight": tof,
            "operations": {"readout": f"{resonator_name}.readout.pulse"},
        }},
        "pulses": {f"{resonator_name}.readout.pulse": {
            "length": length,
            "waveforms": {"I": "wfI", "Q": "wfQ"},
        }},
        "waveforms": {
            "wfI": {"type": "arbitrary", "samples": [0.1] * length},
            "wfQ": {"type": "constant", "sample": 0.0},
        },
    }


def test_the_amendment_opens_the_window_and_stretches_the_readout(backend,
                                                                  roster):
    """THE regression this file exists for, and it is invisible offline any
    other way.

    Both values are read by generate_config(), which the backend calls AFTER
    probe() returns - so the obvious approach (write the QUAM tree in probe(),
    restore in a finally) never reaches the instrument: the program opens its
    window at the very setting under test, and the QUA assembles cleanly
    either way. Found by `scqo run ... --preview` against the real 5Q4C tree,
    2026-09-26, where the embedded config still read time_of_flight 384.

    So the amendment is on the generated CONFIG, and this asserts it there."""
    exp = _experiment(backend, roster, readout_len_ns=500)
    out = exp._amend(_config_with(tof=384, length=800))

    element = out["elements"]["q1.resonator"]
    assert element["time_of_flight"] == 28        # the floor, not the stored 384
    assert out["pulses"]["q1.resonator.readout.pulse"]["length"] == 500
    # the plateau has to outlast the window: a stored envelope shorter than the
    # trace would end inside it and take the plateau with it
    assert len(out["waveforms"]["wfI"]["samples"]) == 500


def test_the_amendment_touches_no_vendor_state(backend, roster):
    """Nothing is borrowed and nothing is restored, so nothing can be left
    wrong by a crash - which for THIS experiment would mean a setup carrying a
    delay deliberately chosen to be the earliest the hardware allows."""
    resonator = backend.machine.qubits["q1"].resonator
    before = (resonator.time_of_flight, resonator.operations["readout"].length)

    exp = _experiment(backend, roster, readout_len_ns=500)
    exp._amend(_config_with())

    assert (resonator.time_of_flight,
            resonator.operations["readout"].length) == before


def test_preview_and_the_run_amend_the_same_way(backend, roster):
    """`--preview` exists to show what WILL run. The two paths reach the config
    through different doors (patch_preview_config vs the probe's own acquire
    callable), so they are pinned to one amendment."""
    exp = _experiment(backend, roster, readout_len_ns=500)
    assert (exp.patch_preview_config(_config_with())
            == exp._amend(_config_with()))


def test_the_probe_runs_one_target_at_a_time(backend, roster, monkeypatch):
    """Two resonators on one feedline share an input port: a multiplexed raw
    capture records both pulses superposed on one trace."""
    seen = {}
    monkeypatch.setattr(
        "scqo_qm.experiments.readout_time_of_flight.build_program",
        lambda machine, qubits, **kw: seen.update(
            batches=[list(b) for b in qubits.batch()]) or ("prog", {}))
    monkeypatch.setattr(backend.machine, "generate_config",
                        lambda: _config_with(), raising=False)

    exp = _experiment(backend, roster, readout_len_ns=500)
    exp.sweep_axes = exp.define_sweep()
    exp.probe()

    assert all(len(batch) == 1 for batch in seen["batches"])


def test_the_probe_hands_its_amended_config_to_its_own_acquire(backend, roster,
                                                               monkeypatch):
    """The 3-tuple form: the backend's shared fetch path would otherwise
    REGENERATE a config, throwing the amendment away — the same shape the
    parametric-drive shells use for their oscillator patch."""
    monkeypatch.setattr(
        "scqo_qm.experiments.readout_time_of_flight.build_program",
        lambda machine, qubits, **kw: ("prog", {"axes": 1}))
    monkeypatch.setattr(backend.machine, "generate_config",
                        lambda: _config_with(), raising=False)

    exp = _experiment(backend, roster, readout_len_ns=500)
    exp.sweep_axes = exp.define_sweep()
    res = exp.probe()

    assert len(res) == 3, "probe must return (prog, axes, acquire)"
    carried = res[2].keywords["config"]
    assert carried["elements"]["q1.resonator"]["time_of_flight"] == 28


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
