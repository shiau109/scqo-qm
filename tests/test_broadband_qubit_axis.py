"""``broadband_qubit_spectroscopy``: the frequency PLAYED is the frequency the dataset
is LABELLED with, whatever IF the target carried into the run.

The probe steps the drive LO, parks the target's ``RF_frequency`` on it (IF = 0) and
hands ``qubit_spectroscopy``'s builder a detuning sweep, which the program plays as
``update_frequency(df + xy.intermediate_frequency)``. The IF in that sum is the one the
tree holds when the program is BUILT - after the move. The probe used to subtract the IF
the target had BEFORE the move, so the whole first sub-band was played ``stored IF`` away
from its label while the dataset kept the requested axis: a qubit known at 2.940851 GHz
(stored IF -59.149 MHz) was reported at 2.88139 GHz (SCQO BACKLOG I57). A target with a
placeholder drive frequency - the case this search exists for - got an arbitrary offset.

On a copy of the live ``quam_state/``, with the builder and the acquisition replaced by
recorders: no cluster, and nothing written to the folder.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

quam_config = pytest.importorskip("quam_config")
pytest.importorskip("qm")

from conftest import recording_device  # noqa: E402
from test_adopt_channel import _session, setup_dir  # noqa: E402,F401

SEARCH = {"start_freq_hz": 4.6e9, "stop_freq_hz": 5.0e9, "num_points_per_lo": 201}


def _target_with_a_stored_if(machine):
    """A live-tree qubit whose drive frequency is NOT its port's LO, and that IF."""
    from scqo_qm._mw_fem import port_lo

    for name, qubit in machine.qubits.items():
        stored_if = float(qubit.xy.RF_frequency) - port_lo(qubit.xy.opx_output)
        if abs(stored_if) > 1.0e6:
            return name, stored_if
    pytest.skip("every live drive frequency sits on its port's LO: nothing to test")


def _search(setup_dir, monkeypatch):
    """Run the probe with recorders in place of the builder and the acquisition.

    Returns ``(target, stored_if, segments, labelled, machine)``; each segment is
    ``(if_played, rf_played)`` as ``update_frequency(df + xy.intermediate_frequency)``
    puts them on the line, read when the program is built.
    """
    import scqo_qm.experiments.qubit_spectroscopy as spectroscopy
    from scqo_qm._mw_fem import port_lo
    from scqo_qm.experiments.broadband_qubit_spectroscopy import QMBroadbandQubitSpectroscopy

    session = _session(setup_dir)
    machine = session.backend.machine
    target, stored_if = _target_with_a_stored_if(machine)
    segments: list[tuple[np.ndarray, np.ndarray]] = []

    def build_program(machine, qubits, *, dfs, **_):
        xy = machine.qubits[target].xy
        if_played = np.asarray(dfs, float) + float(xy.intermediate_frequency)
        segments.append((if_played, port_lo(xy.opx_output) + if_played))
        return object(), {}

    def acquire(machine, prog, sweep_axes, *, num_shots, timeout):
        zeros = np.zeros(segments[-1][0].size)
        return xr.Dataset({"I1": ("detuning", zeros), "Q1": ("detuning", zeros)})

    monkeypatch.setattr(spectroscopy, "build_program", build_program)
    monkeypatch.setattr(spectroscopy, "acquire", acquire)

    exp = QMBroadbandQubitSpectroscopy(
        session.backend, QMBroadbandQubitSpectroscopy.Parameters(targets=[target], **SEARCH))
    exp.device = recording_device(session.backend, session.roster)
    exp.sweep_axes = exp.define_sweep()
    dataset = exp.probe()
    return target, stored_if, segments, np.asarray(dataset["frequency_hz"].values, float), machine


def test_the_played_frequency_is_the_labelled_one_in_every_sub_band(setup_dir, monkeypatch):
    target, stored_if, segments, labelled, _ = _search(setup_dir, monkeypatch)

    assert len(segments) == 2, "two sub-bands: the first one is where the offset was"
    for _, rf_played in segments:
        # each sub-band, against the stretch of the labelled axis it filled
        nearest = labelled[np.abs(labelled[:, None] - rf_played[None, :]).argmin(axis=0)]
        worst = np.abs(nearest - rf_played).max()
        assert worst < 1.0, (
            f"{target} carried a stored IF of {stored_if / 1e6:+.3f} MHz into the run and a "
            f"sub-band was played {worst / 1e6:.3f} MHz from the frequencies it is labelled with")
    everything = np.concatenate([rf for _, rf in segments])
    assert everything.min() == pytest.approx(SEARCH["start_freq_hz"], abs=1.0)
    assert everything.max() == pytest.approx(SEARCH["stop_freq_hz"], abs=1.0)


def test_the_played_if_stays_in_the_window_the_sub_bands_are_laid_out_for(setup_dir, monkeypatch):
    """+50 to +250 MHz above each sub-band's LO. The old offset moved it by the stored IF,
    for a placeholder drive frequency possibly through 0 and onto the other sideband."""
    _, _, segments, _, _ = _search(setup_dir, monkeypatch)

    for if_played, _ in segments:
        assert if_played.min() >= 50.0e6 - 1.0 and if_played.max() <= 250.0e6 + 1.0, (
            f"a sub-band was played at IF {if_played.min() / 1e6:.1f} to "
            f"{if_played.max() / 1e6:.1f} MHz")


def test_the_search_leaves_the_target_where_it_found_it(setup_dir, monkeypatch):
    from scqo_qm._mw_fem import port_lo

    target, stored_if, _, _, machine = _search(setup_dir, monkeypatch)
    xy = machine.qubits[target].xy
    assert float(xy.RF_frequency) - port_lo(xy.opx_output) == pytest.approx(stored_if, abs=1e-3)
    assert float(xy.intermediate_frequency) == pytest.approx(stored_if, abs=1e-3)
