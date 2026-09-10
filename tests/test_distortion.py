"""The QM flux-distortion config-value wrapper (facts -> exponential_filter).

Pure unit tests: the SUM mapping is amplitudes-verbatim + tau s->ns; the CASCADE
path threads scqat's decomposition; apply_exponential_filter writes onto a stub
QUAM (duck-typed qubits[t].z.opx_output). No QOP / instrument needed.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from scqo_qm.backend._distortion import (
    apply_exponential_filter,
    clear_exponential_filter,
    to_exponential_filter,
    to_exponential_filter_cascade,
)


def _machine(existing=None, *, with_z=True):
    """A duck-typed QUAM: machine.qubits['q1'].z.opx_output.exponential_filter."""
    z = None
    if with_z:
        port = SimpleNamespace(exponential_filter=list(existing) if existing else [])
        z = SimpleNamespace(opx_output=port)
    return SimpleNamespace(qubits={"q1": SimpleNamespace(z=z)})


def test_sum_form_maps_tau_to_ns_and_amps_verbatim():
    ef = to_exponential_filter([0.05, -0.03], [100e-9, 3000e-9])
    assert ef == [[0.05, 100.0], [-0.03, 3000.0]]


def test_sum_length_mismatch_refused():
    with pytest.raises(ValueError, match="equal length"):
        to_exponential_filter([0.05], [1e-9, 2e-9])


def test_cascade_shape_and_scale_finite():
    out = to_exponential_filter_cascade([0.05, 0.02], [100e-9, 12e-9])
    ef = out["exponential_filter"]
    assert len(ef) == 2 and all(len(pair) == 2 for pair in ef)
    assert all(np.isfinite(a) for a, _ in ef)
    assert all(tau_ns > 0 for _, tau_ns in ef)  # tau in ns, positive
    assert np.isfinite(out["scale"])


def test_cascade_taus_are_nanoseconds():
    """A ~100 ns cascade tau lands as ~1e2 (ns), not ~1e-7 (s)."""
    out = to_exponential_filter_cascade([0.05], [100e-9])
    tau_ns = out["exponential_filter"][0][1]
    assert 1.0 < tau_ns < 1e5


def test_apply_replaces_by_default_and_maps_tau_to_ns():
    m = _machine(existing=[[0.9, 999.0]])  # a pre-existing tap to be overwritten
    out = apply_exponential_filter(m, "q1", [0.05, -0.03], [100e-9, 3000e-9])
    written = m.qubits["q1"].z.opx_output.exponential_filter
    assert written == [[0.05, 100.0], [-0.03, 3000.0]]  # tau s->ns, amps verbatim
    assert out == {"exponential_filter": written, "scale": 1.0}  # sum form => scale 1


def test_apply_extends_when_replace_false():
    m = _machine(existing=[[0.9, 999.0]])
    apply_exponential_filter(m, "q1", [0.05], [100e-9], replace=False)
    assert m.qubits["q1"].z.opx_output.exponential_filter == [
        [0.9, 999.0], [0.05, 100.0]]  # the old tap kept, the new one appended


def test_apply_does_not_persist():
    """The helper never calls machine.save() — the stub has no save, so a call that
    tried would AttributeError."""
    m = _machine()
    apply_exponential_filter(m, "q1", [0.05], [100e-9])
    assert not hasattr(m, "save")  # nothing added a save; caller owns persistence


def test_apply_unknown_target_refused_by_name():
    m = _machine()
    with pytest.raises(ValueError, match="q9"):
        apply_exponential_filter(m, "q9", [0.05], [100e-9])


def test_apply_target_without_flux_line_refused():
    m = _machine(with_z=False)
    with pytest.raises(ValueError, match="no flux"):
        apply_exponential_filter(m, "q1", [0.05], [100e-9])


def test_apply_cascade_sets_filter_and_returns_finite_scale():
    m = _machine()
    out = apply_exponential_filter(m, "q1", [0.05, 0.02], [100e-9, 12e-9],
                                   form="cascade")
    ef = m.qubits["q1"].z.opx_output.exponential_filter
    assert ef == out["exponential_filter"]
    assert len(ef) == 2 and all(tau_ns > 0 for _, tau_ns in ef)
    assert np.isfinite(out["scale"]) and out["scale"] > 0


def test_apply_cascade_cannot_extend():
    m = _machine(existing=[[0.9, 999.0]])
    with pytest.raises(ValueError, match="cascade"):
        apply_exponential_filter(m, "q1", [0.05], [100e-9], form="cascade",
                                 replace=False)


def test_apply_unknown_form_refused():
    m = _machine()
    with pytest.raises(ValueError, match="unknown form"):
        apply_exponential_filter(m, "q1", [0.05], [100e-9], form="fir")


def test_clear_removes_all_taps_and_reports_them():
    m = _machine(existing=[[0.9, 999.0], [0.1, 5.0]])
    out = clear_exponential_filter(m, "q1")
    assert m.qubits["q1"].z.opx_output.exponential_filter == []
    assert out["removed"] == [[0.9, 999.0], [0.1, 5.0]]


def test_clear_unknown_target_refused_by_name():
    with pytest.raises(ValueError, match="q9"):
        clear_exponential_filter(_machine(), "q9")


def test_clear_target_without_flux_line_refused():
    with pytest.raises(ValueError, match="no flux"):
        clear_exponential_filter(_machine(with_z=False), "q1")


def _opx_plus_machine():
    """A duck-typed QUAM whose z line is on an OPX+ analog output.

    ``OPXPlusAnalogOutputPort`` carries the LF base fields (feedforward/feedback
    filter, offset, delay) and none of the LF-FEM extras — no
    ``exponential_filter``, no ``output_mode``.
    """
    port = SimpleNamespace(feedforward_filter=None, feedback_filter=None,
                           offset=None, delay=0)
    return SimpleNamespace(qubits={"q1": SimpleNamespace(z=SimpleNamespace(opx_output=port))})


def test_apply_on_an_opx_plus_port_is_refused_rather_than_silently_dropped():
    """The write would otherwise SUCCEED and then vanish.

    quam does not object to inventing ``exponential_filter`` on a port class that
    has no such field: the assignment sticks to the instance, ``to_dict()`` omits
    it, and ``get_port_properties()`` never looks. The operator would read a
    printed success, save a state.json with no filter in it, and keep a distorted
    flux line. So the refusal is the correctness fix, not a nicety.
    """
    m = _opx_plus_machine()
    with pytest.raises(ValueError) as err:
        apply_exponential_filter(m, "q1", [0.1], [1e-6])
    message = str(err.value)
    assert "OPX+" in message
    assert "feedforward_filter" in message and "feedback_filter" in message
    assert not hasattr(m.qubits["q1"].z.opx_output, "exponential_filter")


def test_clear_on_an_opx_plus_port_is_refused_the_same_way():
    with pytest.raises(ValueError) as err:
        clear_exponential_filter(_opx_plus_machine(), "q1")
    assert "OPX+" in str(err.value)


def test_the_opx_plus_refusal_lands_before_the_cascade_is_computed():
    """Ordering matters: the decomposition is the expensive half and pulls in
    scqat. An unsupported port must be refused before any of it is spent."""
    with pytest.raises(ValueError, match="OPX\+"):
        apply_exponential_filter(
            _opx_plus_machine(), "q1", [0.1, 0.2], [1e-6, 2e-6], form="cascade")
