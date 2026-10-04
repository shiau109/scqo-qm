"""``qubit_ramsey_flux_crosstalk_pulse``: two flux lines pulsed in one shot.

Two halves:

* the class's params -> builder mapping on the stub tree: the roster LINE named by
  ``source_line`` resolves to the QUAM element that line carries - another qubit's
  ``z`` or the pair's ``TunableCoupler`` - with the bias it stands at, and both
  windows reach the builder in the order asked;
* the program BUILT on the live ``quam_state`` and read back as QUA script: the
  target's own ``z`` and the source element each play ``const`` scaled by their own
  loop variable for the SAME idle, between the y90 and the x90, with the source loop
  outermost; a window past either line's rail is refused before any QUA is emitted;
  under a source lead the source pulse starts that many cycles before the own one
  and is longer by the same amount, so the two end together.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from conftest import make_experiment

STATE = str(Path(__file__).resolve().parents[1] / "quam_state")
NAME = "qubit_ramsey_flux_crosstalk_pulse"


# ------------------------------------------------------------------ the class, on stubs

def _experiment(backend, roster, **kw):
    from scqo_qm.experiments.qubit_ramsey_flux_crosstalk_pulse import (
        QMQubitRamseyFluxCrosstalkPulse as cls,
    )

    kw.setdefault("targets", ["q1"])
    exp = make_experiment(cls, backend, roster, cls.Parameters(**kw))
    exp.sweep_axes = exp.define_sweep()
    return exp


@pytest.fixture
def captured(monkeypatch):
    from scqo_qm.experiments import qubit_ramsey_flux_crosstalk_pulse as module

    seen = {}

    def fake_build(machine, qubits, **kw):
        seen.update(kw, qubits=qubits)
        return "prog", {}

    monkeypatch.setattr(module, "build_program", fake_build)
    return seen


def test_a_qubit_source_line_resolves_to_that_qubits_z(backend, roster, captured):
    backend.machine.qubits["q2"].z.joint_offset = 0.031
    exp = _experiment(backend, roster, source_line="z2")
    exp.probe()
    assert captured["source_z"] is backend.machine.qubits["q2"].z
    assert captured["source_idle_v"] == pytest.approx(0.031)   # the governed joint point
    assert captured["source_name"] == "z2"
    assert [q.name for q in captured["qubits"]] == ["q1"]


def test_a_coupler_source_line_resolves_to_the_tunable_coupler(backend, roster, captured):
    coupler = backend.machine.qubit_pairs["coupler_q1_q2"].coupler
    coupler.decouple_offset = 0.17
    _experiment(backend, roster, source_line="zc").probe()
    assert captured["source_z"] is coupler
    assert captured["source_idle_v"] == pytest.approx(0.17)    # the point it declares ("off")


def test_both_windows_reach_the_builder_in_the_order_asked(backend, roster, captured):
    exp = _experiment(backend, roster, source_line="z2",
                      start_flux_v=0.015, end_flux_v=-0.015, num_flux_points=5,
                      start_source_flux_v=0.04, end_source_flux_v=-0.08,
                      num_source_flux_points=4, num_idle_points=20, num_averages=77,
                      flux_buffer_ns=24)
    exp.probe()
    np.testing.assert_allclose(captured["flux_amps_v"], np.linspace(0.015, -0.015, 5))
    np.testing.assert_allclose(captured["source_amps_v"], np.linspace(0.04, -0.08, 4))
    np.testing.assert_array_equal(captured["idle_cycles"],
                                  np.asarray(exp.sweep_axes["idle_time_ns"]) / 4)
    assert captured["buffer_cycles"] == 6 and captured["num_shots"] == 77
    assert captured["source_lead_cycles"] == 0                 # together by default
    assert captured["reset_type"] == "thermal"
    assert captured["ramp_detuning_hz"] == {"q1": pytest.approx(-4.0e6)}


def test_the_source_lead_reaches_the_builder_in_clock_cycles(backend, roster, captured):
    _experiment(backend, roster, source_line="z2", source_lead_time_ns=1000).probe()
    assert captured["source_lead_cycles"] == 250


def test_active_reset_is_refused_by_name(backend, roster, captured):
    exp = _experiment(backend, roster, source_line="z2", reset_method="active")
    with pytest.raises(ValueError, match="active"):
        exp.probe()
    assert captured == {}                                      # refused before any build


def test_the_neutral_gate_refuses_the_targets_own_line(backend, roster):
    with pytest.raises(ValueError, match="OWN flux line"):
        _experiment(backend, roster, source_line="z1")


# ------------------------------------------------------------------ built on the live tree

quam_config = pytest.importorskip("quam_config")
pytest.importorskip("qm")


@pytest.fixture(scope="module")
def machine():
    return quam_config.Quam.load(STATE)


@pytest.fixture(scope="module")
def target(machine):
    if "q1" not in machine.qubits or "q2" not in machine.qubits:
        pytest.skip("live state has no q1/q2")
    return machine.qubits["q1"]


def _script(machine, prog) -> str:
    from qm import generate_qua_script

    return generate_qua_script(prog, machine.generate_config()).replace(chr(34), chr(39))


def _build(machine, target, source_z, *, flux=None, source=None, buffer_cycles=5,
           use_state=False, source_idle_v=0.0, lead_cycles=0):
    from scqo_qm.experiments._lib import select_qubits
    from scqo_qm.experiments.qubit_ramsey_flux_crosstalk_pulse import build_program

    return build_program(
        machine, select_qubits(machine, [target.name], multiplexed=True),
        flux_amps_v=np.linspace(-0.02, 0.02, 5) if flux is None else flux,
        source_amps_v=np.linspace(-0.05, 0.05, 3) if source is None else source,
        source_z=source_z, source_name="the-source", source_idle_v=source_idle_v,
        idle_cycles=np.arange(4, 44, 4), ramp_detuning_hz={target.name: -4e6},
        buffer_cycles=buffer_cycles, source_lead_cycles=lead_cycles, num_shots=10,
        reset_type="thermal", use_state_discrimination=use_state)


def _play(script, element):
    return [ln for ln in script.splitlines()
            if "play('const'" in ln and f"'{element}'" in ln]


def test_the_live_build_pulses_both_lines_for_the_same_idle(machine, target):
    source = machine.qubits["q2"].z
    prog, axes = _build(machine, target, source)
    script = _script(machine, prog)
    own, other = _play(script, target.z.name), _play(script, source.name)
    assert len(own) == 1 and len(other) == 1, (own, other)
    own_var = re.search(r"amp\(\((v\d+)/", own[0]).group(1)
    other_var = re.search(r"amp\(\((v\d+)/", other[0]).group(1)
    assert own_var != other_var                                # each line its own sweep
    duration = re.search(r"duration=(v\d+)", own[0]).group(1)
    assert f"duration={duration}" in other[0]                  # the same idle
    # between the two pi/2 pulses, and before the readout
    order = [script.index(f"play('y90', '{target.xy.name}')"), script.index(own[0]),
             script.index(other[0]), script.index(f"play('x90', '{target.xy.name}')"),
             script.index("measure(")]
    assert order == sorted(order)
    # the source loop is the OUTER one: its for_each_ opens first
    loops = re.findall(r"for_each_\(\((v\d+)\)", script)
    assert loops.index(other_var) < loops.index(own_var) < loops.index(duration)
    # raw nesting order under the canonical names
    assert list(axes) == ["qubit", "source_flux_v", "flux_bias_v", "idle_time_ns"]
    np.testing.assert_allclose(axes["source_flux_v"].values, np.linspace(-0.05, 0.05, 3))
    np.testing.assert_allclose(axes["idle_time_ns"].values, 4 * np.arange(4, 44, 4))


def test_a_coupler_element_is_pulsed_as_the_source(machine, target):
    pairs = [p for p in machine.qubit_pairs.values() if getattr(p, "coupler", None)]
    if not pairs:
        pytest.skip("live state has no tunable coupler")
    coupler = pairs[0].coupler
    prog, _ = _build(machine, target, coupler, use_state=True)
    script = _script(machine, prog)
    assert len(_play(script, coupler.name)) == 1
    assert len(_play(script, target.z.name)) == 1


def test_a_source_window_past_its_rail_is_refused(machine, target):
    source = machine.qubits["q2"].z
    with pytest.raises(ValueError, match="source line the-source.*full scale"):
        _build(machine, target, source, source=np.linspace(-0.3, 0.3, 3), source_idle_v=0.3)


def test_an_own_window_past_its_rail_is_refused(machine, target):
    with pytest.raises(ValueError, match="full scale|amplitude_scale"):
        _build(machine, target, machine.qubits["q2"].z, flux=np.linspace(-0.6, 0.6, 5))


def test_the_targets_own_z_is_refused_as_a_source(machine, target):
    with pytest.raises(ValueError, match="its own z line"):
        _build(machine, target, target.z)


def test_the_buffer_waits_on_both_lines_and_zero_buffer_plays_without_one(machine, target):
    # the source element sits in no reset wait, so a wait naming it IS the buffer
    source = machine.qubits["q2"].z
    buffered = rf"wait\(5, [^)]*'{re.escape(source.name)}'"
    with_buffer = _script(machine, _build(machine, target, source)[0])
    assert len(re.findall(buffered, with_buffer)) == 2         # before and after the pulses
    without = _script(machine, _build(machine, target, source, buffer_cycles=0)[0])
    assert not re.search(rf"wait\(\d+, [^)]*'{re.escape(source.name)}'", without)


def _timeline(script, target, source):
    """(source start, own start, source duration - own duration) in clock cycles,
    counted from the align() after the reset, read off the generated QUA."""
    own, other = _play(script, target.z.name)[0], _play(script, source.name)[0]
    extra = re.search(r"duration=\((v\d+)\+(\d+)\)", other)
    assert extra and f"duration={extra.group(1)}" in own       # the idle, plus the lead
    body = script[script.index("reset_frame("):script.index(own)]
    source_wait = int(re.search(rf"wait\((\d+), '{re.escape(source.name)}'\)", body).group(1))
    own_waits = [int(n) for n in re.findall(
        rf"wait\((\d+), [^)]*'{re.escape(target.z.name)}'", body)]
    pi2 = int(target.xy.operations["y90"].length) // 4
    return source_wait, sum(own_waits) + pi2, int(extra.group(2))


@pytest.mark.parametrize("lead", [4, 9, 250, 2_500_000])
def test_a_source_lead_starts_the_source_early_and_ends_it_with_the_own_pulse(
        machine, target, lead):
    """Shorter than the y90 + buffer the SOURCE waits, longer the TARGET does; either
    way the source is on `lead` cycles before the own pulse and `lead` cycles longer."""
    source = machine.qubits["q2"].z
    script = _script(machine, _build(machine, target, source, lead_cycles=lead)[0])
    source_start, own_start, longer = _timeline(script, target, source)
    assert own_start - source_start == lead
    assert longer == lead                                      # so they end together
    assert min(source_start, own_start) >= 4                   # every wait is playable
    # the source keeps its own clock: nothing names it between its pulse and the readout
    source_play = _play(script, source.name)[0]
    after = script[script.index(source_play) + len(source_play):script.index("measure(")]
    assert f"'{source.name}'" not in after
    # ... and the pulse is issued before the first pi/2 pulse
    assert script.index(source_play) < script.index(f"play('y90', '{target.xy.name}')")


def test_without_a_lead_the_program_is_the_one_validated_on_hardware(machine, target):
    source = machine.qubits["q2"].z
    plain = _script(machine, _build(machine, target, source)[0])
    own, other = _play(plain, target.z.name)[0], _play(plain, source.name)[0]
    assert "duration=(" not in other                           # the bare idle, no lead added
    # the source pulse follows the own one, after the y90 and the shared buffer
    assert plain.index(f"play('y90', '{target.xy.name}')") < plain.index(own) < plain.index(other)
    assert not re.search(rf"wait\(\d+, '{re.escape(source.name)}'\)", plain)


def test_a_negative_lead_is_refused(machine, target):
    with pytest.raises(ValueError, match="cannot be negative"):
        _build(machine, target, machine.qubits["q2"].z, lead_cycles=-4)
