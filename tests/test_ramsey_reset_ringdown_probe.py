"""Offline build proof + guard census for the Ramsey-after-the-reset QM probe.

Two halves, as in ``test_unidirectional_trotter_probe``:

* the GUARDS run before a single QUA statement is emitted, so they are pinned
  against plain stubs - no QUAM, no config, no QOP;
* the BUILD is rendered from the live ``quam_state``, because the program is
  made of the vendor's own macro (``q2.macros["reset"]``) and there is no honest
  stand-in for it. It skips by name when the state does not carry one.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scqo_qm.experiments.qubit_ramsey_reset_ringdown import build_program

STATE = str(Path(__file__).resolve().parents[1] / "quam_state")

ARMS = ("photon", "reference", "baseline")


# --------------------------------------------------------------- guard stubs


def _qubit(*, xy_ops=("x180", "y90", "x90"), x180_ns=16, macro="reset",
           tone_ns=100, z_ops=("parametric_reset",)):
    macros = {}
    if macro == "reset":
        macros["reset"] = SimpleNamespace(ringdown_ns=0, flux_pulse="parametric_reset")
    elif macro == "foreign":
        macros["reset"] = SimpleNamespace()  # some other macro under that name
    return SimpleNamespace(
        name="q9",
        macros=macros,
        xy=SimpleNamespace(operations={
            op: SimpleNamespace(length=x180_ns) for op in xy_ops}),
        z=SimpleNamespace(name="q9.z", operations={
            op: SimpleNamespace(length=tone_ns) for op in z_ops}),
    )


def _kwargs(qubit=None, **overrides):
    base = dict(
        machine=None,
        qubits=[qubit or _qubit()],
        idle_times_cycles=np.array([4, 6, 8]),
        arms=ARMS[:2],
        frames=np.linspace(0.0, 1.0, 4, endpoint=False),
        num_shots=10,
        reset_type="thermal",
        use_state_discrimination=True,
    )
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (_kwargs(_qubit(macro=None)), "has no macro 'reset'"),
        (_kwargs(reset_operation="dump"), "has no macro 'dump'"),
        # a macro that cannot be played without its wait is not this experiment's
        (_kwargs(_qubit(macro="foreign")), "not a parametric reset"),
        (_kwargs(_qubit(z_ops=())), "which q9.z does not carry"),
        (_kwargs(_qubit(xy_ops=("x180", "x90"))), "no xy operation 'y90'"),
        # an arm that leaves a pulse out has to WAIT its length, exactly
        (_kwargs(_qubit(x180_ns=18)), "q9's x180 is 18 ns long"),
        (_kwargs(_qubit(x180_ns=8)), "q9's x180 is 8 ns long"),
        (_kwargs(_qubit(tone_ns=102)), "q9's reset tone is 102 ns long"),
        (_kwargs(arms=("photon", "control")), r"unknown arm\(s\) \['control'\]"),
    ],
)
def test_guards_refuse_by_name_before_any_qua(kwargs, message):
    """Every refusal names the thing to fix, and fires with no machine at all -
    proof it happens before the builder touches the vendor tree."""
    with pytest.raises(ValueError, match=message):
        build_program(**kwargs)


# ------------------------------------------------------------ requires QUAM

quam_config = pytest.importorskip("quam_config")
pytest.importorskip("qm")

QUBIT = "q2"


@pytest.fixture(scope="module")
def live():
    """The live machine and a qubit carrying a parametric reset, or a skip
    naming what is absent."""
    machine = quam_config.Quam.load(STATE)
    qubit = machine.qubits.get(QUBIT)
    if qubit is None or "reset" not in qubit.macros:
        pytest.skip(f"live {QUBIT} has no 'reset' macro (register_reset_macro.py)")
    return machine


def _render(machine, **overrides):
    """(per-arm statement blocks, sweep axes, buffer lines) of the built program."""
    from qm import generate_qua_script
    from scqo_qm.experiments._lib import select_qubits

    kwargs = _kwargs(machine=machine, arms=ARMS,
                     qubits=select_qubits(machine, [QUBIT], multiplexed=False))
    kwargs["machine"] = machine
    kwargs.update(overrides)
    prog, axes = build_program(**kwargs)
    text = generate_qua_script(prog, machine.generate_config())
    body = text[:text.index("config = ")].splitlines()
    # one inner frame loop per arm, in play order
    starts = [i for i, ln in enumerate(body) if "for_each_" in ln][1:]
    ends = starts[1:] + [next(i for i, ln in enumerate(body) if "stream_processing" in ln)]
    blocks = [[ln.strip() for ln in body[a + 1:b]] for a, b in zip(starts, ends)]
    return blocks, axes, [ln.strip() for ln in body if ".buffer(" in ln]


def _between(block, first, last):
    """The statements from the one containing ``first`` to the one containing ``last``."""
    a = next(i for i, ln in enumerate(block) if first in ln)
    b = next(i for i, ln in enumerate(block) if last in ln)
    return block[a:b + 1]


def test_the_axes_are_idle_arm_frame_and_the_buffers_fold_them_inside_out(live):
    blocks, axes, buffers = _render(live)
    assert list(axes) == ["qubit", "idle_time_ns", "arm", "frame"]
    assert list(axes["idle_time_ns"].values) == [16.0, 24.0, 32.0]
    assert list(axes["arm"].values) == list(ARMS)
    assert len(blocks) == len(ARMS)
    # frame innermost, then arm, then idle: 4 frames, 3 arms, 3 idle times
    assert len(buffers) == 1
    assert '.buffer(4).buffer(3).buffer(3).average().save("state1")' in buffers[0]


def test_each_arm_differs_from_the_next_by_one_pulse(live):
    """photon -> reference drops the x180 alone; reference -> baseline drops the
    tone alone. Each missing pulse is a WAIT of its stored length on the same
    line, so the three arms share one timeline."""
    (photon, reference, baseline), _axes, _buffers = _render(live)
    tone_ns = live.qubits[QUBIT].z.operations["parametric_reset"].length
    x180_ns = live.qubits[QUBIT].xy.operations["x180"].length

    assert any('play("x180", "q2.xy")' in ln for ln in photon)
    for arm in (reference, baseline):
        assert not any('play("x180"' in ln for ln in arm)
        assert f'wait({x180_ns // 4}, "q2.xy")' in arm

    for arm in (photon, reference):
        assert any('play("parametric_reset", "q2.z")' in ln for ln in arm)
    assert not any("parametric_reset" in ln for ln in baseline)
    assert not any("update_frequency" in ln for ln in baseline)
    assert f'wait({tone_ns // 4}, "q2.z")' in baseline

    # the Ramsey pair itself is the same three statements in every arm
    pairs = [_between(arm, 'play("y90"', 'reset_frame') for arm in (photon, reference, baseline)]
    assert pairs[0] == pairs[1] == pairs[2]
    assert [ln.split("(")[0] for ln in pairs[0]] == [
        "play", "wait", "frame_rotation_2pi", "play", "reset_frame"]
    assert 'play("x90", "q2.xy")' in pairs[0][3]


def test_the_opening_pulse_follows_the_tone_with_no_wait_even_when_one_is_stored(live):
    """The run measures what the ring-down wait has to be, so a wait already
    stored with the reset must not sit in front of the opening pulse."""
    macro = live.qubits[QUBIT].macros["reset"]
    saved = macro.ringdown_ns
    try:
        macro.ringdown_ns = 32
        (photon, _reference, _baseline), _axes, _buffers = _render(live)
    finally:
        macro.ringdown_ns = saved
    span = _between(photon, 'play("parametric_reset"', 'play("y90"')
    # the tone, the oscillator handed back, one align of the qubit, the pulse
    assert [ln.split("(")[0] for ln in span] == ["play", "update_frequency", "align", "play"]
    assert not any(ln.startswith("wait(") for ln in span)


def test_two_arms_are_the_default_shape(live):
    blocks, axes, buffers = _render(live, arms=ARMS[:2])
    assert len(blocks) == 2 and list(axes["arm"].values) == ["photon", "reference"]
    assert ".buffer(4).buffer(2).buffer(3).average()" in buffers[0]


def test_raw_iq_streams_fold_the_same_way(live):
    _blocks, _axes, buffers = _render(live, use_state_discrimination=False)
    assert len(buffers) == 2
    assert all(".buffer(4).buffer(3).buffer(3).average()" in ln for ln in buffers)


def test_probe_maps_the_parameters_onto_the_builder(live):
    """QMQubitRamseyResetRingdown.probe() must build the program a direct call
    with the mapped arguments builds: the idle axis in clock cycles, the arms
    from the sweep (the baseline only when asked for), one target at a time."""
    from qm import generate_qua_script
    from scqo.roster import parse_components
    from scqo_qm.backend.qm_backend import QMBackend
    from scqo_qm.backend.roster_gen import roster_toml_for
    from scqo_qm.experiments._lib import select_qubits
    from scqo_qm.experiments.qubit_ramsey_reset_ringdown import QMQubitRamseyResetRingdown

    config = live.generate_config()

    def script(prog):  # drop the volatile "generated at <timestamp>" header line
        return "\n".join(ln for ln in generate_qua_script(prog, config).splitlines()
                         if "generated at" not in ln)

    backend = QMBackend(live, roster=parse_components(roster_toml_for(live)))
    exp = QMQubitRamseyResetRingdown(backend, QMQubitRamseyResetRingdown.Parameters(
        targets=[QUBIT], num_averages=7, max_idle_time_ns=120, num_points=6,
        include_baseline=True, use_state_discrimination=True))
    exp.sweep_axes = exp.define_sweep()
    prog, axes = exp.probe()

    idle_ns = exp.sweep_axes["idle_time_ns"]
    assert list(idle_ns) == [16.0, 36.0, 56.0, 76.0, 96.0, 116.0]
    direct, _ = build_program(
        live, select_qubits(live, [QUBIT], multiplexed=False),
        idle_times_cycles=(idle_ns / 4).astype(int), arms=ARMS,
        frames=np.linspace(0.0, 1.0, 4, endpoint=False), num_shots=7,
        reset_type="thermal", reset_operation="reset", use_state_discrimination=True)
    assert script(prog) == script(direct)
    assert list(axes["arm"].values) == list(ARMS)
    assert list(axes["idle_time_ns"].values) == list(idle_ns)
