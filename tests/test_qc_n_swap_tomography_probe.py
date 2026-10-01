"""qc_n_swap_tomography on QM: the generated QUA, the refusals, the preview, and
the probe's assembly of the fetched arrays into the experiment's contract.

Live-state tests load the repo-relative quam_state and skip when it has no pair
whose control carries a ``stark`` xy op and an iswap-style swap macro."""

from pathlib import Path

import numpy as np
import pytest

quam_config = pytest.importorskip("quam_config")
pytest.importorskip("qm")

from scqo_qm.backend.roster_gen import roster_toml_for  # noqa: E402
from scqo_qm.experiments import qc_n_swap_tomography as mod  # noqa: E402

STATE = str(Path(__file__).resolve().parents[1] / "quam_state")


@pytest.fixture(scope="module")
def machine():
    """The live tree plus ``stark_detuned`` (in memory - the repo state predates it)."""
    from quam.components.pulses import GaussianPulse
    from quam_config.register_stark_detuned import register_stark_detuned

    m = quam_config.Quam.load(STATE)
    # the repo state's `stark` is still the old square tone; 5Q4C's is this Gaussian
    for q in m.qubits.values():
        if q.xy is not None and "stark" in q.xy.operations:
            q.xy.operations["stark"] = GaussianPulse(length=60, sigma=10, amplitude=0.3,
                                                     axis_angle=0.0, subtracted=True)
    register_stark_detuned(m, 50e6)
    return m


def test_a_square_stark_has_no_envelope_to_copy():
    from quam.components.pulses import SquarePulse
    from quam_config.register_stark_detuned import register_stark_detuned

    m = quam_config.Quam.load(STATE)
    q = next(q for q in m.qubits.values() if q.xy is not None)
    q.xy.operations["stark"] = SquarePulse(length=60, amplitude=0.3, axis_angle=0.0)
    with pytest.raises(ValueError, match="GaussianPulse"):
        register_stark_detuned(m, 50e6)


@pytest.fixture(scope="module")
def live_roster(machine):
    from scqo.roster import parse_components

    return parse_components(roster_toml_for(machine))


def _live_choice(machine, live_roster):
    """(target, control role, swap op, vendor pair) of a usable live pair."""
    for _key, qp in machine.qubit_pairs.items():
        ctrl = qp.qubit_control
        if "stark" not in getattr(ctrl.xy, "operations", {}):
            continue
        target = f"{ctrl.name}_{qp.qubit_target.name}"
        ent = live_roster.entities.get(target)
        if ent is None:
            continue
        side = next((r for r in ("high", "low") if ctrl.name in ent.roles.get(r, ())), None)
        if side is None or (ctrl.name, "flux") not in live_roster.defaults:
            continue
        swap_op = next((n for n, m in (qp.macros or {}).items()
                        if isinstance(getattr(m, "flux_pulse", None), str)
                        and getattr(m, "flux_pulse") in ctrl.z.operations), None)
        if swap_op is not None:
            return target, side, swap_op, qp
    pytest.skip("no live pair with a 'stark' xy op + an iswap-style swap macro")


def _script(machine, qp, swap_op, *, calibration_shots=5, high_is_control=True):
    from qm import generate_qua_script

    prog, layout = mod.build_program(
        machine, qp, high_is_control=high_is_control, swap_operation=swap_op,
        stark_operation="stark_detuned", stark_detuning_hz=50e6, rounds_array=[0, 1, 2],
        stark_amps=[0.4, 0.5], bases=mod.BASIS_LABELS, prepared_states=mod.PREPARED_STATES,
        num_shots=3, calibration_shots=calibration_shots, reset_type="thermal",
        operation_gap_ns=20)
    return generate_qua_script(prog, machine.generate_config()), layout


def _pre_rotation_blocks(text: str) -> dict[int, list[tuple[str, str]]]:
    """The basis switch as rendered (an if_/elif_ chain on the basis index):
    ``{k: [(op, element), ...]}`` for the chain after the round's IF restore."""
    import re

    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if 'play("stark_detuned"' in l)
    blocks: dict[int, list[tuple[str, str]]] = {}
    current = None
    for line in lines[start + 1:]:
        head = re.search(r"with (?:if_|elif_)\(\(v\d+==(\d+)\)\)", line)
        if head:
            current = int(head.group(1))
            blocks[current] = []
            continue
        if current is None:
            continue
        play = re.search(r'play\("([^"]+)", "([^"]+)"\)', line)
        if play:
            blocks[current].append((play.group(1), play.group(2)))
        elif "measure(" in line:
            break
    return blocks


@pytest.mark.parametrize("high_is_control", [True, False])
def test_program_plays_the_round_then_the_pre_rotations(machine, live_roster, high_is_control):
    _target, _side, swap_op, qp = _live_choice(machine, live_roster)
    text, layout = _script(machine, qp, swap_op, high_is_control=high_is_control)
    ctrl, tgt = qp.qubit_control.name, qp.qubit_target.name
    assert layout["qubits"] == [ctrl, tgt]
    # the round (swaps with the baked-detuning stark tone) precedes the pre-rotations,
    # then the readout - and no frequency is switched anywhere
    assert "update_frequency(" not in text
    i_stark = text.index('play("stark_detuned"')
    assert i_stark < text.index("measure(", i_stark)
    # every basis plays exactly its two pre-rotations, first character = HIGH member
    high, low = (ctrl, tgt) if high_is_control else (tgt, ctrl)
    blocks = _pre_rotation_blocks(text)
    assert sorted(blocks) == list(range(len(mod.BASIS_LABELS)))
    for k, label in enumerate(mod.BASIS_LABELS):
        expected = [(mod.PRE_ROTATION[axis], f"{member}.xy")
                    for axis, member in zip(label, (high, low))
                    if mod.PRE_ROTATION[axis] is not None]
        assert blocks[k] == expected, label
    assert "cal_state1" in text and "cal_state2" in text
    assert "state1" in text and "state2" in text


def test_every_shot_re_references_both_frames(machine, live_roster):
    """The coherence between the members' frames is read, so every shot resets
    both IF phases and the global phase before the prep (BACKLOG I35)."""
    _target, _side, swap_op, qp = _live_choice(machine, live_roster)
    text, _layout = _script(machine, qp, swap_op, calibration_shots=0)
    shot = text[:text.index('play("stark_detuned"')]
    i_global = shot.rindex("reset_global_phase()")
    for q in (qp.qubit_control, qp.qubit_target):
        assert f'reset_if_phase("{q.xy.name}")' in shot[:i_global]
    assert i_global < shot.rindex('play("x180"')


def test_a_stark_without_baked_detuning_is_refused(machine, live_roster):
    _target, _side, swap_op, qp = _live_choice(machine, live_roster)
    with pytest.raises(ValueError, match="register_stark_detuned"):
        mod.build_program(
            machine, qp, high_is_control=True, swap_operation=swap_op, stark_operation="stark",
            stark_detuning_hz=50e6, rounds_array=[0, 1, 2, 3], stark_amps=[0.5],
            bases=mod.BASIS_LABELS, prepared_states=mod.PREPARED_STATES, num_shots=1,
            calibration_shots=0, reset_type="thermal")
    with pytest.raises(ValueError, match="detuning"):
        mod.build_program(
            machine, qp, high_is_control=True, swap_operation=swap_op,
            stark_operation="stark_detuned", stark_detuning_hz=40e6, rounds_array=[0, 1, 2, 3],
            stark_amps=[0.5], bases=mod.BASIS_LABELS, prepared_states=mod.PREPARED_STATES,
            num_shots=1, calibration_shots=0, reset_type="thermal")


def test_stark_detuned_is_the_stark_envelope_moved_up_by_the_detuning(machine):
    """Same envelope, times exp(+i 2 pi 50 MHz t): the tone update_frequency(+50 MHz)
    plays, without the switch."""
    for q in machine.qubits.values():
        if q.xy is None or "stark" not in q.xy.operations:
            continue
        plain = np.asarray(q.xy.operations["stark"].waveform_function(), dtype=complex)
        moved = np.asarray(q.xy.operations["stark_detuned"].waveform_function(), dtype=complex)
        t = np.arange(plain.size) * 1e-9
        np.testing.assert_allclose(moved, plain * np.exp(2j * np.pi * 50e6 * t), atol=1e-12)


def test_n_stark_amp_still_switches_and_never_resets(machine, live_roster):
    """The populations flavour of the shared round is unchanged."""
    from qm import generate_qua_script
    from scqo_qm.experiments import qc_n_stark_amp

    _target, _side, swap_op, qp = _live_choice(machine, live_roster)
    prog, _ = qc_n_stark_amp.build_program(
        machine, [qp.qubit_control, qp.qubit_target], qp, swap_operation=swap_op,
        stark_operation="stark", stark_detuning_hz=50e6, rounds_array=[0, 1, 2],
        stark_amps=[0.5], num_shots=2, reset_type="thermal", use_state_discrimination=True)
    text = generate_qua_script(prog, machine.generate_config())
    assert text.count("update_frequency(") == 2
    assert "reset_if_phase" not in text and "reset_global_phase" not in text


def test_no_calibration_block_without_shots(machine, live_roster):
    _target, _side, swap_op, qp = _live_choice(machine, live_roster)
    text, layout = _script(machine, qp, swap_op, calibration_shots=0)
    assert "cal_state" not in text and layout["calibration_shots"] == 0


def test_builder_refusals(machine, live_roster):
    _target, _side, swap_op, qp = _live_choice(machine, live_roster)
    kw = dict(high_is_control=True, swap_operation=swap_op, stark_operation="stark_detuned",
              stark_detuning_hz=50e6, rounds_array=[0, 1, 2, 3], stark_amps=[0.5],
              prepared_states=mod.PREPARED_STATES, num_shots=1, calibration_shots=0,
              reset_type="thermal")
    with pytest.raises(ValueError, match="basis label"):
        mod.build_program(machine, qp, bases=["zq"], **kw)
    with pytest.raises(ValueError, match="no macro"):
        mod.build_program(machine, qp, bases=mod.BASIS_LABELS,
                          **{**kw, "swap_operation": "no_such_swap"})
    with pytest.raises(ValueError, match="multiple of 4"):
        mod.build_program(machine, qp, bases=mod.BASIS_LABELS, operation_gap_ns=6, **kw)


def test_members_without_pre_rotations_are_refused():
    from types import SimpleNamespace

    bare = SimpleNamespace(name="qx", xy=SimpleNamespace(operations={"x180": 1}))
    with pytest.raises(ValueError, match="x90"):
        mod._check_members([bare])


def test_calibration_population_counts_joint_codes():
    # 4 shots x 4 prepared states; member levels (high, low)
    high = np.array([[0, 0, 1, 1]] * 3 + [[0, 1, 1, 0]])
    low = np.array([[0, 1, 0, 1]] * 3 + [[0, 1, 0, 1]])
    out = mod.calibration_population(high, low, mod.PREPARED_STATES, ["00", "01", "10", "11"])
    np.testing.assert_allclose(out.sum(axis=1), 1.0)
    np.testing.assert_allclose(out[0], [1.0, 0, 0, 0])
    np.testing.assert_allclose(out[1], [0, 0.75, 0, 0.25])     # one 01 shot read 11
    np.testing.assert_allclose(out[3], [0, 0.25, 0, 0.75])     # one 11 shot read 01


def _experiment(machine, live_roster, side, **params):
    from conftest import make_experiment
    from scqo_qm.backend.qm_backend import QMBackend

    target, _side, swap_op, _qp = _live_choice(machine, live_roster)
    backend = QMBackend(machine, roster=live_roster)
    cls = mod.QMQcNSwapTomography
    exp = make_experiment(cls, backend, live_roster, cls.Parameters(
        targets=[target], swap_operation=swap_op, drive_side=side, flux_side=side,
        swap_counts=[0, 1, 2, 3], stark_amps=[0.4, 0.5], num_averages=6,
        readout_calibration_shots=8, **params))
    exp.sweep_axes = exp.define_sweep()
    return exp, backend


@pytest.mark.parametrize("mode", ["average", "shot"])
def test_probe_assembles_a_conforming_dataset(machine, live_roster, monkeypatch, mode):
    _target, side, _op, _qp = _live_choice(machine, live_roster)
    exp, backend = _experiment(machine, live_roster, side, readout_mode=mode)
    rng = np.random.default_rng(0)

    def fake_acquire(machine_, prog, layout, *, timeout, log=None):
        shape = (layout["num_shots"], 2, 4, len(mod.BASIS_LABELS))
        cal = (layout["calibration_shots"], 4)
        return {"state": [rng.integers(0, 2, shape), rng.integers(0, 2, shape)],
                "cal": [rng.integers(0, 2, cal), rng.integers(0, 2, cal)]}

    monkeypatch.setattr(mod, "acquire", fake_acquire)
    raw = exp.probe()
    ds = backend._to_canonical(raw, exp)
    exp.Contract.validate(ds)
    assert ds["calibration_population"].dims == ("target", "prepared_state", "joint_state")
    np.testing.assert_allclose(ds["calibration_population"].sum("joint_state"), 1.0)
    if mode == "shot":
        assert list(ds["member"].values) == ["high", "low"]


def test_member_order_follows_the_roster_roles(machine, live_roster, monkeypatch):
    """The vendor reads (control, target); the dataset's members are (high, low)."""
    _target, side, _op, _qp = _live_choice(machine, live_roster)
    exp, _backend = _experiment(machine, live_roster, side, readout_mode="shot")
    shape_of = {}

    def fake_acquire(machine_, prog, layout, *, timeout, log=None):
        shape = (layout["num_shots"], 2, 4, len(mod.BASIS_LABELS))
        shape_of["n"] = shape
        return {"state": [np.ones(shape, int), np.zeros(shape, int)],
                "cal": [np.ones((8, 4), int), np.zeros((8, 4), int)]}

    monkeypatch.setattr(mod, "acquire", fake_acquire)
    ds = exp.probe()
    control_is_high = side == "high"
    expected_high = 1 if control_is_high else 0
    assert int(ds["state"].sel(member="high").max()) == expected_high


def test_single_target_preview_builds(machine, live_roster, tmp_path, monkeypatch):
    import socket

    _target, side, _op, _qp = _live_choice(machine, live_roster)
    exp, backend = _experiment(machine, live_roster, side)
    monkeypatch.setattr(socket, "create_connection",
                        lambda *a, **k: pytest.fail("no_simulate must never touch the network"))
    files = backend.preview(exp, tmp_path / "prev", no_simulate=True)
    text = files[0].read_text(encoding="utf-8")
    assert text.startswith("# scqo preview: qc_n_swap_tomography\n# backend: qm\n")
    assert "cal_state1" in text


def test_a_non_control_drive_side_is_refused(machine, live_roster):
    _target, side, _op, _qp = _live_choice(machine, live_roster)
    other = "low" if side == "high" else "high"
    exp, _backend = _experiment(machine, live_roster, other)
    with pytest.raises(ValueError, match="CONTROL"):
        exp.preview_program()
