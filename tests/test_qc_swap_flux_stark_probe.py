"""Offline build proof + guard census for the fixed-N flux x stark QM probe.

Two halves, deliberately:

* the GUARDS run before a single QUA statement is emitted, so they are pinned
  against plain stubs -- no QUAM, no config, no QOP. This probe sweeps BOTH a
  flux window and a stark amplitude factor, so it is the only one that has to
  carry both families of refusal at once, and that is what these pin;
* the BUILD is rendered from the live ``quam_state``, because a QUA program is
  made out of the vendor's own macros and there is no honest stand-in for
  ``pair.macros["iswap"].apply(ctrl_scale=...)``. The committed state carries no
  ``stark`` xy op (it is an operator action via quam_config/register_stark.py),
  so the fixture registers one IN MEMORY and removes it again -- ``Quam.load``
  returns a shared tree.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import make_experiment

AMPS = np.linspace(0.0, 0.05, 5)
STARKS = np.linspace(0.0, 1.0, 5)
SWAP = "iswap"
FLUX_PULSE = "swap_flattop"


# ------------------------------------------------------------------ guard stubs

def _z(amplitude: float = 0.25) -> SimpleNamespace:
    """A flux line carrying only what the flux guard reads: no ``opx_output``
    (so the conservative rail applies) and no ``flux_point`` (idle 0.0)."""
    return SimpleNamespace(
        name="p1_c_z", opx_output=None,
        operations={FLUX_PULSE: SimpleNamespace(amplitude=amplitude)})


def _pair(*, macros=(SWAP,), flux_pulse: str = FLUX_PULSE,
          xy_ops=("x180", "stark"), z_amplitude: float = 0.25) -> SimpleNamespace:
    control = SimpleNamespace(
        name="p1_c", z=_z(z_amplitude),
        xy=SimpleNamespace(operations={op: SimpleNamespace() for op in xy_ops},
                           intermediate_frequency=100_000_000))
    target = SimpleNamespace(name="p1_t")
    return SimpleNamespace(
        name="p1", qubit_control=control, qubit_target=target,
        macros={m: SimpleNamespace(flux_pulse=flux_pulse) for m in macros})


def _build(pair, **overrides):
    """Call the builder with a None machine: every guard fires before the
    ``with program()`` block, so nothing vendor-side is ever touched."""
    from scqo_qm.experiments.qc_swap_flux_stark import build_program

    kwargs = dict(
        swap_operation=SWAP, stark_operation="stark", stark_detuning_hz=50e6,
        swap_count=4, qubit_amplitudes=AMPS, stark_amps=STARKS, num_shots=10,
        reset_type="thermal", use_state_discrimination=True,
    )
    kwargs.update(overrides)
    return build_program(None, [pair.qubit_control, pair.qubit_target], pair, **kwargs)


def test_refuses_a_missing_swap_macro():
    with pytest.raises(ValueError, match="no macro 'iswap'"):
        _build(_pair(macros=()))


def test_refuses_a_macro_whose_flux_pulse_is_not_on_the_control_z():
    with pytest.raises(ValueError, match="no z flux_pulse"):
        _build(_pair(flux_pulse="not_registered"))


def test_refuses_a_flux_window_qua_cannot_express():
    """The macro's stored z amplitude is the divisor of the volts ->
    amplitude_scale conversion, so a window past 2x it cannot be built."""
    with pytest.raises(ValueError, match="amplitude_scale"):
        _build(_pair(z_amplitude=0.01), qubit_amplitudes=np.linspace(0.0, 0.5, 5))


def test_refuses_a_missing_stark_op_and_names_the_register_script():
    with pytest.raises(ValueError, match="register_stark.py"):
        _build(_pair(xy_ops=("x180",)))


def test_refuses_a_stark_factor_outside_the_qua_amplitude_range():
    with pytest.raises(ValueError, match="max_stark_amp"):
        _build(_pair(), stark_amps=np.array([0.0, 2.5]))


def test_refuses_a_gap_off_the_four_nanosecond_clock():
    with pytest.raises(ValueError, match="multiple of 4 ns"):
        _build(_pair(), operation_gap_ns=6)


def test_refuses_a_negative_swap_count():
    with pytest.raises(ValueError, match="swap_count"):
        _build(_pair(), swap_count=-1)


# ------------------------------------------------------------------ role refusal

def test_refuses_non_control_roles(backend, roster):
    """The probe excites the vendor CONTROL member, plays the Stark tone on its
    xy line and sweeps the swap on its flux line, so a role selection resolving
    elsewhere is refused BY NAME before any QUA is built. The stub pair is
    control=q1 while the roster's high=q2, so drive_side='high' resolves to the
    vendor target -- the refusing case."""
    from scqo_qm.experiments.qc_swap_flux_stark import QMQcSwapFluxStark

    exp = make_experiment(QMQcSwapFluxStark, backend, roster,
                          QMQcSwapFluxStark.Parameters(targets=["q1_q2"],
                                                       drive_side="high",
                                                       flux_side="high"))
    exp.sweep_axes = exp.define_sweep()
    with pytest.raises(ValueError, match="CONTROL member"):
        exp.probe()


# ------------------------------------------------------------------ member reshape

def test_member_states_orders_the_axes_by_role():
    """The probe reads out [control, target]; the schema wants (high, low), and
    the two swept amplitudes carry scqo's own coordinate names."""
    import xarray as xr

    from scqo_qm.experiments.qc_swap_flux_stark import _member_states

    raw = xr.DataArray(
        np.arange(2 * 3 * 2 * 4).reshape(2, 3, 2, 4),
        dims=("qubit", "shot", "qubit_amplitude", "stark_amplitude"))
    out = _member_states(raw, "target")
    assert out.dims == ("member", "shot_idx", "flux_amp_v", "stark_amp")
    assert list(out["member"].values) == ["high", "low"]
    # high = the vendor TARGET here, i.e. the probe's second readout row
    np.testing.assert_array_equal(out.sel(member="high").values, raw.isel(qubit=1).values)


# ------------------------------------------------------------------ live build

quam_config = pytest.importorskip("quam_config")
pytest.importorskip("qm")

STATE = str(Path(__file__).resolve().parents[1] / "quam_state")


def _drop_stark(xy):
    if "stark" in xy.operations:
        del xy.operations["stark"]


@pytest.fixture
def live_pair_with_stark():
    """A live machine whose first swap-capable pair's CONTROL qubit carries a
    ``stark`` SquarePulse registered in memory; removed on teardown."""
    from quam.components.pulses import SquarePulse

    machine = quam_config.Quam.load(STATE)
    choice = None
    for _key, qp in machine.qubit_pairs.items():
        ctrl = qp.qubit_control
        swap_op = next((n for n, m in (qp.macros or {}).items()
                        if isinstance(getattr(m, "flux_pulse", None), str)
                        and getattr(m, "flux_pulse") in ctrl.z.operations), None)
        if swap_op is not None and getattr(ctrl, "xy", None) is not None:
            choice = (qp, swap_op)
            break
    if choice is not None:
        ctrl = choice[0].qubit_control
        _drop_stark(ctrl.xy)
        ctrl.xy.operations["stark"] = SquarePulse(length=64, amplitude=0.1, axis_angle=0.0)
    yield machine, choice
    if choice is not None:
        _drop_stark(choice[0].qubit_control.xy)


def test_build_program_renders_the_swap_plus_stark_sequence(live_pair_with_stark):
    from qm import generate_qua_script

    from scqo_qm.experiments.qc_swap_flux_stark import build_program

    machine, choice = live_pair_with_stark
    if choice is None:
        pytest.skip("no live pair with an iswap-style swap macro and an xy line")
    qp, swap_op = choice

    prog, axes = build_program(
        machine, [qp.qubit_control, qp.qubit_target], qp,
        swap_operation=swap_op, stark_operation="stark", stark_detuning_hz=50e6,
        swap_count=3, qubit_amplitudes=AMPS, stark_amps=STARKS,
        num_shots=10, reset_type="thermal", use_state_discrimination=True,
    )
    # The swap count is NOT an axis: only the two amplitudes are swept.
    assert list(axes) == ["qubit", "shot", "qubit_amplitude", "stark_amplitude"]

    text = generate_qua_script(prog, machine.generate_config())
    assert "update_frequency" in text            # the off-resonant detune/restore bracket
    stark_lines = [ln for ln in text.splitlines() if 'play("stark"' in ln]
    assert stark_lines, "stark tone is not played"
    # The tone plays at its natural length -- a duration override would zero-pad
    # the arbitrary waveform instead of stretching it.
    assert all("duration" not in ln for ln in stark_lines), \
        f"stark must play at its natural length, got: {stark_lines}"


def test_single_target_preview_builds(live_pair_with_stark, tmp_path, monkeypatch):
    """The self-acquiring shell is previewable with EXACTLY ONE --target: the
    single program it would build is rendered, no acquire, no network. Builds its
    own live roster rather than borrowing another module's fixture, because the
    stark op only exists inside this file's fixture."""
    import socket

    from scqo.roster import parse_components
    from scqo_qm.backend.qm_backend import QMBackend
    from scqo_qm.backend.roster_gen import roster_toml_for
    from scqo_qm.experiments.qc_swap_flux_stark import QMQcSwapFluxStark

    machine, choice = live_pair_with_stark
    if choice is None:
        pytest.skip("no live pair with an iswap-style swap macro and an xy line")
    qp, swap_op = choice

    live_roster = parse_components(roster_toml_for(machine))
    ctrl = qp.qubit_control
    target = f"{ctrl.name}_{qp.qubit_target.name}"
    ent = live_roster.entities.get(target)
    if ent is None or (ctrl.name, "flux") not in live_roster.defaults:
        pytest.skip(f"{target} is not a roster pair with a control flux channel")
    side = next((r for r in ("high", "low") if ctrl.name in ent.roles.get(r, ())), None)
    if side is None:
        pytest.skip(f"{ctrl.name} carries no roster role on {target}")

    monkeypatch.setattr(socket, "create_connection",
                        lambda *a, **k: pytest.fail(
                            "no_simulate must never touch the network"))
    backend = QMBackend(machine, roster=live_roster)
    exp = make_experiment(
        QMQcSwapFluxStark, backend, live_roster,
        QMQcSwapFluxStark.Parameters(targets=[target], swap_operation=swap_op,
                                     drive_side=side, flux_side=side,
                                     min_flux_amp_v=0.0, max_flux_amp_v=0.05,
                                     num_flux_points=5, num_stark_points=5,
                                     swap_count=2, num_averages=10))
    exp.sweep_axes = exp.define_sweep()
    out_dir = tmp_path / "prev"
    files = backend.preview(exp, out_dir, no_simulate=True)
    assert files == [out_dir / "qua_script.py"]
    text = files[0].read_text(encoding="utf-8")
    assert text.startswith("# scqo preview: qc_swap_flux_stark\n# backend: qm\n")
    assert len(text.splitlines()) > 20  # a real program body, not just header
