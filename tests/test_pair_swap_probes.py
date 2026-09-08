"""``pair_qq_chevron.resolve_amplitudes`` — the chevron probe's amplitude seam.

The chevron plays its flux pulse through TWO different QUA mechanisms depending
on the duration: below 17 ns a baked waveform scaled by ``amp_array``, above it a
stretched ``const`` play scaled by ``(base/const) * a``. Both must emit the same
volts, or the map has a step in it at 17 ns that no fit would flag. That equality
is pure arithmetic over the resolved amplitudes, so it is pinned here against
stub pairs — no QOP, no config, no baking.

``amp_mode="prefactor"`` is what the qualibrate node
(``calibrations/LCH_pair_qq_chevron.py``) uses and must stay byte-identical;
``amp_mode="absolute"`` is what scqo's ``pair_swap_chevron`` drives, where the
sweep values ARE the emitted volts.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scqo_qm.experiments._flux_limits import dac_rail_v
from scqo_qm.experiments.pair_swap_chevron import _flux_qubit, resolve_amplitudes

#: the stubs below expose no opx_output, so they get the conservative rail
_DAC_RAIL = dac_rail_v(None)


def _z(amplitude: float, name: str, output_mode: str | None = None) -> SimpleNamespace:
    port = None if output_mode is None else SimpleNamespace(output_mode=output_mode)
    return SimpleNamespace(name=name, opx_output=port,
                           operations={"const": SimpleNamespace(amplitude=amplitude)})


def _pair(name: str = "p1", *, ctrl_amp: float = 0.25, tgt_amp: float = 0.25,
          physics: bool = False, quad: float = -3e9,
          output_mode: str | None = None,
          z_names: tuple[str, str] | None = None) -> SimpleNamespace:
    """A QUAM ``qubit_pair`` stand-in carrying only what the resolver touches.

    ``physics=False`` OMITS ``anharmonicity`` / ``freq_vs_flux_01_quad_term`` /
    ``xy``: absolute mode must never read them (a bring-up tree may not carry
    them at all), so a leak shows up here as an AttributeError rather than as a
    surprise on the instrument.
    """
    cz, tz = z_names or (f"{name}_c_z", f"{name}_t_z")
    control = SimpleNamespace(name=f"{name}_c", z=_z(ctrl_amp, cz, output_mode))
    target = SimpleNamespace(name=f"{name}_t", z=_z(tgt_amp, tz, output_mode))
    if physics:
        # f_control > f_target so -detuning / quad is positive and the |11>-|02>
        # base amplitude is real
        control.xy = SimpleNamespace(RF_frequency=5.1e9)
        target.xy = SimpleNamespace(RF_frequency=4.8e9)
        target.anharmonicity = -0.2e9
        control.freq_vs_flux_01_quad_term = quad
    return SimpleNamespace(name=name, qubit_control=control, qubit_target=target)


def _emitted(base_level: float, qua_amps: np.ndarray, denom: float):
    """What each QUA branch actually puts on the DAC, per the probe's code.

    baked: the segments are baked at ``base_level`` and ``.run(amp_array=[(z, a)])``
    scales them by ``a``.  play: ``z.play("const", amplitude_scale=(base/denom)*a)``
    emits ``denom`` times that scale.
    """
    baked = base_level * qua_amps
    play = denom * ((base_level / denom) * qua_amps)
    return baked, play


# ------------------------------------------------------------------ absolute mode

def test_absolute_mode_emits_the_swept_volts_on_both_branches():
    amps = np.array([0.05, 0.1, 0.2])
    pairs = [_pair(ctrl_amp=0.25)]
    qua_amps, base_levels, denoms = resolve_amplitudes(
        pairs, amps, amp_mode="absolute", flux_role="control")

    baked, play = _emitted(base_levels["p1"], qua_amps, denoms["p1"])
    np.testing.assert_allclose(baked, amps)
    np.testing.assert_allclose(play, amps)
    # the reference is the sweep's own maximum, so |a| <= 1 by construction and
    # the baked waveform peak IS the largest volt the operator asked for
    assert base_levels["p1"] == pytest.approx(0.2)
    assert float(np.max(np.abs(qua_amps))) == pytest.approx(1.0)


def test_absolute_mode_never_reads_the_quam_physics_fields():
    """The |11>-|02> formula reads anharmonicity + freq_vs_flux_01_quad_term.
    Those are meaningless when the caller names volts, and a bring-up tree may
    not have them — so absolute mode must not consult them at all."""
    pairs = [_pair(physics=False)]  # the stub has neither attribute
    qua_amps, _, _ = resolve_amplitudes(
        pairs, np.array([0.1, 0.2]), amp_mode="absolute", flux_role="control")
    assert qua_amps.size == 2


def test_absolute_mode_uses_one_scalar_reference_for_every_pair():
    """``for_(*from_array(a, ...))`` is ONE loop shared by every multiplexed
    pair, so a per-pair reference is not expressible — every pair bakes at the
    same level even when their `const` amplitudes differ."""
    pairs = [_pair("p1", ctrl_amp=0.25), _pair("p2", ctrl_amp=0.1)]
    amps = np.array([0.05, 0.15])
    _, base_levels, denoms = resolve_amplitudes(
        pairs, amps, amp_mode="absolute", flux_role="control")
    assert base_levels["p1"] == base_levels["p2"] == pytest.approx(0.15)
    assert denoms["p1"] != denoms["p2"]


def test_absolute_sweep_above_the_dac_rail_is_refused():
    with pytest.raises(ValueError, match="full scale"):
        resolve_amplitudes([_pair()], np.array([0.1, 0.6]),
                           amp_mode="absolute", flux_role="control")


# ---------------------------------------------------------------- the rail itself

def test_the_rail_follows_the_port_output_mode():
    """The full scale is a property of the PORT: LF-FEM 'direct' reaches 0.5 V,
    'amplified' 2.5 V. The live chipA state runs every flux port amplified with
    op amplitudes at 1.25 V, so a hardcoded 0.5 would refuse the real chip."""
    assert dac_rail_v(_z(0.25, "z", "direct")) == 0.5
    assert dac_rail_v(_z(1.25, "z", "amplified")) == 2.5
    assert dac_rail_v(_z(0.25, "z", None)) == 0.5   # unknown port -> conservative
    assert dac_rail_v(None) == 0.5


def test_amplified_ports_accept_the_live_chip_amplitudes():
    """1.25 V op amplitudes and a 1 V sweep are legal on an amplified port and
    refused on a direct one — same numbers, different verdict."""
    amps = np.array([0.5, 1.0])
    amplified = [_pair(ctrl_amp=1.25, output_mode="amplified")]
    qua_amps, base_levels, denoms = resolve_amplitudes(
        amplified, amps, amp_mode="absolute", flux_role="control")
    np.testing.assert_allclose(base_levels["p1"] * qua_amps, amps)
    assert denoms["p1"] == pytest.approx(1.25)

    with pytest.raises(ValueError, match="full scale"):
        resolve_amplitudes([_pair(ctrl_amp=1.25, output_mode="direct")], amps,
                           amp_mode="absolute", flux_role="control")


def test_all_zero_absolute_sweep_is_refused():
    with pytest.raises(ValueError, match="no reference"):
        resolve_amplitudes([_pair()], np.zeros(3),
                           amp_mode="absolute", flux_role="control")


# ----------------------------------------------------------------- prefactor mode

def test_prefactor_mode_is_unchanged_from_the_qualibrate_path():
    """The node's own behaviour: `a` reaches QUA verbatim and the baked level is
    the QUAM |11>-|02> amplitude ``sqrt(-detuning / quad_term)``."""
    amps = np.arange(0.8, 1.2, 0.01)
    pairs = [_pair(physics=True)]
    qua_amps, base_levels, denoms = resolve_amplitudes(
        pairs, amps, amp_mode="prefactor", flux_role="control")

    detuning = 5.1e9 - 4.8e9 - (-0.2e9)
    expected = float(np.sqrt(-detuning / -3e9))
    assert base_levels["p1"] == pytest.approx(expected)
    np.testing.assert_allclose(qua_amps, amps)
    # and both branches still agree with each other
    baked, play = _emitted(base_levels["p1"], qua_amps, denoms["p1"])
    np.testing.assert_allclose(baked, play)


@pytest.mark.parametrize("quad,why", [(0.0, "unmeasured"), (3e9, "wrong sign")])
def test_prefactor_mode_refuses_an_unusable_base_level(quad, why):
    """The pre-factor sweep is defined RELATIVE to the |11>-|02> amplitude, so a
    chip whose freq_vs_flux_01_quad_term is 0/None (flux arch never measured —
    7 of the 9 live chipA pairs) has nothing for it to be a factor OF. The old
    code raised a bare ZeroDivisionError or baked a NaN waveform; say what is
    missing and name the way out instead."""
    with pytest.raises(ValueError, match="not a usable level"):
        resolve_amplitudes([_pair(physics=True, quad=quad)], np.array([1.0]),
                           amp_mode="prefactor", flux_role="control")


def test_absolute_mode_runs_where_prefactor_cannot():
    """...and that way out is absolute volts, which never reads the quad term."""
    pairs = [_pair(physics=True, quad=0.0)]
    _, base_levels, _ = resolve_amplitudes(pairs, np.array([0.1, 0.2]),
                                           amp_mode="absolute", flux_role="control")
    assert base_levels["p1"] == pytest.approx(0.2)


def test_one_pair_is_never_reported_as_colliding_with_itself():
    """Regression: the shared-flux-element guard compared each pair against the
    entry it had just inserted, and a NaN base level is != itself — so a single
    pair reported 'pairs X and X share the flux element'."""
    _, base_levels, _ = resolve_amplitudes([_pair(physics=True)], np.array([1.0]),
                                           amp_mode="prefactor", flux_role="control")
    assert set(base_levels) == {"p1"}


def test_prefactor_mode_warns_above_the_rail_rather_than_refusing():
    """The base level is computed from the CHIP's own detuning and a real chip
    can legitimately land above the rail. Raising would break the qualibrate
    node on a config that has always 'worked' — silently clipped — so warn."""
    pairs = [_pair(physics=True, quad=-1e9)]  # -> base 0.707 V, above the rail
    with pytest.warns(RuntimeWarning, match="full scale"):
        _, base_levels, _ = resolve_amplitudes(
            pairs, np.array([0.1, 0.2]), amp_mode="prefactor", flux_role="control")
    assert base_levels["p1"] > _DAC_RAIL


# ----------------------------------------------------------------------- guards

def test_const_op_above_the_dac_rail_is_refused_in_both_modes():
    for mode, amps in (("absolute", np.array([0.1])), ("prefactor", np.array([1.0]))):
        with pytest.raises(ValueError, match="full scale"):
            resolve_amplitudes([_pair(ctrl_amp=0.5, physics=True)], amps,
                               amp_mode=mode, flux_role="control")


def test_prefactor_sweep_outside_the_qua_amplitude_range_is_refused():
    with pytest.raises(ValueError, match="amplitude_scale range"):
        resolve_amplitudes([_pair(physics=True)], np.array([1.0, 2.5]),
                           amp_mode="prefactor", flux_role="control")


def test_play_branch_scale_outside_the_qua_range_is_refused():
    """The >16 ns branch scales `const` by base/const, which can blow the (-2, 2)
    fixed-point range even when the baked scale is fine."""
    with pytest.raises(ValueError, match="16 ns branch"):
        resolve_amplitudes([_pair(ctrl_amp=0.01, physics=True)], np.array([1.0]),
                           amp_mode="prefactor", flux_role="control")


def test_pairs_sharing_a_flux_element_at_different_levels_are_refused():
    """Baking registers `flux_pulse{i}` under the z ELEMENT's own name, so two
    pairs on one flux line at different base levels would silently overwrite
    each other's waveforms in the shared config."""
    shared = ("shared_z", "t_z")
    pairs = [_pair("p1", physics=True, quad=-3e9, z_names=shared),
             _pair("p2", physics=True, quad=-4e9, z_names=shared)]
    with pytest.raises(ValueError, match="share the flux element"):
        resolve_amplitudes(pairs, np.array([1.0]),
                           amp_mode="prefactor", flux_role="control")


def test_flux_role_selects_which_member_carries_the_pulse():
    qp = _pair(ctrl_amp=0.25, tgt_amp=0.1)
    assert _flux_qubit(qp, "control") is qp.qubit_control
    assert _flux_qubit(qp, "target") is qp.qubit_target
    _, _, denoms = resolve_amplitudes([qp], np.array([0.05]),
                                      amp_mode="absolute", flux_role="target")
    assert denoms["p1"] == pytest.approx(0.1)  # the TARGET's const op


def test_a_member_without_a_z_line_is_refused_by_name():
    qp = _pair()
    qp.qubit_target.z = None
    with pytest.raises(ValueError, match="has no z line"):
        resolve_amplitudes([qp], np.array([0.05]),
                           amp_mode="absolute", flux_role="target")


def test_a_missing_const_operation_lists_what_is_available():
    qp = _pair()
    qp.qubit_control.z.operations = {"flattop_cosine": SimpleNamespace(amplitude=0.2)}
    with pytest.raises(ValueError, match="flattop_cosine"):
        resolve_amplitudes([qp], np.array([0.05]),
                           amp_mode="absolute", flux_role="control")


@pytest.mark.parametrize("kwargs,match", [
    ({"amp_mode": "volts", "flux_role": "control"}, "amp_mode"),
    ({"amp_mode": "absolute", "flux_role": "high"}, "flux_role"),
])
def test_bad_mode_names_are_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        resolve_amplitudes([_pair()], np.array([0.1]), **kwargs)


def test_empty_sweep_is_refused():
    with pytest.raises(ValueError, match="empty"):
        resolve_amplitudes([_pair()], np.array([]),
                           amp_mode="absolute", flux_role="control")


# --------------------------------------------------------------------------
# resolve_coupler_plays — the OPTIONAL coupler pulse (the QCQ chevron)
#
# Setting coupler_flux_v turns this map into the phase-free QCQ survey: one
# pulse, so nothing accumulates a between-swap phase the way qc_n_swap_amp and
# pair_swap_angle do. It costs the baking branch, so every refusal below is
# about the two things that then have to hold — a stretchable (constant)
# coupler waveform, and a duration axis on the 4 ns clock.

from quam.components import pulses as _quam_pulses            # noqa: E402

from scqo_qm.components.pulses import FlatTopCosinePulse      # noqa: E402
from scqo_qm.experiments.pair_swap_chevron import resolve_coupler_plays  # noqa: E402

#: on the 4 ns grid and >= 16 ns, i.e. what scqo builds for a coupled run
GRID_TIMES = np.array([16, 20, 40, 100], dtype=int)


def _coupler(amplitude=0.1, *, name="p1_c", shaped=False):
    pulse = (FlatTopCosinePulse(length=40, amplitude=amplitude, edge_width=8)
             if shaped else _quam_pulses.SquarePulse(length=40, amplitude=amplitude))
    return SimpleNamespace(name=name, opx_output=None,
                           operations={"partial_swap_square": pulse})


def _coupled_pair(name="p1", *, coupler=None, macro="partial_swap",
                  flux_pulse="partial_swap_square", **kwargs):
    """``_pair`` plus the coupler and macro the coupler path resolves through."""
    qp = _pair(name, **kwargs)
    qp.coupler = _coupler() if coupler is None else coupler
    qp.macros = {macro: SimpleNamespace(flux_pulse=flux_pulse)}
    return qp


def test_no_coupler_amplitude_leaves_the_coupler_alone():
    """The falsy answer IS the historical chevron: a directly-coupled pair has
    no coupler to declare, and every run before this knob existed took it."""
    assert resolve_coupler_plays([_pair()], None, "partial_swap", GRID_TIMES) == {}


def test_the_scale_divides_by_the_macros_own_coupler_pulse():
    """The SAME conversion ISwapImplementation.apply does, so a coupler
    amplitude found here is the one qc_unidirectional_trotter's
    swap_coupler_flux wants — no unit translation in between."""
    plays = resolve_coupler_plays([_coupled_pair()], 0.05, "partial_swap", GRID_TIMES)
    coupler, pulse_name, scale = plays["p1"]
    assert pulse_name == "partial_swap_square"
    assert coupler.name == "p1_c"
    assert scale == pytest.approx(0.05 / 0.1)


@pytest.mark.parametrize("times,why", [
    (np.array([16, 18, 20]), "18"),        # off the 4 ns clock
    (np.array([8, 16, 20]), "8"),          # under the 16 ns floor
])
def test_a_duration_the_coupler_cannot_be_stretched_to_is_refused(times, why):
    with pytest.raises(ValueError) as err:
        resolve_coupler_plays([_coupled_pair()], 0.05, "partial_swap", times)
    assert why in str(err.value)
    assert "min_swap_time_ns" in str(err.value), "name the knob to lower"


def test_a_shaped_coupler_waveform_is_refused_by_name():
    """play(duration=) stretches a constant pulse and ZERO-PADS a shaped one, so
    a raised-cosine coupler would play its native 40 ns inside every window and
    the map would be of a pulse nobody asked for."""
    qp = _coupled_pair(coupler=_coupler(shaped=True))
    with pytest.raises(ValueError) as err:
        resolve_coupler_plays([qp], 0.05, "partial_swap", GRID_TIMES)
    assert "FlatTopCosinePulse" in str(err.value)
    assert "SquarePulse" in str(err.value)


def test_a_coupler_baked_at_zero_is_refused_here_too():
    """It is the DIVISOR of the volts->amplitude_scale conversion, so a zero is
    unsettable rather than merely weak — the shared _coupler_knob refusal."""
    qp = _coupled_pair(coupler=_coupler(0.0))
    with pytest.raises(ValueError) as err:
        resolve_coupler_plays([qp], 0.05, "partial_swap", GRID_TIMES)
    assert "register_flattop_cosine.py" in str(err.value)


def test_a_missing_macro_is_refused_by_name():
    qp = _coupled_pair(macro="iswap")
    with pytest.raises(ValueError) as err:
        resolve_coupler_plays([qp], 0.05, "partial_swap", GRID_TIMES)
    assert "no macro 'partial_swap'" in str(err.value)
    assert "iswap" in str(err.value), "say what the pair DOES carry"


def test_a_pair_with_no_coupler_is_refused_by_name():
    qp = _coupled_pair()
    qp.coupler = None
    with pytest.raises(ValueError) as err:
        resolve_coupler_plays([qp], 0.05, "partial_swap", GRID_TIMES)
    assert "no coupler" in str(err.value)


def test_volts_past_the_rail_are_refused():
    """The coupler pulse rides the standing bias, so the RELATIVE frame check
    is the right one — and it is the shared _flux_limits one, not a copy."""
    with pytest.raises(ValueError) as err:
        resolve_coupler_plays([_coupled_pair()], 5.0, "partial_swap", GRID_TIMES)
    assert str(_DAC_RAIL) in str(err.value) or "amplitude_scale" in str(err.value)
