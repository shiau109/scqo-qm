"""Flux amplitude validation: what volts a port may emit, in either frame.

Two frames, mirroring scqo's two flux capability mixins:

* ABSOLUTE (``set_dc_offset``) — the swept value IS the line voltage. It REPLACES
  the standing bias, so no idle term is added.
* RELATIVE (``play("const" * amp(v / reference))``) — the swept value is an
  excursion the DAC ADDS to the standing bias.

In the relative frame the stored reference amplitude is the divisor and QUA bounds
the result to (-2, 2). Four ways this goes wrong, each SILENT on hardware:

1. the ``const`` amplitude breaks the ``rail/2`` convention — too small caps the
   reachable window with no error, too large clips inside the DAC;
2. the requested excursion needs ``|amplitude_scale| >= 2``;
3. ``idle + excursion`` exceeds the rail — the DAC emits the SUM;
4. the frames get confused — adding ``idle_v`` in the absolute frame would refuse
   legal sweeps, omitting it in the relative frame would admit clipping ones.

The QM simulator does not show clipping, so every one of these has to be a
build-time refusal.
"""

import pytest

from scqo_qm.experiments._flux_limits import (
    check_flux_bias_absolute,
    check_flux_pulse_relative,
    dac_rail_v,
    declared_idle_offset_v,
    flux_reference_amplitude,
    idle_offset_v,
    rail_remedy,
)

from conftest import _coupler, _flux_line


def _opx_plus_z(**offsets):
    """A flux line on an OPX+ analog output.

    ``OPXPlusAnalogOutputPort`` carries the LF base fields and none of the FEM
    ones — no ``output_mode``, no ``exponential_filter`` — which is exactly what
    makes it indistinguishable from "no port at all" to a check that only looks
    for ``output_mode``.
    """
    z = _flux_line(**offsets)
    z.opx_output = type(
        "OPXPlusPort", (), {"feedforward_filter": None, "feedback_filter": None,
                            "delay": 0, "offset": None})()
    return z


def _z(amplitude, *, mode="direct", operation="const", **offsets):
    """A flux line carrying one op on a port of the given output mode."""
    z = _flux_line(**offsets)
    z.opx_output = type("Port", (), {"output_mode": mode})()
    z.operations = {operation: type("Pulse", (), {"amplitude": amplitude})()}
    return z


# ------------------------------------------------------------- the reference

def test_the_rail_is_read_from_the_PORT_not_a_constant():
    """1.25 V is a legal stored amplitude on an amplified port and past the rail
    on a direct one — the same number, accepted or refused by the port's mode.
    Hardcoding 0.5 would refuse the whole amplified flux config."""
    assert dac_rail_v(_z(1.25, mode="amplified")) == 2.5
    assert flux_reference_amplitude(_z(1.25, mode="amplified"), name="q1") == 1.25
    with pytest.raises(ValueError, match="full scale"):
        flux_reference_amplitude(_z(1.25), name="q1")  # direct rail is 0.5


def test_a_stored_op_at_the_rail_is_refused():
    """At full scale the stored waveform peak is already clipped, before any
    amplitude_scale is applied to it."""
    with pytest.raises(ValueError, match="full scale"):
        flux_reference_amplitude(_z(0.5), name="q2")


def test_the_rail_2_CONVENTION_is_not_enforced_here():
    """An undersized `const` does not clip anything — it only caps how far the
    sweep can reach, and the amplitude_scale bound refuses the moment a probe
    actually asks for more. Enforcing it here would refuse real working configs:
    the live 5Q4C couplers sit at 0.15 V and the chevron's `const` is a free
    parameter. The convention is audited tree-wide by flux_headroom_problems —
    see test_flux_headroom_guard.py."""
    assert flux_reference_amplitude(_z(0.15), name="c12") == 0.15
    assert flux_reference_amplitude(_z(0.01), name="p1_c") == 0.01


def test_a_missing_or_zero_const_is_refused_by_name():
    z = _z(0.25)
    z.operations = {}
    with pytest.raises(ValueError, match="no 'const' operation"):
        flux_reference_amplitude(z, name="q2")
    with pytest.raises(ValueError, match="cannot be zero"):
        flux_reference_amplitude(_z(0.0), name="q2")


def test_a_non_const_op_is_held_to_the_RAIL_but_not_the_convention():
    """A macro's flux pulse (``partial_swap_flattop_cosine`` at 0.152 V on the
    live tree) is a calibrated shape, not the shared workhorse — it legitimately
    sits anywhere below the rail. Applying the rail/2 rule to it, as the single
    old ``const_amp_reference`` would have, refuses every swap probe."""
    z = _z(0.152, operation="partial_swap_flattop_cosine")
    assert flux_reference_amplitude(
        z, name="q1", operation="partial_swap_flattop_cosine") == 0.152
    # ... but the rail still bites
    over = _z(0.6, operation="partial_swap_flattop_cosine")
    with pytest.raises(ValueError, match="full scale"):
        flux_reference_amplitude(over, name="q1", operation="partial_swap_flattop_cosine")


# ----------------------------------------------------------------- the window

def test_a_normal_window_passes_and_returns_the_reference():
    """The real 5Q4C q2 case: 0.107 V idle, +/-0.05 V window."""
    z = _z(0.25, joint=0.10695)
    reference = check_flux_pulse_relative(
        z, name="q2", idle_v=0.10695, amps_v=[-0.05, 0.0, 0.05])
    assert reference == 0.25


def test_an_inexpressible_excursion_is_refused():
    """|v|/reference >= 2 is outside QUA's fixed-point multiplier. With the
    convention that is exactly 'you asked for more than the rail'."""
    z = _z(0.25)
    with pytest.raises(ValueError, match="amplitude_scale"):
        check_flux_pulse_relative(z, name="q2", idle_v=0.0, amps_v=[-0.5, 0.5])


def test_the_combined_rail_is_what_actually_bites():
    """A window that is fine on its own still clips once it rides on a standing
    bias, because the DAC emits the SUM. The simulator shows nothing when it
    happens.

    q2's real numbers: 0.0938 V idle + a +/-0.45 V window reaches 0.544 V, past
    the 0.5 V direct rail.
    """
    z = _z(0.25, joint=0.0938)
    # the window alone is expressible (0.45/0.25 = 1.8 < 2) ...
    check_flux_pulse_relative(z, name="q2", idle_v=0.0, amps_v=[-0.45, 0.45])
    # ... but not once it rides on the idle bias
    with pytest.raises(ValueError, match="full scale") as exc:
        check_flux_pulse_relative(z, name="q2", idle_v=0.0938, amps_v=[-0.45, 0.45])
    assert "SIMULATOR" in str(exc.value)  # the reason this must refuse, not warn


def test_the_combined_rail_check_is_asymmetric():
    """A positive idle bias eats headroom on the positive side only, so the
    check must test both ends rather than the window's magnitude."""
    z = _z(0.25, joint=0.3)
    with pytest.raises(ValueError, match="full scale"):
        check_flux_pulse_relative(z, name="q2", idle_v=0.3, amps_v=[-0.4, 0.4])
    # the same magnitude one-sided downward stays inside the rail
    check_flux_pulse_relative(z, name="q2", idle_v=0.3, amps_v=[-0.4, 0.0])


# ------------------------------------------------------------- the two frames

def test_the_absolute_frame_does_NOT_add_the_idle_bias():
    """``set_dc_offset`` REPLACES the standing bias — it does not ride on it. A
    +/-0.45 V absolute sweep on a line whose idle is 0.0938 V is legal, and is
    exactly the case the relative check above refuses. Sharing one checker
    between the frames would silently refuse every legal resonator flux map."""
    z = _z(0.25, joint=0.0938)
    check_flux_bias_absolute(z, name="q2", bias_v=[-0.45, 0.45])
    with pytest.raises(ValueError, match="full scale"):
        check_flux_pulse_relative(z, name="q2", idle_v=0.0938, amps_v=[-0.45, 0.45])


def test_the_absolute_frame_needs_no_const_op_at_all():
    """A DC bias involves no waveform and no amplitude_scale, so a line with no
    ``const`` is still perfectly able to hold one."""
    z = _flux_line()
    z.opx_output = type("Port", (), {"output_mode": "direct"})()
    z.operations = {}
    check_flux_bias_absolute(z, name="q2", bias_v=[-0.3, 0.3])


def test_the_absolute_frame_still_refuses_past_the_rail():
    z = _z(0.25)
    with pytest.raises(ValueError, match="full scale"):
        check_flux_bias_absolute(z, name="q2", bias_v=[-0.6, 0.6])


# ----------------------------------------------------------------- the remedy

def test_a_direct_port_is_told_to_switch_to_amplified():
    """The whole point of the shared message: name the mode switch AND the const
    value that has to follow it."""
    msg = rail_remedy(_z(0.25), name="q3.z", needed_v=1.4, rail=0.5)
    assert "direct" in msg and 'output_mode="amplified"' in msg
    assert "1.25" in msg  # the const the switch implies


def test_an_already_amplified_port_is_NOT_told_to_amplify():
    """Advising 'run that port in amplified mode' when it already is amplified is
    worse than silence — it was the live wording bug in all three pair probes."""
    msg = rail_remedy(_z(1.25, mode="amplified"), name="q3.z", needed_v=3.0, rail=2.5)
    assert "already" in msg and "amplified" in msg
    assert 'output_mode="amplified"' not in msg
    assert "Narrow the window" in msg


def test_an_unknown_port_says_it_assumed_the_conservative_rail():
    z = _flux_line()
    z.operations = {}
    msg = rail_remedy(z, name="q3.z", needed_v=1.0, rail=0.5)
    assert "does not describe an output port" in msg and "conservative" in msg


def test_an_opx_plus_port_is_told_its_rail_is_fixed_not_that_it_can_be_upgraded():
    """The remedy, not the number, is what the OPX+ case changes.

    Its rail is 0.5 V — arithmetically the LF-FEM's ``direct`` value, so the old
    fallback returned the right volts. What it returned with them was advice to
    "run it in 'amplified' mode", a switch that does not exist on an OPX+, sent to
    an operator who would then go looking for it. Same number, useless sentence.
    """
    z = _opx_plus_z()
    msg = rail_remedy(z, name="q3.z", needed_v=1.0, rail=dac_rail_v(z))
    assert "OPX+" in msg
    assert "amplified" in msg and "no 'amplified' mode" in msg
    assert "Narrow the window" in msg


def test_the_opx_plus_rail_is_reported_as_a_finding_not_as_a_fallback():
    """0.5 V reached two ways that must stay distinguishable: an identified OPX+
    port, and a port the tree does not describe."""
    identified = _opx_plus_z()
    unknown = _flux_line()

    assert dac_rail_v(identified) == dac_rail_v(unknown) == 0.5
    assert "OPX+" in rail_remedy(identified, name="q.z", needed_v=1.0, rail=0.5)
    assert "OPX+" not in rail_remedy(unknown, name="q.z", needed_v=1.0, rail=0.5)


# ------------------------------------------------------------- the idle anchor

def test_the_idle_offset_follows_the_APPLIED_point_not_the_declared_one():
    """``idle_offset_v`` takes the point as an argument on purpose: a probe with
    its own flux_point parameter must validate the bias it is applying. Reading
    ``z.flux_point`` there would rebuild the very gap the factory guard closes —
    note this line DECLARES 'independent' while the probes apply 'joint'."""
    z = _flux_line(joint=0.107, flux_point="independent")
    z.independent_offset = 0.084
    assert idle_offset_v(z, "joint") == 0.107
    assert idle_offset_v(z, "independent") == 0.084
    assert idle_offset_v(z, "zero") == 0.0


def test_an_unknown_flux_point_is_refused_rather_than_defaulted():
    with pytest.raises(ValueError, match="selects no offset"):
        idle_offset_v(_flux_line(), "nonsense")


def test_a_coupler_resolves_its_OWN_flux_point_vocabulary():
    """A TunableCoupler names its points off/on but its attributes
    decouple_offset/interaction_offset, so the <point>_offset convention alone
    raises on every coupler — which is what the pair probes hand it."""
    c = _coupler("c12")
    c.decouple_offset = 0.02
    assert idle_offset_v(c, "off") == 0.02
    assert idle_offset_v(c, "on") == 0.12
    assert idle_offset_v(c, "arbitrary") == 0.0  # shared name, plain convention
    assert idle_offset_v(c, "zero") == 0.0


def test_the_declared_point_is_used_only_where_the_probe_cannot_choose():
    """A probe with no flux_point argument applies whatever the line declares, so
    reading the declaration is the honest answer there — and the coupler default
    ('off') has to resolve through the alias."""
    z = _flux_line(joint=0.107)  # declares 'joint'
    assert declared_idle_offset_v(z) == 0.107
    c = _coupler("c12")
    c.decouple_offset = 0.03
    assert declared_idle_offset_v(c) == 0.03


def test_a_channel_with_no_declared_point_reads_as_zero_bias():
    """A stub or a line that never biases must not refuse — the permissive
    direction is right here because there is no bias to clip against."""
    assert declared_idle_offset_v(object()) == 0.0
