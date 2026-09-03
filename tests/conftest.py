"""Shared fixtures for the QM driver tests (greenfield entity model).

The backend now takes the device ROSTER: a driver serves a view per CHANNEL
ENTITY (``q1_ro`` / ``q1_xy`` / ``q1_z``) over the matching SUBTREE of one QUAM
qubit, and only the roster says what those names mean. :data:`ROSTER_TOML`
describes the fixture chip in the schema-3 vocabulary.

Two machine fixtures, deliberately:

* :func:`machine` is the LIVE ``quam_state/`` (what the probe-equivalence and
  absolute-power tests need — a real QUAM tree with real pulse/reference
  semantics). It SKIPS when the state does not match the root class currently
  toggled in ``quam_config/my_quam.py`` (CLAUDE.md "Key Entrypoints"): a
  legitimate working-tree situation, not a test failure.
* :func:`stub_machine` is a hand-built stand-in with the same SHAPE, so the
  entity-resolution surface (component/components/snapshot, both flux shapes,
  the pair knobs) is covered on every run, toggle or no toggle. It is a stub of
  the vendor TREE only — every neutral conversion under test is the real
  ``scqo_qm.quam_fields`` code.

The stub deliberately names its QUAM pair after the coupler (``coupler_q1_q2``,
which is what quam_builder does) while the roster calls the composite ``q1_q2``
and its coupler mode ``q1_q2_c``: the name join is therefore FALSE on all three
counts, and every resolution has to go through the roster's membership/roles.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

#: The fixture chip in the greenfield schema: ONE multiplexed feedline (the
#: readout riders mint q1_res/q1_ro, ...), a drive wire per qubit, a flux wire
#: for the two flux-tunable qubits and for the coupler MODE (its standing bias
#: is idle_flux on q1_q2_c_z, not a pair field), and a fixed-frequency q3 with
#: no flux rider at all. The pair composite declares one operation, so its
#: per-operation knob family is cz_coupler_flux / cz_vz_high_rad / ...
ROSTER_TOML = """\
schema = 3

[modes.q1]
kind = "flux_transmon"

[modes.q2]
kind = "flux_transmon"

[modes.q3]
kind = "transmon"

[modes.q1_q2_c]
kind = "flux_transmon"

[composites.q1_q2]
kind       = "qubit_pair"
high       = "q2"
low        = "q1"
coupler    = "q1_q2_c"
operations = ["cz", "iswap"]

[lines.fl]
readout = ["q1", "q2", "q3"]

[lines.xy1]
drive = ["q1"]

[lines.xy2]
drive = ["q2"]

[lines.xy3]
drive = ["q3"]

[lines.z1]
flux = ["q1"]

[lines.z2]
flux = ["q2"]

[lines.zc]
flux = ["q1_q2_c"]
"""


@pytest.fixture()
def roster():
    """The fixture chip's validated roster (parsed by the real loader)."""
    from scqo.roster import parse_components

    return parse_components(ROSTER_TOML)


# ---------------------------------------------------------------- stub QUAM tree

class Pulse(SimpleNamespace):
    """A QUAM pulse stand-in: whatever attributes the accessors touch."""


class ReadoutPulse:
    """QUAM ``ReadoutPulse`` stand-in with REAL reference semantics: reading
    ``integration_weights`` resolves the default-weights reference against the
    CURRENT length (exactly what quam does), while the raw slot keeps the stored
    form. The window accessors are meaningless without this."""

    def __init__(self, length=2000, weights=None, angle=0.0,
                 amplitude=0.1, threshold=0.0, rus=0.0):
        from scqo_qm import quam_fields

        self.length = length
        self.amplitude = amplitude
        self.integration_weights_angle = angle
        self.threshold = threshold
        self.rus_exit_threshold = rus
        self._raw_weights = (weights if weights is not None
                             else quam_fields.DEFAULT_INTEGRATION_WEIGHTS_REF)

    @property
    def integration_weights(self):
        from scqo_qm import quam_fields

        if self._raw_weights == quam_fields.DEFAULT_INTEGRATION_WEIGHTS_REF:
            return [(1, self.length)]
        return self._raw_weights

    @integration_weights.setter
    def integration_weights(self, value):
        self._raw_weights = value


def _xy(rf: float) -> SimpleNamespace:
    return SimpleNamespace(
        RF_frequency=rf,
        LO_frequency=rf - 100e6,
        opx_output=SimpleNamespace(full_scale_power_dbm=-11),
        operations={
            "x180": Pulse(amplitude=0.2, length=32, alpha=-0.9),
            "x180_DragCosine": Pulse(amplitude=0.2, length=32, alpha=-0.9),
            "x90_DragCosine": Pulse(amplitude=0.1, length=32, alpha=-0.5),
            "saturation": Pulse(amplitude=0.05, length=1000),
        },
    )


def _resonator(rf: float) -> SimpleNamespace:
    return SimpleNamespace(
        RF_frequency=rf,
        f_01=rf,
        LO_frequency=6.06e9,
        opx_output=SimpleNamespace(full_scale_power_dbm=-11),
        operations={"readout": ReadoutPulse()},
    )


def _flux_line(joint: float = 0.0, *, flux_point: str = "joint") -> SimpleNamespace:
    """A QUAM ``FluxLine`` stand-in — its named points are the QUBIT vocabulary
    (joint/independent/min/arbitrary), which is why the coupler below cannot
    share the accessor.

    ``flux_point`` defaults to the GOVERNED value, so the stub tree is
    scqo-compliant; pass another to exercise ``flux_point_problems``.
    """
    return SimpleNamespace(
        flux_point=flux_point, joint_offset=joint, independent_offset=0.0,
        min_offset=0.0, arbitrary_offset=0.0,
    )


def _coupler(name: str) -> SimpleNamespace:
    """A ``TunableCoupler`` stand-in: its OWN flux-point vocabulary
    (off/on/arbitrary) — the second vendor shape behind ``idle_flux``."""
    return SimpleNamespace(
        id=name, name=name, flux_point="off",
        decouple_offset=0.0, interaction_offset=0.12, arbitrary_offset=0.0,
        settle_time=None,
        # The coupler carries its OWN copy of a named flux pulse. That is where
        # an ISwapImplementation-shaped macro's coupler operating point lives —
        # the macro holds only the NAME (see the iswap stub below).
        operations={"swap_flattop": Pulse(amplitude=0.081, length=96)},
    )


def _qubit(name: str, *, f_01: float, res_rf: float,
           flux: bool = True) -> SimpleNamespace:
    q = SimpleNamespace(
        id=name, name=name, f_01=f_01,
        # the drive RF IS f_01: set_drive_freq writes both to one absolute value
        # and drive_frequency_problems refuses a tree where they differ (exactly
        # how _resonator pairs RF_frequency with f_01)
        xy=_xy(f_01), resonator=_resonator(res_rf),
        # a Thermalizing*Transmon carries this; stock QUAM classes do NOT (the
        # stale_qubit fixture below stands in for one), and the difference is
        # exactly what set_thermalization_time refuses on.
        thermalization_time_ns=None,
    )
    if flux:  # a fixed-frequency transmon genuinely has no z subtree
        q.z = _flux_line()
    return q


def make_stub_machine() -> SimpleNamespace:
    """A QUAM ``Quam`` stand-in matching :data:`ROSTER_TOML`."""
    q1 = _qubit("q1", f_01=4.8e9, res_rf=6.10e9)
    q2 = _qubit("q2", f_01=5.1e9, res_rf=6.20e9)
    q3 = _qubit("q3", f_01=5.4e9, res_rf=6.30e9, flux=False)
    # The QUAM pair is named after its coupler (quam_builder's convention) and
    # its members are control=q1 / target=q2 while the roster says high=q2 /
    # low=q1 — so both the pair join and the virtual-Z side resolution have to
    # go through the roster, never through the name.
    pair = SimpleNamespace(
        id="coupler_q1_q2", name="coupler_q1_q2",
        qubit_control=q1, qubit_target=q2,
        coupler=_coupler("coupler_q1_q2"),
        moving_qubit="control", detuning=0.05,
        macros={
            "CZ": SimpleNamespace(  # QUAM spells the macro "CZ", roster "cz"
                # the VENDOR quam_builder CZGate shape: an explicit Pulse
                coupler_flux_pulse=Pulse(amplitude=-0.125, length=128),
                flux_pulse_qubit=Pulse(amplitude=0.0495, length=128),
                phase_shift_control=0.0, phase_shift_target=0.0,
            ),
            # The LAB's ISwapImplementation shape, and the reason it is here: the
            # <op>_coupler_flux binding was written against the CZ shape only, so
            # a stub carrying just that shape passed while every pair on the real
            # chip — where every macro is an ISwapImplementation — read None and
            # refused writes. It declares NO coupler_flux_pulse and names one
            # flux_pulse played on both the control's z line and the coupler.
            "iswap": SimpleNamespace(
                flux_pulse="swap_flattop",
                phase_shift_control=0.0, phase_shift_target=0.0,
            ),
        },
    )
    return SimpleNamespace(
        qubits={"q1": q1, "q2": q2, "q3": q3},
        qubit_pairs={"coupler_q1_q2": pair},
        # real QUAM always carries this; flux_point_problems reads it to catch an
        # INACTIVE flux-tunable qubit, which the joint path parks at min_offset
        # rather than at the joint_offset its idle_flux claims
        active_qubits=[q1, q2, q3],
    )


@pytest.fixture()
def stub_machine():
    return make_stub_machine()


@pytest.fixture()
def backend(stub_machine, roster):
    """A QMBackend over the stub tree — no instrument, no QUAM package."""
    from scqo_qm.backend.qm_backend import QMBackend

    return QMBackend(stub_machine, roster=roster)


def recording_device(backend, roster):
    """The device surface an experiment reads through — what the Session hands
    to ``exp.device``: channel-entity views over the vendor tree, knobs seeded
    from the instrument (pull), backed by an in-memory store."""
    from scqo.device import RecordingDevice
    from scqo.stores import state_store

    return RecordingDevice(backend.device, roster, state_store(None, roster))


def make_experiment(cls, backend, roster, params):
    """An experiment wired the way the Session wires one: the backend for
    acquisition, the recording device for state."""
    exp = cls(backend, params)
    exp.device = recording_device(backend, roster)
    return exp
