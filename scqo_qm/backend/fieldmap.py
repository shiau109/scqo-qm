"""Declarative field catalog for the QM backend — PURE DATA, no vendor imports.

Keyed by CHANNEL KIND (``drive`` / ``readout`` / ``flux``) since the greenfield
model: knobs live on channel entities (``q1_xy``, ``q1_ro``, ``q1_z``), not on a
single per-qubit component. Per kind, one :class:`scqo.fieldmap.VendorBinding`
per realized KNOB (where it lives on the QUAM tree, in what unit, converted how —
as a DESCRIPTION), one :class:`scqo.fieldmap.Unrealized` per knob this backend
cannot realize, plus the :class:`scqo.fieldmap.VendorOnly` inventory of
calibration-relevant knobs with no neutral counterpart yet. The EXECUTABLE
conversions live in the three channel views of ``backend.py``
(``QMDriveChannel`` / ``QMReadoutChannel`` / ``QMFluxChannel``, via
scqo_qm.quam_fields + power_tools) — this module documents them and is pinned
to the implementation by ``tests/test_scqo_glue.py`` (per kind: bindings |
unrealized == scqo's KNOB fields; imports stay vendor-free).

COMPOSITES are separate: a ``qubit_pair``'s knobs are PER-OPERATION full names
(``iswap_coupler_flux``) instantiated from ``scqo.catalog.OP_KNOBS`` by the
operations the ROSTER declares, so they cannot be tabulated by a static field
name. :data:`OP_KNOB_BINDINGS` / :data:`OP_KNOB_UNREALIZED` are therefore keyed
by the OP_KNOBS SUFFIX (``coupler_flux``, ``vz_high_rad``, ...) and served
through ``QMQubitPair.read_knob``/``write_knob``; ``FIELD_BINDINGS`` stays
channel-kind-only so the per-kind drift alarm keeps its exact meaning.

MONITORS are absent by construction: ``fidelity_g``/``fidelity_e``/``pos_*`` are
measured performance OF the current knobs, never pushed, so they need no vendor
binding and no Unrealized entry (the old ``readout_fidelity`` aggregate is gone;
the per-state pair replaces it). Facts (``flux_offset``, ``flux_per_phi0``,
``distortion_*``) live in physical.json and likewise bind nothing.

Rendered by ``scqo state --fields``; strings reach lab consoles, keep them ASCII.
"""

from __future__ import annotations

from scqo.fieldmap import OperatorCommand, Unrealized, VendorBinding, VendorOnly

FIELD_BINDINGS: dict[str, dict[str, VendorBinding]] = {
    "drive": {
        "drive_freq_hz": VendorBinding(
            path="q.f_01", unit="Hz",
            convert="a write lands the same absolute Hz on q.f_01 and "
                    "q.xy.RF_frequency (the drive line plays RF_frequency; scqo "
                    "reads f_01); the session factory refuses a state.json where "
                    "the two disagree (quam_fields.drive_frequency_problems)"),
        "pi_amp": VendorBinding(
            path="q.xy.operations['x180'].amplitude", unit="",
            convert="a write covers the x180 family's storage nodes; fields that "
                    "are QUAM #-references are skipped (they follow the real node)",
            note="written on the x180_DragCosine storage node; the plain x180 "
                 "entry is usually a QUAM reference alias and follows. Writes the "
                 "x180 family ONLY - the pi/2 is its own knob (pi_amp_x90), never "
                 "derived as half of this one"),
        "pi_amp_x90": VendorBinding(
            path="q.xy.operations['x90_DragCosine'].amplitude", unit="",
            convert="a write also covers -x90_DragCosine (its negative sense comes "
                    "from axis_angle = pi, not a negated amplitude): a literal "
                    "amplitude there is written so the two pi/2 gates cannot drift "
                    "apart, a #-reference one is skipped (it already follows); "
                    "y90/-y90 are aliases and follow x90_DragCosine",
            note="written on the x90_DragCosine storage node; calibrated by "
                 "qubit_deterministic_benchmarking with target_gate=x90. Qblox "
                 "DERIVES X90 from rxy.amp180, so it has no independent home there "
                 "and declines this knob (Unrealized) until a Qblox probe lands"),
        "drag_beta": VendorBinding(


            path="q.xy.operations['x180_DragCosine'].alpha", unit="",
            convert="QM stores DRAG as DragCosinePulse.alpha; written on the "
                    "x180_DragCosine storage node (reference aliases follow)",
            note="calibrated by qubit_drag_equator / qubit_drag_alternating"),
        "drag_beta_x90": VendorBinding(
            path="q.xy.operations['x90_DragCosine'].alpha", unit="",
            convert="QM stores DRAG as DragCosinePulse.alpha; written on the "
                    "x90_DragCosine storage node",
            note="calibrated by qubit_drag_equator / qubit_drag_alternating for x90"),



        "pi_duration_s": VendorBinding(
            path="q.xy.operations['x180'].length", unit="ns",
            convert="seconds -> ns",
            note="positive multiples of 4 ns only (REFUSED otherwise, no silent "
                 "rounding: an off-grid length is unrealizable on QM and would "
                 "de-calibrate the stored pi_amp against a pulse that is not the "
                 "one measured). x90's length stays vendor fine print - it is a "
                 "per-gate value, not the tracked pi length. Qblox counterpart: "
                 "rxy.duration (s, no grid guard there)"),
        "thermalization_time_s": VendorBinding(
            path="q.thermalization_time_ns", unit="ns",
            convert="seconds -> ns, ROUNDED to the 4 ns QUA wait grid (a policy "
                    "wait, not a calibrated pulse - unlike pi_duration_s, which "
                    "refuses off-grid)",
            note="on the QUBIT, not q.xy. Overrides QUAM's derived "
                 "thermalization_time (= thermalization_time_factor * T1), which "
                 "is a read-only property with nowhere to store an absolute "
                 "wait; a qubit scqo has never calibrated falls back to it. "
                 "REQUIRES the qubit's state.json __class__ to name a "
                 "Thermalizing*Transmon. Calibrated by qubit_relaxation "
                 "(thermalization_factor x T1). Qblox counterpart: "
                 "element.reset.duration (s, absolute)"),
        "drive_amp": VendorBinding(
            path="q.xy.operations['saturation'].amplitude", unit="",
            note="the saturation (spec) drive amplitude - the drive_power_dbm "
                 "chain solve's residual"),
        "drive_power_dbm": VendorBinding(
            path="MW-FEM: q.xy.opx_output.full_scale_power_dbm  |  Octave: "
                 "q.xy.frequency_converter_up.gain  -- either one "
                 "+ q.xy.operations['saturation'].amplitude",
            unit="dBm + amp",
            convert="solve the DRIVE chain, coarse knob + amplitude residual, but "
                    "with OPPOSITE policies. MW-FEM: the SMALLEST "
                    "full_scale_power_dbm on the -11..+16 dBm grid (3 dB steps) "
                    "keeping the saturation amplitude <= 0.5. Octave: HOLD the gain "
                    "(-20..+20 dB, 0.5 dB steps) wherever the amplitude (volts, "
                    "< 0.5 V) can absorb the change, and re-stage it only when it "
                    "cannot -- the gain keys the mixer calibration, so moving it "
                    "invalidates that RF output's stored correction. Any other RF "
                    "chain refuses by name",
            coupled=("drive_amp",),
            note="the coarse knob is PORT-level (MW-FEM) or RF-OUTPUT-level "
                 "(Octave) and shared by every xy operation either way: while it "
                 "is off its standing value the stored pi_amp means a different "
                 "power (qubit_spectroscopy sets it and reverts exactly)",
        ),
    },
    "readout": {
        "readout_freq_hz": VendorBinding(
            path="q.resonator.RF_frequency", unit="Hz",
            note="q.resonator.f_01 is kept equal when the resonator carries it"),
        "readout_amp": VendorBinding(
            path="q.resonator.operations['readout'].amplitude", unit=""),
        "readout_power_dbm": VendorBinding(
            path="MW-FEM: q.resonator.opx_output.full_scale_power_dbm  |  Octave: "
                 "q.resonator.frequency_converter_up.gain  -- either one "
                 "+ q.resonator.operations['readout'].amplitude",
            unit="dBm + amp",
            convert="solve the output chain, coarse knob + amplitude residual, but "
                    "with OPPOSITE policies. MW-FEM: the SMALLEST "
                    "full_scale_power_dbm on the -11..+16 dBm grid (3 dB steps) "
                    "keeping the amplitude <= 0.5. Octave: HOLD the gain wherever "
                    "the amplitude (volts, < 0.5 V) can absorb the change, because "
                    "the gain keys the mixer calibration AND is shared by every "
                    "qubit on a multiplexed feedline. Any other RF chain refuses "
                    "by name",
            coupled=("readout_amp",),
            note="coarse grids: MW-FEM -11..+16 dBm in 3 dB steps; Octave gain "
                 "-20..+20 dB in 0.5 dB steps",
        ),
        "readout_duration_s": VendorBinding(
            path="q.resonator.operations['readout'].length", unit="ns",
            convert="seconds -> ns",
            coupled=("readout_integration_s",),
            note="positive multiples of 4 ns only (REFUSED otherwise, no silent "
                 "rounding); shrinking the pulse clamps the integration window "
                 "down with it; a custom weights list is rebuilt constant-window "
                 "around the new length (shaped/optimized weights do not survive)",
        ),
        "readout_integration_s": VendorBinding(
            path="q.resonator.operations['readout'].integration_weights", unit="ns",
            convert="window w -> constant weights [(1, w), (0, length - w)] "
                    "spanning the pulse exactly; the default reference when "
                    "w == length; integration_weights_angle applies on top, untouched",
            note="contract: <= readout_duration_s (weights cannot span past the "
                 "pulse); multiples of 4 ns. Qblox counterpart: "
                 "measure.integration_time (s)",
        ),
        "readout_depletion_s": VendorBinding(
            path="q.resonator.depletion_time", unit="ns",
            convert="seconds -> ns, ROUNDED to the 4 ns QUA wait grid and stored "
                    "as int (a policy wait derived from a fit, like "
                    "thermalization_time_s; 0 is legal and means 'measured, no "
                    "settle needed')",
            note="post-readout photon-depletion wait. Needs NO custom transmon "
                 "class, unlike thermalization_time_s: depletion_time is a plain "
                 "settable int field, and QUAM already spends it after every "
                 "measurement and as depletion_time // 2 in reset_qubit_active. "
                 "Calibrated by resonator_spectroscopy as depletion_factor / "
                 "(2 pi x kappa_tot_hz). Qblox counterpart: "
                 "element.depletion.duration (s, a lab addition to the element)"),
        "readout_rotation_rad": VendorBinding(
            path="q.resonator.operations['readout'].integration_weights_angle",
            unit="rad",
            convert="the ABSOLUTE demod rotation (single_shot_readout proposes "
                    "current - measured delta); a direct edit silently de-calibrates "
                    "it - governed write: scqo set QUBIT.readout_rotation_rad=... . "
                    "Qblox counterpart: acq_rotation (DEGREES)",
            note="acquisition IQ frame, chain-dependent (invalidated by an "
                 "input-chain change such as mw_input gain_db); non-portable",
        ),
        "readout_threshold": VendorBinding(
            path="q.resonator.operations['readout'].threshold", unit="",
            convert="g/e discrimination threshold on the rotated I, in raw demod "
                    "units (NO volts conversion on the scqo path); Qblox counterpart: "
                    "acq_threshold (normalized frame)",
            note="acquisition-frame, chain-dependent; the threshold "
                 "use_state_discrimination applies on the FPGA",
        ),
        "readout_rus_threshold": VendorBinding(
            path="q.resonator.operations['readout'].rus_exit_threshold", unit="",
            note="repeat-until-success (active-reset) exit threshold on the rotated "
                 "I, raw demod units; no Qblox counterpart (Unrealized there)",
        ),
    },
    "flux": {
        "idle_flux": VendorBinding(
            path="q.z.<flux_point>_offset  |  qp.coupler.<flux_point>_offset",
            unit="V",
            convert="the offset SELECTED by the line's flux_point. QUBIT flux "
                    "channel (q1_z): z.flux_point in joint/independent/min/"
                    "arbitrary ('zero' reads 0 V and REFUSES writes). COUPLER "
                    "flux channel (the coupler MODE's own q1_q2_c_z): "
                    "coupler.flux_point off -> decouple_offset, on -> "
                    "interaction_offset, arbitrary -> arbitrary_offset, 'zero' "
                    "likewise 0 V and read-only",
            note="which named flux point is active stays vendor config "
                 "(z.flux_point / coupler.flux_point, catalogued below), BUT "
                 "the scqo path PINS it: z.flux_point must be 'joint' and "
                 "coupler.flux_point 'off', because that is what every probe's "
                 "initialize_qpu applies, and the backend factory REFUSES a "
                 "state that disagrees (quam_fields.flux_point_problems). "
                 "Without that pin the knob reads and writes an offset the "
                 "hardware never holds - live on 5Q4C until 2026-07-29. Given "
                 "the pin, the write lands on hardware at the next "
                 "initialize_qpu (every probe runs it). A coupler's "
                 "standing/decouple bias IS this "
                 "knob on its own flux channel - the old pair-level "
                 "coupler_decouple_v / coupler_interaction_v are gone. On a "
                 "fixed-frequency machine the qubit has no z, so the roster "
                 "declares no flux rider for it and the channel does not exist",
        ),
        "flux_delay_s": VendorBinding(
            path="q.z.opx_output.delay  |  coupler.opx_output.delay",
            unit="ns",
            convert="seconds -> int ns (1 ns resolution, rounded; negative "
                    "refused). ABSOLUTE, not incremental: qubit_xyz_delay writes "
                    "old + fitted peak, unlike the vendored 16a node which did "
                    "q.z.opx_output.delay += fit",
            note="output-path delay of the flux line vs the drive line, "
                 "calibrated so a Z pulse and the XY drive it accompanies "
                 "coincide. PORT-level (LFAnalogOutputPort.delay, shared by "
                 "everything on that DAC output) - on the shared LF base, so "
                 "every baseband port has it, LF-FEM and OPX+ alike; on a "
                 "per-qubit z wire it is per-qubit in "
                 "practice. Qblox counterpart: "
                 "hardware_options.latency_corrections[<port-clock>] (s), "
                 "Unrealized there until a Qblox xyz-delay probe exists",
        ),
    },
}

#: Neutral KNOBS this backend cannot realize, per channel kind (declared, never
#: silent). Empty: QM realizes every drive/readout/flux knob in the catalog —
#: including the discriminator trio that is Unrealized on Qblox. Kept as an
#: explicit empty map so the served-kind set stays visible in one place.
UNREALIZED: dict[str, dict[str, Unrealized]] = {}

# ------------------------------------------------------- composite (pair) knobs

#: Per-OPERATION knob suffixes (``scqo.catalog.OP_KNOBS``) this backend realizes
#: on a ``qubit_pair`` composite, keyed by SUFFIX: the full field name is
#: ``<operation>_<suffix>`` for each operation the ROSTER declares on the pair
#: (``iswap_coupler_flux``), which is why these cannot live in FIELD_BINDINGS
#: (keyed by static catalog field name). The executable conversions are
#: ``QMQubitPair.read_knob``/``write_knob``; ``<op>`` below is the QUAM macro
#: whose name matches the declared operation (case-insensitively: QUAM spells
#: the gate macro "CZ", the roster spells the operation "cz").
OP_KNOB_BINDINGS: dict[str, VendorBinding] = {
    "coupler_flux": VendorBinding(
        path="qp.macros['<op>'].coupler_flux_pulse.amplitude "
             "| qp.coupler.operations[qp.macros['<op>'].flux_pulse].amplitude",
        unit="V",
        note="the flux-activated gate operating point ON THE COUPLER LINE - the "
             "amplitude of the pulse the gate macro plays on qp.coupler while "
             "the moving qubit's z pulse runs. Distinct from the coupler's "
             "STANDING bias, which is idle_flux on the coupler mode's own flux "
             "channel. THREE macro shapes carry it and the resolution walks them "
             "in order (scqo_qm.experiments._coupler_knob.find_coupler_pulse): "
             "the vendor quam_builder CZGate's coupler_flux_pulse holding a Pulse; "
             "the same field holding a pulse NAME; and the lab's "
             "ISwapImplementation, which declares no coupler_flux_pulse at all and "
             "instead plays ONE named flux_pulse on both the control's z line and "
             "the coupler, so the coupler's own copy of it is the operating point. "
             "A macro that DECLARES coupler_flux_pulse and leaves it None is a "
             "fixed-coupler gate: reads None, refuses writes. One that never "
             "declares it is a different shape, not a fixed coupler",
    ),
    "vz_high_rad": VendorBinding(
        path="qp.macros['<op>'].phase_shift_control|phase_shift_target", unit="turns",
        convert="rad -> turns (QM frame_rotation_2pi units): turns = rad / 2pi",
        note="which QM side carries it is resolved from the ROSTER roles: the "
             "pair's high qubit is matched against qp.qubit_control/qubit_target "
             "by NAME (control/target is vendor gate plumbing, never roster "
             "topology). A pair whose QUAM members do not match its roster "
             "high/low pair is refused rather than guessed",
    ),
    "vz_low_rad": VendorBinding(
        path="qp.macros['<op>'].phase_shift_control|phase_shift_target", unit="turns",
        convert="rad -> turns (QM frame_rotation_2pi units): turns = rad / 2pi",
        note="the low qubit's side of the same pair of QUAM attributes; see "
             "vz_high_rad for the role resolution",
    ),
}

#: Per-operation knob suffixes with no QM realization yet (same shape as
#: UNREALIZED; the dataclass attribute spelled ``category`` carries the
#: COMPOSITE KIND). Reads and writes both raise with these reasons.
OP_KNOB_UNREALIZED: dict[str, Unrealized] = {
    "duration_s": Unrealized(
        "qubit_pair", "duration_s",
        "the gate length is carried by TWO simultaneous pulses (the moving "
        "qubit's z pulse and the coupler pulse); writing one without the other "
        "would desync them, and no scqo experiment calibrates a pair duration "
        "yet (Phase 2b chevron/CZ). Promote to a coupled binding when one lands"),
    "drive_freq_hz": Unrealized(
        "qubit_pair", "drive_freq_hz",
        "microwave-activated two-qubit gates are not wired here: the QM macros "
        "in use (CZGate) are FLUX-activated, so there is no gate drive tone to "
        "bind"),
    "amp": Unrealized(
        "qubit_pair", "amp",
        "no overall gate-drive amplitude on a flux-activated macro - the "
        "flux-plane counterpart is <op>_coupler_flux (bound above)"),
    "amp_ratio": Unrealized(
        "qubit_pair", "amp_ratio",
        "two-emission-channel knob: no QM macro here drives a gate from two "
        "channels whose ratio is calibrated"),
    "rel_phase_rad": Unrealized(
        "qubit_pair", "rel_phase_rad",
        "two-emission-channel knob: see amp_ratio - nothing to bind on a "
        "flux-activated macro"),
    "waveform": Unrealized(
        "qubit_pair", "waveform",
        "optimized-pulse samples: QM stores the gate pulse as a typed Pulse "
        "object (SquarePulse/FlatTopGaussian), not a sample array; binding an "
        "arbitrary waveform means switching the macro's pulse class, which no "
        "scqo experiment asks for yet"),
    "waveform_dt_s": Unrealized(
        "qubit_pair", "waveform_dt_s",
        "the mandatory companion of <op>_waveform - Unrealized with it"),
}

#: Backend-unique calibration knobs, vendor-owned and untracked by SCQO (edit in
#: the setup's state.json with QUAM tools). Each entry carries its placement-rule
#: kind (scqo state --rule): realizer / candidate / vendor / unique. Doubles as
#: the neutral-field promotion backlog (candidates pre-declare their convention).
VENDOR_ONLY_COMMON: dict[str, VendorOnly] = {
    "readout_length": VendorOnly(
        path="q.resonator.operations['readout'].length", unit="ns", kind="realizer",
        doc="readout pulse length - realizes the TRACKED readout_duration_s. The "
            "integration window is NOT fused to it: readout_integration_s owns "
            "the weights support (default weights only LOOK fused - they span "
            "the pulse by reference)",
        edit="scqo set QUBIT.readout_duration_s=... - a direct edit silently "
             "de-calibrates it"),
    "readout_integration_weights": VendorOnly(
        path="q.resonator.operations['readout'].integration_weights", unit="",
        kind="realizer",
        doc="integration-weights list - its nonzero SUPPORT realizes the "
            "TRACKED readout_integration_s. The SHAPE within the window stays "
            "vendor territory (a future weight-optimization node may write it; "
            "any later window write rebuilds constant weights)",
        edit="scqo set QUBIT.readout_integration_s=... - the setter writes "
             "constant zero-padded weights"),
    "time_of_flight": VendorOnly(
        path="q.resonator.time_of_flight", unit="ns", kind="vendor",
        doc="acquisition latency compensation - aligns the instrument's receive "
            "path with its own transmit path. The TOF measurement's product is "
            "written HERE, in NANOSECONDS, offline - never a neutral field",
        counterpart="measure.acq_delay (s)"),
    # NOTE: depletion_time is no longer VendorOnly - it REALIZES the tracked
    # readout_depletion_s (binding above). It was a hand-set policy value sitting
    # at QUAM's 16 ns default with nothing governing it, while QUAM spent it in
    # four places (after every measurement, and depletion_time // 2 inside
    # reset_qubit_active). resonator_spectroscopy now calibrates it from the
    # measured linewidth; the governed write is
    # scqo set QUBIT.readout_depletion_s=... .
    "x180_length": VendorOnly(
        path="q.xy.operations['x180'].length", unit="ns", kind="realizer",
        doc="pi/x180 pulse length - it REALIZES the tracked pi_duration_s "
            "(promoted to a neutral drive knob in the greenfield catalog; "
            "binding above). Multiple of 4 ns (chipA: 32 ns here vs 200 ns on "
            "Qblox - genuinely per-chain calibrated)",
        edit="scqo set QUBIT.pi_duration_s=... - a direct edit silently "
             "de-calibrates the stored pi_amp with it",
        counterpart="rxy.duration (s, no grid guard there)"),
    "x90_length": VendorOnly(
        path="q.xy.operations['x90_DragCosine'].length", unit="ns", kind="vendor",
        doc="pi/2 pulse length - a PER-GATE vendor value, deliberately NOT "
            "locked to the tracked pi_duration_s (the neutral knob is the pi "
            "pulse's length only)",
        edit="edit state.json directly when a chip wants a different x90 "
             "envelope - nothing governs this one"),
    "drag_alpha": VendorOnly(
        path="q.xy.operations['<gate>_DragCosine'].alpha", unit="", kind="realizer",
        doc="PER-GATE DRAG coefficient (chipA: x180 -0.94, x90 -0.50). The x180 "
            "node REALIZES the tracked neutral drag_beta (binding above); the "
            "OTHER gates' alpha values remain vendor fine print",
        edit="scqo set QUBIT.drag_beta=... for the x180 node; the other gates' "
             "alpha stay direct state.json edits",
        counterpart="rxy.beta (derivative scale, different math convention)"),
    "per_gate_detuning": VendorOnly(
        path="q.xy.operations['x90_DragCosine'].detuning", unit="Hz", kind="unique",
        doc="per-gate drive detuning (chipA: -300 kHz on x90 vs 0 on x180) - "
            "no Qblox counterpart (one shared rxy op set there): experiments "
            "depending on it run ONLY on QM"),
    # ------------------------------------------------- flux points + couplers
    "flux_point": VendorOnly(
        path="q.z.flux_point", unit="", kind="vendor",
        doc="which named qubit flux point idles (joint/independent/min/"
            "arbitrary/zero) - SELECTS which offset the tracked idle_flux "
            "reads and writes on q1_z. A mode switch, not a calibration "
            "outcome",
        edit="PINNED: the backend factory REFUSES any value but 'joint' (the "
             "point every probe's initialize_qpu applies) - a declaration that "
             "disagrees with the applied bias makes idle_flux inert. Flipping "
             "it re-points idle_flux at a different stored number, so re-seed "
             "after changing it"),
    "coupler_flux_point": VendorOnly(
        path="qp.coupler.flux_point", unit="", kind="vendor",
        doc="which named coupler point idles (off/on/arbitrary/zero) - SELECTS "
            "which offset the tracked idle_flux reads and writes on the "
            "COUPLER mode's flux channel (q1_q2_c_z). A mode switch, not a "
            "calibration outcome",
        edit="PINNED to 'off' by the same factory audit; flipping it re-points "
             "the coupler's idle_flux at a different stored number"),
    "coupler_decouple_offset": VendorOnly(
        path="qp.coupler.decouple_offset", unit="V", kind="realizer",
        doc="the interaction-OFF coupler standing bias (pair_zz_coupler's "
            "product - the ZZ zero crossing). It REALIZES the tracked "
            "idle_flux of the coupler mode's flux channel while "
            "coupler.flux_point == 'off' (the old pair-level coupler_decouple_v "
            "neutral field is GONE)",
        edit="scqo set <coupler>_z.idle_flux=..."),
    "coupler_interaction_offset": VendorOnly(
        path="qp.coupler.interaction_offset", unit="V", kind="realizer",
        doc="the interaction-ON coupler standing bias (gate operating point). "
            "It REALIZES the tracked idle_flux of the coupler mode's flux "
            "channel while coupler.flux_point == 'on' - which the factory pins "
            "OFF, so there is no reachable governed write and `edit` is "
            "deliberately empty. A per-GATE operating point is NOT this: that "
            "is the composite knob <operation>_coupler_flux (bound above)"),
    "coupler_arbitrary_offset": VendorOnly(
        path="qp.coupler.arbitrary_offset", unit="V", kind="realizer",
        doc="free-form coupler bias for exploratory work - realizes idle_flux "
            "while coupler.flux_point == 'arbitrary', which the factory pins "
            "OFF, so there is no reachable governed write and `edit` is "
            "deliberately empty"),
    "coupler_settle_time": VendorOnly(
        path="qp.coupler.settle_time", unit="ns", kind="vendor",
        doc="coupler flux settle wait - an instrument-response policy value, "
            "not a calibration outcome"),
    # ----------------------------------------------------------- qubit pairs (QCQ)
    "pair_detuning": VendorOnly(
        path="qp.detuning", unit="V", kind="candidate",
        doc="flux amplitude bringing the two qubits to equal energy (the gate "
            "resonance condition) - neutral candidate for a per-operation "
            "composite knob; promoted when a scqo experiment (chevron) "
            "calibrates it"),
    "pair_moving_qubit": VendorOnly(
        path="qp.moving_qubit", unit="", kind="vendor",
        doc="which vendor side (control/target) carries the flux pulse in 2Q "
            "gates - a PER-OPERATION fact the driver reads; roster roles are "
            "high/low and never store this (settled pair-role decision). The "
            "composite view maps high/low onto control/target by NAME"),
    "pair_mutual_flux_bias": VendorOnly(
        path="qp.mutual_flux_bias", unit="V", kind="vendor",
        doc="two-element per-qubit z biases for the pair's mutual idle "
            "(to_mutual_idle); vendor-owned gate plumbing"),
    "pair_macro_flux_pulse": VendorOnly(
        path="qp.macros['<op>'].flux_pulse_qubit", unit="", kind="candidate",
        doc="the MOVING qubit's z pulse for a gate macro (name or Pulse; the "
            "coupler's twin, coupler_flux_pulse.amplitude, is the bound "
            "<op>_coupler_flux). Its amplitude/length become neutral "
            "per-operation knobs when a scqo chevron/CZ experiment calibrates "
            "them - Phase 2b"),
    "pair_confusion": VendorOnly(
        path="qp.confusion", unit="", kind="vendor",
        doc="4x4 two-qubit assignment confusion matrix - a stored measured "
            "artifact, DEAD to SCQO per the placement rule (never read, never "
            "written by it); portable traces live in run records"),
}

#: The MW-FEM's own port knobs. Present ONLY on a tree whose drive/readout
#: channels are MWChannels: an Octave channel has no ``opx_output`` at all, so
#: every path below is an AttributeError there, and listing them would send an
#: operator after state.json keys their instrument does not have.
VENDOR_ONLY_MW_FEM: dict[str, VendorOnly] = {
    "readout_upconverter_frequency": VendorOnly(
        path="q.resonator.opx_output.upconverter_frequency", unit="Hz", kind="vendor",
        doc="readout LO - the MW-FEM upconverter, PORT-level (state.json "
            "ports.mw_outputs.<con>.<fem>.<port>) and shared by everything on "
            "that output; many LO/IF splits give the SAME RF, so SCQO owns only "
            "the RF (readout_freq_hz) and never moves the LO in a chain solve",
        coupled=("downconverter_frequency", "readout_band"),
        edit="move it so IF = RF - LO stays in range and readout_band covers "
             "the target; downconverter_frequency MUST move with it or "
             "demodulation breaks",
        counterpart="modulation_frequencies lo_freq"),
    "drive_upconverter_frequency": VendorOnly(
        path="q.xy.opx_output.upconverter_frequency", unit="Hz", kind="vendor",
        doc="drive LO - PORT-level MW-FEM upconverter, shared by everything on "
            "that output",
        coupled=("drive_band",),
        edit="keep IF = f_01 - LO in range and drive_band matching"),
    "downconverter_frequency": VendorOnly(
        path="q.resonator.opx_input.downconverter_frequency", unit="Hz", kind="vendor",
        doc="receive-side downconversion LO on the MW input port, equal to the "
            "readout upconverter - PORT-level. No example value quoted on "
            "purpose: the old 'chipA: 6.06 GHz' had rotted two revisions deep "
            "(the repo's dev state runs 5.95 GHz, the live chipA config "
            "5.1 GHz) while the EQUALITY, which is the durable claim, held "
            "throughout",
        coupled=("readout_upconverter_frequency", "downconverter_band"),
        edit="it MUST track readout_upconverter_frequency or demodulation "
             "breaks, and downconverter_band must cover it",
        counterpart="none - Qblox has no separate knob (NCO handles it)"),
    "readout_band": VendorOnly(
        path="q.resonator.opx_output.band", unit="", kind="vendor",
        doc="which MW-FEM Nyquist band the READOUT output port runs in "
            "(chipA: 2) - a PORT-level hardware MODE chosen so the band covers "
            "the readout LO, not a calibration outcome. broadband_resonator_"
            "spectroscopy reads it live to derive the LO limits it may step "
            "within. Coverage is the INSTRUMENT's call: a band that does not "
            "cover the frequency comes back as a QM error, so no table here "
            "second-guesses it",
        coupled=("readout_upconverter_frequency",),
        edit="state.json ports.mw_outputs.<con>.<fem>.<port>.band, offline with "
             "QUAM tools. The MW-FEM pairs ports (2,3) (4,5) (6,7) and BOTH "
             "ports of a pair must carry the same band (see experiments/"
             "broadband_qubit_spectroscopy.py::_partner_port_id); "
             "quam_config/populate_quam_lf_mw_fems.py::get_band(freq) is the "
             "derivation the lab seeds from"),
    "drive_band": VendorOnly(
        path="q.xy.opx_output.band", unit="", kind="vendor",
        doc="the same PORT-level hardware mode on the DRIVE output port "
            "(chipA: 1), chosen so the band covers the drive LO. "
            "broadband_qubit_spectroscopy WRITES it run-scoped - one value per "
            "frequency segment - and restores the original in a finally. "
            "Coverage is the INSTRUMENT's call: a band that does not cover the "
            "frequency comes back as a QM error",
        coupled=("drive_upconverter_frequency",),
        edit="state.json ports.mw_outputs.<con>.<fem>.<port>.band, offline with "
             "QUAM tools; BOTH ports of a MW-FEM pair (2,3) (4,5) (6,7) must "
             "carry the same band"),
    "downconverter_band": VendorOnly(
        path="q.resonator.opx_input.band", unit="", kind="vendor",
        doc="which MW-FEM band the readout INPUT port runs in (chipA: 2, "
            "matching the output side) - PORT-level; the receive band must "
            "cover downconverter_frequency, which the instrument enforces",
        coupled=("downconverter_frequency",),
        edit="state.json ports.mw_inputs.<con>.<fem>.<port>.band, offline with "
             "QUAM tools"),
    "full_scale_power_dbm": VendorOnly(
        path="q.resonator.opx_output.full_scale_power_dbm", unit="dBm", kind="realizer",
        doc="the coarse readout power knob (grid -11..+16 in 3 dB steps, "
            "PORT-level - shared like the LO) - it REALIZES the tracked "
            "readout_power_dbm (binding above)",
        edit="scqo set QUBIT.readout_power_dbm=... (solves the chain, keeps "
             "readout_amp coupled, recorded); a direct edit silently "
             "de-calibrates the absolute power, and any later readout_power_dbm "
             "write re-solves and overwrites a forced value"),
    "drive_full_scale_power_dbm": VendorOnly(
        path="q.xy.opx_output.full_scale_power_dbm", unit="dBm", kind="realizer",
        doc="the coarse DRIVE power knob (grid -11..+16 in 3 dB steps, "
            "PORT-level - shared by every xy operation) - it REALIZES the "
            "tracked drive_power_dbm (binding above)",
        edit="scqo set QUBIT.drive_power_dbm=... (solves the chain, keeps "
             "drive_amp coupled, recorded); a direct edit silently re-scales "
             "what every stored pi_amp AND the absolute drive power mean",
        counterpart="drive-port output_att"),
}

#: The Octave's own knobs - the analog-upconversion counterpart of the block
#: above. Same job (put the tone at the right frequency at the right power),
#: entirely different vendor objects: the LO and the coarse power live on an
#: ``OctaveUpConverter`` COMPONENT rather than on a port, and the chain carries a
#: mixer calibration the MW-FEM has no equivalent of.
VENDOR_ONLY_OCTAVE: dict[str, VendorOnly] = {
    "readout_octave_lo_frequency": VendorOnly(
        path="q.resonator.frequency_converter_up.LO_frequency", unit="Hz",
        kind="vendor",
        doc="readout LO - the Octave up-converter's synthesizer. Grid "
            "[2 : 0.250 : 18] GHz, and IF = RF - LO must stay within "
            "+/-400 MHz. Like the MW-FEM upconverter it is shared by everything "
            "on that RF output, and many LO/IF splits give the SAME RF, so SCQO "
            "owns only the RF (readout_freq_hz) and never moves the LO in a "
            "chain solve",
        coupled=("octave_downconverter_lo_frequency",),
        edit="state.json octaves.<name>.RF_outputs.<n>.LO_frequency, offline "
             "with QUAM tools. The Octave SHARES synthesizers between RF "
             "outputs - synth1: RF1 + RFin1, synth2: RF2 + RF3, synth3: RF4 + "
             "RF5 - so two lines on one synth are FORCED to the same LO and "
             "moving one moves the other. The down-converter LO must move with "
             "it, and the mixer calibration for the new (output, LO, gain) has "
             "to be re-run",
        counterpart="readout_upconverter_frequency on an MW-FEM tree"),
    "drive_octave_lo_frequency": VendorOnly(
        path="q.xy.frequency_converter_up.LO_frequency", unit="Hz", kind="vendor",
        doc="drive LO - the same Octave synthesizer knob on the xy line, with "
            "the same [2 : 0.250 : 18] GHz grid and +/-400 MHz IF window",
        edit="state.json octaves.<name>.RF_outputs.<n>.LO_frequency, offline "
             "with QUAM tools; mind the shared synthesizers (RF2 + RF3, RF4 + "
             "RF5) and re-run the mixer calibration afterwards",
        counterpart="drive_upconverter_frequency on an MW-FEM tree"),
    "octave_downconverter_lo_frequency": VendorOnly(
        path="q.resonator.frequency_converter_down.LO_frequency", unit="Hz",
        kind="vendor",
        doc="receive-side LO on the Octave down-converter. It must equal the "
            "readout up-converter's LO or demodulation breaks. UNSET IS THE "
            "DANGEROUS CASE, and it is silent: quam's apply_to_config OMITS the "
            "whole RF_inputs entry when this is not a number, so the generated "
            "config simply has no receive path and reports no error",
        coupled=("readout_octave_lo_frequency",),
        edit="state.json octaves.<name>.RF_inputs.<n>.LO_frequency - normally a "
             "QUAM reference to the up-converter's LO, which is what keeps the "
             "two from drifting apart; prefer repairing that reference over "
             "writing a second literal",
        counterpart="downconverter_frequency on an MW-FEM tree"),
    "readout_octave_gain": VendorOnly(
        path="q.resonator.frequency_converter_up.gain", unit="dB", kind="realizer",
        doc="the coarse readout power knob on an Octave (grid -20..+20 dB in "
            "0.5 dB steps, per RF OUTPUT and shared by every channel on it) - "
            "it REALIZES the tracked readout_power_dbm together with "
            "readout_amp, which carries the residual in VOLTS",
        edit="scqo set QUBIT.readout_power_dbm=... (solves the chain and HOLDS "
             "this value wherever the amplitude can absorb the change - the "
             "gain is part of the mixer calibration's cache key, so moving it "
             "invalidates the stored LO-leakage correction for that RF output "
             "and, on a multiplexed feedline, changes every other channel's "
             "power with it)",
        counterpart="full_scale_power_dbm on an MW-FEM tree (dBm, 3 dB grid, "
                    "and free to move - it keys no calibration)"),
    "drive_octave_gain": VendorOnly(
        path="q.xy.frequency_converter_up.gain", unit="dB", kind="realizer",
        doc="the same coarse power knob on the drive line - it REALIZES the "
            "tracked drive_power_dbm together with drive_amp",
        edit="scqo set QUBIT.drive_power_dbm=... (solves the chain and HOLDS "
             "this value wherever the amplitude can absorb the change - the "
             "gain is part of the mixer calibration's cache key, so moving it "
             "invalidates the stored LO-leakage correction for that RF output)",
        counterpart="drive_full_scale_power_dbm on an MW-FEM tree"),
    "readout_octave_output_mode": VendorOnly(
        path="q.resonator.frequency_converter_up.output_mode", unit="",
        kind="vendor",
        doc="the Octave RF switch on the readout output: always_on, "
            "always_off, triggered or triggered_reversed. THE DEFAULT IS "
            "always_off - a tree assembled without setting it emits nothing at "
            "all, with no error anywhere, which is the first thing to check "
            "when a brand-new Octave setup measures a flat line",
        edit="state.json octaves.<name>.RF_outputs.<n>.output_mode; "
             "quam_config/populate_quam_opxp_octave.py sets always_on, and "
             "quam_builder's own transmon builder sets it when it builds the "
             "IQ path",
        counterpart="none - an MW-FEM has no RF switch"),
    "drive_octave_output_mode": VendorOnly(
        path="q.xy.frequency_converter_up.output_mode", unit="", kind="vendor",
        doc="the same RF switch on the drive output, with the same always_off "
            "default and the same silent consequence",
        edit="state.json octaves.<name>.RF_outputs.<n>.output_mode",
        counterpart="none - an MW-FEM has no RF switch"),
    "octave_downconverter_if_mode": VendorOnly(
        path="q.resonator.frequency_converter_down.IF_mode_I / .IF_mode_Q",
        unit="", kind="vendor",
        doc="how each down-converted quadrature reaches the OPX analog input: "
            "direct, envelope, mixer or off. Both quadratures normally run "
            "direct; anything else is a deliberate receive-path experiment",
        edit="state.json octaves.<name>.RF_inputs.<n>.IF_mode_I / IF_mode_Q",
        counterpart="none - the MW-FEM demodulates on the FEM itself"),
    "octave_calibration_db_path": VendorOnly(
        path="machine.octaves[<name>].calibration_db_path", unit="", kind="vendor",
        doc="where the Octave's MIXER CALIBRATION lives - the LO-leakage and "
            "image corrections, cached per (RF output, LO, gain) and per (that "
            "LO, IF). A SEPARATE file from state.json, so a setup snapshot does "
            "not contain it and two runs with identical QUAM state can still "
            "have been taken with different mixer corrections",
        edit="state.json octaves.<name>.calibration_db_path - point it at the "
             "setup's own backend_config/ folder. UNSET IS THE TRAP: quam falls "
             "back to os.getcwd(), so the calibration lands in whatever "
             "directory the process happened to start in and a later run "
             "silently finds none",
        counterpart="none - an MW-FEM needs no mixer calibration"),
}

#: The COMPLETE inventory, whatever the tree: what :func:`vendor_only_for`
#: filters, and what the structural self-checks run against. Kept whole on
#: purpose - a ``coupled`` name may cross families (an Octave LO couples to its
#: own down-converter LO), and the union is where every such name resolves.
VENDOR_ONLY: dict[str, VendorOnly] = {
    **VENDOR_ONLY_COMMON,
    **VENDOR_ONLY_MW_FEM,
    **VENDOR_ONLY_OCTAVE,
}

#: Which family block each RF chain brings in. Keyed by the strings
#: ``scqo_qm._family.rf_chain`` returns.
_VENDOR_ONLY_BY_CHAIN = {
    "mw_fem": VENDOR_ONLY_MW_FEM,
    "octave": VENDOR_ONLY_OCTAVE,
}


def vendor_only_for(rf_chains) -> dict[str, VendorOnly]:
    """The inventory for a tree whose channels run on ``rf_chains``.

    ``scqo state --fields`` is the only place an operator DISCOVERS these knobs,
    so it has to describe the instrument in front of them: an MW-FEM ``band`` on
    an Octave tree names a state.json key that hardware does not have, and the
    Octave's gain is invisible on an MW one. A tree that mixes chains gets both
    blocks, because it genuinely has both.

    An unrecognized chain contributes nothing and is not an error - the common
    block still describes everything that does not depend on the chain.
    """
    out = dict(VENDOR_ONLY_COMMON)
    for chain in rf_chains:
        out.update(_VENDOR_ONLY_BY_CHAIN.get(chain, {}))
    return out


#: The vendor OPERATOR CLIs this driver ships. They are not scqo subcommands
#: (scqo run <name> is the single entry point, and a QM-specific verb could only
#: be refused on Qblox), so `scqo -h` cannot list them - `scqo state --fields`
#: renders this inventory instead, which is the only place an operator discovers
#: them rather than memorizing them. Declared HERE and not in qm_backend.py
#: because it is pure declarative vendor metadata of the same class as
#: VENDOR_ONLY, and this module's import guard is what proves it stays
#: vendor-free. Every string below compresses the target module's own docstring.
OPERATOR_COMMANDS: tuple[OperatorCommand, ...] = (
    OperatorCommand(
        name="apply_distortion",
        command="python -m scqo_qm.backend.apply_distortion --target <target> "
                "[--run <run_id>]",
        doc="LF-FEM flux lines only. Write accepted cryoscope taps "
            "(distortion_amp / distortion_tau_s "
            "are FACTS - accepting them records the measurement and pushes "
            "NOTHING) into the target's z-output exponential filter. Run it "
            "after a cryoscope run's facts are accepted; it is the same command "
            "the cryoscope writeback hint prints for you, run-addressed. Fully "
            "offline - it never opens a QuantumMachinesManager.",
        options="--run RUN_ID (taps from that run's fit - the iteration door)  "
                "--extend (refine a residual instead of overwriting)  "
                "--form {sum,cascade} (QOP >= 3.3 / 3.4.1)  "
                "--clear (fresh-line reset before a clean-slate "
                "characterization)  --dry-run  --config PATH"),
    OperatorCommand(
        name="calibrate_octave",
        command="python -m scqo_qm.backend.calibrate_octave [--target <qubit>...]",
        doc="Octave trees only. Calibrate the Octave up-conversion mixers (LO "
            "leakage and image) for the active device/setup. An Octave mixes a "
            "baseband IQ pair up with an analog mixer, so every output carries "
            "leakage and an image sideband until it is calibrated at the exact "
            "(RF output, LO, gain) and (LO, IF) it will run at; an MW-FEM "
            "synthesizes microwave directly and has no such step. Re-run after "
            "any LO change (including one forced by a shared synthesizer), any "
            "GAIN change (gain is part of the cache key - unlike an MW-FEM's "
            "full_scale_power_dbm, which keys nothing), a new IF, or a cold "
            "start. Takes the cluster while it runs; not destructive.",
        options="--target QUBIT... (default: every active qubit)  "
                "--readout-only / --drive-only  --timeout S  --dry-run (say "
                "what and where, calibrate nothing)  --config PATH",
        caution="The results land in calibration_db.json, which is NOT part of "
                "state.json - a setup snapshot does not capture it, so two runs "
                "with identical QUAM state can carry different mixer "
                "corrections. If octaves.<name>.calibration_db_path is unset, "
                "quam falls back to os.getcwd() and the file follows whatever "
                "directory you launched from."),
    OperatorCommand(
        name="close_qm",
        command="python -m scqo_qm.backend.close_qm",
        doc="Halt running jobs and close the open Quantum Machines on the "
            "cluster serving the ACTIVE scqo device/setup - the recovery door "
            "when a crashed or abandoned session still holds the cluster's "
            "locks (symptom: a job that stalls forever, or an open that never "
            "returns). NOT a wedged-gateway fix: a cluster in "
            "DEADLINE_EXCEEDED needs a restart from its web UI.",
        options="--qm-id ID (just this one)  --dry-run (list what is open, "
                "close nothing)  --config PATH",
        caution="DESTRUCTIVE and there is NO confirmation prompt - halting a "
                "job discards data it had not yet streamed out, including a "
                "measurement someone else started. Run --dry-run first unless "
                "you know the cluster is idle."),
    OperatorCommand(
        name="register_partial_swap",
        command="python -m scqo_qm.backend.register_partial_swap --pair <pair> "
                "--name partial_swap_<t> --z-amp <V> --coupler-amp <V>",
        doc="Add or retune a square partial-swap operation on one qubit pair: "
            "the control qubit's z pulse (the swap resonance) and the coupler "
            "pulse (the angle), both named partial_swap_square_<t>, plus the "
            "ISwapImplementation macro partial_swap_<t> that experiments play "
            "by name (<t> = target angle x 100, three digits). Step 2 of "
            "SCQO/procedures/pair-partial-swap. The live state.json is "
            "replaced only after a staged save proves nothing else changes. "
            "Fully offline.",
        options="--update (retune an existing operation's amplitudes)  "
                "--length NS (new operation only, default 40)  --list (show "
                "the pairs' partial swaps, write nothing)  --dry-run  "
                "--config PATH",
        caution="Writes the setup's live state.json: a run in progress "
                "records the edit as setup-snapshot drift, so run it between "
                "measurements."),
)
