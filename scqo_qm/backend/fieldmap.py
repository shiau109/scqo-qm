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

from scqo.fieldmap import Unrealized, VendorBinding, VendorOnly

FIELD_BINDINGS: dict[str, dict[str, VendorBinding]] = {
    "drive": {
        "drive_freq_hz": VendorBinding(
            path="q.f_01", unit="Hz",
            convert="a write also shifts q.xy.RF_frequency by the same delta "
                    "(quam_fields.set_drive_freq keeps the drive line on the qubit)"),
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
            path="q.xy.opx_output.full_scale_power_dbm "
                 "+ q.xy.operations['saturation'].amplitude",
            unit="dBm + amp",
            convert="solve the DRIVE chain (power_tools): SMALLEST full_scale_power_dbm "
                    "on the -11..+16 dBm grid (3 dB steps) keeping the saturation "
                    "amplitude <= 0.5; the amplitude carries the exact residual",
            coupled=("drive_amp",),
            note="the xy full scale is PORT-level and shared by every xy operation: "
                 "while it is off its standing value the stored pi_amp means a "
                 "different power (qubit_spectroscopy sets it and reverts exactly)",
        ),
    },
    "readout": {
        "readout_freq_hz": VendorBinding(
            path="q.resonator.RF_frequency", unit="Hz",
            note="q.resonator.f_01 is kept equal when the resonator carries it"),
        "readout_amp": VendorBinding(
            path="q.resonator.operations['readout'].amplitude", unit=""),
        "readout_power_dbm": VendorBinding(
            path="q.resonator.opx_output.full_scale_power_dbm "
                 "+ q.resonator.operations['readout'].amplitude",
            unit="dBm + amp",
            convert="solve the output chain (power_tools): SMALLEST full_scale_power_dbm "
                    "on the -11..+16 dBm grid (3 dB steps) keeping the amplitude <= 0.5; "
                    "the amplitude carries the exact residual",
            coupled=("readout_amp",),
            note="MW-FEM full-scale grid: -11..+16 dBm in 3 dB steps",
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
                 "coincide. PORT-level (LFFEMAnalogOutputPort.delay, shared by "
                 "everything on that DAC output) - same class as "
                 "full_scale_power_dbm; on a per-qubit z wire it is per-qubit in "
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
VENDOR_ONLY: dict[str, VendorOnly] = {
    "readout_length": VendorOnly(
        path="q.resonator.operations['readout'].length", unit="ns", kind="realizer",
        doc="readout pulse length - realizes the TRACKED readout_duration_s "
            "(a direct edit silently de-calibrates it; the governed write is "
            "scqo set QUBIT.readout_duration_s=...). The integration window is "
            "NOT fused to it: readout_integration_s owns the weights support "
            "(default weights only LOOK fused - they span the pulse by reference)"),
    "readout_integration_weights": VendorOnly(
        path="q.resonator.operations['readout'].integration_weights", unit="",
        kind="realizer",
        doc="integration-weights list - its nonzero SUPPORT realizes the "
            "TRACKED readout_integration_s (governed write: scqo set "
            "QUBIT.readout_integration_s=...; the setter writes constant "
            "zero-padded weights). The SHAPE within the window stays vendor "
            "territory (a future weight-optimization node may write it; any "
            "later window write rebuilds constant weights)"),
    "time_of_flight": VendorOnly(
        path="q.resonator.time_of_flight", unit="ns", kind="vendor",
        doc="acquisition latency compensation - aligns the instrument's receive "
            "path with its own transmit path. The TOF measurement's product is "
            "written HERE, in NANOSECONDS, offline - never a neutral field. "
            "Qblox counterpart: measure.acq_delay (s)"),
    # NOTE: depletion_time is no longer VendorOnly - it REALIZES the tracked
    # readout_depletion_s (binding above). It was a hand-set policy value sitting
    # at QUAM's 16 ns default with nothing governing it, while QUAM spent it in
    # four places (after every measurement, and depletion_time // 2 inside
    # reset_qubit_active). resonator_spectroscopy now calibrates it from the
    # measured linewidth; the governed write is
    # scqo set QUBIT.readout_depletion_s=... .
    "readout_upconverter_frequency": VendorOnly(
        path="q.resonator.opx_output.upconverter_frequency", unit="Hz", kind="vendor",
        doc="readout LO - the MW-FEM upconverter, PORT-level (state.json "
            "ports.mw_outputs.<con>.<fem>.<port>) and shared by everything on "
            "that output; many LO/IF splits give the SAME RF, so SCQO owns only "
            "the RF (readout_freq_hz) and never moves the LO in a chain solve. "
            "Move it so IF = RF - LO stays in range, the port band must cover "
            "the target, and downconverter_frequency MUST move with it or "
            "demodulation breaks. Qblox counterpart: modulation_frequencies "
            "lo_freq"),
    "drive_upconverter_frequency": VendorOnly(
        path="q.xy.opx_output.upconverter_frequency", unit="Hz", kind="vendor",
        doc="drive LO - PORT-level MW-FEM upconverter, shared; keep "
            "IF = f_01 - LO in range and the port band matching"),
    "downconverter_frequency": VendorOnly(
        path="q.resonator.opx_input.downconverter_frequency", unit="Hz", kind="vendor",
        doc="receive-side downconversion LO on the MW input port (chipA: "
            "6.06 GHz, equal to the readout upconverter) - MUST track "
            "readout_upconverter_frequency or demodulation breaks; PORT-level, "
            "band-constrained. Qblox has no separate knob (NCO handles it)"),
    "full_scale_power_dbm": VendorOnly(
        path="q.resonator.opx_output.full_scale_power_dbm", unit="dBm", kind="realizer",
        doc="the coarse readout power knob (grid -11..+16 in 3 dB steps, "
            "PORT-level - shared like the LO) - it REALIZES the tracked "
            "readout_power_dbm (binding above). Change power with "
            "`scqo set QUBIT.readout_power_dbm=...` (solves the chain, keeps "
            "readout_amp coupled, recorded); a direct edit silently "
            "de-calibrates the absolute power, and any later readout_power_dbm "
            "write re-solves and overwrites a forced value"),
    "drive_full_scale_power_dbm": VendorOnly(
        path="q.xy.opx_output.full_scale_power_dbm", unit="dBm", kind="realizer",
        doc="the coarse DRIVE power knob (grid -11..+16 in 3 dB steps, "
            "PORT-level - shared by every xy operation) - it REALIZES the "
            "tracked drive_power_dbm (binding above). Change power with "
            "`scqo set QUBIT.drive_power_dbm=...` (solves the chain, keeps "
            "drive_amp coupled, recorded); a direct edit silently re-scales "
            "what every stored pi_amp AND the absolute drive power mean. "
            "Qblox counterpart: drive-port output_att"),
    "x180_length": VendorOnly(
        path="q.xy.operations['x180'].length", unit="ns", kind="realizer",
        doc="pi/x180 pulse length - it REALIZES the tracked pi_duration_s "
            "(promoted to a neutral drive knob in the greenfield catalog; "
            "binding above). The governed write is scqo set "
            "QUBIT.pi_duration_s=...; a direct edit silently de-calibrates the "
            "stored pi_amp with it. Multiple of 4 ns (chipA: 32 ns here vs "
            "200 ns on Qblox - genuinely per-chain calibrated)"),
    "x90_length": VendorOnly(
        path="q.xy.operations['x90_DragCosine'].length", unit="ns", kind="vendor",
        doc="pi/2 pulse length - a PER-GATE vendor value, deliberately NOT "
            "locked to the tracked pi_duration_s (the neutral knob is the pi "
            "pulse's length only); edit it directly when a chip wants a "
            "different x90 envelope"),
    "drag_alpha": VendorOnly(
        path="q.xy.operations['<gate>_DragCosine'].alpha", unit="", kind="realizer",
        doc="PER-GATE DRAG coefficient (chipA: x180 -0.94, x90 -0.50). The x180 "
            "node now REALIZES the tracked neutral drag_beta (binding above; "
            "governed write: scqo set QUBIT.drag_beta=...); the OTHER gates' "
            "alpha values remain vendor fine print edited directly. Qblox "
            "counterpart: rxy.beta (derivative scale, different math convention)"),
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
            "outcome; flipping it re-points idle_flux at a different stored "
            "number, so re-seed after changing it. Under scqo it is PINNED to "
            "'joint' (the point every probe's initialize_qpu applies) and the "
            "backend factory refuses anything else - a declaration that "
            "disagrees with the applied bias makes idle_flux inert"),
    "coupler_flux_point": VendorOnly(
        path="qp.coupler.flux_point", unit="", kind="vendor",
        doc="which named coupler point idles (off/on/arbitrary/zero) - SELECTS "
            "which offset the tracked idle_flux reads and writes on the "
            "COUPLER mode's flux channel (q1_q2_c_z). A mode switch, not a "
            "calibration outcome"),
    "coupler_decouple_offset": VendorOnly(
        path="qp.coupler.decouple_offset", unit="V", kind="realizer",
        doc="the interaction-OFF coupler standing bias (pair_zz_coupler's "
            "product - the ZZ zero crossing). It REALIZES the tracked "
            "idle_flux of the coupler mode's flux channel while "
            "coupler.flux_point == 'off' (the old pair-level coupler_decouple_v "
            "neutral field is GONE - the governed write is now "
            "scqo set <coupler>_z.idle_flux=...)"),
    "coupler_interaction_offset": VendorOnly(
        path="qp.coupler.interaction_offset", unit="V", kind="realizer",
        doc="the interaction-ON coupler standing bias (gate operating point). "
            "It REALIZES the tracked idle_flux of the coupler mode's flux "
            "channel while coupler.flux_point == 'on'. A per-GATE operating "
            "point is NOT this: that is the composite knob "
            "<operation>_coupler_flux (bound above)"),
    "coupler_arbitrary_offset": VendorOnly(
        path="qp.coupler.arbitrary_offset", unit="V", kind="realizer",
        doc="free-form coupler bias for exploratory work - realizes idle_flux "
            "while coupler.flux_point == 'arbitrary'"),
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
