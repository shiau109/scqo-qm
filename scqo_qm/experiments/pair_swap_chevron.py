"""Single-excitation flux-chevron acquisition probe: vendor code only (qm/quam) -
no qualibrate, no scqo, no scqat.

Adapted from `calibrations/19_chevron_11_02.py` (the two-qubit CZ chevron). The
flux pulse still sweeps amplitude x duration on the control qubit's z line and
both qubits are still read out, but only **one** qubit of the pair is excited with
`x180` (selected by `drive_role`, default the control qubit) instead of preparing
|11>. There is no fit/state-writeback downstream; the node renders a 2D color map.

With state discrimination both qubits are read out 2-level and the saved data is the
joint two-qubit populations P00, P01, P10, P11 (first digit = control, second =
target) as variables `state_gg/state_ge/state_eg/state_ee` -- not the independent
single-qubit averages. Without state discrimination the raw I/Q of each qubit is saved.

The sub-4ns flux-pulse granularity uses the baking tool (`qualang_tools.bakery`):
short segments (1..16 ns) are baked into the config, and longer pulses combine a
baked tail with a dynamically stretched (multiple-of-4ns) `play`.

The amplitude sweep is in VOLTS, mirroring `pair_swap_flux_map`: the sweep values
ARE the emitted flux-pulse amplitudes. The QUAM |11>-|02> resonance formula is not
consulted at all — it is meaningless when the caller names volts, and a bring-up
tree may not carry the fields it reads. (The unitless pre-factor mode the retired
qualibrate node used went with it; the last release carrying it is v3.13.0.)

`flux_role` selects which qubit's z line carries the flux pulse; it defaults to
the control qubit, which is what this probe used to hardwire.

`coupler_amp` (volts, optional) holds the pair's COUPLER at a fixed amplitude for
the same window, playing the coupler-side waveform of the named pair macro
`swap_operation` -- so on a QCQ pair, where the swap only exists while the
coupler pulse plays, the arch is taken AT that coupler setting. It rescales the
same way the macro itself does (`cplr_amp / stored amplitude`), so a value found
here is the value `qc_unidirectional_trotter`'s `swap_coupler_flux` wants.
Engaging it SWITCHES OFF the baking branch: a baked segment plays on ONE
element, so the coupler could not be sample-aligned with it, and the whole
duration axis is therefore restricted to stretched `play(duration=)` on the 4 ns
clock. scqo builds the axis on that grid; this builder refuses anything else.

QM single-excitation swap chevron for scqo — supplies ``probe()``.

Parameters, the record-only map summary and the (absent) writeback are
inherited from ``scqo.experiments.PairSwapChevron``; the joint-population
reduction comes from :class:`JointPopulationMixin`. scqo sweeps
``(flux_amp_v, swap_time_ns)`` in absolute volts and whole nanoseconds; the
QM builder sweeps a QUA amplitude scale x a 1 ns-granular pulse duration
(baked below 4 ns), so this adapter maps the neutral high/low roles onto the
vendor's control/target. ``coupler_flux_v`` / ``swap_operation`` pass straight through:
when the first is set the builder also plays that macro's coupler pulse at that
fixed amplitude and the baking branch is not built, which is why scqo puts the
duration axis on the 4 ns grid for those runs.

Unlike every other QM adapter here, ``probe()`` ACQUIRES and returns a ready
``xr.Dataset``: the program only runs against the probe's own baked config
(which carries the ``flux_pulse1..16`` operations), and the backend's shared
fetch path would regenerate a config without them.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import xarray as xr
from qm.qua import *

from qualang_tools.bakery import baking
from qualang_tools.loops import from_array

from quam.components import pulses as quam_pulses

from scqo_qm.experiments._amp_limits import MAX_AMP_SCALE
from scqo_qm.experiments._coupler_knob import guard_coupler_amplitudes
from scqo_qm.experiments._lib import acquire as _acquire
from scqo_qm.experiments._flux_limits import (
    check_flux_pulse_relative,
    declared_idle_offset_v,
    flux_reference_amplitude,
)


def _flux_qubit(qp, flux_role: str):
    """Return the qubit of the pair whose z line carries the flux pulse."""
    return qp.qubit_target if flux_role == "target" else qp.qubit_control


def resolve_amplitudes(qubit_pairs, amplitudes, *, flux_role: str):
    """Resolve the amplitude sweep into what the QUA program actually needs.

    Returns `(qua_amps, base_levels, denoms)`:
      - `qua_amps` is the array the single shared QUA loop iterates (the value the
        baked pulse's `amp_array` scales by);
      - `base_levels[pair]` is the level the 1..16 ns segments are baked at;
      - `denoms[pair]` is the stored amplitude of the z line's `const` operation,
        which the >16 ns `play` branch divides by.

    Both QUA branches emit `base_level * qua_amp` volts -- the baked branch scales
    the baked waveform directly, the play branch plays `const` at
    `(base_level/denom) * qua_amp`. That equality is the invariant this function
    has to preserve, and it is what `tests/test_pair_swap_probes.py` pins.

    The sweep is volts: one
    reference `amp_ref = max|amplitudes|` becomes the baked level for every pair
    and `qua_amps = amplitudes / amp_ref`, so the emitted volts equal the swept
    value on both branches. The reference must be a SCALAR because
    `for_(*from_array(a, ...))` is one loop shared by every multiplexed pair;
    deriving it from the sweep also makes the stored waveform peak equal the
    sweep's own maximum, so the DAC-rail guard below becomes the physically
    meaningful one ("you asked to emit more than the rail can").

    Pure: no QUA, no config, no baking -- callable from a test with stub pairs.
    """
    if flux_role not in ("control", "target"):
        raise ValueError(f"flux_role must be 'control' or 'target', got {flux_role!r}")

    amplitudes = np.asarray(amplitudes, dtype=float)
    if amplitudes.size == 0:
        raise ValueError("amplitudes is empty; nothing to sweep.")

    # The z line carrying the pulse must exist and own the `const` operation the
    # >16 ns branch stretches. `flux_reference_amplitude` enforces the rail/2
    # convention on it (the rail being PER PORT, direct 0.5 V vs amplified 2.5 V).
    denoms = {}
    for qp in qubit_pairs:
        fq = _flux_qubit(qp, flux_role)
        if fq.z is None:
            raise ValueError(
                f"{flux_role} qubit of {qp.name} ({fq.name}) has no z line; cannot play a flux pulse."
            )
        if "const" not in fq.z.operations:
            raise ValueError(
                f"z line of {fq.name} has no operation 'const'; available: {list(fq.z.operations)}"
            )
        denoms[qp.name] = flux_reference_amplitude(fq.z, name=fq.name, operation="const")

    amp_ref = float(np.max(np.abs(amplitudes)))
    if amp_ref == 0.0:
        raise ValueError(
            "an all-zero amplitude sweep has no reference to bake against; "
            "widen the window."
        )
    for qp in qubit_pairs:
        z = _flux_qubit(qp, flux_role).z
        check_flux_pulse_relative(
            z, name=f"{qp.name} flux sweep on {_flux_qubit(qp, flux_role).name}.z",
            idle_v=declared_idle_offset_v(z), amps_v=amplitudes, operation="const")
    base_levels = {qp.name: amp_ref for qp in qubit_pairs}
    qua_amps = amplitudes / amp_ref

    # THREE guards left with the pre-factor mode rather than being kept as
    # defence in depth, because none of them could fire any more, and a check
    # that cannot fire is the defect BACKLOG I17 is about:
    #   * two pairs on one flux ELEMENT needing different baked levels -- the
    #     baked level is now ONE shared reference, so they cannot differ;
    #   * `max|qua_amps| >= MAX_AMP_SCALE` -- qua_amps is
    #     `amplitudes / max|amplitudes|`, so its maximum is exactly 1.0;
    #   * the >16 ns branch's `amp_ref/const >= MAX_AMP_SCALE` -- that is
    #     `peak/reference` against the same stored `const` op, which
    #     `check_flux_pulse_relative` above already refuses, FIRST and with the
    #     reachable excursion spelled out.
    return qua_amps, base_levels, denoms


#: QM plays a stretched pulse in whole 4 ns clock cycles, with a 4-cycle floor.
_CLOCK_NS = 4
_MIN_FLUX_NS = 16


def resolve_coupler_plays(qubit_pairs, coupler_amp, swap_operation: str,
                          times_cycles) -> dict:
    """``{pair name: (coupler element, pulse name, amplitude_scale)}``, or ``{}``.

    ``{}`` -- the falsy answer -- is a run that leaves the coupler alone, which is
    every chevron this probe played before the knob existed. Otherwise every pair
    contributes one entry and the map is truthy, so callers branch on it directly.

    Three kinds of refusal, all BEFORE any QUA is built, each with its own fix:

    * the shared :func:`guard_coupler_amplitudes` covers the macro (missing), the
      pair (no coupler), the macro's coupler side (no playable pulse), the stored
      amplitude (baked at zero -- unsettable, since it is the divisor) and the
      port (rail, idle sum, and QUA's amplitude_scale bound);
    * a SHAPED coupler waveform is refused here. ``play(duration=)`` stretches a
      constant pulse and ZERO-PADS everything else, so a raised-cosine coupler
      would silently play its native length inside a longer window and the map
      would be of a pulse nobody asked for;
    * a duration off the 4 ns grid (or under 16 ns) is refused for the same
      reason the bakes are skipped: with the coupler engaged there is no
      sub-clock path left. scqo already builds the axis on that grid, so this is
      the vendor-side backstop, not the primary check.

    The scale is a Python float and stays a compile-time constant: the coupler
    amplitude is FIXED for the whole map, unlike the member's swept one.
    """
    if coupler_amp is None:
        return {}
    amp = float(coupler_amp)
    bad = [int(t) for t in np.asarray(times_cycles, dtype=int)
           if int(t) % _CLOCK_NS or int(t) < _MIN_FLUX_NS]
    if bad:
        raise ValueError(
            f"coupler_amp is set, so every duration must be a whole multiple of "
            f"{_CLOCK_NS} ns and at least {_MIN_FLUX_NS} ns -- a coupler pulse can "
            f"only be STRETCHED, never baked sub-clock alongside the member's. "
            f"Off-grid: {sorted(set(bad))[:8]}. Raise min_swap_time_ns to "
            f"{_MIN_FLUX_NS} or leave coupler_flux_v unset.")
    plays = {}
    for qp in qubit_pairs:
        coupler, pulse_name = guard_coupler_amplitudes(
            qp, swap_operation, [amp],
            why=" This map holds it at a fixed amplitude while it sweeps the "
                "member's flux pulse, so the arch is the one that pair has AT "
                "that coupler setting.",
            label=f"{qp.name} macro {swap_operation!r} coupler_flux_v")
        pulse = coupler.operations[pulse_name]
        if not isinstance(pulse, quam_pulses.SquarePulse):
            raise ValueError(
                f"{qp.name}: the coupler pulse {pulse_name!r} of macro "
                f"{swap_operation!r} is a {type(pulse).__name__}, not a SquarePulse. "
                f"This map overrides the pulse DURATION, which stretches a constant "
                f"waveform but ZERO-PADS a shaped one -- the coupler would play its "
                f"native length inside every window and the map would be of a pulse "
                f"you did not ask for. Use a macro whose coupler pulse is square "
                f"(register one with quam_config/register_swap_macro.py), or leave "
                f"coupler_flux_v unset.")
        plays[qp.name] = (coupler, pulse_name, amp / float(pulse.amplitude))
    return plays


def baked_waveform(qubit, baked_config, base_level: float = 0.5, max_samples: int = 16):
    """Create truncated baked waveforms for the chevron flux pulse.

    Generates a list of baking objects, each containing an incrementally longer flux pulse
    (1..max_samples samples) at the specified base_level. Each baked pulse is registered
    as an operation named "flux_pulse{i}" on the provided qubit z line.

    Copied from `calibration_utils.chevron_cz.parameters.baked_waveform` so this probe
    stays free of qualibrate imports.

    Returns a list of baking objects; index i corresponds to a pulse of i+1 samples.
    """
    pulse_segments = []
    waveform = [base_level] * max_samples
    for i in range(1, max_samples + 1):
        with baking(baked_config, padding_method="right") as b:
            wf = waveform[:i]
            b.add_op(f"flux_pulse{i}", qubit.z.name, wf)
            b.play(f"flux_pulse{i}", qubit.z.name)
        pulse_segments.append(b)
    return pulse_segments


def build_program(
    machine,
    qubit_pairs,
    *,
    amplitudes,
    times_cycles,
    num_shots: int,
    reset_type: str,
    use_state_discrimination: bool,
    flux_role: str = "control",
    drive_role: str = "control",
    coupler_amp: Optional[float] = None,
    swap_operation: str = "partial_swap",
    simulate: bool = False,
):
    """Build the single-excitation flux-chevron QUA program.

    Returns (program, sweep_axes, baked_config). The returned `baked_config` carries
    the baked flux-pulse operations and MUST be the config used to execute (pass it to
    `acquire(..., config=baked_config)`); a freshly generated config would lack them.

    `amplitudes` is the flux-pulse amplitude sweep, in absolute volts (see the module
    docstring and `resolve_amplitudes`).
    `times_cycles` is the pulse-duration sweep in ns; `qubit_pairs` is a BatchableList
    of qubit pairs (`qubit_pairs.batch()` / `.get_names()`). `flux_role` selects which
    qubit of each pair carries the flux pulse and `drive_role` which receives the x180
    ("control" or "target"); the two are independent.

    `coupler_amp` is the FIXED coupler amplitude in absolute volts, or None to
    leave the coupler alone (the historical behaviour). When it is set, the
    coupler-side pulse of `swap_operation` plays for the same window on every
    pair and every duration must be a whole number of 4 ns clock cycles -- the
    baked sub-clock branch is not built at all, because a bake carries one
    element and the coupler would drift out of it.
    """
    num_qubit_pairs = len(qubit_pairs)

    qua_amps, base_levels, denoms = resolve_amplitudes(
        qubit_pairs, amplitudes, flux_role=flux_role
    )
    coupler_plays = resolve_coupler_plays(
        qubit_pairs, coupler_amp, swap_operation, times_cycles)

    sweep_axes = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        # the axis carries what the CALLER swept, not the QUA scale factor
        "amplitude": xr.DataArray(
            np.asarray(amplitudes, dtype=float),
            attrs={"long_name": "amplitudes of the flux pulse", "units": "V"},
        ),
        "time": xr.DataArray(times_cycles, attrs={"long_name": "pulse duration", "units": "ns"}),
    }

    baked_config = machine.generate_config()

    # Pre-compute the baked short segments (1..16 samples) for each flux qubit in
    # the pairs -- but ONLY on the coupler-free path. A bake plays one element, so
    # with the coupler engaged those segments are unreachable (the duration axis is
    # on the 4 ns grid) and baking them would upload waveforms nothing can play.
    baked_signals = {} if coupler_plays else {
        _flux_qubit(qp, flux_role).name: baked_waveform(
            _flux_qubit(qp, flux_role), baked_config,
            base_level=base_levels[qp.name], max_samples=16
        )
        for qp in qubit_pairs
    }

    with program() as prog:
        t = declare(int)  # QUA variable for the flux pulse segment index
        a = declare(fixed)
        t_left_ns = declare(int)
        t_cycles = declare(int)
        # Each pair's stretched-branch amplitude_scale, (base_level / denom) * a,
        # assigned AHEAD of the reset rather than computed in front of the z play:
        # there it would hold back the z pulse alone, and on the coupled path the
        # coupler pulse beside it (a compile-time constant) would not wait.
        z_scl = [declare(fixed) for _ in range(num_qubit_pairs)]
        I_c, I_c_st, Q_c, Q_c_st, n, n_st = machine.declare_qua_variables()
        I_t, I_t_st, Q_t, Q_t_st, _, _ = machine.declare_qua_variables()
        if use_state_discrimination:
            # Per-shot single-qubit outcomes (both qubits read out 2-level -> {0, 1}).
            state_c = [declare(int) for _ in range(num_qubit_pairs)]
            state_t = [declare(int) for _ in range(num_qubit_pairs)]
            # Joint two-qubit indicators (first digit = control, second = target);
            # averaged over shots they give the P00/P01/P10/P11 populations.
            ind_gg = declare(int)  # 00
            ind_ge = declare(int)  # 01
            ind_eg = declare(int)  # 10
            ind_ee = declare(int)  # 11
            state_gg_st = [declare_stream() for _ in range(num_qubit_pairs)]
            state_ge_st = [declare_stream() for _ in range(num_qubit_pairs)]
            state_eg_st = [declare_stream() for _ in range(num_qubit_pairs)]
            state_ee_st = [declare_stream() for _ in range(num_qubit_pairs)]

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qp in multiplexed_qubit_pairs.values():
                machine.initialize_qpu(target=qp.qubit_control)
                machine.initialize_qpu(target=qp.qubit_target)
            align()
            # Averaging loop
            with for_(n, 0, n < num_shots, n + 1):
                save(n, n_st)
                # Pulse amplitude loop (the QUA scale factor; see resolve_amplitudes)
                with for_(*from_array(a, qua_amps)):
                    ################################################################################################
                    # The duration argument in the play command can only produce pulses with duration multiple of  #
                    # 4ns. To overcome this limitation we use the baking tool from the qualang-tools package to    #
                    # generate pulses with 1ns granularity. To avoid creating custom waveforms for each iteration  #
                    # we combine baked pulses with dynamically stretched (multiple of 4ns) pulses.                 #
                    ################################################################################################
                    with for_(*from_array(t, times_cycles)):
                        for ii, qp in multiplexed_qubit_pairs.items():
                            fq = _flux_qubit(qp, flux_role)
                            # the stretched play's scale, computed before the reset
                            # (see the declaration above)
                            assign(z_scl[ii], (base_levels[qp.name] / denoms[qp.name]) * a)
                            # Qubit initialization
                            qp.qubit_control.reset(reset_type, simulate)
                            qp.qubit_target.reset(reset_type, simulate)
                            align()
                            # Excite only one qubit of the pair (single-excitation chevron).
                            if drive_role == "target":
                                qp.qubit_target.xy.play("x180")
                            else:
                                qp.qubit_control.xy.play("x180")

                            align()

                            if not coupler_plays:
                                # For the first 16ns we play baked pulses exclusively. Loop the time index until 16.
                                with if_(t <= 16):
                                    with switch_(t):
                                        # Switch case to select the baked pulse with duration t ns
                                        for j in range(1, 17):
                                            with case_(j):
                                                baked_signals[fq.name][j - 1].run(
                                                    amp_array=[(fq.z.name, a)]
                                                )

                                # For pulse durations above 16ns we combine baking with regular play statements.
                                with else_():
                                    # We calculate the closest lower multiple of 4 of the time index
                                    assign(t_cycles, t >> 2)  # Right shift by 2 is a quick way to divide by 4
                                    # Calculate the duration to add to pulse multiple of 4.
                                    assign(t_left_ns, t - (t_cycles << 2))  # left shift by 2 to multiply by 4
                                    # Switch case with the 4 possible sequences:
                                    with switch_(t_left_ns):
                                        # Play only the pulse multiple of 4
                                        with case_(0):
                                            align()
                                            fq.z.play(
                                                "const",
                                                duration=t_cycles,
                                                amplitude_scale=z_scl[ii],
                                            )
                                        # Play the pulse multiple of 4 followed by the baked pulse of the missing duration
                                        for j in range(1, 4):
                                            with case_(j):
                                                align()
                                                with strict_timing_():
                                                    fq.z.play(
                                                        "const",
                                                        duration=t_cycles,
                                                        amplitude_scale=z_scl[ii],
                                                    )
                                                    baked_signals[fq.name][j - 1].run(
                                                        amp_array=[(fq.z.name, a)]
                                                    )
                            else:
                                # The COUPLED path: no baking, so no sub-clock
                                # branch and no `t_left_ns` remainder -- the axis
                                # is on the 4 ns grid (resolve_coupler_plays
                                # refuses anything else), so `t >> 2` IS the
                                # duration. The two plays are issued back to back
                                # with no align between them, so they start on the
                                # same edge on their two elements -- the same
                                # shape pair_swap_flux_map uses.
                                align()
                                assign(t_cycles, t >> 2)
                                coupler, coupler_pulse, c_scale = coupler_plays[qp.name]
                                fq.z.play(
                                    "const",
                                    duration=t_cycles,
                                    amplitude_scale=z_scl[ii],
                                )
                                coupler.play(
                                    coupler_pulse,
                                    duration=t_cycles,
                                    amplitude_scale=c_scale,
                                )
                            align()

                            if use_state_discrimination:
                                qp.qubit_control.readout_state(state_c[ii])
                                qp.qubit_target.readout_state(state_t[ii])
                                # Joint-state indicators from the two binary outcomes:
                                #   ee(11)=c*t, eg(10)=c-ee, ge(01)=t-ee, gg(00)=1-c-t+ee
                                assign(ind_ee, state_c[ii] * state_t[ii])
                                assign(ind_eg, state_c[ii] - ind_ee)
                                assign(ind_ge, state_t[ii] - ind_ee)
                                assign(ind_gg, 1 - state_c[ii] - state_t[ii] + ind_ee)
                                save(ind_gg, state_gg_st[ii])
                                save(ind_ge, state_ge_st[ii])
                                save(ind_eg, state_eg_st[ii])
                                save(ind_ee, state_ee_st[ii])
                            else:
                                qp.qubit_control.resonator.measure("readout", qua_vars=(I_c[ii], Q_c[ii]))
                                qp.qubit_target.resonator.measure("readout", qua_vars=(I_t[ii], Q_t[ii]))
                                save(I_c[ii], I_c_st[ii])
                                save(Q_c[ii], Q_c_st[ii])
                                save(I_t[ii], I_t_st[ii])
                                save(Q_t[ii], Q_t_st[ii])

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubit_pairs):
                if use_state_discrimination:
                    # Averaging the 0/1 indicators over shots yields the joint populations.
                    state_gg_st[i].buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(f"state_gg{i}")
                    state_ge_st[i].buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(f"state_ge{i}")
                    state_eg_st[i].buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(f"state_eg{i}")
                    state_ee_st[i].buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(f"state_ee{i}")
                else:
                    I_c_st[i].buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(f"I_control{i}")
                    Q_c_st[i].buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(f"Q_control{i}")
                    I_t_st[i].buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(f"I_target{i}")
                    Q_t_st[i].buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(f"Q_target{i}")

    return prog, sweep_axes, baked_config


def acquire(
    machine,
    prog,
    sweep_axes,
    *,
    num_shots: int,
    timeout: float,
    log: Optional[Callable] = None,
    config: Optional[dict] = None,
) -> xr.Dataset:
    """Connect to the QOP, execute the program and fetch the raw xr.Dataset.

    Pass the baked config returned by `build_program` as `config`; the shared
    execute-and-fetch helper would otherwise regenerate a config without the baked ops.
    """
    return _acquire(machine, prog, sweep_axes, num_shots=num_shots, timeout=timeout, log=log, config=config)


from typing import Any

import numpy as np
import xarray as xr
from scqo import register
from scqo.experiments import PairSwapChevron

from ._pair_roles import JointPopulationMixin


@register
class QMPairSwapChevron(JointPopulationMixin, PairSwapChevron):
    """Build, run and fetch the multiplexed swap chevron on the QM OPX."""

    # preview opt-out (backend.SELF_ACQUIRING_ATTR): truthy reason = refuse
    probe_self_acquires = ("it bakes a per-call config for the sub-17 ns "
                           "branch and fetches against it inside probe()")

    def probe(self) -> Any:
        from ._reset import check_reset_method
        from ._vendor import role_side, vendor_pair_name
        from scqo_qm.experiments._lib import select_qubit_pairs

        machine = self.backend.machine  # type: ignore[attr-defined]
        # targets are ROSTER composite names; the probe helper selects by QUAM
        # pair key (QM names its pairs after the coupler), so translate first —
        # order preserved, which is what the axis relabel below relies on.
        vendor_names = [vendor_pair_name(self, p) for p in self.params.targets]
        pairs = select_qubit_pairs(machine, vendor_names, multiplexed=True)

        # Resolved BEFORE any QUA is built, so a roster/params mismatch refuses
        # without costing instrument time. `high` fixes the reduce_raw
        # orientation; the two selectors are independent of each other.
        self._high_side = role_side(self, "high", field="targets")
        drive_role = role_side(self, self.params.drive_side, field="drive_side")
        flux_role = role_side(self, self.params.flux_side, field="flux_side",
                              needs_flux=True)

        # The probe's `times_cycles` argument is ns despite its name (it plays
        # 1..16 ns baked segments below the 4 ns clock), and it must be an
        # integer array — from_array loops it into an int QUA variable.
        times_ns = np.round(self.sweep_axes["swap_time_ns"]).astype(int)
        prog, axes, baked_config = build_program(
            machine,
            pairs,
            amplitudes=self.sweep_axes["flux_amp_v"],
            times_cycles=times_ns,
            num_shots=self.params.num_averages,
            reset_type=check_reset_method(self),
            use_state_discrimination=True,
            flux_role=flux_role,
            drive_role=drive_role,
            coupler_amp=self.params.coupler_flux_v,
            swap_operation=self.params.swap_operation,
        )
        # The canonical time axis is the probe's REAL grid: re-declare it so
        # sizes and values match the raw data exactly.
        self.sweep_axes["swap_time_ns"] = axes["time"].values.astype(float)
        sweep_axes = {
            # The probe labels its target axis with VENDOR pair keys; scqo's
            # dataset (and estimate()) key on the ROSTER composite names the
            # operator asked for. Same order by construction (vendor_names was
            # built from targets), so relabel rather than rename downstream.
            "qubit_pair": xr.DataArray(list(self.params.targets)),
            "flux_amp_v": axes["amplitude"],
            "swap_time_ns": axes["time"],
        }
        # Acquire here: the baked config is per-call and cannot be reached
        # through the backend's (program, axes, module) shape.
        shots = getattr(self.params, "num_averages", None) or getattr(
            self.params, "num_shots", 1)
        return acquire(
            machine, prog, sweep_axes,
            num_shots=int(shots), timeout=self.backend._timeout,
            config=baked_config,
        )
