"""Fixed-time qubit-flux x coupler-flux acquisition probe: vendor code only (qm/quam) -
no qualibrate, no scqo, no scqat.

A fixed-time 2D variant of `pair_swap_chevron`. Instead of sweeping a flux
amplitude x duration, here the **duration is fixed** and **two flux amplitudes are
swept**, forming the x/y axes of a 2D color map:

  - x axis: the **coupler** flux amplitude (played on `qp.coupler`),
  - y axis: a **qubit** flux amplitude (played on the `flux_role` qubit's `z` line,
    control or target).

Both flux pulses play simultaneously over the same fixed window (the dual-flux pattern
from the archived `calibrations/exclude/LCH_iswap_fixed_time_search.py`). One qubit of the pair is excited
with `x180` (selected by `drive_role`, default the control qubit); both qubits are read
out. There is no fit/state-writeback downstream; the node renders a 2D color map.

The flux pulses ride on top of the idle DC biases set by `machine.initialize_qpu`. The
duration is fixed (a multiple of 4 ns), so no baking is needed -- the QUA
`play(..., duration=...)` override is enough.

Amplitude sweep (applied to BOTH channels): the sweep values are pulse amplitudes in
volts. Each channel plays `amplitude_scale = a / ref` where `ref` is that channel op's
stored amplitude, so the emitted pulse equals the swept value. The resulting scale must
stay inside QUA's (-2, 2) dynamic-amplitude range or a ValueError is raised before
building. The division is assigned to a per-pair variable AHEAD of the reset, so the two
flux plays carry plain variables: computed in front of each play instead, it would hold
back that element alone, and the qubit and coupler pulses of one swap could start on
different clock edges. (The unitless pre-factor mode the retired qualibrate side used
went with it; the last release carrying it is v3.13.0.)

With state discrimination both qubits are read out 2-level and the saved data is the
joint two-qubit populations P00/P01/P10/P11 (first digit = control, second = target) as
variables `state_gg/state_ge/state_eg/state_ee`. Without state discrimination the raw
I/Q of each qubit is saved.

QM fixed-time coupler-flux x qubit-flux swap map for scqo — supplies ``probe()``.

Parameters, the record-only map summary and the (absent) writeback are inherited
from ``scqo.experiments.PairSwapFluxMap``; the joint-population reduction comes
from :class:`JointPopulationMixin`. scqo sweeps ``(qubit_flux_v,
coupler_flux_v)`` in absolute volts — which is exactly what this builder sweeps,
so the adapter's work is the role mapping, the neutral pulse SHAPE -> QUAM
operation name, and quantizing the fixed duration onto the 4 ns clock.

The probe's debug-only path ``swap_via_macro`` is deliberately not reachable from
here: it exists to isolate a vendor macro against a direct play.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import xarray as xr
from qm.qua import *

from qualang_tools.loops import from_array

from scqo_qm.experiments._lib import acquire as _acquire
from scqo_qm.experiments._flux_limits import (
    check_flux_pulse_relative,
    declared_idle_offset_v,
)


def _flux_qubit(qp, flux_role: str):
    """Return the qubit of the pair whose z line carries the qubit flux pulse."""
    return qp.qubit_target if flux_role == "target" else qp.qubit_control


def build_program(
    machine,
    qubit_pairs,
    *,
    coupler_amplitudes,
    qubit_amplitudes,
    coupler_operation: str = "const",
    qubit_operation: str = "const",
    flux_time: Optional[int] = None,
    flux_role: str = "control",
    num_shots: int,
    reset_type: str,
    use_state_discrimination: bool,
    drive_role: str = "control",
    swap_via_macro: bool = False,
    swap_operation: str = "iswap",
    simulate: bool = False,
):
    """Build the fixed-time qubit-flux x coupler-flux QUA program.

    Returns (program, sweep_axes). The plain `machine.generate_config()` already carries
    both flux operations, so no special (baked) config is needed -- execute with
    `acquire(..., config=None)`.

    `coupler_amplitudes` / `qubit_amplitudes` are the two flux sweeps (x / y of the 2D
    map), in absolute volts (see module docstring). `coupler_operation`
    is the coupler op (default "swap_01_10_square"); `qubit_operation` is the qubit z op
    (default "const"). `flux_role` selects which qubit's z carries the qubit flux pulse;
    `drive_role` selects which qubit receives the x180. `flux_time` is the shared fixed
    pulse duration in ns (multiple of 4, >= 16); None uses each op's native length.

    Debug isolation: with `swap_via_macro=True` the swap is played through the QUAM
    `qp.macros[swap_operation]` `.apply()` path (exactly what `qc_swap_reset`/`qc_N_swap`
    call) instead of the direct flux play -- to test whether the macro itself reproduces
    the 2D-map swap. In this mode the qubit-z amplitude follows the y-sweep (absolute
    volts, handed to the macro as `ctrl_scale` once divided by its z pulse's stored
    amplitude) while the coupler plays bare at its baked amplitude, so the coupler (x)
    sweep and `flux_time` are ignored.
    """
    num_qubit_pairs = len(qubit_pairs)

    if flux_role not in ("control", "target"):
        raise ValueError(f"flux_role must be 'control' or 'target', got {flux_role!r}")

    # Resolve the per-pair coupler and qubit-z op reference amplitudes (verify they exist).
    coupler_refs = {}
    qubit_refs = {}
    for qp in qubit_pairs:
        if qp.coupler is None:
            raise ValueError(f"Qubit pair {qp.name} has no coupler; cannot run a coupler-flux sweep.")
        if coupler_operation not in qp.coupler.operations:
            raise ValueError(
                f"Coupler of {qp.name} has no operation {coupler_operation!r}; "
                f"available: {list(qp.coupler.operations)}"
            )
        fq = _flux_qubit(qp, flux_role)
        if fq.z is None:
            raise ValueError(f"{flux_role} qubit of {qp.name} ({fq.name}) has no z line; cannot run a qubit-flux sweep.")
        if qubit_operation not in fq.z.operations:
            raise ValueError(
                f"z line of {fq.name} has no operation {qubit_operation!r}; "
                f"available: {list(fq.z.operations)}"
            )
        coupler_refs[qp.name] = float(qp.coupler.operations[coupler_operation].amplitude)
        qubit_refs[qp.name] = float(fq.z.operations[qubit_operation].amplitude)

    coupler_amplitudes = np.asarray(coupler_amplitudes, dtype=float)
    qubit_amplitudes = np.asarray(qubit_amplitudes, dtype=float)

    # The swept values are volts on each channel, riding on whatever standing bias
    # initialize_qpu applied (this probe takes no flux_point argument, so the
    # declaration is what runs). One shared helper does the reference contract,
    # QUA's amplitude_scale bound and the idle + excursion sum.
    if not swap_via_macro:
        for qp in qubit_pairs:
            for what, op, channel, amps in (
                ("coupler", coupler_operation, qp.coupler, coupler_amplitudes),
                (f"{flux_role}-qubit z", qubit_operation,
                 _flux_qubit(qp, flux_role).z, qubit_amplitudes),
            ):
                check_flux_pulse_relative(
                    channel, name=f"{qp.name} {what} op {op!r}",
                    idle_v=declared_idle_offset_v(channel),
                    amps_v=amps, operation=op)

    # Optional debug: play the swap through the QUAM macro instead of the direct flux play.
    macro_refs = {}
    if swap_via_macro:
        for qp in qubit_pairs:
            if swap_operation not in qp.macros:
                raise ValueError(f"Pair {qp.name} has no macro {swap_operation!r}; available: {list(qp.macros)}.")
            flux_pulse_name = getattr(qp.macros[swap_operation], "flux_pulse", None)
            ops = qp.qubit_control.z.operations
            if not isinstance(flux_pulse_name, str) or flux_pulse_name not in ops:
                raise ValueError(
                    f"Macro {swap_operation!r} on {qp.name} has no z flux_pulse playable in macro mode "
                    f"(flux_pulse={flux_pulse_name!r})."
                )
            # Same shared guard as qc_N_swap_amp, which sweeps this exact knob: the
            # macro's z pulse is the amplitude_scale reference (not a `const`, so the
            # rail/2 convention does not apply), and the swept volts ride on the
            # declared standing bias. The reference it returns is what the volts are
            # divided by -- ahead of the swap, see the hoisted assigns below.
            z = qp.qubit_control.z
            macro_refs[qp.name] = check_flux_pulse_relative(
                z,
                name=f"{qp.name} macro {swap_operation!r} on {qp.qubit_control.name}.z",
                idle_v=declared_idle_offset_v(z),
                amps_v=qubit_amplitudes,
                operation=flux_pulse_name,
            )

    # Shared fixed pulse duration in clock cycles (4 ns), or None to use each op's native length.
    duration_cycles = None
    if flux_time is not None:
        if flux_time % 4 != 0 or flux_time < 16:
            raise ValueError(f"flux_time must be a multiple of 4 ns and >= 16 ns, got {flux_time}.")
        duration_cycles = flux_time // 4

    sweep_axes = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        # Outer loop -> y axis.
        "qubit_amplitude": xr.DataArray(
            qubit_amplitudes, attrs={"long_name": f"{flux_role} qubit flux amplitude", "units": "V"}
        ),
        # Inner loop -> x axis.
        "coupler_amplitude": xr.DataArray(
            coupler_amplitudes, attrs={"long_name": "coupler flux amplitude", "units": "V"}
        ),
    }

    with program() as prog:
        c_a = declare(fixed)  # swept coupler amplitude, in volts
        q_a = declare(fixed)  # swept qubit-z amplitude, in volts
        # Per-pair amplitude_scales, assigned AHEAD of the swap (see the loop
        # body). One per pair because multiplexed pairs share the volts loop
        # variable but not their stored reference amplitudes.
        q_scl = [declare(fixed) for _ in range(num_qubit_pairs)]
        c_scl = [declare(fixed) for _ in range(num_qubit_pairs)]
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
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers).
            for qp in multiplexed_qubit_pairs.values():
                machine.initialize_qpu(target=qp.qubit_control)
                machine.initialize_qpu(target=qp.qubit_target)
            align()
            # Averaging loop
            with for_(n, 0, n < num_shots, n + 1):
                save(n, n_st)
                # Qubit-flux amplitude loop (outer -> y axis)
                with for_(*from_array(q_a, qubit_amplitudes)):
                    # Coupler-flux amplitude loop (inner -> x axis)
                    with for_(*from_array(c_a, coupler_amplitudes)):
                        for ii, qp in multiplexed_qubit_pairs.items():
                            # volts -> amplitude_scale, divided HERE, before the
                            # reset, and never between the prep's align() and the
                            # two flux plays: there each division would hold back
                            # its own element's play, and the z and coupler pulses
                            # of one swap could start on different clock edges.
                            if swap_via_macro:
                                assign(q_scl[ii], q_a / macro_refs[qp.name])
                            else:
                                assign(q_scl[ii], q_a / qubit_refs[qp.name])
                                assign(c_scl[ii], c_a / coupler_refs[qp.name])
                            # Qubit initialization
                            qp.qubit_control.reset(reset_type, simulate)
                            qp.qubit_target.reset(reset_type, simulate)
                            align()
                            # Excite only one qubit of the pair (single excitation).
                            if drive_role == "target":
                                qp.qubit_target.xy.play("x180")
                            else:
                                qp.qubit_control.xy.play("x180")
                            align()

                            # Two simultaneous flux pulses at the fixed duration, swept amplitudes.
                            fq = _flux_qubit(qp, flux_role)
                            if swap_via_macro:
                                # Debug: exercise the QUAM swap macro's .apply() path (as
                                # qc_swap_reset does) instead of the direct flux play. The qubit-z
                                # amplitude follows the y-sweep (absolute volts, already divided
                                # into a scale above); the coupler is played bare at the macro
                                # pulse's baked amplitude, so the coupler (x) sweep and flux_time
                                # are ignored in this mode.
                                qp.macros[swap_operation].apply(ctrl_scale=q_scl[ii])
                            else:
                                # plain variables: every division was done ahead of the reset
                                c_scale = c_scl[ii]
                                q_scale = q_scl[ii]
                                if duration_cycles is None:
                                    fq.z.play(qubit_operation, amplitude_scale=q_scale)
                                    qp.coupler.play(coupler_operation, amplitude_scale=c_scale)
                                else:
                                    fq.z.play(qubit_operation, amplitude_scale=q_scale, duration=duration_cycles)
                                    qp.coupler.play(coupler_operation, amplitude_scale=c_scale, duration=duration_cycles)
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
                    # Inner buffer = coupler amplitude (x), outer buffer = qubit amplitude (y).
                    state_gg_st[i].buffer(len(coupler_amplitudes)).buffer(len(qubit_amplitudes)).average().save(f"state_gg{i}")
                    state_ge_st[i].buffer(len(coupler_amplitudes)).buffer(len(qubit_amplitudes)).average().save(f"state_ge{i}")
                    state_eg_st[i].buffer(len(coupler_amplitudes)).buffer(len(qubit_amplitudes)).average().save(f"state_eg{i}")
                    state_ee_st[i].buffer(len(coupler_amplitudes)).buffer(len(qubit_amplitudes)).average().save(f"state_ee{i}")
                else:
                    I_c_st[i].buffer(len(coupler_amplitudes)).buffer(len(qubit_amplitudes)).average().save(f"I_control{i}")
                    Q_c_st[i].buffer(len(coupler_amplitudes)).buffer(len(qubit_amplitudes)).average().save(f"Q_control{i}")
                    I_t_st[i].buffer(len(coupler_amplitudes)).buffer(len(qubit_amplitudes)).average().save(f"I_target{i}")
                    Q_t_st[i].buffer(len(coupler_amplitudes)).buffer(len(qubit_amplitudes)).average().save(f"Q_target{i}")

    return prog, sweep_axes


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

    No baking is involved, so `config` may be left as None and the shared helper falls
    back to `machine.generate_config()`.
    """
    return _acquire(machine, prog, sweep_axes, num_shots=num_shots, timeout=timeout, log=log, config=config)


from typing import Any

import xarray as xr
from scqo import register
from scqo.experiments import PairSwapFluxMap

from ._pair_roles import JointPopulationMixin

#: neutral waveform shape -> the QUAM operation name it is played as. The op
#: must exist on BOTH the coupler and the selected member's z line; register
#: the shaped one with `python quam_config/register_flattop_cosine.py` before
#: selecting it (the probe refuses with the available list if it is missing).
_SHAPE_OPS = {"square": "const", "flattop_cosine": "flattop_cosine"}

#: QM plays flux pulses on a 4 ns clock, with a 16 ns floor (4 cycles).
_CLOCK_NS = 4
_MIN_FLUX_NS = 16


@register
class QMPairSwapFluxMap(JointPopulationMixin, PairSwapFluxMap):
    """Build the multiplexed fixed-time 2D swap map on the QM OPX (QCQ pairs)."""

    def probe(self) -> Any:
        from ._reset import check_reset_method
        from ._vendor import role_side, vendor_pair_name
        from scqo_qm.experiments._lib import select_qubit_pairs

        machine = self.backend.machine  # type: ignore[attr-defined]
        vendor_names = [vendor_pair_name(self, p) for p in self.params.targets]
        pairs = select_qubit_pairs(machine, vendor_names, multiplexed=True)

        self._high_side = role_side(self, "high", field="targets")
        drive_role = role_side(self, self.params.drive_side, field="drive_side")
        flux_role = role_side(self, self.params.flux_side, field="flux_side",
                              needs_flux=True)

        operation = _SHAPE_OPS[self.params.flux_pulse_shape]
        flux_time = None
        if self.params.swap_time_ns is not None:
            flux_time = max(
                _MIN_FLUX_NS,
                int(round(self.params.swap_time_ns / _CLOCK_NS)) * _CLOCK_NS)
        # what the instrument ACTUALLY plays — estimate() records this, not the
        # unquantized request (None = each operation's own native length).
        self._flux_time_ns = float(flux_time) if flux_time is not None else None

        prog, axes = build_program(
            machine,
            pairs,
            coupler_amplitudes=self.sweep_axes["coupler_flux_v"],
            qubit_amplitudes=self.sweep_axes["qubit_flux_v"],
            coupler_operation=operation,
            qubit_operation=operation,
            flux_time=flux_time,
            flux_role=flux_role,
            num_shots=self.params.num_averages,
            reset_type=check_reset_method(self),
            use_state_discrimination=True,
            drive_role=drive_role,
        )
        return prog, {
            # roster composite names, not the vendor pair keys the probe used
            "qubit_pair": xr.DataArray(list(self.params.targets)),
            # raw nesting order: the qubit amplitude is the OUTER loop
            "qubit_flux_v": axes["qubit_amplitude"],
            "coupler_flux_v": axes["coupler_amplitude"],
        }
