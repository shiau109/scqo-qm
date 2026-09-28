"""Residual-ZZ-vs-coupler-frequency acquisition probe: vendor code only (qm/quam) - no qualibrate, no scqo, no scqat.

Hahn echo (with virtual detuning) on a QCQ pair while the coupler plays two gated `const` pulses at
a swept bias amplitude and interaction time; both qubits are read out. Each coupler-flux slice is a
Ramsey-like decay fitted downstream to extract the signed residual ZZ vs coupler bias.

QM residual-ZZ vs coupler bias for scqo - supplies ``probe()`` + the raw
joint-state reduction.

Parameters, the per-bias echo-fringe fit and the writeback (the decouple point
as ``idle_flux`` on the COUPLER's own flux LINE, plus the residual
``zz_hz`` fact on the pair) are inherited from
``scqo.experiments.PairZZCoupler``. scqo sweeps
``(coupler_bias_v, idle_time_ns)``; the QM builder sweeps ``amplitudes`` (V on
the pair's tunable coupler) x ``durations`` (interaction time, clock cycles) with
a Hahn echo + virtual detuning on ONE pair member and joint two-qubit state
readout. The neutral ``measure`` role (high/low, roster-declared) is mapped onto
the vendor's control/target here; ``reduce_raw`` turns the joint populations into
the canonical ``signal`` (the measured qubit's excited-state probability).
"""

from __future__ import annotations

from typing import Callable, Optional

import xarray as xr
from qm.qua import *
from qualang_tools.loops import from_array

from scqo_qm.experiments._lib import acquire as _acquire
from scqo_qm.experiments._flux_limits import check_flux_pulse_relative, declared_idle_offset_v


def build_program(
    machine,
    qubit_pairs,
    *,
    amplitudes,
    durations,
    detuning_hz: int,
    num_shots: int,
    reset_type: str,
    use_state_discrimination: bool,
    measure_qubit: str = "target",
    simulate: bool = False,
):
    """Build the ZZ-vs-coupler-frequency QUA program. Returns (program, sweep_axes).

    `amplitudes` is the coupler bias sweep (V), `durations` the interaction-time sweep in clock
    cycles (4 ns); `detuning_hz` is the virtual detuning in Hz. `measure_qubit` ("control" or
    "target") selects which qubit's signal is fitted. `qubit_pairs` is a BatchableList of pairs
    (see `qualibration_libs.parameters.get_qubit_pairs`).
    """
    num_qubit_pairs = len(qubit_pairs)

    # The coupler plays `const` rescaled by amp/const.amplitude, on top of whatever
    # standing bias initialize_qpu applied (this probe takes no flux_point argument,
    # so the coupler's declaration -- normally "off" -> decouple_offset -- is what
    # runs). Rail, amplitude_scale bound and the idle + excursion sum, all shared.
    for qp in qubit_pairs:
        coupler = getattr(qp, "coupler", None)
        if coupler is None:
            raise ValueError(
                f"Qubit pair {qp.name} has no coupler; this probe sweeps a coupler bias.")
        check_flux_pulse_relative(
            coupler, name=f"{qp.name} coupler", idle_v=declared_idle_offset_v(coupler),
            amps_v=amplitudes, operation="const")

    sweep_axes = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "amp": xr.DataArray(
            amplitudes, attrs={"long_name": "coupler bias amplitude (tunes coupler frequency)", "units": "V"}
        ),
        "time": xr.DataArray(durations * 4, attrs={"long_name": "interaction time", "units": "ns"}),
    }

    with program() as prog:
        # Both qubits of each pair are read out (control + target), so declare two IQ sets.
        I_c, I_c_st, Q_c, Q_c_st, n, n_st = machine.declare_qua_variables(num_IQ_pairs=num_qubit_pairs)
        I_t, I_t_st, Q_t, Q_t_st, _, _ = machine.declare_qua_variables(num_IQ_pairs=num_qubit_pairs)
        virtual_detuning_phase = declare(fixed)
        amp = declare(fixed)
        t = declare(int)
        if use_state_discrimination:
            # Read out BOTH qubits 2-level and form the joint two-qubit populations P00/P01/P10/P11
            # (first digit = control, second = target).
            state_c = [declare(int) for _ in range(num_qubit_pairs)]
            state_t = [declare(int) for _ in range(num_qubit_pairs)]
            ind_gg = declare(int)  # 00
            ind_ge = declare(int)  # 01
            ind_eg = declare(int)  # 10
            ind_ee = declare(int)  # 11
            state_gg_st = [declare_stream() for _ in range(num_qubit_pairs)]
            state_ge_st = [declare_stream() for _ in range(num_qubit_pairs)]
            state_eg_st = [declare_stream() for _ in range(num_qubit_pairs)]
            state_ee_st = [declare_stream() for _ in range(num_qubit_pairs)]

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            # Initialize the QPU
            for qp in multiplexed_qubit_pairs.values():
                machine.initialize_qpu(target=qp.qubit_control)
                machine.initialize_qpu(target=qp.qubit_target)
            align()

            measured_qubits_map = {
                ii: qp.qubit_control if measure_qubit == "control" else qp.qubit_target
                for ii, qp in multiplexed_qubit_pairs.items()
            }
            partner_qubits_map = {
                ii: qp.qubit_target if measure_qubit == "control" else qp.qubit_control
                for ii, qp in multiplexed_qubit_pairs.items()
            }

            with for_(n, 0, n < num_shots, n + 1):
                save(n, n_st)
                with for_(*from_array(amp, amplitudes)):
                    with for_(*from_array(t, durations)):
                        # The ramp is NEGATED on purpose -- this is not a typo. QUA's
                        # frame_rotation_2pi rotates the ELEMENT FRAME, the opposite
                        # handedness to a pulse-axis phase (Qblox's X90(phase=...)). Both
                        # consumers recover the residual ZZ as `zz = f_fit - detuning`
                        # (scqo's neutral estimate(), the node's analysis.py), which only
                        # means the PHYSICAL zeta if the fringe sits at detuning + zeta.
                        # Un-negated it reports -zeta. The zero crossing that estimate()
                        # actually writes back is sign-invariant, so this is a reporting
                        # fix, not a writeback one. Official 19_zz_off_jazz.py leaves the
                        # ramp un-negated -- it only takes argmin|zeta|, which cannot tell
                        # the difference, so do not copy its sign into a probe.
                        assign(virtual_detuning_phase, Cast.mul_fixed_by_int(-detuning_hz * 1e-9, 4 * t))

                        # Reset
                        for ii, qp in multiplexed_qubit_pairs.items():
                            qp.qubit_control.reset(reset_type, simulate)
                            qp.qubit_target.reset(reset_type, simulate)
                            reset_frame(qp.qubit_target.xy.name)
                            reset_frame(qp.qubit_control.xy.name)
                        align()

                        # Qubit manipulation (Hahn echo with virtual detuning + coupler pulses)
                        for ii, qp in multiplexed_qubit_pairs.items():
                            measured_qubit = measured_qubits_map[ii]
                            partner_qubit = partner_qubits_map[ii]
                            measured_qubit.xy.play("x90")
                            qp.coupler.wait(measured_qubit.xy.operations["x90"].length // 4)
                            partner_qubit.wait(measured_qubit.xy.operations["x90"].length // 4)
                            qp.coupler.play(
                                "const",
                                amplitude_scale=amp / qp.coupler.operations["const"].amplitude,
                                duration=t,
                            )
                            measured_qubit.xy.wait(t)
                            partner_qubit.xy.wait(t)
                            measured_qubit.xy.play("x180")
                            partner_qubit.xy.play("x180")
                            qp.coupler.wait(measured_qubit.xy.operations["x180"].length // 4)
                            measured_qubit.xy.frame_rotation_2pi(virtual_detuning_phase)
                            qp.coupler.play(
                                "const",
                                amplitude_scale=amp / qp.coupler.operations["const"].amplitude,
                                duration=t,
                            )
                            measured_qubit.xy.wait(t)
                            partner_qubit.xy.wait(t)
                            measured_qubit.xy.play("x90")
                        align()

                        # Qubit readout — measure BOTH qubits of the pair
                        for ii, qp in multiplexed_qubit_pairs.items():
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
                        align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubit_pairs):
                if use_state_discrimination:
                    state_gg_st[i].buffer(len(durations)).buffer(len(amplitudes)).average().save(f"state_gg{i + 1}")
                    state_ge_st[i].buffer(len(durations)).buffer(len(amplitudes)).average().save(f"state_ge{i + 1}")
                    state_eg_st[i].buffer(len(durations)).buffer(len(amplitudes)).average().save(f"state_eg{i + 1}")
                    state_ee_st[i].buffer(len(durations)).buffer(len(amplitudes)).average().save(f"state_ee{i + 1}")
                else:
                    I_c_st[i].buffer(len(durations)).buffer(len(amplitudes)).average().save(f"I_control{i + 1}")
                    Q_c_st[i].buffer(len(durations)).buffer(len(amplitudes)).average().save(f"Q_control{i + 1}")
                    I_t_st[i].buffer(len(durations)).buffer(len(amplitudes)).average().save(f"I_target{i + 1}")
                    Q_t_st[i].buffer(len(durations)).buffer(len(amplitudes)).average().save(f"Q_target{i + 1}")

    return prog, sweep_axes


def acquire(
    machine,
    prog,
    sweep_axes,
    *,
    num_shots: int,
    timeout: float,
    log: Optional[Callable] = None,
) -> xr.Dataset:
    """Connect to the QOP, execute the program and fetch the raw xr.Dataset."""
    return _acquire(machine, prog, sweep_axes, num_shots=num_shots, timeout=timeout, log=log)


from typing import Any

import numpy as np
import xarray as xr
from scqo import register
from scqo.experiments import PairZZCoupler


@register
class QMPairZZCoupler(PairZZCoupler):
    """Build the multiplexed ZZ-vs-coupler QUA program on the QM OPX (QCQ pairs)."""

    def _measure_side(self, machine: Any) -> str:
        """Map the neutral ``measure`` role (high/low) onto vendor control/target.

        The mapping itself is shared with the pair swap maps — see
        ``_vendor.role_side`` for why the roster's declared roles are the only
        source and why a mixed mapping refuses. ``machine`` is accepted (and
        unused) so the call site reads like the probe helpers around it."""
        from ._vendor import role_side

        return role_side(self, self.params.measure, field="measure")

    def _vendor_pair(self, machine: Any, name: str) -> Any:
        """The QUAM qubit_pair behind a ROSTER composite name.

        ``machine`` is accepted (and unused) so the call site reads like the
        probe helpers around it; the resolution itself is the backend's, which
        joins roster composite -> QUAM pair by MEMBERSHIP."""
        from ._vendor import vendor_pair

        return vendor_pair(self, name)

    def probe(self) -> Any:
        from ._reset import check_reset_method
        from scqo_qm.experiments._lib import select_qubit_pairs

        from ._vendor import vendor_pair_name

        machine = self.backend.machine  # type: ignore[attr-defined]
        # targets are ROSTER composite names; the probe helper selects by QUAM
        # pair key (QM names its pairs after the coupler), so translate first —
        # order preserved, which is what the axis relabel below relies on.
        vendor_names = [vendor_pair_name(self, p) for p in self.params.targets]
        pairs = select_qubit_pairs(machine, vendor_names, multiplexed=True)
        self._side = self._measure_side(machine)

        # Canonical idle times (ns) -> clock cycles; the raw time axis is the
        # QUANTIZED grid (durations*4 ns), which estimate() reads from coords.
        cycles = np.unique(np.clip(
            np.round(self.sweep_axes["idle_time_ns"] / 4).astype(int), 4, None))
        amplitudes = self.sweep_axes["coupler_bias_v"]

        prog, axes = build_program(
            machine,
            pairs,
            amplitudes=amplitudes,
            durations=cycles,
            detuning_hz=int(self.params.detuning_hz),
            num_shots=self.params.num_averages,
            reset_type=check_reset_method(self),
            use_state_discrimination=True,
            measure_qubit=self._side,
        )
        # The canonical time axis is the probe's REAL quantized grid: re-declare
        # it so sizes and values match the raw data exactly.
        self.sweep_axes["idle_time_ns"] = axes["time"].values.astype(float)
        sweep_axes = {
            # The probe labels its target axis with VENDOR pair keys; scqo's
            # dataset (and estimate()) key on the ROSTER composite names the
            # operator asked for. Same order by construction (vendor_names was
            # built from targets), so relabel rather than rename downstream.
            "qubit_pair": xr.DataArray(list(self.params.targets)),
            "coupler_bias_v": axes["amp"],
            "idle_time_ns": axes["time"],
        }
        return prog, sweep_axes

    def reduce_raw(self, raw: xr.Dataset) -> xr.Dataset:
        """Joint two-qubit populations -> the measured qubit's excited-state
        probability (the canonical ``signal``). First digit = control."""
        if "state_ee" in raw.data_vars:
            sig = (raw["state_eg"] + raw["state_ee"] if self._side == "control"
                   else raw["state_ge"] + raw["state_ee"])
        else:  # IQ fallback (no state discrimination): fit the I fringe
            sig = raw["I_control"] if self._side == "control" else raw["I_target"]
        return sig.to_dataset(name="signal")
