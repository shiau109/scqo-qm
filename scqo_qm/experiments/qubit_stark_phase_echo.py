"""AC-Stark phase echo acquisition probe: vendor code only (qm/quam) - no qualibrate, no scqo, no scqat.

Hahn echo with an off-resonant AC-Stark tone filling the SECOND free-evolution arm::

    y90 - wait(D) - x180 - stark(D, off-resonant) - {x90 | -y90} - readout
          arm 1 (idle)      arm 2 (stark)

Both arms are the same length ``D`` = the registered ``stark`` operation's OWN
baked length (arm 1's idle is derived from it), so the central x180 refocuses
static dephasing and the surviving phase is the AC-Stark shift the tone imprints
in arm 2. The tone therefore fully REPLACES the free evolution in arm 2 - it is
played at its natural length with NO dynamic-duration override, because the stark
waveform is arbitrary (a SquarePulse bakes to samples, not a constant) and QUA
would ZERO-PAD an arbitrary pulse to a longer requested duration, leaving free
evolution in arm 2. To change D, re-register the stark op at a new length
(quam_config/register_stark.py); arm 1 follows automatically.

The tone plays at an off-resonant IF (``base_if + stark_detuning_hz``) via
``update_frequency`` and is RESTORED to the resonant IF before the closing pulse
and readout (both need resonance). Only the tone AMPLITUDE is swept
(``amplitude_scale``); the detuning is a fixed per-run scalar.

The accumulated phase is read out in TWO bases (``meas_basis``): close with x90
(-> <Z> = sin phi) or -y90 (-> <Z> = cos phi), so the scqat estimator recovers phi
unambiguously as ``atan2(sin, cos)``. -y90 is realized as ``y90`` with a negated
amplitude_scale (the reversed pi/2 rotation).

The PREP is ``y90`` (not x90): it starts the equatorial state on +x, aligned with
the measurement's phi=0 reference, so at amp=0 (no stark) the prepared phase is
exactly 0 and the measured ABSOLUTE phase is the Stark-induced phase (nothing to
subtract). Hardware-validated on 5Q4C (run 20260813-011304).

FRAME/SIGN: there is no virtual-detuning ramp here (the phase is physical, from the
Stark tone), so no cross-repo sign convention is coupled - the closing pulses are
plain gates and the estimator anchors phi=0 at the smallest amplitude.

``probe()`` also reports the stark op's BAKED amplitude per qubit
(``probe_stark_amp``): the swept factor is a multiplier of it, and this driver is
the only layer that can read a named operation's own amplitude, so without the
report the run has no absolute amplitude scale. scqo turns it into the
``digital_amp`` coordinate on the dataset.

QM AC-Stark phase echo for scqo - supplies only ``probe()``. Parameters, the
two-quadrature phase estimator and the (absent) writeback are inherited from
``scqo.experiments.QubitStarkPhaseEcho``. scqo sweeps ``stark_amp`` (amplitude
factor) x ``meas_basis`` (2 points); this builder realizes both on the same coord
names, which the backend's ``_to_canonical`` maps through.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import xarray as xr
from qm.qua import *
from qualang_tools.loops import from_array

from scqo_qm.experiments._amp_limits import check_amp_scale_window


def build_program(
    machine,
    qubits,
    *,
    stark_amps,
    stark_detuning_hz: float,
    stark_operation: str,
    num_shots: int,
    reset_type: str,
    reset_max_attempts: int = 15,
    use_state_discrimination: bool = False,
    simulate: bool = False,
    log: Optional[Callable] = None,
):
    """Build the AC-Stark phase echo QUA program. Returns (program, sweep_axes).

    ``stark_amps`` is the swept amplitude factor (QUA ``amplitude_scale`` of the
    baked ``stark_operation``). The per-arm length is the stark op's OWN baked
    length: arm 1 idles for it and arm 2 plays the tone at its natural length, so
    the echo is balanced with the RF tone fully replacing the second free
    evolution (no dynamic-duration override -> no zero-padding of the arbitrary
    stark waveform). ``qubits`` is a BatchableList (see ``_lib.select_qubits``).
    """
    stark_amps = np.asarray(stark_amps, dtype=float)
    bases = np.array([0, 1])
    num_qubits = len(qubits)

    # Guards BEFORE any QUA + the per-qubit constants. Refuse a missing named op
    # and an amplitude factor QUA cannot express (|factor| >= 2), BY NAME and
    # before instrument time is booked. Arm 1's idle (in 4 ns clock cycles) is the
    # stark tone's OWN baked length, floored at 4 cycles (16 ns) for QUA's wait().
    stark_cycles: dict[str, int] = {}
    base_ifs: dict[str, int] = {}
    stark_ifs: dict[str, int] = {}
    for i in range(num_qubits):
        q = qubits[i]
        for op in (stark_operation, "y90"):
            if op not in q.xy.operations:
                hint = ("  Register it (quam_config/register_stark.py)."
                        if op == stark_operation else "")
                raise ValueError(f"{q.name}.xy has no '{op}' operation.{hint}")
        check_amp_scale_window(stark_amps, name=f"{q.name}.xy '{stark_operation}'",
                               knob="max_stark_amp")
        stark_cycles[q.name] = max(4, int(q.xy.operations[stark_operation].length) // 4)
        base_if = int(q.xy.intermediate_frequency)
        base_ifs[q.name] = base_if
        stark_ifs[q.name] = int(stark_detuning_hz) + base_if

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "stark_amp": xr.DataArray(
            stark_amps, attrs={"long_name": "stark amplitude factor", "units": ""}),
        "meas_basis": xr.DataArray(
            bases, attrs={"long_name": "measurement basis (0=x90 sin, 1=-y90 cos)", "units": ""}),
    }

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        a = declare(fixed)  # swept stark amplitude factor (amplitude_scale)
        b = declare(int)    # measurement basis: 0 = close x90, 1 = close -y90

        if use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < num_shots, n + 1):
                save(n, n_st)

                with for_(*from_array(a, stark_amps)):       # outer: stark amplitude
                    with for_(*from_array(b, bases)):        # inner: measurement basis
                        for i, qubit in multiplexed_qubits.items():
                            qubit.reset(reset_type, simulate, log_callable=log,
                                        max_attempts=reset_max_attempts)
                        align()
                        # Echo with the stark tone in arm 2, then the basis close.
                        for i, qubit in multiplexed_qubits.items():
                            d = stark_cycles[qubit.name]
                            base_if = base_ifs[qubit.name]
                            stark_if = stark_ifs[qubit.name]
                            qubit.align()
                            qubit.xy.play("y90")
                            qubit.xy.wait(d)                        # arm 1: idle = stark tone length
                            qubit.xy.play("x180")                   # echo refocus
                            qubit.xy.update_frequency(stark_if)     # detune the tone
                            qubit.xy.play(stark_operation, amplitude_scale=a)  # arm 2: full stark tone
                            qubit.xy.update_frequency(base_if)      # RESTORE before close + readout
                            with if_(b == 0):
                                qubit.xy.play("x90")                # basis 0 -> reads sin phi
                            with else_():
                                qubit.xy.play("y90", amplitude_scale=-1.0)  # -y90 -> reads cos phi
                        align()
                        for i, qubit in multiplexed_qubits.items():
                            if use_state_discrimination:
                                qubit.readout_state(state[i])
                                save(state[i], state_st[i])
                            else:
                                qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                                save(I[i], I_st[i])
                                save(Q[i], Q_st[i])
                        align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                if use_state_discrimination:
                    state_st[i].buffer(len(bases)).buffer(len(stark_amps)).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(bases)).buffer(len(stark_amps)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(bases)).buffer(len(stark_amps)).average().save(f"Q{i + 1}")

    return prog, sweep_axes


from typing import Any

import numpy as np

from scqo import register
from scqo.experiments import QubitStarkPhaseEcho


def _baked_stark_amps(qubits, stark_operation: str) -> dict[str, float]:
    """``{qubit: baked amplitude}`` of the stark op — the ABSOLUTE reference the
    swept factor multiplies.

    scqo attaches this as the ``digital_amp`` coordinate, so the run's figures carry
    an absolute-amplitude axis and the full-turn answer is reported in the frame an
    operator can act on. It has to come from HERE: the reference is a named
    OPERATION's own amplitude, not a channel knob, so no roster field addresses it
    and no later device read recovers what actually played.

    Provenance only, so it degrades instead of raising: a pulse class with no scalar
    ``amplitude`` (nothing bans an arbitrary-sample stark waveform) has no absolute
    reference to report, and scqo simply leaves the axis off. Call this only AFTER
    ``build_program``, whose guard refuses a missing operation BY NAME.
    """
    baked: dict[str, float] = {}
    for i in range(len(qubits)):
        q = qubits[i]
        amplitude = getattr(q.xy.operations[stark_operation], "amplitude", None)
        if amplitude is not None:
            baked[q.name] = float(amplitude)
    return baked


@register
class QMQubitStarkPhaseEcho(QubitStarkPhaseEcho):
    """Build a multiplexed AC-Stark phase echo QUA program on the QM OPX."""

    def probe(self) -> Any:
        from ._reset import check_reset_method, reset_max_attempts
        from scqo_qm.experiments._lib import select_qubits

        machine = self.backend.machine  # type: ignore[attr-defined]
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)
        stark_amps = np.asarray(self.sweep_axes["stark_amp"], dtype=float)

        built = build_program(
            machine,
            qubits,
            stark_amps=stark_amps,
            stark_detuning_hz=float(self.params.stark_detuning_hz),
            stark_operation=self.params.stark_operation,
            num_shots=self.params.num_averages,
            reset_type=check_reset_method(self),
            reset_max_attempts=reset_max_attempts(self),
            use_state_discrimination=bool(self.params.use_state_discrimination),
        )
        self.probe_stark_amp = _baked_stark_amps(qubits, self.params.stark_operation)
        return built
