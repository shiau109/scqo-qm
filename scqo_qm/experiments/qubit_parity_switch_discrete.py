"""Discrete charge-parity monitor acquisition probe: vendor code only (qm/quam) - no qualibrate, no scqo, no scqat.

Per cycle: measure M1 - depletion wait - x90 - fixed idle - y90 - measure M2 -
pad wait, repeated as `num_shots` back-to-back cycles. M1 PROJECTS the qubit
(measurement-based initialization — the reason this probe carries no
`qubit.reset(...)` of any kind), the 90-degree-shifted pulse pair maps the
parity onto the pole, and M2 reads it: the parity of each cycle is the
WITHIN-CYCLE difference m1 XOR m2, so — unlike the continuous sibling — decay
during the pad wait corrupts no parity sample and the cycle may be padded
arbitrarily long (`tau_wait_cycles`, from scqo's `cycle_period_ns`).

BOTH measurements are recorded, into the SAME streams: two saves per cycle,
`buffer(2).buffer(num_shots)` and NO `.average()`, so each stream comes back
shaped (shot_idx, meas_idx) with meas_idx 0 = M1. Handle base names stay
digit-free (`state1`/`I1`/`Q1`), which is what lets the fetcher group them per
qubit.

Like the continuous probe, this does NOT call `qubit.readout_state()` in
discriminated mode: that helper ends with its own depletion wait, which would
double-count the governed wait and break the scheduled cycle period. The
threshold assign is inlined instead (the threshold still comes from the QUAM
readout operation, i.e. scqo's accepted `readout_threshold` knob).

`idle_cycles`, `depletion_cycles` and `tau_wait_cycles` are per-qubit-name
dicts of 4 ns clock cycles; the caller (the scqo shell) owns the ns->cycles
conversion, the cycle-period padding math and its too-short refusal.

QM discrete charge-parity monitor for scqo - supplies only ``probe()``.

Parameters (including ``cycle_period_ns``), the REFUSE gates, the
within-cycle telegraph-PSD fit and the ``parity_rate_hz`` writeback are all
inherited from ``scqo.experiments.QubitParitySwitchDiscrete``. This class
converts the resolved ns values into QUA clock cycles, computes the pad wait
that stretches each cycle to exactly the requested period — refusing BY NAME
when ``cycle_period_ns`` is shorter than the scheduled sequence — and reports
the exact scheduled cycle period back to the neutral layer.

PER-MEASUREMENT contract: both measurements of every cycle are recorded
individually — the probe's streams are ``buffer(2).buffer(num_shots)`` with
NO ``.average()``.

NOT MULTIPLEXED, for the same reason as the continuous shell: each qubit's
cycle period is its own telegraph timebase.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import xarray as xr
from qm.qua import *

from scqo_qm.experiments._lib import acquire as _acquire


def build_program(
    machine,
    qubits,
    *,
    idle_cycles: Dict[str, int],
    depletion_cycles: Dict[str, int],
    tau_wait_cycles: Dict[str, int],
    num_shots: int,
    use_state_discrimination: bool = False,
    simulate: bool = False,
):
    """Build the discrete parity-monitor QUA program. Returns (program, sweep_axes).

    `qubits` is a BatchableList (see `_lib.select_qubits`); the scqo shell
    selects it with `multiplexed=False` so each qubit gets its own batch and
    therefore its own uncoupled cycle period — the quantity the rate is
    divided by.
    """
    num_qubits = len(qubits)

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "shot_idx": xr.DataArray(np.arange(1, num_shots + 1),
                                 attrs={"long_name": "cycle index"}),
        "meas_idx": xr.DataArray(np.arange(2),
                                 attrs={"long_name": "measurement index (0 = M1)"}),
    }

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        if use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        def _measure_and_save(multiplexed_qubits):
            for i, qubit in multiplexed_qubits.items():
                qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                if use_state_discrimination:
                    # readout_state()'s body WITHOUT its trailing depletion
                    # wait — see the module docstring.
                    assign(state[i],
                           Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold))
                    save(state[i], state_st[i])
                else:
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < num_shots, n + 1):
                save(n, n_st)

                # M1: the projective initialization — deliberately no
                # qubit.reset() anywhere in this loop.
                _measure_and_save(multiplexed_qubits)
                align()

                # wait out M1's readout photons before the coherent block
                for i, qubit in multiplexed_qubits.items():
                    cycles = depletion_cycles[qubit.name]
                    if cycles:
                        qubit.resonator.wait(cycles)
                align()

                # x90 - fixed idle - y90 (the picture's pulse order; equivalent
                # to the continuous y90-first order up to a telegraph sign flip)
                for i, qubit in multiplexed_qubits.items():
                    qubit.xy.play("x90")
                    qubit.xy.wait(idle_cycles[qubit.name])
                    qubit.xy.play("y90")
                align()

                # M2: the parity readout, into the SAME streams as M1
                _measure_and_save(multiplexed_qubits)
                align()

                # pad the cycle to the requested period (0 = minimal cycle)
                for i, qubit in multiplexed_qubits.items():
                    cycles = tau_wait_cycles[qubit.name]
                    if cycles:
                        qubit.resonator.wait(cycles)
                align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                # two saves per cycle -> meas_idx (len 2) is the innermost
                # buffered axis, then shot_idx; NO .average() — every
                # measurement is its own sample.
                if use_state_discrimination:
                    state_st[i].buffer(2).buffer(num_shots).save(f"state{i + 1}")
                else:
                    I_st[i].buffer(2).buffer(num_shots).save(f"I{i + 1}")
                    Q_st[i].buffer(2).buffer(num_shots).save(f"Q{i + 1}")

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


from typing import Any, ClassVar

from scqo import register
from scqo.experiments import QubitParitySwitchDiscrete
from scqo.experiments._depletion import depletion_wait_ns
from scqo.requirements import Requirement

from ._requires import X90_AMPLITUDE

from .qubit_parity_switch_continuous import _cycles

_PERIOD_TOO_SHORT = (
    "cycle_period_ns = {want:.0f} ns is shorter than the scheduled sequence "
    "on {target}: M1 + depletion + x90 + idle + y90 + M2 already takes "
    "{sequence:.0f} ns. Raise cycle_period_ns to at least that (or leave it "
    "None for the minimal, unpadded cycle)."
)


@register
class QMQubitParitySwitchDiscrete(QubitParitySwitchDiscrete):
    """Build a two-measurement-per-cycle parity-monitor QUA program on the QM
    OPX."""

    #: what only this backend consumes, on top of the neutral requirements
    requires: ClassVar[tuple[Requirement, ...]] = (
        *QubitParitySwitchDiscrete.requires, X90_AMPLITUDE)
    #: true of THIS probe only; scqo shows them in
    #: `scqo run qubit_parity_switch_discrete --help`
    backend_notes: ClassVar[tuple[str, ...]] = (
        "the pi/2 pulses are the x90 and y90 operations, whose amplitude is pi_amp_x90",
        "targets are measured one after another, each at its own cycle period",
        "with use_state_discrimination a measurement is |1> when its I is above "
        "readout_threshold",
        "the pad that fills the cycle is on the 4 ns grid, and one shorter than 16 ns is "
        "raised to 16 ns; the reported cycle period includes it",
    )

    def probe(self) -> Any:
        from scqo_qm.experiments._lib import select_qubits

        machine = self.backend.machine  # type: ignore[attr-defined]
        # one qubit per batch: independent cycle periods (module docstring)
        qubits = select_qubits(machine, self.params.targets, multiplexed=False)

        idle_cycles: dict[str, int] = {}
        depletion_cycles: dict[str, int] = {}
        tau_wait_cycles: dict[str, int] = {}
        self.probe_shot_period_s: dict[str, float] = {}
        for target in self.params.targets:
            idle_cycles[target] = _cycles(self.resolved_idle_ns(target))
            # THE precedence point; never None — define_sweep refused a target
            # without a governed value. 0 is legal and means no wait.
            depletion_ns = float(depletion_wait_ns(self, target))
            depletion_cycles[target] = 0 if depletion_ns <= 0 else _cycles(depletion_ns)
            sequence_ns = self._sequence_ns(
                machine, target, idle_cycles[target], depletion_cycles[target])
            tau_wait_cycles[target] = self._tau_wait_cycles(
                self.params.cycle_period_ns, sequence_ns, target)
            # the EXACT scheduled cycle period, after the pad's grid snap —
            # this is the telegraph timebase the rate is divided by.
            self.probe_shot_period_s[target] = (
                sequence_ns + tau_wait_cycles[target] * 4.0) * 1e-9

        return build_program(
            machine,
            qubits,
            idle_cycles=idle_cycles,
            depletion_cycles=depletion_cycles,
            tau_wait_cycles=tau_wait_cycles,
            num_shots=self.resolved_num_shots(),
            use_state_discrimination=bool(self.params.use_state_discrimination),
        )

    @staticmethod
    def _sequence_ns(machine: Any, target: str, idle_cycles: int,
                     depletion_cycles: int) -> float:
        """The scheduled per-cycle sequence duration WITHOUT the pad: M1 +
        depletion + x90 + idle + y90 + M2, from the same numbers passed to
        ``build_program``. The QUA ``align()``s add tens of ns of sequencer
        overhead not counted here — constant, so it biases the rate by a
        sub-percent fraction rather than distorting the spectrum."""
        qubit = machine.qubits[target]
        pi2 = qubit.xy.operations["x90"].length + qubit.xy.operations["y90"].length
        readout = qubit.resonator.operations["readout"].length
        return ((idle_cycles + depletion_cycles) * 4.0
                + float(pi2) + 2.0 * float(readout))

    @staticmethod
    def _tau_wait_cycles(cycle_period_ns: float | None, sequence_ns: float,
                         target: str) -> int:
        """The pad that stretches the cycle to ``cycle_period_ns``, in 4 ns
        cycles. None or an exact fit pads nothing; a period SHORTER than the
        sequence is refused by name; a positive sub-16 ns remainder snaps UP
        to the 16 ns floor (the reported period reflects the snap)."""
        if cycle_period_ns is None:
            return 0
        remainder = float(cycle_period_ns) - float(sequence_ns)
        if remainder < 0:
            raise ValueError(_PERIOD_TOO_SHORT.format(
                want=float(cycle_period_ns), sequence=float(sequence_ns),
                target=target))
        if remainder == 0:
            return 0
        return _cycles(remainder)
