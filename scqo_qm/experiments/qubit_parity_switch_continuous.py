"""Continuous charge-parity monitor acquisition probe: vendor code only (qm/quam) - no qualibrate, no scqo, no scqat.

y90 - fixed idle - x90 - measure, repeated as `num_shots` back-to-back single
shots. (The two-measurement-per-cycle sibling is
`qubit_parity_switch_discrete.py`.) The 90-degree-SHIFTED second pulse measures sin of the parity-dependent
phase (+/- pi/2 at the intended idle), which is ODD in parity, so the two charge
parities land on opposite readout poles; a same-axis pair would measure the even
cos and see no parity at all.

Two deliberate absences, both load-bearing:

* **No `qubit.reset(...)`.** Between shots only the RESONATOR is waited out
  (its depletion time), and the absence of a qubit reset is REQUIRED rather
  than an optimization: the sequence inverts with the pole the previous shot
  left behind, so the readout is the running XOR of the parity and the
  consecutive-pair difference is the parity telegraph the rate is fitted from.
  Resetting to |0> each shot would sever that chain. The shot cadence is also
  the telegraph timebase — a thermal wait would stretch it by orders of
  magnitude and scale every reported rate with it.
* **No `.average()` in the stream processing.** Every shot is its own sample of
  the telegraph; averaging them is exactly the information being measured.
  (The legacy `calibrations/exclude/LCH_parity_switch_ramsey.py` node carried a
  `.buffer(n).average()` that was a no-op only because a single buffer was ever
  produced.)

`idle_cycles` and `depletion_cycles` are per-qubit-name dicts of 4 ns clock
cycles; the caller (the scqo shell) owns the ns->cycles conversion and the
precedence between a per-run override and the standing QUAM value.

WHY THIS DOES NOT CALL `qubit.readout_state()` in discriminated mode, unlike
every other per-shot probe here: that helper ENDS with its own
`wait(resonator.depletion_time // 4)`. On top of this program's own
between-shot wait that is two depletion waits per shot — the cadence, and
therefore every reported rate, off by the difference — and it would also make
the per-run `readout_depletion_ns` override apply to only one of them. The
threshold assign is three lines of the same vendor API, so the wait keeps
exactly one owner. (The threshold itself still comes from the QUAM readout
operation, i.e. scqo's accepted `readout_threshold` knob.)

QM continuous charge-parity monitor for scqo - supplies only ``probe()``.

Parameters, the REFUSE gates (a stored ``parity_delta_f_hz`` for the fixed
idle, a governed depletion wait for the shot cadence, stored blob centers for
the trace discrimination), the telegraph-PSD fit and the ``parity_rate_hz``
writeback are all inherited from
``scqo.experiments.QubitParitySwitchContinuous``. This class only converts the
resolved ns values into QUA clock cycles and reports the scheduled shot period
back to the neutral layer.

PER-SHOT contract: every shot is recorded individually — the probe's streams
are ``buffer(num_shots)`` with NO ``.average()``.

NOT MULTIPLEXED, on purpose. Each qubit gets its own batch, so its shot cadence
is its own; in a multiplexed batch the ``align()``s tie every qubit's period to
the slowest member, and that period is exactly what each qubit's switching rate
is divided by.
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
    num_shots: int,
    use_state_discrimination: bool = False,
    simulate: bool = False,
):
    """Build the parity-monitor QUA program. Returns (program, sweep_axes).

    `qubits` is a BatchableList (see `_lib.select_qubits`); the scqo shell
    selects it with `multiplexed=False` so each qubit gets its own batch and
    therefore its own uncoupled shot cadence — a multiplexed batch's aligns
    would tie every qubit's period to the slowest member, and the period is the
    quantity the rate is divided by.
    """
    num_qubits = len(qubits)

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "shot_idx": xr.DataArray(np.arange(1, num_shots + 1),
                                 attrs={"long_name": "shot index"}),
    }

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        if use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < num_shots, n + 1):
                save(n, n_st)

                # Resonator-only reset: wait out the readout photons of the
                # PREVIOUS shot. There is deliberately no qubit.reset() here.
                for i, qubit in multiplexed_qubits.items():
                    cycles = depletion_cycles[qubit.name]
                    if cycles:
                        qubit.resonator.wait(cycles)
                align()

                # y90 - fixed idle - x90
                for i, qubit in multiplexed_qubits.items():
                    qubit.xy.play("y90")
                    qubit.xy.wait(idle_cycles[qubit.name])
                    qubit.xy.play("x90")
                align()

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
                align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                # buffer(num_shots) and NO .average(): every shot is a sample of
                # the telegraph (see the module docstring).
                if use_state_discrimination:
                    state_st[i].buffer(num_shots).save(f"state{i + 1}")
                else:
                    I_st[i].buffer(num_shots).save(f"I{i + 1}")
                    Q_st[i].buffer(num_shots).save(f"Q{i + 1}")

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
from scqo.experiments import QubitParitySwitchContinuous
from scqo.experiments._depletion import depletion_wait_ns
from scqo.requirements import Requirement

from ._requires import X90_AMPLITUDE

#: QUA plays waits on a 4 ns clock with a 16 ns (4-cycle) floor.
_MIN_CYCLES = 4


def _cycles(ns: float) -> int:
    """ns -> QUA clock cycles, floored at the 16 ns minimum wait."""
    return max(_MIN_CYCLES, int(round(float(ns) / 4.0)))


@register
class QMQubitParitySwitchContinuous(QubitParitySwitchContinuous):
    """Build a per-shot parity-monitor QUA program on the QM OPX."""

    #: what only this backend consumes, on top of the neutral requirements
    requires: ClassVar[tuple[Requirement, ...]] = (
        *QubitParitySwitchContinuous.requires, X90_AMPLITUDE)
    #: true of THIS probe only; scqo shows them in
    #: `scqo run qubit_parity_switch_continuous --help`
    backend_notes: ClassVar[tuple[str, ...]] = (
        "the pi/2 pulses are the y90 and x90 operations, whose amplitude is pi_amp_x90",
        "targets are measured one after another, each at its own shot period",
        "with use_state_discrimination a shot is |1> when its I is above readout_threshold",
        "the reported shot period sums the played durations and leaves out the "
        "sequencer's alignment overhead, some tens of ns per shot",
    )

    def probe(self) -> Any:
        from scqo_qm.experiments._lib import select_qubits

        machine = self.backend.machine  # type: ignore[attr-defined]
        # one qubit per batch: independent shot cadences (see the module docstring)
        qubits = select_qubits(machine, self.params.targets, multiplexed=False)

        idle_cycles: dict[str, int] = {}
        depletion_cycles: dict[str, int] = {}
        self.probe_shot_period_s: dict[str, float] = {}
        for target in self.params.targets:
            idle_cycles[target] = _cycles(self.resolved_idle_ns(target))
            # THE precedence point (per-run override else the standing knob,
            # which binds to QUAM's resonator.depletion_time); never None —
            # define_sweep refused a target without a governed value. 0 is
            # legal and means no wait, so it must survive the floor.
            depletion_ns = float(depletion_wait_ns(self, target))
            depletion_cycles[target] = 0 if depletion_ns <= 0 else _cycles(depletion_ns)
            self.probe_shot_period_s[target] = self._shot_period_s(
                machine, target, idle_cycles[target], depletion_cycles[target])

        return build_program(
            machine,
            qubits,
            idle_cycles=idle_cycles,
            depletion_cycles=depletion_cycles,
            # resolved by the neutral layer from record_time_s (or an explicit
            # override); params.num_shots is None in the normal case
            num_shots=self.resolved_num_shots(),
            use_state_discrimination=bool(self.params.use_state_discrimination),
        )

    @staticmethod
    def _shot_period_s(machine: Any, target: str, idle_cycles: int,
                       depletion_cycles: int) -> float:
        """The scheduled shot-to-shot period from the durations the program
        above actually plays: depletion + y90 + idle + x90 + readout.

        This is the telegraph timebase — the neutral layer converts a per-shot
        switching probability into Hz with it — so it is summed from the same
        numbers passed to ``build_program`` rather than re-derived from knobs.
        The QUA ``align()``s between the blocks add tens of ns of sequencer
        overhead that is not counted here; on a multi-microsecond period that
        is well under a percent, and it is constant, so it biases the rate by
        the same fraction rather than distorting the spectrum.
        """
        qubit = machine.qubits[target]
        pi2 = qubit.xy.operations["y90"].length + qubit.xy.operations["x90"].length
        readout = qubit.resonator.operations["readout"].length
        return ((idle_cycles + depletion_cycles) * 4.0 + float(pi2) + float(readout)) * 1e-9
