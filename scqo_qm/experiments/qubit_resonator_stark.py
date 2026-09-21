"""Qubit spectroscopy under a resonator Stark tone: vendor code only (qm/quam) - no qualibrate, no scqat.

At every Stark-tone amplitude, sweep the qubit drive detuning with a saturation
pulse while a tone on the readout resonator fills it with photons; read out once
they have gone. The line's AC-Stark shift and broadening are fitted downstream.

ONE SEQUENCE, both backends (scqo ``experiments/_stark_tone.py`` owns it):

    resonator : [==== Stark tone: amp_prefactor x readout op ====]              [## measure ##]
    xy        :               [====== saturation drive ==========]
                |<- ring_up ->|<----------- drive_len ---------->|<-depletion->|

WHAT MAKES THE TONES OVERLAP: there is NO ``align()`` between the Stark tone and
the drive. Both element timelines run from the one shared ``align()`` above them;
the drive waits the ring-up, and the tone is exactly ring-up + drive long, so the
two END together. The ``align()`` AFTER them is the second anchor: the resonator
then waits the depletion and measures with the standard readout — the same op,
unscaled, in every row.

THE STARK TONE IS THE READOUT OPERATION, PLAYED: ``resonator.play("readout",
duration=..., amplitude_scale=a)``. The readout op is a constant pulse, so it
stretches to any length, and its stored amplitude IS the ``readout_amp`` knob — so
``a`` is exactly scqo's ``amp_prefactor``. QUA's dynamic amplitude is bounded to
(-2, 2) (``_amp_limits``).

NOTE (a SILENT failure): the qubit drive (``xy``) and its ``resonator`` must be on
DIFFERENT QM cores/threads for the tones to overlap at all. Same-core elements are
SERIALIZED — the Stark tone plays to completion first, the drive sees only its
ring-down, and the map comes back with no shift and a clean fit. Check the analog
traces in the OPX simulator (``--preview``) before trusting a result from a new
wiring, and repartition the FEM rather than working around it here.

TARGETS ARE MEASURED ONE AT A TIME (``multiplexed=False``), as the Qblox probe
does: on one feedline, N concurrent Stark tones would add on the DAC and each
resonator would sit in the others' tones — a different experiment from the
per-target one Qblox runs.

QM qubit_resonator_stark for scqo - supplies only ``probe()``. Parameters, the
Stark fit and the record-only writeback are inherited from
``scqo.experiments.QubitResonatorStark``; the timing is NOT computed here —
``define_sweep()`` resolved it through ``_stark_tone.stark_windows`` and this shell
reads ``resolved_windows()``.

Drive power contract: the core ``run()`` already solved the drive chain for
``drive_power_dbm`` (recorded set -> acquire -> revert), parking the exact
amplitude on the saturation op — so the probe plays it at ``amplitude_scale=1.0``.
"""

from __future__ import annotations

from typing import Callable, Optional

import xarray as xr
from qm.qua import *
from qualang_tools.loops import from_array

from scqo_qm.experiments._amp_limits import check_amp_scale_window
from scqo_qm.experiments._lib import acquire as _acquire


def _cycles(name: str, value_ns) -> int:
    """ns -> QUA clock cycles, refusing anything off the 4 ns grid.

    scqo's ``_stark_tone`` already put every time on the grid; refusing here keeps
    a future caller from silently rounding (never ``u.ns``, whose value depends on
    whether a ``program()`` happens to be in scope).
    """
    value_ns = int(round(float(value_ns)))
    if value_ns % 4:
        raise ValueError(f"{name}={value_ns} ns is not a multiple of the 4 ns QUA clock cycle")
    return value_ns // 4


def build_program(
    machine,
    qubits,
    *,
    amps,
    dfs,
    tone_len_ns,
    ring_up_ns,
    drive_len_ns,
    depletion_ns,
    num_shots: int,
    reset_type: str,
    reset_max_attempts: int = 15,
    operation: str = "saturation",
    ro_operation: str = "readout",
    simulate: bool = False,
    log: Optional[Callable] = None,
):
    """Build the Stark-tone spectroscopy QUA program. Returns ``(program, sweep_axes)``.

    `amps` is the Stark-tone amplitude prefactor sweep (multiplies the readout op's
    stored amplitude; refused by name outside QUA's (-2, 2)); `dfs` is the
    drive-detuning sweep in Hz; `qubits` is a BatchableList (see
    `_lib.select_qubits`) — the scqo shell passes one target per batch. All times
    are ns, multiples of 4, as scqo's ``stark_windows`` resolved them.
    """
    check_amp_scale_window(amps, name=", ".join(qubits.get_names()))
    num_qubits = len(qubits)
    tone_cycles = _cycles("tone_len_ns", tone_len_ns)
    ring_cycles = _cycles("ring_up_ns", ring_up_ns)
    drive_cycles = _cycles("drive_len_ns", drive_len_ns)
    depletion_cycles = _cycles("depletion_ns", depletion_ns)
    if drive_cycles < 1:
        raise ValueError(f"drive_len_ns={drive_len_ns} ns is shorter than one clock cycle")
    if tone_cycles != ring_cycles + drive_cycles:
        raise ValueError(
            f"tone_len_ns={tone_len_ns} must equal ring_up_ns + drive_len_ns "
            f"({ring_up_ns} + {drive_len_ns}): the drive ENDS with the tone")

    # the canonical names directly, amplitude outer: the raw nesting IS scqo's order
    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "amp_prefactor": xr.DataArray(amps, attrs={"long_name": "Stark-tone amplitude prefactor"}),
        "detuning_hz": xr.DataArray(dfs, attrs={"long_name": "qubit drive detuning", "units": "Hz"}),
    }

    with program() as prog:
        # Macro to declare I, Q, n and their respective streams for a given number of qubit
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        a = declare(fixed)  # QUA variable for the Stark-tone amplitude prefactor
        df = declare(int)  # QUA variable for the qubit drive detuning

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < num_shots, n + 1):
                save(n, n_st)
                with for_(*from_array(a, amps)):
                    with for_(*from_array(df, dfs)):
                        for i, qubit in multiplexed_qubits.items():
                            # Update the qubit frequency while the drive is still off
                            qubit.xy.update_frequency(df + qubit.xy.intermediate_frequency)
                            # a real state reset: the previous point's drive has ended
                            qubit.reset(reset_type, simulate, log_callable=log,
                                        max_attempts=reset_max_attempts)

                        # THE shared edge. The Stark tone and the drive both run from
                        # here and only the drive's own wait separates them - do not
                        # add an align() inside the block, that would serialize them.
                        align()

                        for i, qubit in multiplexed_qubits.items():
                            # the Stark tone: the readout op, stretched and scaled
                            qubit.resonator.play(ro_operation, duration=tone_cycles, amplitude_scale=a)
                            if ring_cycles:
                                # the photons build up before the drive probes the line
                                qubit.xy.wait(ring_cycles)
                            # the saturation drive, ending WITH the tone
                            qubit.xy.play(operation, amplitude_scale=1.0, duration=drive_cycles)

                        # the second anchor: tone and drive are both over here
                        align()

                        for i, qubit in multiplexed_qubits.items():
                            if depletion_cycles:
                                # the Stark photons leave before the readout
                                qubit.resonator.wait(depletion_cycles)
                            # the STANDARD readout, the same in every row
                            qubit.resonator.measure(ro_operation, qua_vars=(I[i], Q[i]))
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])
                        align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_st[i].buffer(len(dfs)).buffer(len(amps)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(dfs)).buffer(len(amps)).average().save(f"Q{i + 1}")

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


from typing import Any  # noqa: E402

from scqo import register  # noqa: E402
from scqo.experiments import QubitResonatorStark  # noqa: E402

from ._reset import _FACTORY_DEPLETION_NS  # noqa: E402


@register
class QMQubitResonatorStark(QubitResonatorStark):
    """Build the Stark-tone spectroscopy QUA program on the QM OPX, one target at
    a time. Subclasses the CORE class, not ``QMQubitSpectroscopy``: that shell opts
    into active reset, and a subclass would inherit the opt-in silently."""

    def probe(self) -> Any:
        from ._reset import check_reset_method, reset_max_attempts
        from scqo_qm.experiments._lib import select_qubits

        machine = self.backend.machine  # type: ignore[attr-defined]
        targets = list(self.params.targets)
        reset_type = check_reset_method(self)
        self._refuse_the_factory_depletion(targets)
        window = self.resolved_windows()
        return build_program(
            machine,
            select_qubits(machine, targets, multiplexed=False),
            amps=self.sweep_axes["amp_prefactor"],
            dfs=self.sweep_axes["detuning_hz"],
            tone_len_ns=window.tone_len_ns,
            ring_up_ns=window.ring_up_ns,
            drive_len_ns=window.drive_len_ns,
            depletion_ns=window.depletion_ns,
            num_shots=self.params.num_averages,
            reset_type=reset_type,
            reset_max_attempts=reset_max_attempts(self),
        )

    def _refuse_the_factory_depletion(self, targets: list[str]) -> None:
        """QUAM's ``depletion_time`` is never None — an ungoverned one sits at the
        16 ns factory default, which scqo's neutral helper cannot tell from a
        measured value. Refuse it (the rule active reset applies), unless this run
        states its own wait: without it the readout would start with the Stark
        photons still in the resonator, while Qblox refuses the same knob as NaN."""
        if self.params.readout_depletion_ns is not None:
            return
        factory = [t for t in targets
                   if abs(self.device.channel(t, "readout").readout_depletion_s * 1e9
                          - _FACTORY_DEPLETION_NS) < 1e-6]
        if factory:
            names = ", ".join(factory)
            raise ValueError(
                f"{self.name}: readout_depletion_s on {names} is QUAM's "
                f"{_FACTORY_DEPLETION_NS:g} ns factory default, i.e. never calibrated. "
                f"The Stark tone's photons must leave the resonator before the readout, "
                f"so run resonator_spectroscopy on {names} and accept its "
                f"readout_depletion_s proposal, or pass readout_depletion_ns= for this run."
            )
