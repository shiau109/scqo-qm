"""Qubit-spectroscopy acquisition probes: vendor code only (qm/quam) - no qualibrate, no scqo, no scqat.

Sweep the qubit drive detuning while playing a (typically weak, long) saturation pulse and
reading out the resonator; the qubit line is fitted downstream.

TWO SEQUENCES, ONE EXPERIMENT. The saturation drive always ENDS at an anchor and
starts ``drive_len_ns`` earlier; ``readout_overlap`` picks the anchor:

    readout_overlap = False              readout_overlap = True
    [==== drive ====]                       [======== drive ========]
                    [## readout ##]   [## readout tone ############]
                    ^ anchor                                anchor ^
                                            [acq_lead_ns] [== ADC ==]

``build_program`` is the sequential one (an ``align()`` between the drive and the
measurement); ``build_overlap_program`` is the concurrent one. Nothing is bounded
against anything: a 20 us drive against a 2 us tone simply starts before the tone
and runs through it, and whichever element starts SECOND is given a ``wait`` of
the lead scqo computed.

WHAT MAKES THE TONES OVERLAP: there is NO ``align()`` between the qubit drive and
the measurement in ``build_overlap_program``. Both element timelines run from the
one shared ``align()`` above them, so only their own ``wait``s separate them.

WHAT MAKES THE ADC OPEN LATE: QUAM's ``measure()`` takes no acquisition-delay
argument - the ADC follows its own pulse by ``resonator.time_of_flight`` and
nothing else - so the lead is a PRE-TONE instead. The same readout operation is
played for ``acq_lead_ns`` immediately before ``measure()``, contiguous on the
resonator's timeline, so the resonator sees one seamless tone of
``acq_lead_ns + readout_len`` and the ADC integrates only its tail.
``time_of_flight`` is untouched: it is the cable, not the lead.

NOTE (a SILENT failure, overlap mode only): the qubit drive (``xy``) and its
``resonator`` must be on DIFFERENT QM cores/threads for the tones to overlap at
all. Same-core elements are SERIALIZED - the drive plays to completion first, you
get the sequential experiment back, and the fit still converges and looks clean.
Check the analog traces in the OPX simulator before trusting a result from a new
wiring, and if they are serialized, repartition the FEM rather than working
around it here.

QM qubit spectroscopy for scqo - supplies only ``probe()``.

Parameters, peak fitting and the drive_freq_hz writeback are inherited from
``scqo.experiments.QubitSpectroscopy``. scqo sweeps ``detuning_hz``; the QM builder
builds the same sweep on coord ``detuning``, which the backend's ``_to_canonical``
renames back. The overlap timing is NOT computed here:
``scqo.experiments._overlap.overlap_windows`` resolves the tone length, the ADC
lead and the two start leads from the readout channel's own knobs and refuses
off-grid values, so this shell and the Qblox probe cannot drift apart on what the
same Parameters mean.

Drive power contract: the core ``run()`` already solved the drive chain for
``drive_power_dbm`` (recorded set -> acquire -> revert), parking the exact
amplitude on the saturation op — so the probe plays it at ``amplitude_scale=1.0``
(exact in QUA fixed point). The shared probe keeps its ``operation_amp`` argument
for the qualibrate node, a separate consumer with its own explicit amps.
"""

from __future__ import annotations

from typing import Callable, Optional

import xarray as xr
from qm.qua import *
from qualang_tools.loops import from_array

from scqo_qm.experiments._lib import acquire as _acquire


def _cycles(name: str, value_ns) -> int:
    """ns -> QUA clock cycles, refusing anything off the 4 ns grid.

    Refused rather than rounded: a silent round here would make the same
    requested timing mean one thing on QM and another on Qblox, which is the
    whole failure ``scqo.experiments._overlap`` exists to prevent.
    """
    value_ns = int(round(float(value_ns)))
    if value_ns % 4:
        raise ValueError(
            f"{name}={value_ns} ns is not a multiple of the 4 ns QUA clock cycle"
        )
    return value_ns // 4


def build_program(
    machine,
    qubits,
    *,
    dfs,
    operation: str,
    operation_len,
    operation_amp: float,
    num_shots: int,
    reset_type: str,
    reset_max_attempts: int = 15,
    drive_qubit: Optional[str] = None,
    simulate: bool = False,
    log: Optional[Callable] = None,
):
    """Build the SEQUENTIAL qubit-spectroscopy QUA program. Returns (program, sweep_axes).

    `dfs` is the drive-detuning sweep in Hz; `qubits` is a BatchableList (see
    `_lib.select_qubits`). When `drive_qubit` is None every qubit is driven;
    otherwise only that qubit plays the drive. `operation_len` (ns) overrides the
    operation's configured length when not None — the scqo path always passes a
    number (the drive length is a Parameter), so the fallback is there for the
    qualibrate node, which is a separate consumer with its own inputs.

    The ``align()`` between the drive block and the measurement block is the
    sequence: the drive is over before the readout tone starts, which is what
    lets the line be measured with no readout photons present.
    """
    num_qubits = len(qubits)

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "readout frequency", "units": "Hz"}),
    }

    with program() as prog:
        # Macro to declare I, Q, n and their respective streams for a given number of qubit
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        df = declare(int)  # QUA variable for the qubit frequency

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < num_shots, n + 1):
                save(n, n_st)
                with for_(*from_array(df, dfs)):
                    for i, qubit in multiplexed_qubits.items():
                        qubit.reset(reset_type, simulate, log_callable=log,
                                    max_attempts=reset_max_attempts)

                    for i, qubit in multiplexed_qubits.items():
                        if drive_qubit is None or qubit.name == drive_qubit:
                            # Get the duration of the operation from the node parameters or the state
                            duration = operation_len if operation_len is not None else qubit.xy.operations[operation].length
                            # Update the qubit frequency
                            qubit.xy.update_frequency(df + qubit.xy.intermediate_frequency)
                            # Play the saturation pulse
                            qubit.xy.play(
                                operation,
                                amplitude_scale=operation_amp,
                                duration=duration // 4,
                            )
                    align()

                    for i, qubit in multiplexed_qubits.items():
                        # readout the resonator
                        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        # save data
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
                    align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_st[i].buffer(len(dfs)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(dfs)).average().save(f"Q{i + 1}")

    return prog, sweep_axes


def build_overlap_program(
    machine,
    qubits,
    *,
    dfs,
    operation: str,
    drive_len_ns,
    operation_amp: float,
    acq_lead_ns,
    drive_lead_ns,
    readout_lead_ns,
    ro_operation: str,
    num_shots: int,
    reset_type: str,
    reset_max_attempts: int = 15,
    simulate: bool = False,
    log: Optional[Callable] = None,
):
    """Build the CONCURRENT-tone qubit-spectroscopy QUA program.

    Returns ``(program, sweep_axes)``.

    `dfs` is the drive-detuning sweep in Hz; `qubits` is a BatchableList (see
    `_lib.select_qubits`). `drive_len_ns` is the saturation length and
    `acq_lead_ns` is how long the readout tone runs before the ADC opens. The
    drive and the tone END together, so whichever starts second waits: the
    resonator waits `drive_lead_ns` (drive longer than the tone) or the drive
    waits `readout_lead_ns` (tone longer than the drive). Exactly one of those is
    non-zero; scqo's ``_overlap.overlap_windows`` computes both, so this file
    never subtracts two lengths. All times are ns, all multiples of 4.
    """
    num_qubits = len(qubits)
    drive_cycles = _cycles("drive_len_ns", drive_len_ns)
    lead_cycles = _cycles("acq_lead_ns", acq_lead_ns)
    drive_lead_cycles = _cycles("drive_lead_ns", drive_lead_ns)
    readout_lead_cycles = _cycles("readout_lead_ns", readout_lead_ns)
    if drive_cycles < 1:
        raise ValueError(f"drive_len_ns={drive_len_ns} ns is shorter than one clock cycle")

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "qubit drive detuning", "units": "Hz"}),
    }

    with program() as prog:
        # Macro to declare I, Q, n and their respective streams for a given number of qubit
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        df = declare(int)  # QUA variable for the qubit frequency

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < num_shots, n + 1):
                save(n, n_st)
                with for_(*from_array(df, dfs)):
                    for i, qubit in multiplexed_qubits.items():
                        # Update the qubit frequency while the drive is still off
                        qubit.xy.update_frequency(df + qubit.xy.intermediate_frequency)
                        # Wait for the qubit to decay to the ground state. This is a
                        # real state reset in both modes: the drive is a finite pulse
                        # that ended before this point, so nothing is driving here.
                        qubit.reset(reset_type, simulate, log_callable=log,
                                    max_attempts=reset_max_attempts)

                    # THE shared edge. Every element below runs from here and only
                    # its own wait separates it - do not add an align() inside the
                    # block, that is what would serialize the two tones.
                    align()

                    for i, qubit in multiplexed_qubits.items():
                        if drive_lead_cycles:
                            # the drive is longer than the whole tone: it started
                            # first, so the resonator sits out the difference
                            qubit.resonator.wait(drive_lead_cycles)
                        if lead_cycles:
                            # the ADC lead: the same readout operation, played
                            # back-to-back into measure() so the resonator sees
                            # one seamless tone and only its tail is integrated
                            qubit.resonator.play(ro_operation, duration=lead_cycles)
                        if readout_lead_cycles:
                            # the tone is longer than the drive: the drive starts
                            # late so that the two END together
                            qubit.xy.wait(readout_lead_cycles)
                        # the saturation drive, on its own element's timeline
                        qubit.xy.play(
                            operation,
                            amplitude_scale=operation_amp,
                            duration=drive_cycles,
                        )
                        # readout the resonator (the ADC opens here, acq_lead_ns
                        # after the tone onset, plus the element's time_of_flight)
                        qubit.resonator.measure(ro_operation, qua_vars=(I[i], Q[i]))
                        # save data
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
                    align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_st[i].buffer(len(dfs)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(dfs)).average().save(f"Q{i + 1}")

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
from scqo.experiments import QubitSpectroscopy
from scqo.experiments._overlap import overlap_windows


@register
class QMQubitSpectroscopy(QubitSpectroscopy):
    """Build a multiplexed two-tone spectroscopy QUA program on the QM OPX.

    One shell, two sequences, chosen by ``readout_overlap`` — see the module
    docstring for the anchor rule and the FEM-core caveat that decides whether
    the concurrent mode really overlaps.
    """

    #: Both sequences play a FINITE saturation pulse that is over before the
    #: reset, so the reset is a genuine state reset and active reset is valid
    #: here. The readout condition is frozen for the whole run (only the DRIVE
    #: frequency sweeps), which is the other half of the rule in _reset.py.
    #: NOT yet validated on the instrument — see the hardware ledger.
    supports_active_reset: ClassVar[bool] = True

    def probe(self) -> Any:
        from ._reset import check_reset_method, reset_max_attempts
        from scqo_qm.experiments._lib import select_qubits

        machine = self.backend.machine  # type: ignore[attr-defined]
        targets = list(self.params.targets)
        qubits = select_qubits(machine, targets, multiplexed=True)
        reset_type = check_reset_method(self)

        if not self.params.readout_overlap:
            return build_program(
                machine,
                qubits,
                dfs=self.sweep_axes["detuning_hz"],
                operation="saturation",
                operation_len=int(self.params.drive_len_ns),
                operation_amp=1.0,  # run() parked the exact amplitude on the saturation op
                num_shots=self.params.num_averages,
                reset_type=reset_type,
                reset_max_attempts=reset_max_attempts(self),
            )

        # One multiplexed program plays ONE set of times, so every target has to
        # agree on them. They can only differ through the per-target readout
        # knobs, and silently taking the first target's would put the others'
        # ADC in the wrong place — visible as a weaker, shifted peak on exactly
        # the qubits nobody was looking at.
        windows = {q: overlap_windows(self, q) for q in targets}
        distinct = {(w.tone_len_ns, w.acq_start_ns, w.drive_len_ns) for w in windows.values()}
        if len(distinct) > 1:
            detail = ", ".join(
                f"{q}: tone={w.tone_len_ns:g} ns, acq_start={w.acq_start_ns:g} ns, "
                f"drive={w.drive_len_ns:g} ns"
                for q, w in windows.items()
            )
            raise ValueError(
                f"qubit_spectroscopy (readout_overlap=true): the targets resolve to "
                f"different concurrent-tone windows, which one multiplexed program "
                f"cannot play ({detail}). They differ through readout_duration_s / "
                f"readout_integration_s — equalize those, or run the targets in "
                f"separate runs."
            )
        window = windows[targets[0]]

        return build_overlap_program(
            machine,
            qubits,
            dfs=self.sweep_axes["detuning_hz"],
            operation="saturation",
            drive_len_ns=window.drive_len_ns,
            operation_amp=1.0,  # run() parked the exact amplitude on the saturation op
            acq_lead_ns=window.acq_start_ns,
            drive_lead_ns=window.drive_lead_ns,
            readout_lead_ns=window.readout_lead_ns,
            ro_operation="readout",
            num_shots=self.params.num_averages,
            reset_type=reset_type,
            reset_max_attempts=reset_max_attempts(self),
        )
