"""Qubit-spectroscopy-vs-flux acquisition probe: vendor code only (qm/quam) - no qualibrate, no scqo, no scqat.

Sweep a flux bias and the drive frequency and read out each qubit; the qubit line is fitted
flux-by-flux downstream to give a frequency(flux) trace. Optional single flux / xy source qubits:
when `z_source_qubit` / `xy_source_qubit` are None every measured qubit fluxes / drives itself.

PULSE CONTRACT: `dcs` are VOLTS measured from the standing DC bias that
`initialize_qpu` applies -- the z `const` pulse rides on top of that offset, so
the axis is idle-relative and 0 V means "stay parked". `flux_point` is passed
EXPLICITLY (rather than leaning on quam_builder's "joint" default) because the
same value feeds the rail validation: the DAC emits idle + excursion, so the
number that was checked must be the number that plays.

QM qubit spectroscopy vs PULSED flux for scqo - supplies only ``probe()``.

Parameters, the transmon-arch fit and reporting are inherited from
``scqo.experiments.QubitSpectroscopyFluxPulse``. scqo sweeps ``(flux_bias_v,
detuning_hz)``; the QM builder sweeps ``dcs`` (flux, V) and ``dfs`` (drive
detuning, Hz) with each qubit fluxing/driving its OWN lines (z/xy source = None,
as in the official 03b node).

PULSE CONTRACT: this probe conforms to the ``_pulse`` name by construction — the
flux is a z PULSE played only alongside the saturation drive
(``build_program`` below: ``qubit.z.play(...,
duration=operation_duration)`` then ``align()`` then ``measure``), so every
readout happens at idle flux and the neutral ``estimate()`` reduces the map
against ONE global IQ reference. The DAC adds that pulse to the standing offset
``initialize_qpu`` applies, which is what makes ``flux_bias_v`` an excursion FROM
``idle_flux`` (0 V = stay parked) and why the neutral ``estimate()``
re-references the fitted sweet spot to an absolute set-point before writing
``flux_offset``. ``flux_point`` is passed explicitly so the bias the probe
rail-validated is the bias that plays. The probe MODULE keeps its historical name
(it is shared with the qualibrate shell path).

AXIS-ORDER NOTE (do not "fix" the order below): the probe's QUA loops nest df
(outer) over dc (inner) and its streams are ``buffer(len(dcs)).buffer(len(dfs))``,
so the raw per-qubit array is shaped (detuning, flux) — the OPPOSITE of scqo's
declared sweep order (flux_bias_v, detuning_hz). A positional rename in
``_to_canonical`` would therefore swap the axes (silently, whenever the two sweep
lengths happen to be equal). Instead we re-key the probe's sweep_axes with the
canonical scqo names IN RAW NESTING ORDER; ``_to_canonical`` then takes its
name-based path (no rename, per-name size checks) and ``estimate()`` transposes
by name.
"""

from __future__ import annotations

from typing import Callable, Optional

import xarray as xr
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.units import unit

from scqo_qm.experiments._lib import acquire as _acquire
from scqo_qm.experiments._flux_limits import check_flux_pulse_relative, idle_offset_v


def build_program(
    machine,
    qubits,
    *,
    dfs,
    dcs,
    operation: str,
    operation_len,
    num_shots: int,
    z_source_qubit: Optional[str] = None,
    xy_source_qubit: Optional[str] = None,
    multiplexed: bool = False,
    flux_point: str = "joint",
):
    """Build the qubit-spectroscopy-vs-flux QUA program. Returns (program, sweep_axes).

    `dfs` is the drive-detuning sweep (Hz), `dcs` the flux-PULSE sweep in volts
    RELATIVE to `flux_point`'s standing offset (see the module docstring's PULSE
    CONTRACT); `qubits` is a BatchableList (see `_lib.select_qubits`).
    `z_source_qubit` / `xy_source_qubit` (names on `machine`, or None) select a
    single flux / xy source. `operation_len` (ns) overrides the driving qubit's
    operation length when not None.
    """
    u = unit(coerce_to_integer=True)
    num_qubits = len(qubits)

    # Resolve the single flux / xy source qubits (if any). When None, each measured
    # qubit drives its own z- / xy-line (identical to the official 03b node).
    z_source = None if z_source_qubit is None else machine.qubits[z_source_qubit]
    xy_source = None if xy_source_qubit is None else machine.qubits[xy_source_qubit]

    # Validate the line(s) that will actually carry the pulse: the assigned
    # source when there is one, else every measured qubit's own z. The division
    # below was always correct; what was missing is the check that the requested
    # window is expressible AND that idle + excursion stays under the rail.
    flux_carriers = [z_source] if z_source is not None else list(qubits)
    amp_ref = {}
    for carrier in flux_carriers:
        z = getattr(carrier, "z", None)
        if z is None:
            raise ValueError(
                f"{carrier.name}: no flux line, but this probe plays the flux "
                f"as a z pulse on it")
        amp_ref[carrier.name] = check_flux_pulse_relative(
            z, name=carrier.name, idle_v=idle_offset_v(z, flux_point), amps_v=dcs)

    # Saturation duration (Python-level): uniform when operation_len is given, else
    # taken from the driving qubit's operation length.
    ref_qubit = xy_source if xy_source is not None else next(iter(qubits))
    operation_duration = (
        operation_len * u.ns if operation_len is not None
        else ref_qubit.xy.operations[operation].length * u.ns
    )

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "qubit frequency", "units": "Hz"}),
        "flux_bias": xr.DataArray(
            dcs,
            attrs={"long_name": "flux pulse amplitude relative to the idle bias", "units": "V"},
        ),
    }

    with program() as prog:
        # Macro to declare I, Q, n and their respective streams for a given number of qubit
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        df = declare(int)  # QUA variable for the qubit drive frequency detuning
        dc = declare(fixed)  # QUA variable for the flux dc level

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                machine.initialize_qpu(target=qubit, flux_point=flux_point)
            align()

            with for_(n, 0, n < num_shots, n + 1):
                save(n, n_st)
                with for_(*from_array(df, dfs)):
                    with for_(*from_array(dc, dcs)):
                        # Qubit initialization: thermalize to the ground state.
                        for i, qubit in multiplexed_qubits.items():
                            qubit.reset_qubit_thermal()
                        # Update the drive frequency on whichever xy-line plays.
                        if xy_source is None:
                            for i, qubit in multiplexed_qubits.items():
                                qubit.xy.update_frequency(df + qubit.xy.intermediate_frequency)
                        else:
                            xy_source.xy.update_frequency(df + xy_source.xy.intermediate_frequency)
                        align()

                        # Bring the qubit(s) to the flux point during the saturation pulse.
                        if z_source is None:
                            for i, qubit in multiplexed_qubits.items():
                                qubit.z.play(
                                    "const",
                                    amplitude_scale=dc / qubit.z.operations["const"].amplitude,
                                    duration=operation_duration,
                                )
                        else:
                            z_source.z.play(
                                "const",
                                amplitude_scale=dc / z_source.z.operations["const"].amplitude,
                                duration=operation_duration,
                            )
                        # Apply the saturation drive: from each qubit, or a single xy source.
                        if xy_source is None:
                            for i, qubit in multiplexed_qubits.items():
                                qubit.xy.play(operation, amplitude_scale=1.0, duration=operation_duration)
                        else:
                            xy_source.xy.play(operation, amplitude_scale=1.0, duration=operation_duration)
                        align()

                        # Readout every measured qubit's resonator.
                        for i, qubit in multiplexed_qubits.items():
                            qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])

            # Measure sequentially
            if not multiplexed:
                align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_st[i].buffer(len(dcs)).buffer(len(dfs)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(dcs)).buffer(len(dfs)).average().save(f"Q{i + 1}")

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

from scqo import register
from scqo.experiments import QubitSpectroscopyFluxPulse


@register
class QMQubitSpectroscopyFluxPulse(QubitSpectroscopyFluxPulse):
    """Build a multiplexed 2D (pulsed-flux x detuning) spectroscopy QUA program on the QM OPX."""

    def probe(self) -> Any:
        from scqo_qm.experiments._lib import select_qubits
        from scqo_qm.quam_fields import GOVERNED_FLUX_POINT

        from ._vendor import flux_source_name

        machine = self.backend.machine  # type: ignore[attr-defined]
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)

        # A foreign flux source is a ROSTER entity; this probe plays the flux as
        # a z PULSE on a QUAM qubit, so _vendor refuses a coupler here with the
        # reason (qubit_only) instead of failing inside the QUA build.
        z_source = (None if self.params.flux_component is None
                    else flux_source_name(self, self.params.flux_component,
                                          qubit_only=True))

        prog, axes = build_program(
            machine,
            qubits,
            dfs=self.sweep_axes["detuning_hz"],
            dcs=self.sweep_axes["flux_bias_v"],
            operation="saturation",
            operation_len=None,  # use each qubit's own saturation-pulse length
            num_shots=self.params.num_averages,
            z_source_qubit=z_source,
            xy_source_qubit=None,  # None = every measured qubit drives its own xy line
            flux_point=GOVERNED_FLUX_POINT,
        )
        # Canonical names in RAW nesting order (detuning outer, flux inner — see
        # module docstring); the DataArray values (incl. units attrs) are reused.
        sweep_axes = {
            "qubit": axes["qubit"],
            "detuning_hz": axes["detuning"],
            "flux_bias_v": axes["flux_bias"],
        }
        return prog, sweep_axes
