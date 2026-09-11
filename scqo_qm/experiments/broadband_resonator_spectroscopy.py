"""QM broadband resonator spectroscopy for scqo — supplies only ``probe()``.

Parameters, fitting, simulation are inherited from
``scqo.experiments.BroadbandResonatorSpectroscopy``.
"""

from __future__ import annotations

import warnings

from typing import Any
import numpy as np
import xarray as xr

from scqo_qm._family import RF_OCTAVE, rf_chain
from scqo_qm._octave import LO_MAX_HZ, LO_MIN_HZ, rf_outputs_sharing_synth, snap_lo

#: MW-FEM LO range per Nyquist band (Hz), for the READOUT output port. The drive
#: probe's table differs at the band-1 edges and deliberately stays its own: these
#: are the ranges each probe was commissioned against, not a shared vendor fact.
_MW_FEM_BANDS: dict[int, tuple[float, float]] = {
    1: (0.5e9, 4.5e9),
    2: (4.5e9, 7.5e9),
    3: (7.5e9, 10.5e9),
}

from scqo import register
from scqo.experiments import BroadbandResonatorSpectroscopy


def _make_int_sweep(f_start: float, f_stop: float, n_pts: int) -> np.ndarray:
    """Generate an exact integer-stepped array to satisfy QUA from_array linearity checks."""
    n = max(2, int(n_pts))
    start_int = int(round(f_start))
    stop_int = int(round(f_stop))
    if stop_int <= start_int:
        return np.array([start_int], dtype=np.int64)
    step = max(1, int(round((stop_int - start_int) / (n - 1))))
    return start_int + np.arange(n, dtype=np.int64) * step


@register
class QMBroadbandResonatorSpectroscopy(BroadbandResonatorSpectroscopy):
    """Build and execute wideband resonator spectroscopy across stepped LO sub-bands on the QM OPX."""

    # preview opt-out (backend.SELF_ACQUIRING_ATTR): truthy reason = refuse
    probe_self_acquires = "broadband spectroscopy steps LO frequencies across sub-bands"

    def probe(self) -> xr.Dataset:
        from scqo_qm.experiments._lib import acquire, select_qubits
        from scqo_qm.experiments._resonator_spectroscopy import build_program

        machine = self.backend.machine  # type: ignore[attr-defined]
        primary_target = self.params.targets[0]
        qubit = machine.qubits[primary_target]
        qubits = select_qubits(machine, [primary_target], multiplexed=False)

        start = float(self.params.start_freq_hz)
        stop = float(self.params.stop_freq_hz)
        bw = float(self.params.bandwidth_per_lo_hz)
        pts_per_lo = int(self.params.num_points_per_lo)
        gap = float(self.params.lo_gap_hz)
        num_shots = int(self.params.num_averages)

        lo_step = bw
        n_segments = max(1, int(np.ceil((stop - start) / lo_step)))
        lo_centers = [start + (i + 0.5) * lo_step for i in range(n_segments)]

        # Save original LO and RF settings to restore in finally
        rr = qubit.resonator
        orig_mw_up = getattr(getattr(rr, "opx_output", None), "upconverter_frequency", None)
        orig_mw_down = getattr(getattr(rr, "opx_input", None), "downconverter_frequency", None)
        orig_oct_up = getattr(getattr(rr, "frequency_converter_up", None), "LO_frequency", None)
        orig_oct_down = getattr(getattr(rr, "frequency_converter_down", None), "LO_frequency", None)

        orig_rf_frequencies: dict[str, float | None] = {}
        for q_name, q_obj in machine.qubits.items():
            if hasattr(q_obj, "resonator") and hasattr(q_obj.resonator, "RF_frequency"):
                orig_rf_frequencies[q_name] = q_obj.resonator.RF_frequency

        # The LO limits this probe may step within. It STEPS an LO, so an
        # unknown range is not something to degrade around: the old [0, inf]
        # fallback let an Octave tree walk its synthesizer below 2 GHz and label
        # the result as though it had worked.
        chain = rf_chain(rr)
        is_octave = chain == RF_OCTAVE
        band = getattr(getattr(rr, "opx_output", None), "band", None)
        if is_octave:
            min_lo, max_lo = LO_MIN_HZ, LO_MAX_HZ
        elif band in _MW_FEM_BANDS:
            min_lo, max_lo = _MW_FEM_BANDS[band]
        else:
            raise ValueError(
                f"{primary_target}: this probe steps the readout LO, and the LO "
                f"range of its RF chain ({chain or 'undeclared'}) is unknown - an "
                f"MW-FEM declares it through opx_output.band (this port's is "
                f"{band!r}) and an Octave through its synthesizer's [2, 18] GHz. "
                f"Refusing rather than sweeping with open limits, which would "
                f"produce a labelled spectrum from LOs the hardware never played.")

        if is_octave:
            # Moving this LO physically moves any RF output sharing its
            # synthesizer, and this probe tracks only the readout line - so it
            # would restore its own LO and leave the partner's parked wherever
            # the last segment put it. Say so rather than let it happen quietly.
            # (A normal OPX+ readout sits on RF1, which has synth1 to itself, so
            # this is a wiring remark, not a routine one.)
            rf_output = getattr(getattr(rr, "frequency_converter_up", None), "id", None)
            partners = (rf_outputs_sharing_synth(int(rf_output))
                        if rf_output is not None else ())
            if partners:
                warnings.warn(
                    f"{primary_target}: the readout is on Octave RF output "
                    f"{rf_output}, which shares a synthesizer with RF output(s) "
                    f"{', '.join(str(p) for p in partners)}. Stepping the readout "
                    f"LO moves theirs too, and this probe restores only its own - "
                    f"any line on those outputs will be left at the last "
                    f"segment's LO. Re-seed them from state.json afterwards.",
                    RuntimeWarning, stacklevel=2)

        # Guard band and slice parameters: IF strictly stays in [min_if, max_if]
        # or [-max_if, -min_if]. 400 MHz happens to be BOTH the Octave's hardware
        # IF ceiling and this probe's MW-FEM convention, so one number serves.
        min_if = max(20.0e6, gap / 2.0)
        max_if = min(400.0e6, min_if + bw)
        span_per_lo = max_if - min_if

        # Sub-band slices: (f_start, f_stop, lo_freq)
        slices: list[tuple[float, float, float]] = []
        curr_f = start
        while curr_f < stop:
            next_f = min(stop, curr_f + span_per_lo)
            candidate_lo = curr_f - min_if
            if candidate_lo >= min_lo:
                lo = min(candidate_lo, max_lo)
            else:
                lo = min_lo
                if next_f > lo - min_if:
                    next_f = min(stop, lo - min_if)

            if curr_f >= next_f:
                curr_f = lo + min_if
                continue

            slices.append((curr_f, next_f, lo))
            curr_f = next_f

        all_rf_freqs: list[np.ndarray] = []
        all_i: list[np.ndarray] = []
        all_q: list[np.ndarray] = []

        try:
            for f_a, f_b, lo in slices:
                slice_span = f_b - f_a
                if slice_span <= 0:
                    continue

                clamped_lo = min(max(lo, min_lo), max_lo)
                if is_octave:
                    # Snap BEFORE the axis is derived from it: dfs below is
                    # (rf - clamped_lo_int), so an LO the synthesizer would
                    # silently round leaves every point of this segment
                    # mislabelled by the rounding.
                    clamped_lo = min(max(snap_lo(clamped_lo), min_lo), max_lo)
                clamped_lo_int = int(round(clamped_lo))

                n_pts = max(2, int(round(pts_per_lo * (slice_span / span_per_lo))))
                rf_seg = _make_int_sweep(f_a, f_b, n_pts)

                valid_mask = (rf_seg >= start) & (rf_seg <= stop)
                if not np.any(valid_mask):
                    continue
                clipped_rf = rf_seg[valid_mask]
                if len(clipped_rf) > 1 and len(clipped_rf) != len(rf_seg):
                    clipped_rf = _make_int_sweep(clipped_rf[0], clipped_rf[-1], len(clipped_rf))

                dfs = clipped_rf - clamped_lo_int

                # Synchronize all resonators' RF_frequency to clamped_lo so base IF = 0 Hz for all elements
                for q_obj in machine.qubits.values():
                    if hasattr(q_obj, "resonator") and hasattr(q_obj.resonator, "RF_frequency"):
                        q_obj.resonator.RF_frequency = clamped_lo

                # Update LO frequency on MW-FEM or Octave converters
                if hasattr(rr, "opx_output") and hasattr(rr.opx_output, "upconverter_frequency"):
                    rr.opx_output.upconverter_frequency = clamped_lo
                if hasattr(rr, "opx_input") and hasattr(rr.opx_input, "downconverter_frequency"):
                    rr.opx_input.downconverter_frequency = clamped_lo
                if hasattr(rr, "frequency_converter_up") and hasattr(rr.frequency_converter_up, "LO_frequency"):
                    rr.frequency_converter_up.LO_frequency = clamped_lo
                if hasattr(rr, "frequency_converter_down") and hasattr(rr.frequency_converter_down, "LO_frequency"):
                    rr.frequency_converter_down.LO_frequency = clamped_lo

                prog, sweep_axes = build_program(
                    machine,
                    qubits,
                    dfs=dfs,
                    num_shots=num_shots,
                )

                sub_ds = acquire(
                    machine,
                    prog,
                    sweep_axes,
                    num_shots=num_shots,
                    timeout=self.backend._timeout,
                )

                if "I1" in sub_ds:
                    i_vals = np.asarray(sub_ds["I1"].values).squeeze()
                    q_vals = np.asarray(sub_ds["Q1"].values).squeeze()
                elif "I" in sub_ds:
                    i_vals = np.asarray(sub_ds["I"].values).squeeze()
                    q_vals = np.asarray(sub_ds["Q"].values).squeeze()
                else:
                    i_var = [v for v in sub_ds.data_vars if "I" in v][0]
                    q_var = [v for v in sub_ds.data_vars if "Q" in v][0]
                    i_vals = np.asarray(sub_ds[i_var].values).squeeze()
                    q_vals = np.asarray(sub_ds[q_var].values).squeeze()

                all_rf_freqs.append(clipped_rf)
                all_i.append(i_vals.ravel())
                all_q.append(q_vals.ravel())
        finally:
            # LO first, then RF_frequency: an RF is only meaningful once its LO is
            # correct, and restoring in the other order briefly asks for an IF the
            # chain cannot carry. Each step is guarded on its own, because one
            # channel that refuses a write must not strand the REST of the tree at
            # a swept LO - the worst possible exit, since the session would end
            # "cleanly" with resonators parked mid-sweep and the next experiment
            # would measure them there without complaint.
            restore_failures: list[str] = []

            def _restore(what: str, apply) -> None:
                try:
                    apply()
                except Exception as exc:
                    restore_failures.append(f"{what}: {type(exc).__name__}: {exc}")

            if orig_mw_up is not None and getattr(rr, "opx_output", None) is not None:
                _restore("readout upconverter LO",
                         lambda: setattr(rr.opx_output, "upconverter_frequency",
                                         orig_mw_up))
            if orig_mw_down is not None and getattr(rr, "opx_input", None) is not None:
                _restore("readout downconverter LO",
                         lambda: setattr(rr.opx_input, "downconverter_frequency",
                                         orig_mw_down))
            if orig_oct_up is not None and getattr(rr, "frequency_converter_up", None) is not None:
                _restore("Octave up-converter LO",
                         lambda: setattr(rr.frequency_converter_up, "LO_frequency",
                                         orig_oct_up))
            if orig_oct_down is not None and getattr(rr, "frequency_converter_down", None) is not None:
                _restore("Octave down-converter LO",
                         lambda: setattr(rr.frequency_converter_down, "LO_frequency",
                                         orig_oct_down))

            for q_name, rf_val in orig_rf_frequencies.items():
                if rf_val is None:
                    continue
                resonator = getattr(machine.qubits[q_name], "resonator", None)
                if resonator is None:
                    continue
                _restore(f"{q_name} readout RF",
                         lambda r=resonator, v=rf_val: setattr(r, "RF_frequency", v))

            if restore_failures:
                warnings.warn(
                    "broadband_resonator_spectroscopy could not restore "
                    + "; ".join(restore_failures)
                    + " - those readout channels are still parked at a sub-band "
                      "frequency and the next experiment would measure them "
                      "there. Re-seed from the setup's state.json.",
                    RuntimeWarning, stacklevel=2)

        if not all_rf_freqs:
            raise RuntimeError("no frequency sub-bands were measured")

        # Stitch full spectrum
        stitched_freqs = np.concatenate(all_rf_freqs)
        stitched_i = np.concatenate(all_i)
        stitched_q = np.concatenate(all_q)

        # Sort and deduplicate by frequency
        order = np.argsort(stitched_freqs)
        unique_indices = np.unique(stitched_freqs[order], return_index=True)[1]
        sorted_indices = order[unique_indices]

        final_freqs = stitched_freqs[sorted_indices]
        final_i = stitched_i[sorted_indices]
        final_q = stitched_q[sorted_indices]

        # Broadcast across all requested targets
        targets = self.params.targets
        n_targets = len(targets)

        i_2d = np.tile(final_i, (n_targets, 1))
        q_2d = np.tile(final_q, (n_targets, 1))

        dataset = xr.Dataset(
            data_vars={
                "I": (("target", "frequency_hz"), i_2d),
                "Q": (("target", "frequency_hz"), q_2d),
            },
            coords={
                "target": targets,
                "frequency_hz": final_freqs,
            },
        )
        return dataset


