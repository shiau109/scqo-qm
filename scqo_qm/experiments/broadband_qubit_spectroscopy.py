"""QM broadband qubit spectroscopy for scqo — supplies only ``probe()``.

Parameters, fitting, simulation are inherited from
``scqo.experiments.BroadbandQubitSpectroscopy``.

Multi-band support
------------------
When the requested RF range crosses MW-FEM hardware band boundaries the
probe automatically:
  1. Splits [start_freq_hz, stop_freq_hz] into non-overlapping band segments
     using the cutoffs below.
  2. For each segment switches ``opx_output.band`` on all target drive ports
     before setting the sub-band LO.
  3. Restores every changed attribute (band, LO, RF_frequency) in a ``finally``
     block so the machine is left in its original state even on error.

For non-MW-FEM hardware (no ``opx_output.band`` attribute) the multi-band
path is skipped and the code falls back to the original single-band behaviour.

MW-FEM band definitions (lo = upconverter_frequency range):
  Band 1: LO 0.05 – 5.5  GHz, recommended for RF < 5.0 GHz
  Band 2: LO 4.5  – 7.5  GHz, recommended for RF 5.0 – 7.0 GHz
  Band 3: LO 6.5  – 10.5 GHz, recommended for RF > 7.0 GHz
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from scqo import register
from scqo.experiments import BroadbandQubitSpectroscopy

# ---------------------------------------------------------------------------
# MW-FEM band table: band -> (lo_min, lo_max)
# ---------------------------------------------------------------------------
_MW_FEM_BANDS: dict[int, tuple[float, float]] = {
    1: (0.05e9, 5.5e9),
    2: (4.5e9,  7.5e9),
    3: (6.5e9,  10.5e9),
}

# Non-overlapping RF segments for cross-band scanning.
# Band 2 is included because port-pair switching (see _partner_port_id) ensures
# that both ports in a pair (2,3)(4,5)(6,7) are always set to the same band,
# satisfying the QM FEM constraint.
#
# Physical RF coverage with min_if=50 MHz, max_if=250 MHz:
#   Band 1: RF up to  lo_max + max_if = 5.5  + 0.25 = 5.75 GHz
#   Band 2: RF  5.75 GHz to 6.55 GHz  (the gap Band 1/3 cannot reach alone)
#   Band 3: RF from   lo_min + min_if = 6.5  + 0.05 = 6.55 GHz
_RF_BAND_SEGMENTS: list[tuple[int, float, float]] = [
    (1, 0.0,    5.75e9),           # Band 1: RF up to 5.75 GHz
    (2, 5.75e9, 6.55e9),           # Band 2: RF 5.75 - 6.55 GHz (gap only Band 2 covers)
    (3, 6.55e9, float("inf")),     # Band 3: RF from 6.55 GHz
]


def _make_int_sweep(f_start: float, f_stop: float, n_pts: int) -> np.ndarray:
    """Generate an exact integer-stepped array to satisfy QUA from_array linearity checks."""
    n = max(2, int(n_pts))
    start_int = int(round(f_start))
    stop_int = int(round(f_stop))
    if stop_int <= start_int:
        return np.array([start_int], dtype=np.int64)
    step = max(1, int(round((stop_int - start_int) / (n - 1))))
    return start_int + np.arange(n, dtype=np.int64) * step


def _get_port_info(opx_out) -> tuple[str, int, int] | None:
    """Return (controller, fem, port_id) from an opx_output object, or None.

    Tries several common QuAM attribute names so the helper is robust across
    different QuAM versions.
    """
    ctrl = (
        getattr(opx_out, "controller_id", None)
        or getattr(opx_out, "controller", None)
    )
    fem = (
        getattr(opx_out, "fem_id", None)
        or getattr(opx_out, "fem", None)
    )
    port = (
        getattr(opx_out, "port_id", None)
        or getattr(opx_out, "port", None)
    )
    if ctrl is None or fem is None or port is None:
        return None
    return (str(ctrl), int(fem), int(port))


def _partner_port_id(port_id: int) -> int:
    """Return the port that is paired with port_id under the MW-FEM constraint.

    QM MW-FEM requires that port pairs (2,3) (4,5) (6,7) always share the same
    band.  Even-numbered ports are paired with the next odd port; odd ports with
    the previous even port.
    """
    return port_id + 1 if port_id % 2 == 0 else port_id - 1


def _build_slices(
    start: float,
    stop: float,
    min_if: float,
    max_if: float,
    span_per_lo: float,
    has_band_switching: bool,
    current_band: int | None,
    current_min_lo: float,
    current_max_lo: float,
) -> list[tuple[int, float, float, float, float, float]]:
    """Return all (band, f_start, f_stop, lo, lo_min, lo_max) slice tuples.

    When *has_band_switching* is True, the RF range is split into per-band
    segments using ``_RF_BAND_SEGMENTS`` and each segment is sliced using that
    band's LO limits.  Otherwise, the entire range is sliced with the single
    band's limits.
    """
    if has_band_switching:
        # Build one (band, rf_start, rf_stop, lo_min, lo_max) entry per active segment
        active_segs: list[tuple[int, float, float, float, float]] = []
        for seg_band, seg_rf_min, seg_rf_max in _RF_BAND_SEGMENTS:
            seg_start = max(start, seg_rf_min)
            seg_stop = min(stop, seg_rf_max)
            if seg_start >= seg_stop:
                continue
            if seg_band not in _MW_FEM_BANDS:
                continue
            lo_min, lo_max = _MW_FEM_BANDS[seg_band]
            active_segs.append((seg_band, seg_start, seg_stop, lo_min, lo_max))
    else:
        active_segs = [(current_band or 0, start, stop, current_min_lo, current_max_lo)]

    slices: list[tuple[int, float, float, float, float, float]] = []
    for seg_band, seg_start, seg_stop, lo_min, lo_max in active_segs:
        curr_f = seg_start
        while curr_f < seg_stop:
            next_f = min(seg_stop, curr_f + span_per_lo)
            candidate_lo = curr_f - min_if  # LO = RF - min_if (RF > LO, positive IF)

            if candidate_lo >= lo_min:
                lo = min(candidate_lo, lo_max)
            else:
                lo = lo_min
                allowed_min_rf = lo_min + min_if
                allowed_max_rf = lo_min + max_if
                if next_f <= allowed_min_rf:
                    curr_f = allowed_min_rf
                    continue
                if curr_f < allowed_min_rf:
                    curr_f = allowed_min_rf
                next_f = min(next_f, allowed_max_rf)

            if curr_f >= next_f:
                break

            slices.append((seg_band, curr_f, next_f, lo, lo_min, lo_max))
            curr_f = next_f

    return slices


@register
class QMBroadbandQubitSpectroscopy(BroadbandQubitSpectroscopy):
    """Build and execute wideband two-tone qubit spectroscopy across stepped drive LOs.

    Supports automatic band switching for MW-FEM hardware so that the scan
    range can span multiple hardware bands (e.g. Band 1 + Band 2).
    """

    # preview opt-out (backend.SELF_ACQUIRING_ATTR): truthy reason = refuse
    probe_self_acquires = "broadband qubit spectroscopy steps drive LO frequencies across sub-bands"

    def probe(self) -> xr.Dataset:
        from scqo_qm.experiments._lib import select_qubits
        from scqo_qm.experiments._reset import check_reset_method
        from scqo_qm.experiments.qubit_spectroscopy import acquire, build_program

        machine = self.backend.machine  # type: ignore[attr-defined]
        targets = self.params.targets
        primary_target = targets[0]
        primary_qubit = machine.qubits[primary_target]
        qubits = select_qubits(machine, targets, multiplexed=True)
        reset_type = check_reset_method(self)
        operation_len = int(self.params.drive_len_ns)

        start = float(self.params.start_freq_hz)
        stop = float(self.params.stop_freq_hz)
        bw = float(self.params.bandwidth_per_lo_hz)
        pts_per_lo = int(self.params.num_points_per_lo)
        gap = float(self.params.lo_gap_hz)
        num_shots = int(self.params.num_averages)

        # Save original drive LO, RF, and band settings for all qubits
        orig_mw_up: dict[str, float | None] = {}
        orig_oct_up: dict[str, float | None] = {}
        orig_rf_frequencies: dict[str, float | None] = {}
        orig_bands: dict[str, int | None] = {}

        for q_name, q_obj in machine.qubits.items():
            if hasattr(q_obj, "xy"):
                orig_rf_frequencies[q_name] = getattr(q_obj.xy, "RF_frequency", None)
                orig_mw_up[q_name] = getattr(getattr(q_obj.xy, "opx_output", None), "upconverter_frequency", None)
                orig_oct_up[q_name] = getattr(getattr(q_obj.xy, "frequency_converter_up", None), "LO_frequency", None)
                orig_bands[q_name] = getattr(getattr(q_obj.xy, "opx_output", None), "band", None)

        # Detect whether the primary drive port supports runtime band switching
        primary_opx_out = getattr(primary_qubit.xy, "opx_output", None)
        has_band_switching: bool = (
            primary_opx_out is not None and hasattr(primary_opx_out, "band")
        )
        current_band: int | None = getattr(primary_opx_out, "band", None)

        # Single-band fallback limits (used when no band switching is available)
        single_lo_min, single_lo_max = 0.0, float("inf")
        if current_band == 1:
            single_lo_min, single_lo_max = _MW_FEM_BANDS[1]
        elif current_band == 2:
            single_lo_min, single_lo_max = _MW_FEM_BANDS[2]
        elif current_band == 3:
            single_lo_min, single_lo_max = _MW_FEM_BANDS[3]

        # Guard band: IF stays in [min_if, max_if] (positive; RF > LO for drive port)
        min_if = max(50.0e6, gap / 2.0)
        max_if = min(250.0e6, min_if + bw)
        span_per_lo = max_if - min_if

        # Build all (band, f_start, f_stop, lo, lo_min, lo_max) slices
        all_slices = _build_slices(
            start, stop, min_if, max_if, span_per_lo,
            has_band_switching, current_band,
            single_lo_min, single_lo_max,
        )

        all_rf_freqs: list[np.ndarray] = []
        all_i_by_target: dict[str, list[np.ndarray]] = {t: [] for t in targets}
        all_q_by_target: dict[str, list[np.ndarray]] = {t: [] for t in targets}

        # Build the set of qubits that must change band together with the targets.
        # MW-FEM constraint: port pairs (2,3)(4,5)(6,7) must always be in the same band.
        # We collect every qubit whose drive port is the hardware partner of any target port.
        port_pair_members: set[str] = set(targets)
        if has_band_switching:
            target_fem_ports: set[tuple[str, int, int]] = set()
            for target_name in targets:
                opx_out = getattr(getattr(machine.qubits[target_name], "xy", None), "opx_output", None)
                if opx_out is not None:
                    info = _get_port_info(opx_out)
                    if info is not None:
                        ctrl, fem, port = info
                        target_fem_ports.add(info)                              # target itself
                        target_fem_ports.add((ctrl, fem, _partner_port_id(port)))  # its pair partner

            if target_fem_ports:
                for q_name, q_obj in machine.qubits.items():
                    opx_out = getattr(getattr(q_obj, "xy", None), "opx_output", None)
                    if opx_out is None or not hasattr(opx_out, "band"):
                        continue
                    if _get_port_info(opx_out) in target_fem_ports:
                        port_pair_members.add(q_name)

        # Track the currently-active band to avoid redundant switches
        active_band: int | None = current_band

        try:
            for seg_band, f_a, f_b, lo, lo_min, lo_max in all_slices:
                slice_span = f_b - f_a
                if slice_span <= 0:
                    continue

                clamped_lo = min(max(lo, lo_min), lo_max)
                clamped_lo_int = int(round(clamped_lo))

                n_pts = max(2, int(round(pts_per_lo * (slice_span / span_per_lo))))
                rf_seg = _make_int_sweep(f_a, f_b, n_pts)

                valid_mask = (rf_seg >= start) & (rf_seg <= stop)
                if not np.any(valid_mask):
                    continue
                clipped_rf = rf_seg[valid_mask]
                if len(clipped_rf) > 1 and len(clipped_rf) != len(rf_seg):
                    clipped_rf = _make_int_sweep(clipped_rf[0], clipped_rf[-1], len(clipped_rf))

                # Switch hardware band if needed (MW-FEM only).
                # Switch the target AND its paired port together so the port-pair
                # band constraint ((2,3)(4,5)(6,7) must share the same band) is met.
                # For non-target partners: also park their LO within the new band's
                # valid range (the QM config validator rejects LO values that are
                # outside the configured band).  Targets' LOs are synchronized to
                # clamped_lo in the step below; the finally block restores everything.
                if has_band_switching and seg_band != active_band:
                    lo_min_new, _ = _MW_FEM_BANDS.get(seg_band, (0.0, float("inf")))
                    for member_name in port_pair_members:
                        member_q = machine.qubits[member_name]
                        member_xy = getattr(member_q, "xy", None)
                        if member_xy is None:
                            continue
                        opx_out = getattr(member_xy, "opx_output", None)
                        if opx_out is not None and hasattr(opx_out, "band"):
                            opx_out.band = seg_band
                        # Non-target members: park LO at lo_min of the new band so
                        # the upconverter frequency is inside the new band's range.
                        # Target members' LOs are updated to clamped_lo below.
                        if member_name not in targets:
                            if opx_out is not None and hasattr(opx_out, "upconverter_frequency"):
                                opx_out.upconverter_frequency = lo_min_new
                            if hasattr(member_xy, "RF_frequency"):
                                member_xy.RF_frequency = lo_min_new  # IF = 0
                    active_band = seg_band


                # dfs = detuning relative to intermediate_frequency, as expected by
                # build_program (qubit_spectroscopy.py: update_frequency(df + IF)).
                # Read BEFORE setting RF_frequency so we capture the current IF.
                nominal_if = int(round(primary_qubit.xy.intermediate_frequency))
                dfs = (clipped_rf - clamped_lo_int - nominal_if).astype(np.int32)

                # Synchronize target qubits to clamped_lo so base IF = 0 Hz
                for target_name in targets:
                    target_q = machine.qubits[target_name]
                    if hasattr(target_q, "xy"):
                        target_xy = target_q.xy
                        if hasattr(target_xy, "RF_frequency"):
                            target_xy.RF_frequency = clamped_lo
                        if hasattr(target_xy, "opx_output") and hasattr(target_xy.opx_output, "upconverter_frequency"):
                            target_xy.opx_output.upconverter_frequency = clamped_lo
                        if hasattr(target_xy, "frequency_converter_up") and hasattr(target_xy.frequency_converter_up, "LO_frequency"):
                            target_xy.frequency_converter_up.LO_frequency = clamped_lo

                # Synchronize idle qubits to their own upconverter LO
                for q_name, q_obj in machine.qubits.items():
                    if q_name not in targets and hasattr(q_obj, "xy") and hasattr(q_obj.xy, "RF_frequency"):
                        idle_lo = getattr(getattr(q_obj.xy, "opx_output", None), "upconverter_frequency", None)
                        if idle_lo is None:
                            idle_lo = getattr(getattr(q_obj.xy, "frequency_converter_up", None), "LO_frequency", None)
                        if idle_lo is not None:
                            q_obj.xy.RF_frequency = idle_lo

                prog, sweep_axes = build_program(
                    machine,
                    qubits,
                    dfs=dfs,
                    operation="saturation",
                    operation_len=operation_len,
                    operation_amp=1.0,
                    num_shots=num_shots,
                    reset_type=reset_type,
                )

                sub_ds = acquire(
                    machine,
                    prog,
                    sweep_axes,
                    num_shots=num_shots,
                    timeout=self.backend._timeout,
                )

                all_rf_freqs.append(clipped_rf)
                for t_idx, target in enumerate(targets):
                    i_name = f"I{t_idx + 1}"
                    q_name_var = f"Q{t_idx + 1}"
                    if i_name in sub_ds and q_name_var in sub_ds:
                        i_vals = np.asarray(sub_ds[i_name].values).squeeze()
                        q_vals = np.asarray(sub_ds[q_name_var].values).squeeze()
                    elif "I" in sub_ds and "Q" in sub_ds:
                        i_raw = np.asarray(sub_ds["I"].values).squeeze()
                        q_raw = np.asarray(sub_ds["Q"].values).squeeze()
                        if i_raw.ndim > 1:
                            i_vals = i_raw[t_idx]
                            q_vals = q_raw[t_idx]
                        else:
                            i_vals = i_raw
                            q_vals = q_raw
                    else:
                        i_var = [v for v in sub_ds.data_vars if "I" in v][t_idx]
                        q_var = [v for v in sub_ds.data_vars if "Q" in v][t_idx]
                        i_vals = np.asarray(sub_ds[i_var].values).squeeze()
                        q_vals = np.asarray(sub_ds[q_var].values).squeeze()

                    all_i_by_target[target].append(i_vals.ravel())
                    all_q_by_target[target].append(q_vals.ravel())

        finally:
            # Restore all qubits' original band, LO, and RF_frequency
            for q_name, q_obj in machine.qubits.items():
                if not hasattr(q_obj, "xy"):
                    continue
                target_xy = q_obj.xy

                # Restore band first so subsequent LO writes land in the right range
                if has_band_switching:
                    orig_b = orig_bands.get(q_name)
                    opx_out = getattr(target_xy, "opx_output", None)
                    if orig_b is not None and opx_out is not None and hasattr(opx_out, "band"):
                        opx_out.band = orig_b

                # Restore upconverter / Octave LO
                mw_val = orig_mw_up.get(q_name)
                if mw_val is not None and hasattr(target_xy, "opx_output"):
                    target_xy.opx_output.upconverter_frequency = mw_val
                oct_val = orig_oct_up.get(q_name)
                if oct_val is not None and hasattr(target_xy, "frequency_converter_up"):
                    target_xy.frequency_converter_up.LO_frequency = oct_val

                # Restore RF_frequency last (depends on LO being correct)
                rf_val = orig_rf_frequencies.get(q_name)
                if rf_val is not None and hasattr(target_xy, "RF_frequency"):
                    target_xy.RF_frequency = rf_val

        if not all_rf_freqs:
            raise RuntimeError("no frequency sub-bands were measured")

        # Stitch full spectrum
        stitched_freqs = np.concatenate(all_rf_freqs)
        order = np.argsort(stitched_freqs)
        unique_indices = np.unique(stitched_freqs[order], return_index=True)[1]
        sorted_indices = order[unique_indices]
        final_freqs = stitched_freqs[sorted_indices]

        n_targets = len(targets)
        n_freqs = len(final_freqs)
        final_i = np.empty((n_targets, n_freqs), dtype=float)
        final_q = np.empty((n_targets, n_freqs), dtype=float)

        for t_idx, target in enumerate(targets):
            t_i = np.concatenate(all_i_by_target[target])[sorted_indices]
            t_q = np.concatenate(all_q_by_target[target])[sorted_indices]
            final_i[t_idx] = t_i
            final_q[t_idx] = t_q

        return xr.Dataset(
            data_vars={
                "I": (("target", "frequency_hz"), final_i),
                "Q": (("target", "frequency_hz"), final_q),
            },
            coords={
                "target": targets,
                "frequency_hz": final_freqs,
            },
        )
