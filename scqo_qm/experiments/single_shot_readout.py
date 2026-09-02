"""QM single-shot readout fidelity for scqo - supplies only ``probe()``.

Parameters, the two-Gaussian-mixture fit and reporting are inherited from
``scqo.experiments.SingleShotReadout``. PER-SHOT contract: every readout shot's
I/Q point is recorded individually — the probe's streams are
``buffer(2).buffer(num_shots)`` with NO ``.average()``.

AXIS-ORDER NOTE: the probe's QUA loops nest the shot loop (outer) over the
prepared-state loop (inner), so the raw per-qubit array is shaped
(shot_idx, prepared_state) — the OPPOSITE of scqo's declared sweep order
(prepared_state, shot_idx). The probe's sweep_axes already carry the canonical
names ``shot_idx``/``prepared_state`` in that raw nesting order, so the backend's
``_to_canonical`` takes its name-based path (no positional rename — which would
try to swap the axes) and ``estimate()`` transposes by name.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scqo import Outcome, register
from scqo.experiments import SingleShotReadout


@register
class QMSingleShotReadout(SingleShotReadout):
    """Build a multiplexed per-shot |g>/|e> readout QUA program on the QM OPX, and
    propose the readout discriminator (rotation + thresholds) as governed
    suggestions from the measured blobs."""

    def probe(self) -> Any:
        from ._reset import check_reset_method
        from scqo_qm.experiments._lib import select_qubits
        from scqo_qm.experiments import _readout_fidelity as fidelity_probe

        machine = self.backend.machine  # type: ignore[attr-defined]
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)

        return fidelity_probe.build_program(
            machine,
            qubits,
            operation="readout",
            num_shots=self.params.num_shots,
            reset_type=check_reset_method(self),
        )

    def update(self) -> None:
        """The inherited per-state fidelity + blob-center MONITOR writes
        (``fidelity_g``/``fidelity_e``/``pos_*`` on the readout channel; the old
        aggregate ``readout_fidelity`` is gone) plus the QM readout
        DISCRIMINATOR as governed suggestions.

        For every SUCCESSFUL qubit this computes the demod rotation + thresholds from
        the measured blobs (``scqat.tools.compute_ge_discriminator`` — the solve is
        shared with the Qblox backend, which realizes the same two knobs) and PROPOSES
        them through ``self.device`` as the neutral fields ``readout_rotation_rad`` /
        ``readout_threshold`` / ``readout_rus_threshold`` on the target's READOUT
        channel (``self.device.channel(qubit, "readout")``) — captured as pending
        suggestions, decided with ``scqo accept`` and pushed to the QUAM readout
        operation on accept, exactly like ``drive_freq_hz`` / ``pi_amp``. The run itself
        never touches the vendor config, so the figure always shows the data in the
        frame it was measured in; to check a new rotation, accept it and re-run.

        The rotation field is ABSOLUTE: the measured ``delta`` is relative to the
        current weights rotation, so the proposal is ``current - delta`` (the field's
        current value seeded from the vendor in pull mode). An UNSET rotation reads
        as 0.0 — the honest baseline for a channel whose discriminator has never been
        calibrated, which is precisely when this runs first."""
        super().update()  # fidelity_g/e + pos_* monitors (through self.device)

        if self.result is None or self.dataset is None:
            return
        from scqat.tools import compute_ge_discriminator

        for qubit in self.params.targets:
            if self.result.outcomes.get(qubit) is not Outcome.SUCCESSFUL:
                continue
            fit = self.result.fit[qubit]
            mean_g = (fit["mean_g_i"], fit["mean_g_q"])
            mean_e = (fit["mean_e_i"], fit["mean_e_q"])
            if not np.all(np.isfinite([*mean_g, *mean_e])):
                print(f"[single_shot_readout] {qubit}: degenerate blobs; no discriminator proposal")
                continue

            sq = self.dataset.sel(target=qubit)
            shots_g = (sq["I"].sel(prepared_state=0).values, sq["Q"].sel(prepared_state=0).values)
            shots_e = (sq["I"].sel(prepared_state=1).values, sq["Q"].sel(prepared_state=1).values)
            d = compute_ge_discriminator(mean_g, mean_e, shots_g, shots_e)

            # The discriminator trio lives on the READOUT channel entity
            # (q1_ro), addressed by its target's default channel — the qubit
            # MODE name carries no knobs since the greenfield split.
            view = self.device.channel(qubit, "readout")
            # An UNSET rotation means no rotation has been applied yet, so the
            # absolute baseline is 0.0 — not an error. It is the normal state of a
            # channel whose discriminator has never been calibrated, which is
            # exactly when the first single-shot readout runs; floating None
            # crashed update() there and cost the run its whole suggestion set.
            stored = view.readout_rotation_rad
            current = 0.0 if stored is None else float(stored)
            new_rotation = current - d["delta_angle_rad"]  # accumulate as an absolute proposal
            view.readout_rotation_rad = new_rotation
            view.readout_threshold = d["ge_threshold"]
            view.readout_rus_threshold = d["rus_exit_threshold"]
            print(
                f"[single_shot_readout] {qubit}: discriminator PROPOSED "
                f"(readout_rotation_rad {current:.4f} -> {new_rotation:.4f} rad, "
                f"readout_threshold {d['ge_threshold']:.4g}) — review with scqo accept, "
                f"then re-run to confirm"
            )
