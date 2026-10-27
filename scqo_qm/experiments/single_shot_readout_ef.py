"""QM single-shot readout of g, e and f for scqo - supplies only ``probe()``.

Parameters, the three-blob fit and the record-only result are inherited from
``scqo.experiments.SingleShotReadoutEf``. The program is the two-state readout's
(``_readout_fidelity.build_program``) with prepared state 2 added: the x180, then
the qubit's ``EF_x180`` operation on the same xy element retuned to the 1->2 IF
(``drive_freq_ef_hz`` = QUAM ``q.f_12``, minus the LO), the 0->1 IF restored right
after it - the same 1->2 pulse ``qubit_power_rabi_ef`` calibrates, at its stored
amplitude. PER-SHOT contract: ``buffer(3).buffer(num_shots)``, no ``.average()``.

AXIS ORDER: the shot loop is outer and the prepared-state loop inner, as in
``single_shot_readout``; the sweep axes carry the canonical names, so the
backend maps them by name and ``estimate()`` transposes by name.
"""

from __future__ import annotations

from typing import Any, ClassVar

from scqo import register
from scqo.experiments import SingleShotReadoutEf


@register
class QMSingleShotReadoutEf(SingleShotReadoutEf):
    """Build a multiplexed per-shot |g>/|e>/|f> readout QUA program on the QM OPX
    (thermal reset only; the core Parameters refuse an active one)."""

    backend_notes: ClassVar[tuple[str, ...]] = (
        "the 1->2 pi pulse is the qubit's EF_x180 operation (its amplitude is pi_amp_ef, "
        "its length pi_duration_ef_s), played on the qubit's own xy element retuned with "
        "update_frequency; the 0->1 IF is restored right after it",
        "drive_freq_ef_hz must lie within +-400 MHz of the xy LO (one upconverter's band)",
    )

    def probe(self) -> Any:
        from ._reset import check_reset_method
        from scqo_qm.experiments import _readout_fidelity as fidelity_probe
        from scqo_qm.experiments._lib import select_qubits
        from scqo_qm.experiments.qubit_power_rabi_ef import ef_intermediate_frequency
        from scqo_qm.quam_fields import EF_OPERATION

        machine = self.backend.machine  # type: ignore[attr-defined]
        targets = list(self.params.targets)
        qubits = select_qubits(machine, targets, multiplexed=True)
        reset_type = check_reset_method(self)
        ef_ifs = {q: ef_intermediate_frequency(
                      machine.qubits[q], self.device.channel(q, "drive").drive_freq_ef_hz)
                  for q in targets}
        return fidelity_probe.build_program(
            machine,
            qubits,
            operation="readout",
            num_shots=self.params.num_shots,
            reset_type=reset_type,
            prepared_states=(0, 1, 2),
            ef_ifs=ef_ifs,
            ef_operation=EF_OPERATION,
        )
