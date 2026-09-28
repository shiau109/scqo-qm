"""The Quantum Machines backend for the scqo experiment API.

Importing this package wires the QM stack into scqo: it exposes :class:`QMBackend`
and, as an import side effect, registers every QM experiment into the scqo
catalog. `qm`/`quam` are needed only for real acquisition (the vendor imports are
lazy, so `import scqo_qm` itself stays light and offline-friendly). The vendored
official qualibrate nodes never import this package.
"""

from scqo_qm.backend.qm_backend import (
    QMBackend,
    QMDeviceModel,
    QMDriveChannel,
    QMFluxChannel,
    QMFluxLine,
    QMOperation,
    QMReadoutChannel,
)
from scqo_qm import experiments  # noqa: F401  (import side effect: @register)

__all__ = [
    "QMBackend", "QMDeviceModel",
    # one view class per CHANNEL KIND, the flux LINE, and the operation surface
    "QMDriveChannel", "QMReadoutChannel", "QMFluxChannel", "QMFluxLine",
    "QMOperation",
]
