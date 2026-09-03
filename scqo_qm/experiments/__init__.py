"""Importing this package registers every QM experiment into the scqo catalog.

Add a line here for each new experiment module so its ``@register`` runs
(manual on purpose; ``tests/test_experiment_registration.py`` enforces
completeness in both directions).

Fused experiment modules carry their QUA builders at module level, so this
import pulls in ``qm.qua`` eagerly (the QUA DSL's star-import cannot be
function-local). qm's import-time logger writes to STDOUT by default, which
would corrupt the scqo CLI's JSON contract — the block below flips qm's own
switch and re-homes the same records on stderr before the first qm import.
"""

import logging as _logging
import os as _os
import sys as _sys

_os.environ.setdefault("QM_DISABLE_STREAMOUTPUT", "1")
_qm_logger = _logging.getLogger("qm")
if not any(getattr(_h, "stream", None) is _sys.stderr for _h in _qm_logger.handlers):
    _handler = _logging.StreamHandler(_sys.stderr)
    _handler.setFormatter(_logging.Formatter("%(asctime)s - qm - %(levelname)-8s - %(message)s"))
    _qm_logger.addHandler(_handler)
    _qm_logger.setLevel(_logging.INFO)

from . import pair_swap_chevron  # noqa: F401  (import side effect: @register)
from . import pair_swap_angle  # noqa: F401  (import side effect: @register)
from . import pair_swap_flux_map  # noqa: F401  (import side effect: @register)
from . import pair_zz_coupler  # noqa: F401  (import side effect: @register)
from . import qc_n_stark_amp  # noqa: F401  (import side effect: @register)
from . import qc_n_swap_amp  # noqa: F401  (import side effect: @register)
from . import qc_trotter_compensation  # noqa: F401  (import side effect: @register)
from . import qc_unidirectional_trotter  # noqa: F401  (import side effect: @register)
from . import qubit_ramsey_cryoscope  # noqa: F401  (import side effect: @register)
from . import qubit_deterministic_benchmarking  # noqa: F401  (import side effect: @register)
from . import qubit_drag_alternating  # noqa: F401  (import side effect: @register)

from . import qubit_drag_equator  # noqa: F401  (import side effect: @register)
from . import qubit_echo  # noqa: F401  (import side effect: @register)
from . import qubit_echo_flux_pulse  # noqa: F401  (import side effect: @register)
from . import qubit_parametric_drive_amp  # noqa: F401  (import side effect: @register)
from . import qubit_parametric_drive_time  # noqa: F401  (import side effect: @register)
from . import qubit_parity_switch_continuous  # noqa: F401  (import side effect: @register)
from . import qubit_parity_switch_discrete  # noqa: F401  (import side effect: @register)
from . import qubit_pi_pulse_error  # noqa: F401  (import side effect: @register)
from . import qubit_power_rabi  # noqa: F401  (import side effect: @register)
from . import qubit_ramsey  # noqa: F401  (import side effect: @register)
from . import qubit_ramsey_phasor  # noqa: F401  (import side effect: @register)
from . import qubit_relaxation  # noqa: F401  (import side effect: @register)
from . import qubit_relaxation_flux_pulse  # noqa: F401  (import side effect: @register)
from . import qubit_spectroscopy  # noqa: F401  (import side effect: @register)
from . import qubit_spectroscopy_cryoscope  # noqa: F401  (import side effect: @register)
from . import qubit_sqrb  # noqa: F401  (import side effect: @register)
from . import qubit_stark_phase_echo  # noqa: F401  (import side effect: @register)
from . import qubit_t1_ade  # noqa: F401  (import side effect: @register)
from . import qubit_t1_bayesian  # noqa: F401  (import side effect: @register)
from . import qubit_thermal_population  # noqa: F401  (import side effect: @register)
from . import qubit_tomography  # noqa: F401  (import side effect: @register)
from . import qubit_xyz_delay  # noqa: F401  (import side effect: @register)
from . import qubit_spectroscopy_flux_pulse  # noqa: F401  (import side effect: @register)
from . import readout_frequency  # noqa: F401  (import side effect: @register)
from . import readout_power  # noqa: F401  (import side effect: @register)
from . import broadband_resonator_spectroscopy  # noqa: F401  (import side effect: @register)
from . import broadband_qubit_spectroscopy  # noqa: F401  (import side effect: @register)
from . import resonator_spectroscopy  # noqa: F401  (import side effect: @register)
from . import resonator_spectroscopy_flux  # noqa: F401  (import side effect: @register)
from . import resonator_spectroscopy_power_chain  # noqa: F401  (import side effect: @register)
from . import resonator_spectroscopy_power_amp  # noqa: F401  (import side effect: @register)
from . import single_shot_readout  # noqa: F401  (import side effect: @register)
from . import single_shot_readout_gef  # noqa: F401  (import side effect: @register)

