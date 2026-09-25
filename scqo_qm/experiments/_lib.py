"""Shared plumbing for probes: target selection and the execute-and-fetch half.

``BatchableList`` and ``XarrayDataFetcher`` come from
``scqo_qm._vendored.qualibration_libs`` — a copy of the two modules this file needs,
because the upstream distribution requires ``qualibrate`` and this driver does not run
it (see that package's README; ``tests/test_lib_fetcher.py`` pins the contract).

Flux amplitude/rail validation lives in ``_flux_limits.py`` — a probe asking
"may this port emit these volts?" imports from there, not here.
"""

from typing import Callable, List, Optional

import xarray as xr
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter

from scqo_qm._vendored.qualibration_libs import BatchableList, XarrayDataFetcher


def select_qubits(machine, names: Optional[List[str]] = None, *, multiplexed: bool = False) -> BatchableList:
    """Node-free replacement for `qualibration_libs.parameters.get_qubits(node)`.

    Selects qubits from the machine by name (or `machine.active_qubits` when
    `names` is None/empty) and wraps them in the same `BatchableList` the
    qualibrate helper produces, so probes can iterate `qubits.batch()` /
    `qubits.get_names()` identically in both shells.
    """
    if not names:
        qubits = machine.active_qubits
    else:
        qubits = [machine.qubits[q] for q in names]
    if multiplexed:
        batched_groups = [list(range(len(qubits)))]
    else:
        batched_groups = [[i] for i in range(len(qubits))]
    return BatchableList(qubits, batched_groups)


def select_qubit_pairs(machine, names: Optional[List[str]] = None, *,
                       multiplexed: bool = False) -> BatchableList:
    """Node-free replacement for `qualibration_libs.parameters.get_qubit_pairs(node)`:
    selects pairs from the machine by name (or `machine.active_qubit_pairs` when
    `names` is None/empty) and wraps them in the same `BatchableList`."""
    if not names:
        pairs = machine.active_qubit_pairs
    else:
        pairs = [machine.qubit_pairs[p] for p in names]
    if multiplexed:
        batched_groups = [list(range(len(pairs)))]
    else:
        batched_groups = [[i] for i in range(len(pairs))]
    return BatchableList(pairs, batched_groups)


def acquire(
    machine,
    prog,
    sweep_axes,
    *,
    num_shots: int,
    timeout: float,
    log: Optional[Callable] = None,
    config: Optional[dict] = None,
) -> xr.Dataset:
    """Connect to the QOP, execute the program and fetch the raw xr.Dataset.

    The execute-and-fetch half is identical for every swept experiment, so all
    probes share this one implementation. `config` defaults to
    `machine.generate_config()`; pass an explicit config when the program needs a
    pre-built one (e.g. a baked config carrying baking ops the fresh config lacks).
    """
    qmm = machine.connect()
    config = config if config is not None else machine.generate_config()
    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    with qm_session(qmm, config, timeout=timeout) as qm:
        job = qm.execute(prog)
        data_fetcher = XarrayDataFetcher(job, sweep_axes)
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher.get("n", 0),
                num_shots,
                start_time=data_fetcher.t_start,
            )
        # Expose possible runtime errors. execution_report is a method on some QM
        # API versions and a property on others — tolerate both.
        if log:
            rep = getattr(job, "execution_report", None)
            if callable(rep):
                log(rep())
            elif rep is not None:
                log(rep)
    return dataset
