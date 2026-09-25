"""The contract ``_lib`` depends on from the vendored `qualibration-libs` pieces.

`scqo_qm/_vendored/qualibration_libs/` is a COPY (see its README): the upstream
distribution requires `qualibrate`, which this driver does not run. A copy can go stale
without anything failing loudly, so this file pins what `_lib.select_qubits` and
`_lib.acquire` actually use — the batching shape every probe iterates, and the fetcher's
dataset, scalar handles and `t_start`. Refresh the copy against upstream and this is what
says whether the refresh is safe.

The QM job and `qualang_tools`' fetching_tool are doubled: the fetcher's own job plumbing
is upstream's business, while the SHAPE it returns is what this driver reads.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scqo_qm._vendored.qualibration_libs import BatchableList, XarrayDataFetcher
from scqo_qm._vendored.qualibration_libs import fetcher as fetcher_module


def _qubit(name):
    return SimpleNamespace(name=name)


# --- BatchableList: what select_qubits builds and every probe iterates ------------

def test_one_group_per_item_is_the_sequential_shape():
    q1, q2 = _qubit("q1"), _qubit("q2")
    batchable = BatchableList([q1, q2], [[0], [1]])

    assert batchable.get_names() == ["q1", "q2"]
    assert [dict(group) for group in batchable.batch()] == [{0: q1}, {1: q2}]


def test_one_group_with_every_index_is_the_multiplexed_shape():
    q1, q2 = _qubit("q1"), _qubit("q2")

    assert [dict(g) for g in BatchableList([q1, q2], [[0, 1]]).batch()] == [{0: q1, 1: q2}]


def test_a_group_map_that_misses_an_item_is_refused():
    with pytest.raises(ValueError):
        BatchableList([_qubit("q1"), _qubit("q2")], [[0]])


# --- XarrayDataFetcher: the dataset _lib.acquire returns -------------------------

class _FetchingToolDouble:
    """qualang_tools' live fetching_tool, reduced to what the fetcher calls."""

    def __init__(self, values, rounds=1):
        self._values = values
        self._left = rounds

    def is_processing(self):
        self._left -= 1
        return self._left > 0

    def fetch_all(self):
        return list(self._values)

    def get_start_time(self):
        return 1234.5


@pytest.fixture()
def fetched(monkeypatch):
    """One 2-qubit x 3-point acquisition, fetched through the double."""
    handles = ["n", "I1", "Q1"]
    job = SimpleNamespace(result_handles={k: None for k in handles})
    i_data = np.arange(6, dtype=float).reshape(2, 3)
    q_data = -i_data
    monkeypatch.setattr(
        fetcher_module, "fetching_tool",
        lambda job, keys, mode: _FetchingToolDouble([7, i_data, q_data]))

    axes = {"qubit": np.array(["q1", "q2"]), "freq": np.arange(3)}
    fetcher = XarrayDataFetcher(job, axes)
    datasets = list(fetcher)
    return fetcher, datasets, i_data


def test_the_fetcher_yields_a_dataset_over_the_declared_axes(fetched):
    _, datasets, i_data = fetched

    dataset = datasets[-1]
    assert list(dataset.coords) == ["qubit", "freq"]
    assert dataset["I1"].dims == ("qubit", "freq")
    np.testing.assert_allclose(dataset["I1"].values, i_data)
    np.testing.assert_allclose(dataset["Q1"].values, -i_data)


def test_a_scalar_handle_is_readable_by_name_and_t_start_is_set(fetched):
    fetcher, _, _ = fetched

    # _lib.acquire feeds both to qualang_tools' progress_counter
    assert fetcher.get("n", 0) == 7
    assert fetcher.get("missing", "default") == "default"
    assert fetcher.t_start is not None
