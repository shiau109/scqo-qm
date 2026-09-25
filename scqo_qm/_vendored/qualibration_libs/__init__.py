"""The two pieces of ``qualibration-libs`` this driver uses (see README.md).

``BatchableList`` is the shape ``_lib.select_qubits`` hands every probe, and
``XarrayDataFetcher`` is the execute-and-fetch half of ``_lib.acquire``. Import them
from here, never from the installed ``qualibration_libs``: that distribution requires
``qualibrate``, and this driver does not.
"""

from scqo_qm._vendored.qualibration_libs.batchable_list import BatchableList
from scqo_qm._vendored.qualibration_libs.fetcher import XarrayDataFetcher

__all__ = ["BatchableList", "XarrayDataFetcher"]
