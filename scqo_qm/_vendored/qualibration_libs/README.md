# Vendored from `qualibration-libs`

Upstream: <https://github.com/qua-platform/qualibration-libs>, version **0.2.1**, commit
**09fc7357b7233b313827931733ae4b7d493e71b8** — the commit `scqo-qm/pyproject.toml` pinned
while the package was still a dependency. BSD 3-Clause, `LICENSE` beside this file.

## Why these files are here

`scqo_qm/experiments/_lib.py` needs exactly two classes: `BatchableList` (the batching
shape every probe iterates) and `XarrayDataFetcher` (the execute-and-fetch half of
`acquire`). Neither module imports `qualibrate` — only `qualibration_libs.parameters`
does — but the DISTRIBUTION requires `qualibrate>=1.0.2`, so depending on it installs the
whole GUI stack (fastapi, uvicorn, sqlalchemy, psycopg2) into every QM environment. This
driver does not run qualibrate nodes, so it carries the ~500 lines instead.

| here | upstream path |
|---|---|
| `batchable_list.py` | `qualibration_libs/core/batchable_list.py` |
| `exceptions.py` | `qualibration_libs/core/exceptions.py` (only `format_available_items`, which the other two use) |
| `fetcher.py` | `qualibration_libs/data/fetcher.py` |

## What was changed

Nothing but the intra-package import, which cannot resolve once the files move: in
`batchable_list.py` and `fetcher.py`,

```
from qualibration_libs.core.exceptions import format_available_items
```

became the same import from `scqo_qm._vendored.qualibration_libs.exceptions`. The files are
otherwise byte-identical to upstream, so a later `diff` against it is meaningful — keep it
that way, and fix behaviour in `scqo_qm/` rather than here (the repo's rule for vendored
code).

## Refreshing

Install the upstream version you want in a scratch environment, copy the three files over,
re-apply the import edit, run the scqo-qm suite (`tests/test_lib_fetcher.py` pins the
contract `_lib.acquire` depends on), and update the commit above.
