# AGENTS.md — scqo-qm

**You are reading the contributor brief.** If you are working in the maintainer's lab
tree, read [CLAUDE.md](CLAUDE.md) instead — it is the far fuller document (every physics
and backend invariant lives there) and its rules assume a shared, live checkout.

## What this repo is

Two products in one repo:

1. **`scqo_qm/`** — the Quantum Machines OPX1000 backend for
   [SCQO](https://github.com/shiau109/SCQO), the vendor-neutral experiment API. Its Qblox
   sibling is [scqo-qblox](https://github.com/shiau109/scqo-qblox); never import from it.
2. **Vendored official qualibrate calibrations** (`calibrations/` + `calibration_utils/`),
   copied in by `sync_official.py`. Official nodes only.

Vendor stack: **qm-qua → quam → qualibrate**.

## Setup — a standalone fork of this repo cannot install

`pyproject.toml` resolves scqo as `{ path = "../SCQO", editable = true }`. You need SCQO
and scqat cloned as **siblings** under one parent, under their own names:

```
<parent>/
  SCQO/
  scqat/
  scqo-qm/
```

This repo pins a Python 3.11 lockfile and pulls two dependencies straight from git, so
build it from the lockfile rather than from `pyproject.toml`:

```bash
cd <parent>
uv venv .venv-qm --python 3.11
uv pip install --python .venv-qm/bin/python -r ./scqo-qm/requirements-qm.lock.txt
uv pip install --python .venv-qm/bin/python -e ./scqat -e ./SCQO -e ./scqo-qm --no-deps
```

Windows: `.venv-qm\Scripts\python.exe`.

## Editable installs freeze their version at install time

After changing branches or pulling, re-run the install lines. Entry points — the `scqo`
command and the backend registration — also register only at install time, so a checkout
alone leaves them stale.

## Adding an experiment

1. ONE file, `scqo_qm/experiments/<name>.py`, subclassing the backend-free experiment
   from `scqo.experiments.<name>`.
2. Write the QUA program in a module-level `build_program(machine, qubits, *, ...)` and
   keep the class's `probe()` a thin params→builder mapping. That builder seam is the
   house style — QUA programs are long, and it is what the live-state tests and the
   preview exercise.
3. An experiment whose streams `XarrayDataFetcher` cannot fetch defines its own
   module-level `acquire()`, and `probe()` returns `(prog, sweep_axes, acquire)` — the
   callable.
4. `@register` it and add its import line to `scqo_qm/experiments/__init__.py`.
   `tests/test_experiment_registration.py` refuses a module missing its line.
5. **Two census literals react to a new experiment** and are the authority for their
   sets — update them if yours belongs: `tests/test_preview.py`'s `SELF_ACQUIRING`
   (probes that acquire inside `probe()`) and `tests/test_reset_method.py`'s `CARRIERS`
   (active-reset opt-in).
6. May import `qm.qua` (the DSL star-import cannot be function-local), `quam`,
   `qualang_tools`, `qualibration_libs.core`/`.data`, `scqo`. **Never** `qualibrate`,
   **never** `scqat`.

A qualibrate node is NOT part of adding an experiment — the GUI serves official nodes only.

## Two invariants that fail SILENTLY

Read the whole *Physics + backend invariants* section of `CLAUDE.md` before touching a
probe. These two bite hardest:

- **Virtual-detuning sign.** A probe realizing `frequency_detuning_hz` must ramp the
  phase **negative**. Get it backwards and every accepted update *doubles* the residual
  detuning while the fit still looks clean.
- **Flux frames.** `check_flux_bias_absolute` (the swept value replaces the bias) versus
  `check_flux_pulse_relative` (the excursion rides on the standing bias). Confusing them
  is silent in both directions. The DAC rail is a property of the **port**, not a
  constant — no builder carries its own rail number.

## Testing

Run with the venv interpreter directly:

```bash
.venv-qm/bin/python -m pytest tests/ -q          # Windows: .venv-qm\Scripts\python.exe
```

Avoid bare `uv run` — its sync would rebuild the env from `pyproject.toml` and displace
the lockfile's pin authority. `uv run --no-sync` helps only with `UV_PROJECT_ENVIRONMENT`
pointed at `.venv-qm`; by default it still targets `scqo-qm/.venv`, which is not a thing
this repo has. Which environment for which repo: [ENVIRONMENTS.md](ENVIRONMENTS.md). The suite
is small enough that the **full run is the targeted run**; run it before every commit.

**This repo has no CI** — its git-sourced dependencies and pinned py3.11 lockfile make it
impractical. So your pasted local result is the only evidence a reviewer has. Report the
exact command and which interpreter produced it.

## What you can and cannot verify

You **can** run the offline suite and `python scripts/check_real_config.py <folder>`
against your own lab's `state.json` + `wiring.json` — it works on a temporary copy and
never writes to your originals.

You **cannot** validate against the maintainer's OPX1000. Every PR records `offline`,
`hardware <chip> <date>`, or `unverified`.

## Branch and PR

1. `feature/<slug>`, never `main`.
2. **Same branch name in every repo you touch.**
3. Drivers merge **last**: `scqat → SCQO → drivers`.

Full detail: [SCQO's CONTRIBUTING.md](https://github.com/shiau109/SCQO/blob/main/CONTRIBUTING.md).

## Do not

- **Do not edit vendored official files** (`calibrations/` non-graph files,
  `calibration_utils/`). Change behavior in `scqo_qm/`, or update upstream and re-sync.
- **Never touch the frozen archive** (`customized/`, `calibrations/exclude/`). It is
  packaged so `exclude/` stays importable; that is not permission to edit it.
- Editable code lives in `scqo_qm/`, `quam_config/`, `scripts/`,
  `calibrations/offline_graph/`. Everything else is vendored, generated, or frozen.
- Do not rename or move the lab QUAM classes in `quam_builder/` or `components/` without
  a migration. They are persisted **by dotted path** in every `state.json`, and QUAM
  falls back to base classes silently on a missing import — never trust absence of error.
