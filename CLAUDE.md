# scqo-qm — project guide

## Project Overview
Two products in one repo (renamed from LCHQMDriver in the v1 restructure):
1. **`scqo_qm/`** — the Quantum Machines OPX1000 backend for **`scqo`**, the vendor-neutral
   experiment API shared with the Qblox driver ([scqo-qblox](https://github.com/shiau109/scqo-qblox)), so the same experiment
   runs on either instrument through one `Session`. scqo is a HARD dependency.
2. **Vendored official qualibrate calibrations** (`calibrations/` + `calibration_utils/`, copied in
   by `sync_official.py`) — the qualibrate GUI path, official nodes only. The custom LCH_* qualibrate
   shells were RETIRED in the v1 restructure; `calibrations/exclude/` + its `customized/node/`
   packages are a frozen archive (never edit, never import from live code, not runnable).

Vendor stack: **qm-qua** → **quam** → **qualibrate** (read-only in the workspace). Design, the
`Session` contract, and cross-repo terminology (Experiment = probe + estimator) live in
`SCQO\CLAUDE.md`; analysis runs through scqat **estimators** inherited via scqo.

## Layout
```
scqo_qm/
  scqo_backend.py        # the `scqo.backends` entry-point factory (name "qm"):
                         #   build_backend(cfg, setup, roster) fires the state_sync="pull"
                         #   guard BEFORE any QUAM state is touched, loads the setup's vendor
                         #   folder (canonical names state.json + wiring.json; loud SystemExit
                         #   when missing), audits flux points/headroom, threads the ROSTER
  backend/
    qm_backend.py        # QMBackend (scqo.Backend) + QMDeviceModel + ONE view class per
                         #   CHANNEL KIND (QMDriveChannel/QMReadoutChannel/QMFluxChannel)
                         #   + QMQubitPair (composite view over the QUAM qubit_pair);
                         #   acquire()/preview() live here
    fieldmap.py          # declarative neutral->vendor field catalog (pure data, per channel kind)
    roster_gen.py        # roster_toml_for(machine): derive a schema-3 roster from a live QUAM
                         #   tree (test fixtures + scripts/check_real_config.py; the REAL roster
                         #   is <data_root>/<device>/components.toml)
    _distortion.py       # flux-distortion facts -> exponential-filter arithmetic (pure)
    apply_distortion.py  # operator CLI: python -m scqo_qm.backend.apply_distortion
                         #   (QMBackend.distortion_apply_command hands scqo's two
                         #   cryoscopes this command line as their writeback hint)
    close_qm.py          # operator CLI: python -m scqo_qm.backend.close_qm - halt
                         #   jobs + close open QMs when a dead session still holds
                         #   the cluster's locks (QMBackend.close_qm does the work)
  experiments/
    __init__.py          # one import line per experiment module so @register runs (manual;
                         #   tests/test_experiment_registration.py enforces completeness both
                         #   directions) + the qm-logging stderr re-home (see Operational Notes)
    <name>.py            # ONE FILE PER EXPERIMENT (one per registered scqo experiment): merged physics docstring, a
                         #   module-level build_program(...) (+ own acquire() where the streams
                         #   are heterogeneous), and the registered QM<Name>(<Name>) class whose
                         #   probe() maps self.params -> the local builder
    _lib.py              # select_qubits/select_qubit_pairs + the shared execute-and-fetch acquire
    _flux_limits.py      # the flux rail per PORT, two frames (absolute/relative), idle sums
    _amp_limits.py       # the QUA amplitude_scale bound (one home; AST-scan enforced)
    _reset.py            # the active-reset door: check_reset_method/reset_max_attempts
    _vendor.py           # the one door out of the neutral surface (raw QUAM element by ROSTER)
    _pair_roles.py       # joint-population digit reordering for the pair maps
    _qc_populations.py   # shared swap-reset population math
    _readout_fidelity.py # shared SSRO builder (single_shot_readout / _gef / thermal_population)
    _resonator_spectroscopy.py  # shared 1D builder (resonator_spectroscopy / _power_chain)
  quam_fields.py         # the single neutral-field <-> QUAM mapping + whole-tree audits
  components/            # lab pulse shapes + macros (FlatTopCosinePulse, ISwapImplementation,
                         #   ParametricReset - PERSISTED as __class__ in state.json: moving or
                         #   renaming them requires scripts/migrate_state_scqo_qm.py-style care)
  quam_builder/          # lab QUAM classes (MixedTransmonQuam root, Thermalizing* transmons -
                         #   ALSO persisted as __class__ in state.json)
quam_config/             # QUAM class entrypoint (my_quam.py: Quam(MixedTransmonQuam)) + the
                         #   register_* scripts that materialize lab operations into a state
quam_state/              # serialized instrument config (state.json/wiring.json; gitignored)
calibrations/            # OFFICIAL vendored nodes only + exclude/ (frozen archive)
                         #   + offline_graph/ (manual LCH_graph_* post-processing scripts)
calibration_utils/       # vendored official support code (regenerate via sync_official.py)
customized/              # FROZEN RUMP of the retired qualibrate era: node/ packages the
                         #   exclude/ archive imports + common_parameters + read_data.
                         #   Never edit; never import from live code.
scripts/                 # check_real_config.py, migrate_state_scqo_qm.py (the package-rename
                         #   state migration), migrate_root_class.py, migrate_thermalizing_*.py
```

## Adding an experiment
1. Subclass the backend-free experiment from `scqo.experiments.<name>` in ONE file
   `scqo_qm/experiments/<name>.py`. Write the QUA program in a module-level
   `build_program(machine, qubits, *, ...)` and keep the class's `probe()` a thin
   params→builder mapping (this file-local builder is the QM house style — QUA programs are
   long and the builder seam is what the live-state tests and preview exercise). An
   experiment whose streams `XarrayDataFetcher` cannot fetch also defines a module-level
   `acquire()` and returns the 3-tuple `(prog, sweep_axes, acquire)` — the CALLABLE.
2. `@register` the class and add its import line in `scqo_qm/experiments/__init__.py`
   (manual — `tests/test_experiment_registration.py` refuses a module missing its line).
3. Read device state through the CHANNEL views (`self.device.channel(target, "readout")...`);
   vendor-only bits come from `_vendor.vendor_element(...)`. Shared guards: `_flux_limits`,
   `_amp_limits`, `_reset.check_reset_method`.
4. May import qm.qua (module-level star-import is the DSL's requirement), quam, qualang_tools,
   `qualibration_libs.core`/`.data`, scqo; NEVER qualibrate, never scqat.
5. Census literals that react to a new experiment: `tests/test_preview.py` `SELF_ACQUIRING`
   (probe_self_acquires shells), `tests/test_reset_method.py` `CARRIERS` (active-reset opt-in).
A qualibrate node is NOT part of adding an experiment (the GUI serves official nodes only).

## Physics + backend invariants (verified against the working tree)

**Virtual-detuning sign — a SILENT failure.** A probe realizing scqo's `frequency_detuning_hz`
must ramp the phase **negative on EVERY backend**: the second pi/2's phase has to run BACKWARD
relative to the free precession of a qubit sitting above its drive, so the observed fringe is
`applied + err` — scqo's shared `estimate()` writes `drive_freq += (f_fit − applied)` for both
backends. A frame rotation and a pulse-axis phase are the SAME handedness, so the negation is not
compensating a cross-vendor asymmetry — it is genuinely required on both. Get it backwards and
every accepted update DOUBLES the residual detuning while the fit still looks clean (chipA q1,
2026-07-28: +47.9k → +95.7k → +191.5k Hz). BOTH frame-ramping builders carry the negation:
`scqo_qm/experiments/qubit_ramsey.py` and `scqo_qm/experiments/pair_zz_coupler.py` (there the
zero crossing feeding the writeback is sign-invariant). The official nodes leave the ramp
un-negated — do not copy either sign into a builder.

### Flux headroom — `scqo_qm/experiments/_flux_limits.py`
**The DAC rail is a property of the PORT, not a constant** (±0.5 V `direct`, ±2.5 V `amplified`;
a stored waveform at/above full scale clips on hardware while the SIMULATOR SHOWS NOTHING). No
builder carries its own rail constant. Two frames, two entry points, mirroring scqo's parameters
1:1: `check_flux_bias_absolute` (set_dc_offset REPLACES the bias — no idle term) and
`check_flux_pulse_relative` (play rides ON the bias — check `|idle + excursion|`). Confusing them
is silent in both directions. Couplers name their points `off`/`on` but their attributes
`decouple_offset`/`interaction_offset`. Severity split (load-bearing): *clipping* → refuse;
*reach* (const = rail/2 convention) → advisory only, enforced ONLY in
`quam_fields.flux_headroom_warnings`. The whole-tree audits run once from `scqo_backend.py`.

**Flux-amplitude sweeps: absolute volts or prefactor.** Both pair swap experiments take
`amp_mode="absolute"|"prefactor"` (+ `flux_role`). scqo drives them `"absolute"` (the swept
values ARE the emitted volts). Prefactor mode needs `freq_vs_flux_01_quad_term` (7 of 9 live
chipA pairs have it unset — refused naming the field). The chevron's two QUA branches (baked
below 17 ns, stretched `const` above) must emit the same volts — `resolve_amplitudes` is pure and
pinned by `tests/test_pair_swap_probes.py`. The partial-swap workflow is SCQO TUTORIAL §12.

**Readout output at the scqo boundary** (the readout schema — SCQO TUTORIAL §11): shot axis
`shot_idx`; per-shot discriminated data stays `state` (integer LEVELS); FPGA-averaged
discriminated data is `population` (the backend renames when the contract accepts it). Pair maps
store `joint_population` over role-ordered `joint_state` labels (digits high,low) via
`_pair_roles.JointPopulationMixin`; `qc_n_swap_amp` reduces per-shot member states through scqo's
shared `states_to_joint_population` and refuses non-control `drive_side`/`flux_side` by name.

**T1-tracking experiments (`qubit_t1_ade` / `qubit_t1_bayesian`):** on-FPGA arithmetic
(`Math.div/ln/sqrt/exp` on `fixed` ∈ [-8, 8); numeric-range rationale in each module docstring).
Heterogeneous streams → each module defines its OWN `acquire()` and `probe()` returns
`(prog, sweep_axes, acquire)` — **the callable** (the backend unpacks it directly; same contract
as `qubit_tomography`). ms→cycles conversions MUST use `Cast.mul_int_by_fixed` (the fixed product
wraps modulo 16 — pinned by `tests/test_t1_tracking_shells.py`). Both refuse a missing readout
threshold, the Bayesian one a missing confusion matrix, BY NAME before any QUA is built.

**Active reset** (`reset_method="active"`) lives in `scqo_qm/experiments/_reset.py`, the ONE door
(`check_reset_method`), with `QMBackend.acquire` re-checking before `probe()`. Opt-in is per
experiment (`supports_active_reset`, default DENY), limited to the coherent-drive carriers
whose readout condition is fixed for the whole run; everything else refuses BY NAME.
`tests/test_reset_method.py`'s `CARRIERS` literal is the authority for WHICH - do not restate
the list here, and do not assume it matches Qblox's (it does not: Qblox denies
`qubit_ramsey_phasor` pending its hardware run). The
sequence is QUAM's `reset_qubit_active` (repeat-until-success). Four QM-specific rules, each a
SILENT failure if broken: (1) `active_reset_rounds` → QUAM `max_attempts` is an UPPER bound;
(2) BOTH `readout_threshold` AND `readout_rus_threshold` are required even at rounds=1, and an
uncalibrated value is `None` — the guard refuses first; (3) `readout_depletion_s` must be governed
(QUAM's 16 ns factory default is refused); (4) `thermalization_time_ns` + `active` is refused, not
ignored. Offline proves policy + program build; the feedback loop is hardware (chipA walkthrough
still owed).

**Sequence preview** (`scqo run <name> --preview` → `QMBackend.preview`): dumps
`generate_qua_script` to `qua_script.py` (offline), then AUTO-TRIES the gateway simulator →
`simulated_waveforms.html` (2 s TCP probe gates the attempt; any failure degrades to script-only
with a PreviewWarning). `--no-simulate` = guaranteed offline; `--simulate-ns` widens the 20 µs
default window. Some experiments acquire INSIDE `probe()` and are refused by name BEFORE it runs,
each declaring `probe_self_acquires = "<why>"` (default-ALLOW polarity — contrast
`supports_active_reset`). `tests/test_preview.py`'s `SELF_ACQUIRING` literal is the authority
and enforces the set both directions - read it there rather than trusting a list in prose,
which is exactly how this one went stale at 6 while the real set had grown to 9.

**Placement rule** (`scqo state --rule`; SCQO TUTORIAL §10, "Where does a value live?"): QUAM-tree copies of physics the tree
operationally CONSUMES (T1 for thermalization, anharmonicity for DRAG) are CACHES with scqo's
physical.json as truth; QUAM's stored measured artifacts (confusion_matrix, gate_fidelity, ...)
are dead to SCQO.

### State authority (`state_sync` rule)
scqo's `RecordingDevice` owns its own state JSON; the QUAM tree is the vendor store. The LCH
qualibrate writers are RETIRED, but official nodes run through the GUI can still write QUAM, so
**QM sessions keep `state_sync="pull"`** (the vendor wins at startup; scqo pushes only what it
freshly measures — `scqo_qm/scqo_backend.py` enforces this before any QUAM state is loaded).
Flipping a device to `"push"` is a per-device ops decision for when the GUI is retired from
calibration writing on that device. **Which QUAM state loads** is decided by the device's cooldown
setup alone (`<device>/<cycle>/<name>/backend_config/` holding canonical `state.json` +
`wiring.json`); keep qualibrate's own `[quam] state_path` pointed at the same folder on machines
running both stacks.

### scqo student surface
Students use the **`scqo` command** from any directory in `.venv-qm`
(`scqo user --device <name> [--setup <name>]`; `scqo run <name>` is the one way to run an
experiment — never add per-command wrappers). `simulated` is the practice mode. The qualibrate GUI
(`qm.bat`) serves the OFFICIAL vendored nodes only.

## Key Entrypoints
- `quam_config/my_quam.py` → `class Quam(MixedTransmonQuam)` — the QUAM class every path loads.
  The root class + the lab transmon/pulse/macro classes are PERSISTED by dotted path in
  state.json (`scqo_qm.*` since the v1 restructure): re-check `__class__` after any QUAM save,
  and gate config edits offline with `m.generate_config()` under `warnings.simplefilter("error")`.
- `scripts/migrate_state_scqo_qm.py` → the `customized.*` → `scqo_qm.*` state migration with
  positive verification (QUAM silently falls back to base classes on a missing import — never
  trust absence-of-error).

## Operational Notes (verified against the working tree)
- **Official code is VENDORED (copied), and committed.** `calibrations/<name>.py` +
  `calibration_utils/<name>/` come from `sync_official.py` (`calibration_links.toml`;
  `official_sync.json` records the vendored upstream commit). Do not edit vendored files in place.
  Updating (~every 2 months): pull `qua-libs_official` → `python sync_official.py` → review diff →
  commit. `customized/`'s frozen archive and `scqo_qm/` are never touched by the sync.
- **`calibrations/offline_graph/`** holds manual `LCH_graph_*.py` post-processing scripts
  (editable lab code; qualibrate does not list them).
- **Environments:** the scqo path runs in the shared `.venv-qm` (rebuildable from
  `requirements-qm.lock.txt`); siblings `.venv-view` (no instrument libs) and `.venv-qblox`.
  `qm.bat` activates `.venv-qm` and runs `qualibrate start` (GUI).
- **qm logging vs the CLI JSON contract:** fused experiment modules import `qm.qua` at module
  level (the DSL star-import cannot be function-local), and qm's import-time logger writes to
  STDOUT by default — `scqo_qm/experiments/__init__.py` flips qm's own
  `QM_DISABLE_STREAMOUTPUT` switch and re-homes the records on stderr BEFORE the first qm import.
  Machines without the QM stack: scqo's entry-point discovery skips this driver gracefully.
- **Packaging:** dist `scqo-qm`; wheel packages `calibrations`, `calibration_utils`,
  `quam_config`, `customized` (frozen archive travels for exclude/ importability), `scqo_qm`.
  Entry points: `scqo.experiments` → `scqo_qm.experiments`; `scqo.backends` →
  `qm = scqo_qm.scqo_backend:build_backend`. Entry points register at INSTALL time — re-run
  `uv pip install -e` after changing them. Python `>=3.10,<3.13`, black `line-length = 120`.

## Tests

Run as **`.venv\Scripts\python.exe -m pytest tests\ -q`** (the repo venv is built from
`requirements-qm.lock.txt` + editable scqo/scqat; avoid bare `uv run` — its sync would rebuild
the env from pyproject, displacing the lockfile pin authority. `uv run --no-sync` is the
acceptable alternative). **The full suite IS the targeted run here** - it is small enough that a
selection map would cost more attention than it saves; run it before every commit. If the repo
venv is missing or stale, use the shared one: the v3.0.0 release notes record the repo venv
failing to collect for want of `typing_extensions`, and both recent cuts were validated with the
shared venv. No test count is quoted here on purpose - see the `OFFLINE-VALIDATED` line in the
matching RELEASES.toml block for what each release actually ran. Live-state tests load the repo-relative `quam_state/` (hermetic — no
`~/.qualibrate` dependency).

| File | Covers | Needs QM stack? |
|---|---|---|
| `test_quam_fields.py`, `test_flux_headroom_guard.py`, `test_flux_point_guard.py` | the neutral-field mapping + whole-tree audits | no |
| `test_flux_pulse_amplitude.py`, `test_amp_limits.py` | the two flux frames, rails, the one-home amplitude bound (AST scan over experiments/) | no |
| `test_reset_method.py` | the active-reset door: opt-in census, refusals, the acquire() backstop, the .reset-literal scan | no |
| `test_experiment_registration.py` | every experiment module has its __init__ import line (both directions) | no |
| `test_preview.py` | the probe_self_acquires census + preview refusal ordering | no |
| `test_qc_populations.py`, `test_pair_swap_probes.py`, `test_parity_switch_shell.py`, `test_t1_tracking_shells.py`, `test_ramsey_cryoscope_probe.py`, `test_spectroscopy_cryoscope_probe.py` | pure builder math, param mapping, AST properties of the fused modules | no |
| `test_mixed_quam.py`, `test_distortion.py`, `test_apply_distortion.py` | the lab QUAM root + distortion arithmetic | partly |
| `test_close_qm.py` | the best-effort cluster-cleanup hook + its operator CLI (doubles, no cluster) | yes |
| `test_experiment_surface.py` | `_vendor.py` — the one door out of the neutral surface | yes |
| `test_qm_backend.py` | entity surface on the stub; builder-vs-class mapping equivalence, baked-config self-acquisition, active-reset + tracker builds on the LIVE quam_state; preview | yes |
| `test_overlap_probe.py` | concurrent-tone timing asserted on generated QUA (quote-agnostic vs qm versions) | yes |
| `test_scqo_glue.py` | the `scqo` CLI works in THIS venv + the qm factory (slowest) | yes |
| `test_check_real_config.py` | `scripts/check_real_config.py` end-to-end to its PASS line on the live quam_state (subprocess, ~14 s — exit code + final line asserted, never a pipeline fragment) | yes |

## Workspace Packages (Read-Only)
The vendor stack (`qm` → `quam` → `quam_builder` → `qualibrate`) is available read-only; do NOT
The sibling repos are [SCQO](https://github.com/shiau109/SCQO) (the vendor-neutral core, a hard dependency resolved as `../SCQO`), [scqat](https://github.com/shiau109/scqat) (analysis) and [scqo-qblox](https://github.com/shiau109/scqo-qblox) (the Qblox backend - never import from it).

## Rules for the AI assistant

Rules 1–5 are about **what may be edited** and hold everywhere, forks included. Rule 6 is a
*lab-tree* rule: it exists because the maintainer's checkout is shared and live (editable
installs, sometimes a running hardware session). In your own fork it does not apply — work on a
feature branch and open a PR, as [AGENTS.md](AGENTS.md) describes.

1. **Do NOT edit vendored official files** (`calibrations/` non-graph files, `calibration_utils/`).
   Change behavior in `scqo_qm/` instead, or update upstream and re-sync.
2. **Editable code lives in:** `scqo_qm/`, `quam_config/`, `scripts/`,
   `calibrations/offline_graph/`. Everything else is vendored, generated, or frozen.
3. **Never touch the frozen archive** (`customized/`, `calibrations/exclude/`) — it exists for
   history, not for running. It *is* packaged (`pyproject.toml` ships it so `exclude/` stays
   importable), which is not permission to edit it.
4. **Skip `data/`** — data storage only, and gitignored, so a fresh clone has none.
5. **No qualibrate nodes.** New experiments are fused files in `scqo_qm/experiments/`
   (**Adding an experiment**); qualibrate scaffolding returns only on explicit request.
6. **In the maintainer's tree only:** present a plan and get the maintainer's approval before
   modifying code, call out any critical vendor dependency you add or change, and report
   working-tree/instruction conflicts before changing anything. Several agents share that tree
   at once, so a file someone else has modified is off limits.
