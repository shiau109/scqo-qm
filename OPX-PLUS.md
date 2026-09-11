# Running scqo on an OPX+ with an Octave

This driver serves both Quantum Machines RF chains through one backend name
(`qm`). Nothing in `cooldowns.toml` changes: a setup still declares
`backend = "qm"`, and the driver reads which hardware it is looking at out of the
QUAM tree.

## The axis is not "OPX+ vs OPX1000"

Two things vary independently, and an OPX1000 can drive an Octave
(`quam_config/wiring_examples/wiring_lffem_octave.py`), so the chassis is the
wrong thing to reason about:

| | what varies | what it decides |
|---|---|---|
| **RF chain** | `MWChannel` + MW-FEM port, vs `IQChannel` + `OctaveUpConverter` | absolute power, where the LO lives, whether there is a mixer to calibrate, `band` |
| **Baseband flux port** | `LFFEMAnalogOutputPort` vs `OPXPlusAnalogOutputPort` | the DAC rail, which predistortion field exists |

`scqo state --fields` shows the knobs of the chain you actually have, and hides
the ones you do not. Every run record stamps `readout_rf_chain` /
`drive_rf_chain`, so a later reader can tell which instrument produced it.

## What one OPX+ and one Octave can hold

Verified by running the vendor's own allocator, not estimated:

| topology | ceiling | what runs out |
|---|---|---|
| shared readout + xy + z | **2 qubits** | 10 analog outputs (`2 + 3N <= 10`) |
| shared readout + xy only (fixed frequency) | **4 qubits** | analog outputs and the Octave's 5 RF outputs, together |
| a second readout feedline | **impossible** | both analog inputs go to the first one |

The Octave also **shares LO synthesizers between RF outputs** — synth1: RF1 +
RFin1, synth2: RF2 + RF3, synth3: RF4 + RF5. Two drive lines on one synth are
forced to the same LO. This has no MW-FEM analogue (an MW-FEM pairs ports only
for their `band`, and each keeps its own `upconverter_frequency`), so plan the
wiring around it: `scripts/make_opxp_fixture.py` puts its two drives on RF2 and
RF4 for exactly this reason.

A startup audit refuses a tree that asks one synthesizer for two LOs.

## Building the QUAM tree

```bash
python scripts/make_opxp_fixture.py --out <setup>/backend_config
```

That script is the headless, reproducible form of
`quam_config/wiring_examples/wiring_opxp_octave.py` +
`quam_config/populate_quam_opxp_octave.py` (both of which block on `input()` and
a matplotlib window). Edit the frequencies and powers at the top of it, or use
it as the template for your own wiring.

Three things it sets that a hand-assembled tree usually gets wrong, each of
which the startup audits now catch:

- **`output_mode`**. `OctaveUpConverter` defaults to `always_off`. A tree that
  never set it emits **nothing** — the run completes and returns a clean flat
  line, with no error anywhere.
- **the down-converter LO**. Unset, quam omits the whole `RF_inputs` entry from
  the generated config: no receive path, no error.
- **`calibration_db_path`**. Unset, quam falls back to `os.getcwd()`, so your
  mixer calibration follows whatever directory you launched from and a run
  started elsewhere silently finds none. Point it at the setup's own
  `backend_config/`.

Then register the setup as usual (`SCQO/INSTALL.md` §2):

```toml
[cd1.setup.qm_opxp]
backend = "qm"
note = "OPX+ + Octave, 2 flux-tunable qubits"
```

## Hardware validation walkthrough

Record the run id at each step; `scqo restore <run_id> --setup <name>` rebuilds
any of them.

**0. Open a session.** It will refuse loudly if the tree asks for an LO the
synthesizer cannot produce, an IF outside ±400 MHz, an RF switch that is off, or
an `RF_frequency` stored as a QUAM reference. Fix what it names before going on —
each of those is silent on hardware.

**1. Calibrate the mixers. Do this first; everything after depends on it.**

```bash
python -m scqo_qm.backend.calibrate_octave --dry-run
python -m scqo_qm.backend.calibrate_octave
```

An Octave up-converts with an analog mixer, so every output carries LO leakage
and an image sideband until this runs. There is no MW-FEM equivalent, which is
why it is easy to forget.

**2. Readout chain.** `resonator_spectroscopy` → `readout_frequency` →
`readout_power`.

**3. Drive chain.** `qubit_spectroscopy` → `qubit_power_rabi` → `qubit_ramsey` →
`qubit_relaxation`.

**4. Discrimination.** `single_shot_readout`.

**5. Flux, if there are z lines.** `resonator_spectroscopy_flux` →
`qubit_spectroscopy_flux_pulse`. Note the OPX+ rail is a **fixed ±0.5 V** with no
`amplified` mode — five times less than an LF-FEM's 2.5 V — so flux amplitudes
from an OPX1000 setup do not transfer.

**6. After any `*_power_dbm` write, check whether the gain moved.** The driver
holds the Octave gain and spends the amplitude wherever it can, precisely because
the gain is part of the mixer calibration's cache key. When it cannot, it warns
by name — and then step 1 has to be repeated for that RF output. On a multiplexed
feedline the gain is shared, so the move landed on every qubit on that line.

## Known gaps

- **Flux predistortion is refused, not implemented.** `apply_distortion` writes
  an LF-FEM `exponential_filter`; an OPX+ analog output has no such field and
  uses `feedforward_filter` (FIR) + `feedback_filter` (IIR) instead — different
  arithmetic, not a renamed field. The command refuses by name and is hidden from
  `scqo state --fields` on an OPX+-only flux tree. The cryoscopes still measure
  the distortion; only the writeback is missing.
- **`broadband_resonator_spectroscopy` does not track a shared synthesizer.** It
  steps only the readout LO, so if the readout sits on an RF output that shares a
  synth with a drive line, that drive line is left at the last segment's LO. It
  warns by name. A normal readout sits on RF1, which has synth1 to itself.
- **External analog mixers** (`wiring_opxp_external_mixers.py`) are detected and
  refused by name for absolute power. Their power lives on the LO source, outside
  anything this driver reaches.
