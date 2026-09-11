"""Build the OPX+ / Octave QUAM fixture tree, headlessly and reproducibly.

``quam_state_opxp/`` is this repo's only Octave tree. Every other state on disk --
``quam_state/``, ``quam_state_6q/``, ``state_lib/*`` and both live device configs
-- is MW-FEM, so without it the whole Octave half of the driver (the power solve,
the frequency audits, the fieldmap split, the calibration command, the broadband
LO snapping) could only ever be exercised against hand-built stubs.

It is generated rather than hand-written on purpose: a hand-written tree would
drift from what ``quam_builder`` actually produces, and the shapes that matter
here are exactly the ones the builder decides -- ``RF_frequency`` stored as a
literal with the IF as the reference, ``output_mode`` set away from its
``always_off`` default, the down-converter LO referencing the up-converter's.

The two interactive scripts it is derived from cannot be used directly:
``quam_config/wiring_examples/wiring_opxp_octave.py`` calls ``plt.show(block=True)``
and ``input()``, and ``quam_config/populate_quam_opxp_octave.py`` writes into
whatever ``QUAM_STATE_PATH`` happens to be. Their CONTENT is mirrored here; when
either changes, re-run this.

Usage::

    python scripts/make_opxp_fixture.py                  # -> quam_state_opxp/
    python scripts/make_opxp_fixture.py --out some/dir

Wiring: 1 OPX+ + 1 Octave, two flux-tunable qubits sharing one multiplexed
readout line. That is the real ceiling for this pairing -- 2 + 3N <= 10 analog
outputs -- so the fixture is also a standing statement of what fits.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

#: Readout: one multiplexed feedline through Octave RF1 / RFin1 (synth1, which it
#: has to itself). Drive: q1 on RF2, q2 on RF4 -- deliberately NOT RF2 + RF3,
#: which share synth2 and would be forced to one LO.
RR_LO_HZ = 6.0e9
RR_FREQS_HZ = (6.05e9, 6.12e9)
XY_LO_HZ = (5.0e9, 5.25e9)
XY_FREQS_HZ = (5.1e9, 5.3e9)
ANHARMONICITY_HZ = 200e6

READOUT_POWER_DBM = -40.0
DRIVE_POWER_DBM = -10.0


def octave_gain_and_amplitude(desired_dbm: float, max_amplitude: float = 0.125):
    """The lab's own chain solve, from ``populate_quam_opxp_octave.py``."""
    from qualang_tools.units import unit

    u = unit(coerce_to_integer=True)
    resulting = desired_dbm - u.volts2dBm(max_amplitude)
    if resulting < 0:
        gain = round(max(resulting + 0.5, -20) * 2) / 2
    else:
        gain = round(min(resulting + 0.5, 20) * 2) / 2
    amplitude = u.dBm2volts(desired_dbm - gain)
    if not (-20 <= gain <= 20 and -0.5 <= amplitude < 0.5):
        raise ValueError(f"power {desired_dbm} dBm is outside the Octave chain "
                         f"(got gain {gain} dB, amplitude {amplitude} V)")
    return gain, amplitude


def build(out_dir: Path) -> None:
    from qualang_tools.wirer import Connectivity, Instruments, allocate_wiring
    from qualang_tools.wirer.wirer.channel_specs import (
        octave_spec,
        opx_iq_octave_spec,
        opx_spec,
    )
    from quam_builder.builder.qop_connectivity import build_quam_wiring
    from quam_builder.builder.superconducting import build_quam

    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["QUAM_STATE_PATH"] = str(out_dir)

    instruments = Instruments()
    instruments.add_opx_plus(controllers=[1])
    instruments.add_octave(indices=1)

    qubits = [1, 2]
    connectivity = Connectivity()
    connectivity.add_resonator_line(
        qubits=qubits, constraints=octave_spec(index=1, rf_out=1, rf_in=1))
    connectivity.add_qubit_drive_lines(
        qubits=qubits[0],
        constraints=opx_iq_octave_spec(con=1, out_port_i=3, out_port_q=4, rf_out=2))
    connectivity.add_qubit_drive_lines(
        qubits=qubits[1],
        constraints=opx_iq_octave_spec(con=1, out_port_i=7, out_port_q=8, rf_out=4))
    connectivity.add_qubit_flux_lines(qubits=qubits[0],
                                      constraints=opx_spec(con=1, out_port=5))
    connectivity.add_qubit_flux_lines(qubits=qubits[1],
                                      constraints=opx_spec(con=1, out_port=9))
    allocate_wiring(connectivity, instruments)

    from quam_config import Quam

    machine = Quam()
    build_quam_wiring(connectivity, "127.0.0.1", "fixture_cluster", machine)
    machine = Quam.load()
    # The calibration DB belongs beside the tree it calibrates, not in whatever
    # directory a process happened to start in (quam's own fallback).
    build_quam(machine, str(out_dir))

    populate(machine)
    machine.save()


def populate(machine) -> None:
    """Mirror of ``quam_config/populate_quam_opxp_octave.py`` for this wiring."""
    from quam_builder.builder.superconducting.add_default_pulses import (
        add_DragCosine_pulses,
    )

    rr_gain, rr_amp = octave_gain_and_amplitude(
        READOUT_POWER_DBM, max_amplitude=0.125 / len(machine.qubits))
    xy_gain, xy_amp = octave_gain_and_amplitude(DRIVE_POWER_DBM)

    for index, (name, qubit) in enumerate(machine.qubits.items()):
        qubit.resonator.f_01 = RR_FREQS_HZ[index]
        qubit.resonator.RF_frequency = RR_FREQS_HZ[index]
        qubit.resonator.frequency_converter_up.LO_frequency = RR_LO_HZ
        qubit.resonator.frequency_converter_up.gain = rr_gain
        qubit.resonator.frequency_converter_up.output_mode = "always_on"
        qubit.resonator.operations["readout"].length = 2000
        qubit.resonator.operations["readout"].amplitude = rr_amp
        qubit.resonator.depletion_time = 1000

        qubit.f_01 = XY_FREQS_HZ[index]
        qubit.xy.RF_frequency = XY_FREQS_HZ[index]
        qubit.xy.frequency_converter_up.LO_frequency = XY_LO_HZ[index]
        qubit.xy.frequency_converter_up.gain = xy_gain
        qubit.xy.frequency_converter_up.output_mode = "always_on"
        qubit.xy.operations["saturation"].length = 20_000
        qubit.xy.operations["saturation"].amplitude = 0.1
        qubit.grid_location = f"{index},0"

        # quam_builder leaves a fresh flux line at 'independent', but every scqo
        # probe biases at 'joint' (joint_offset) -- so a tree straight from the
        # builder trips flux_point_problems at session start. The live lab trees
        # all carry 'joint'; a fixture that does not would be testing against a
        # config no session could open. Nothing Octave-specific: a freshly built
        # LF-FEM tree lands here too.
        qubit.z.flux_point = "joint"

        add_DragCosine_pulses(qubit, amplitude=xy_amp, length=40,
                              anharmonicity=ANHARMONICITY_HZ, alpha=0.0, detuning=0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="quam_state_opxp",
                        help="destination folder (default: quam_state_opxp)")
    parser.add_argument("--force", action="store_true",
                        help="replace the folder if it already exists")
    args = parser.parse_args(argv)

    out_dir = Path(args.out).resolve()
    if out_dir.exists():
        if not args.force:
            print(f"{out_dir} already exists; pass --force to rebuild it")
            return 2
        shutil.rmtree(out_dir)

    build(out_dir)

    from quam_config import Quam

    machine = Quam.load(str(out_dir))
    config = machine.generate_config()
    print(f"built {out_dir}")
    print(f"  qubits   : {list(machine.qubits)}")
    print(f"  octaves  : {list(machine.octaves)}")
    print(f"  elements : {len(config.get('elements', {}))}")
    print(f"  octave cfg: {json.dumps(list(config.get('octaves', {})))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
