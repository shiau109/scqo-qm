"""Register `stark_detuned`: the `stark` tone with its detuning baked into the waveform.

For every qubit whose xy line carries a `stark` operation (a GaussianPulse), adds
`q.xy.operations["stark_detuned"]`: a `DragGaussianPulse` with the SAME length,
sigma, amplitude, axis angle and subtraction, alpha = 0 (no DRAG term) and
`detuning` = the stark detuning. QUAM modulates the envelope by
exp(+i 2 pi detuning t), which moves the tone up by `detuning` exactly as
`update_frequency(base_IF + detuning)` does - so it is the same physical tone as
qc_n_stark_amp's `stark` played at the stark IF, without a frequency switch.

Why it exists: qc_n_swap_tomography reads the coherence between the two pair
members, so it re-references both drive frames (and the global phase) every shot,
and an `update_frequency` round trip does not mix with that reset (SCQO BACKLOG
I35). Re-run this after recalibrating `stark` (it copies the current envelope).

    python quam_config/register_stark_detuned.py <state folder> [--detuning-hz 50e6]

After running, per the repo CLAUDE.md: re-check the pulse `__class__` path survived
the save, and gate the config offline with `machine.generate_config()`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

SOURCE_OP = "stark"
OP = "stark_detuned"


def register_stark_detuned(machine, detuning_hz: float, *, source_op: str = SOURCE_OP,
                           op: str = OP) -> list[str]:
    """Add ``op`` beside every ``source_op`` GaussianPulse; returns the qubits done.

    Refuses a ``source_op`` that is not a Gaussian (its envelope would not carry over)."""
    from quam.components.pulses import DragGaussianPulse, GaussianPulse

    done = []
    for q in machine.qubits.values():
        if q.xy is None or source_op not in q.xy.operations:
            continue
        src = q.xy.operations[source_op]
        if not isinstance(src, GaussianPulse):
            raise ValueError(f"{q.name}.xy {source_op!r} is a {type(src).__name__}, not a "
                             f"GaussianPulse; nothing to copy the envelope from")
        q.xy.operations[op] = DragGaussianPulse(
            length=src.length, amplitude=src.amplitude, sigma=src.sigma,
            axis_angle=src.axis_angle if src.axis_angle is not None else 0.0,
            subtracted=src.subtracted, alpha=0.0, anharmonicity=-200e6,
            detuning=float(detuning_hz))
        done.append(q.name)
    return done


def main(argv: list[str] | None = None) -> None:
    from scqo_qm.quam_io import load_state, save_state

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("state_path", help="the QUAM state folder to edit (a setup's backend_config/)")
    parser.add_argument("--detuning-hz", type=float, default=50e6,
                        help="the stark detuning baked into the waveform (default 50e6)")
    args = parser.parse_args(argv)
    folder = Path(args.state_path).expanduser().resolve()
    if not (folder / "state.json").is_file():
        raise SystemExit(f"no state.json in {folder}")
    machine = load_state(str(folder))
    done = register_stark_detuned(machine, args.detuning_hz)
    if not done:
        raise SystemExit(f"no qubit carries a {SOURCE_OP!r} xy operation; nothing registered")
    machine.generate_config()        # refuse to save a tree QM would reject
    save_state(machine, str(folder))
    print(f"Saved. {OP!r} (detuning {args.detuning_hz:.0f} Hz) registered on {done}.")


if __name__ == "__main__":
    main()
