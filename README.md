# scqo-qm

The Quantum Machines backend for the vendor-neutral `scqo` experiment API
(`scqo_qm/` — one fused file per experiment). It serves BOTH QM RF chains through one
backend name: MW-FEM (OPX1000) and Octave (OPX+, or an OPX1000 with LF-FEMs). Built on
the qm-qua → QUAM stack.

Every experiment runs through `scqo run`, and the vendor tools through `scqo-qm
<command>`. There is no GUI: the vendored `qua-libs` qualibrate nodes this repo used to
carry were removed after v3.13.0, which is the last release that has them.

See [CLAUDE.md](CLAUDE.md) for the full architecture, conventions, and operating rules.
