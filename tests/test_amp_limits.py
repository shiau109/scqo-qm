"""What QUA's dynamic ``amp()`` can express, for the DRIVE/READOUT sweeps.

The flux probes have guarded this bound for a while; the three amplitude sweeps
(power Rabi, pi-pulse error amplification, deterministic benchmarking) did not,
so an over-range window reached the QOP compiler — after instrument time was
already booked, with the simulator showing nothing.
"""

import numpy as np
import pytest

from scqo_qm.experiments._amp_limits import MAX_AMP_SCALE, check_amp_scale_window


def test_a_window_inside_the_qua_range_passes():
    check_amp_scale_window(np.linspace(0.0, 1.9, 21), name="q1")


def test_the_bound_is_exclusive_because_2_is_unrepresentable():
    """QUA's fixed-point multiplier spans [-2, 2 - 2**-16], so 2.0 itself has no
    representation — the check is >=, not >. scqo's default top factor is 1.9
    for exactly this reason."""
    with pytest.raises(ValueError):
        check_amp_scale_window([2.0], name="q1")
    check_amp_scale_window([2.0 - 1e-3], name="q1")


def test_negative_factors_are_bounded_by_magnitude():
    with pytest.raises(ValueError):
        check_amp_scale_window([-2.5], name="q1")


def test_the_message_names_the_target_and_the_knob():
    with pytest.raises(ValueError) as err:
        check_amp_scale_window([0.5, 3.0], name="q1, q2")
    message = str(err.value)
    assert "q1, q2" in message
    assert "start_amp_factor/end_amp_factor" in message
    assert str(MAX_AMP_SCALE) in message


def test_an_empty_sweep_is_not_an_error():
    check_amp_scale_window([], name="q1")


def test_only_one_module_defines_the_bound():
    """One vendor fact, one home.

    Five modules used to carry their own ``_MAX_AMP_SCALE = 2.0``. A per-probe
    copy is exactly the kind of constant that drifts from the vendor while every
    test still passes, which is why ``_flux_limits`` already forbids per-probe
    DAC-rail constants — same rule, applied to ``amp()``'s bound.

    Static (AST) rather than import-based: a copy would be caught even in a probe
    no test imports. Scans the fused scqo experiments
    dir, where every experiment lives.
    """
    import ast
    import pathlib

    pkg = pathlib.Path(__file__).resolve().parents[1] / "scqo_qm"
    offenders = []
    for folder in (pkg / "experiments",):
        for path in sorted(folder.glob("*.py")):
            if path.name == "_amp_limits.py":
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not isinstance(node, ast.Assign):
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.lstrip("_") == "MAX_AMP_SCALE":
                        offenders.append(f"{path.relative_to(pkg)}:{node.lineno}")
    assert offenders == [], (
        "these define their own copy of the QUA amplitude_scale bound; import "
        f"MAX_AMP_SCALE from scqo_qm.experiments._amp_limits instead: {offenders}"
    )
