"""Every QUAM load and save goes through ``scqo_qm.quam_io`` — an AST scan.

Two failures this catches, both silent:

* a bare ``Quam.load()`` resolves through ``QUAM_STATE_PATH`` and then qualibrate's
  ``[quam] state_path``, so it reads whichever tree those name. On a machine whose
  repos have moved that is a folder nobody else loads, and the script says nothing.
* a ``save()`` that does not state ``include_defaults`` makes QUAM read
  ``~/.qualibrate/config.toml`` and raise when it is absent — after the values are
  already on the instrument (issue #38).

Both are shaped like a working call, so only a scan finds them. The precedent is
``tests/test_amp_limits.py``'s one-home scan.

A QUA stream save (``I_st.save("I1")``) always names its tag, and that positional
argument is what tells the two apart — QUAM's save takes none.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: Scanned trees. ``_vendored/`` is upstream code, kept byte-identical on purpose.
SCANNED = ("scqo_qm", "quam_config", "scripts", "tests")

#: The one module allowed to call QUAM's own load/save: it IS the door.
DOOR = "scqo_qm/quam_io.py"

#: Build scripts, allowed a bare load/save because quam_builder itself saves the tree
#: it is building and takes no path — each sets QUAM_STATE_PATH to a folder it owns
#: first, which is the only reason that env var is still written anywhere.
BUILD_SCRIPTS = {
    "quam_config/generate_quam.py":
        "sets QUAM_STATE_PATH to the folder it creates, then build_quam_wiring / "
        "build_quam save through it",
    "scripts/make_opxp_fixture.py":
        "the same shape, for the Octave tree the tests build",
}

#: Wiring templates: the build shape above, copied per instrument layout.
EXAMPLE_DIR = "quam_config/wiring_examples"

#: Receivers whose ``.save()`` is not a QUAM tree save. Deliberately a CENSUS: a
#: no-argument ``.save()`` on any other name fails the scan, so a new QUAM saver
#: cannot slip in under an unfamiliar variable name - the author either routes it
#: through the door or names it here with its reason.
NON_QUAM_SAVERS = {
    "device": "scqo's DeviceModel.save() - the device surface, which itself calls the door",
}


def _relpath(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def _python_files():
    for tree in SCANNED:
        for path in sorted((REPO / tree).rglob("*.py")):
            rel = _relpath(path)
            if "_vendored/" in rel or rel == DOOR or rel in BUILD_SCRIPTS:
                continue
            if rel.startswith(EXAMPLE_DIR + "/"):
                continue
            yield rel, path


def _calls(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


def _attr_name(node: ast.Call) -> str | None:
    return node.func.attr if isinstance(node.func, ast.Attribute) else None


def _receiver(node: ast.Call) -> str:
    value = node.func.value
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute):
        return value.attr
    return ""


def test_no_bare_quam_load_outside_the_door_and_the_build_scripts():
    offenders = [
        f"{rel}:{node.lineno}"
        for rel, path in _python_files()
        for node in _calls(path)
        if _attr_name(node) == "load" and _receiver(node).endswith("Quam")
        and not node.args and not node.keywords
    ]
    assert not offenders, (
        "bare Quam.load() reads whichever tree QUAM_STATE_PATH or qualibrate's "
        "state_path names; load the setup's folder through scqo_qm.quam_io.load_state: "
        + ", ".join(offenders))


def test_every_quam_save_states_include_defaults():
    offenders = [
        f"{rel}:{node.lineno}"
        for rel, path in _python_files()
        for node in _calls(path)
        if _attr_name(node) == "save" and not node.args
        and _receiver(node) not in NON_QUAM_SAVERS
        and not any(kw.arg == "include_defaults" for kw in node.keywords)
    ]
    assert not offenders, (
        "a QUAM save that does not state include_defaults reads ~/.qualibrate and "
        "raises without it (issue #38); save through scqo_qm.quam_io.save_state: "
        + ", ".join(offenders))


def test_the_door_and_the_exceptions_are_real_files():
    assert (REPO / DOOR).is_file()
    for rel in BUILD_SCRIPTS:
        assert (REPO / rel).is_file(), rel
    assert (REPO / EXAMPLE_DIR).is_dir()


@pytest.mark.parametrize("snippet, offends", [
    ("machine.save()", True),
    ("machine.save(path=folder)", True),         # the path is not the missing half
    ("tree.save()", True),                       # an unfamiliar name is NOT excused
    ("I_st.save('I1')", False),                  # a QUA stream tag, not a QUAM save
    ("machine.save(path=f, include_defaults=True)", False),
    ("backend.device.save()", False),            # the census entry
])
def test_the_save_rule_tells_a_quam_save_from_the_others(snippet, offends):
    """The positional tag is the whole difference from a QUA stream save; pin it, so a
    future edit cannot quietly widen the rule into every stream save in every probe,
    nor narrow it into "only variables called machine"."""
    node = ast.parse(snippet).body[0].value
    flagged = (_attr_name(node) == "save" and not node.args
               and _receiver(node) not in NON_QUAM_SAVERS
               and not any(kw.arg == "include_defaults" for kw in node.keywords))
    assert flagged is offends
