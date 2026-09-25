"""How scqo-qm writes a QUAM tree back to disk — the one home for the save convention.

QUAM leaves two things about a save to its configuration file,
``~/.qualibrate/config.toml``: WHERE a bare ``machine.save()`` writes (``[quam]
state_path``, after the ``QUAM_STATE_PATH`` env var), and whether fields at their
default value are written (``[quam.serialization] include_defaults``). The second is
looked up on EVERY save that does not pass it, before the path is even considered,
and a missing file raises ``FileNotFoundError`` — on a machine that has never run
qualibrate, ``scqo set`` pushed the new value to the instrument and then failed to
persist it (issue #38). Nothing here consults that file.

``INCLUDE_DEFAULTS`` is ``True`` because that is what the qualibrate stack writes (QUAM's
config default and its own fallback), so a state saved by scqo and one saved by a
qualibrate node stay byte-comparable. The lab QUAM root pins it on its serialiser too
(``MixedTransmonQuam.get_serialiser``), which covers the bare saves inside
quam_builder's own builders that no scqo call site can pass it to.

Kept free of any scqo_qm import: the QUAM root class imports it.
"""

from __future__ import annotations

from typing import Any

#: Every scqo-side QUAM save passes this explicitly (see the module docstring).
INCLUDE_DEFAULTS = True


def load_state(path: str) -> Any:
    """Load the QUAM tree in ``path`` — the folder, always named.

    ``Quam.load()`` with no argument resolves through ``QUAM_STATE_PATH`` and then
    qualibrate's ``[quam] state_path``, so a bare load reads whichever tree those
    happen to name — on a machine whose repos have moved, one nobody else loads. The
    folder a session works on is a fact of its setup, so it is passed, never resolved.
    """
    from quam_config import Quam  # lazy: keep this module import-light

    return Quam.load(str(path))


def save_state(machine: Any, path: str | None = None) -> None:
    """Save ``machine`` to ``path`` without consulting any qualibrate configuration.

    ``path`` is the folder the tree was loaded from; pass it whenever it is known.
    ``QuamRoot.load(path)`` does not remember where it read from, so without one a
    save goes wherever ``QUAM_STATE_PATH`` (or qualibrate's ``state_path``) points at
    the time — not necessarily the setup this session loaded.
    """
    machine.save(path=path, include_defaults=INCLUDE_DEFAULTS)
