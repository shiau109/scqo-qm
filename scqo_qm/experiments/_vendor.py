"""Raw-vendor access for the QM probes — the one place that leaves the neutral surface.

Since the greenfield model a probe reads its NEUTRAL state through the channel
entity that owns the knob (``self.device.channel(target, "readout")``), which
serves the SCQO runtime value whether ``self.device`` is the Session's
RecordingDevice or a bare :class:`~scqo_qm.backend.qm_backend.QMDeviceModel`.

The QUAM tree itself is not resolved here — every probe takes the whole
``backend.machine`` plus a ``BatchableList`` of QUAM qubits
(``scqo_qm/experiments/_lib.select_qubits``), because a QUA program is built out of
the vendor's own pulse macros. What DOES need resolving is the handful of places
where scqo hands a ROSTER name and the probe wants a VENDOR one:

* a foreign flux source (``flux_component``) — scqo names a roster entity (a
  qubit mode, or a tracked coupler mode like ``q1_q2_c``), while the probes'
  ``z_source`` names a QUAM object (``machine.qubits[...]`` or
  ``machine.qubit_pairs[...].coupler``);
* a pair target — scqo names a composite (``q1_q2``), while QM names its pairs
  after the coupler (``coupler_q1_q2``), so the two genuinely differ.

Both joins go through the roster and the backend's own resolution, never through
string arithmetic on the entity name. Entity names are resolved the core
power_chain way: ask the roster for the target's DEFAULT channel of a kind, then
address the vendor tree by that name (``DeviceModel`` has no ``channel()``; only
``RecordingDevice`` does). Going through the channel KIND is also the capability
guard — an entity with no flux channel fails on a clear roster error here instead
of deep inside a QUA program.
"""

from __future__ import annotations

from typing import Any


def _key_of(mapping: Any, obj: Any, *, attr: str | None = None) -> list[str]:
    """The key(s) of a QUAM container whose entry IS ``obj`` (or whose entry's
    ``attr`` is). Identity, never name equality: a QUAM component's ``name`` is
    derived from its ``id`` and need not be the key its parent files it under."""
    return [key for key, value in (mapping or {}).items()
            if (getattr(value, attr, None) if attr else value) is obj]


def vendor_pair(experiment: Any, composite: str) -> Any:
    """The raw QUAM ``qubit_pair`` behind a roster COMPOSITE name.

    Resolved by the backend (by name first, else by the composite's high/low
    membership): QM names its pairs after the coupler — ``coupler_q1_q2`` — so
    a roster composite named ``q1_q2`` is NOT a key of ``machine.qubit_pairs``
    and must never be used as one. Not through ``component()``: since SCQO
    4.0.0 a composite carries facts only and has no view - its knobs live on
    its declared operations, and a pair probe needs the pair whether or not the
    roster declares any.
    """
    return experiment.backend.device.pair(composite)


def vendor_pair_name(experiment: Any, composite: str) -> str:
    """The KEY of ``machine.qubit_pairs`` for a roster composite — what the
    probe helpers (``_lib.select_qubit_pairs``) select pairs by."""
    pairs = getattr(experiment.backend.machine, "qubit_pairs", {}) or {}
    hits = _key_of(pairs, vendor_pair(experiment, composite))
    if len(hits) == 1:
        return hits[0]
    raise ValueError(
        f"{composite!r}: the QUAM qubit_pair it resolves to is not a unique "
        f"entry of machine.qubit_pairs (matches {sorted(hits)}; pairs: "
        f"{sorted(pairs)})")


def vendor_qubit(experiment: Any, mode: str, *, field: str,
                 kind: str = "drive") -> Any:
    """The raw QUAM qubit behind a roster qubit MODE name.

    The single-mode twin of :func:`vendor_pair`, for probes that address a qubit
    scqo named in a PARAMETER rather than in ``targets`` (a chain's prep or reset
    qubit): ``machine.qubits[mode]`` would be string arithmetic on an entity
    name, so the join goes through the roster's default channel of ``kind`` and
    that channel view's ``qubit`` — the same route :func:`flux_source_name`
    takes. Going through the channel KIND is also the capability guard: a mode
    with no channel of that kind fails here, on a clear roster error, instead of
    deep inside a QUA program.

    ``kind`` is the channel the probe will actually play on — ``"drive"`` for an
    xy pulse, ``"flux"`` for a z-line technique — so the refusal names the wire
    that is missing. ``field`` is the Parameters field the name came from and
    appears in every refusal.
    """
    roster = experiment.device.roster
    try:
        channel = roster.default_channel(mode, kind)
    except Exception as err:
        raise ValueError(
            f"{field}={mode!r}: no default {kind} channel in this roster "
            f"({err})") from err
    qubit = getattr(experiment.backend.device.component(channel), "qubit", None)
    if qubit is None:
        raise ValueError(
            f"{field}={mode!r}: its {kind} channel {channel!r} serves no QUAM "
            f"qubit (a tunable coupler has none) - name a qubit mode")
    return qubit


def vendor_borrowed(experiment: Any, address: str) -> Any:
    """The QUAM element that REALIZES a borrowed drive channel, by its roster
    ADDRESS (``xy2.q1_q2_c``) - resolved by the backend's device model, which
    files adopted elements under that address and refuses (KeyError, naming
    ``scqo-qm adopt-channel``) one the state has not adopted. The Session's gate
    has normally refused that case already, before any run folder exists."""
    view = experiment.backend.device.component(address)
    element = getattr(view, "vendor", None)
    if element is None:  # pragma: no cover - component() raises first
        raise ValueError(f"{address!r}: its view serves no QUAM element")
    return element


def role_member(roster: Any, pair: str, role: str) -> str:
    """The ONE mode filling a pair's ``high``/``low`` role (roster truth)."""
    members = roster.entities[pair].roles.get(role, ())
    if len(members) != 1:
        raise ValueError(
            f"{pair}: role {role!r} has {len(members)} member(s) "
            f"{list(members)} - the selected member must be exactly one")
    return members[0]


def role_side(experiment: Any, role: str, *, field: str,
              needs_flux: bool = False) -> str:
    """Map a neutral high/low ROLE onto the vendor pair's control/target.

    The roster's declared high/low ROLES are the governed truth and the only
    source: high/low is design-nominal topology, while a live-f_01 ordering
    legitimately crosses during tuning (greenfield schema section 4), so there
    is nothing to fall back to. ``targets`` are ROSTER composite names; the QUAM
    pair behind one is resolved by the backend (QM names its pairs after the
    coupler, so the roster name rarely matches), never by indexing
    ``machine.qubit_pairs`` with the roster name.

    ``field`` is the Parameters field the role came from - it appears in every
    refusal, so an error says WHICH knob to split the run on. The pair probes
    take ONE side for the whole program, so a role that maps onto DIFFERENT
    vendor sides across the selected pairs is refused.

    ``needs_flux=True`` also asserts the selected member owns a flux channel.
    That gate is params-AWARE, and ``Experiment.validate_targets`` is a
    classmethod over ``(roster, targets)`` with no view of params - so the
    roster hook can only ask "could some member carry it", and this is where
    "the one you chose does" is checked. Still pre-probe: it fires before any
    QUA is built.
    """
    roster = experiment.device.roster
    targets = list(experiment.params.targets)
    if not targets:
        raise ValueError(f"{field}={role!r}: no targets selected")
    sides: dict[str, str] = {}
    for pair_name in targets:
        qp = vendor_pair(experiment, pair_name)
        control = qp.qubit_control.name
        target = qp.qubit_target.name
        member = role_member(roster, pair_name, role)
        if needs_flux and (member, "flux") not in roster.defaults:
            raise ValueError(
                f"{pair_name}: {field}={role!r} selects member {member!r}, which "
                f"has no flux channel in this roster - nothing to play the flux "
                f"pulse on. Choose the other member, or wire this one's flux line.")
        if member == control:
            sides[pair_name] = "control"
        elif member == target:
            sides[pair_name] = "target"
        else:
            raise ValueError(
                f"{pair_name}: roster member {field}={member!r} is neither the "
                f"vendor pair's control ({control}) nor target ({target}) - "
                f"roster/vendor naming mismatch")
    if len(set(sides.values())) > 1:
        raise ValueError(
            f"{field}={role!r} maps onto DIFFERENT vendor sides across the "
            f"selected pairs ({sides}); the probe plays one side per program - "
            f"run these pairs in separate commands")
    return next(iter(sides.values()))


def flux_source_name(experiment: Any, entity: str, *,
                     qubit_only: bool = False) -> str:
    """The VENDOR name a foreign flux source is addressed by in the probes.

    ``entity`` is the roster name the operator typed (``scqo run ... --set
    flux_component=q1_q2_c``); the returned string is what the probe's
    ``z_source`` / ``z_source_qubit`` argument takes — a key of
    ``machine.qubits`` for a qubit's own z line, or a key of
    ``machine.qubit_pairs`` for a coupler (the probe reaches the coupler as
    ``machine.qubit_pairs[<name>].coupler``).

    ``qubit_only=True`` for probes that play a z PULSE through
    ``machine.qubits[...].z`` and therefore cannot drive a coupler at all
    (``scqo_qm/experiments/qubit_spectroscopy_flux_pulse.py``): a coupler entity is
    refused here, with the reason, rather than raising a bare KeyError inside
    the probe.
    """
    machine = experiment.backend.machine
    view = experiment.backend.device.component(
        experiment.device.roster.default_channel(entity, "flux"))
    qubit = getattr(view, "qubit", None)
    if qubit is not None:  # an ordinary qubit's own z line
        hits = _key_of(getattr(machine, "qubits", {}), qubit)
        if len(hits) == 1:
            return hits[0]
        raise ValueError(
            f"flux_component={entity!r}: its QUAM qubit is not a unique entry "
            f"of machine.qubits (matches {sorted(hits)})")
    if qubit_only:
        raise ValueError(
            f"flux_component={entity!r} is a coupler: this probe plays its "
            f"flux as a z PULSE on a QUAM qubit, and a tunable coupler has no "
            f"qubit to play it on — sweep a qubit's z line, or use "
            f"resonator_spectroscopy_flux (whose probe drives the coupler "
            f"element directly)")
    # A coupler: the probe reaches it through its PAIR. Match the vendor object
    # the backend resolved, never the name (QM names its pairs after the
    # coupler, e.g. `coupler_q1_q2`, which the roster need not repeat).
    pairs = getattr(machine, "qubit_pairs", {}) or {}
    hits = _key_of(pairs, view.vendor, attr="coupler")
    if len(hits) == 1:
        return hits[0]
    raise ValueError(
        f"flux_component={entity!r} resolves to a coupler the probe cannot "
        f"address: " + (f"several QUAM pairs carry it ({sorted(hits)})"
                        if hits else
                        f"no QUAM qubit_pair in this state owns it "
                        f"(pairs: {sorted(pairs)})"))
