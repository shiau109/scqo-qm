"""A QUAM root that tolerates fixed-frequency and flux-tunable qubits together.

chipA is fixed-frequency with isolated qubits; 5Q4C is flux-tunable with four pairs
in a linear array. Future chips will be PARTLY each. The vendor roots cannot express
that: ``FixedFrequencyQuam`` and ``FluxTunableQuam`` each NARROW ``BaseQuam``'s
``Dict[str, AnyTransmon]`` to one concrete type, so a device must be all one or all
the other.

The data model already allows the mix — ``BaseQuam`` is a Union, and each qubit
serializes its own ``__class__``, so a state.json can already hold both kinds. What
breaks is ``FluxTunableQuam``'s flux helpers, which assume every qubit has a flux
line. ``set_all_fluxes`` (reached from ``initialize_qpu``, which EVERY probe calls)
does an unguarded ``target.z.settle()``, and ``apply_all_flux_to_zero`` an unguarded
``q.z.to_zero()`` — both ``AttributeError`` on a fixed-frequency qubit.

This class widens the annotations back and guards every flux access through
:func:`flux_line`. The semantic change worth naming: a qubit without a flux line is
NORMAL here, not a degraded case. The vendor's guarded helpers ``warnings.warn`` per
fixed qubit; on a mostly-fixed chip that is pure noise, so those warnings are gone.
Nothing is silently skipped that a caller actually asked for — asking for a
flux-specific operation on a qubit with no flux still REFUSES by name.

Selecting this class is a per-device decision in the device's ``state.json`` root
``__class__``, exactly as the transmon subclasses are per-qubit. It must be a
CONCRETE class path: the loader honours the stored ``__class__`` over the class
``.load()`` was called on (``quam/core/quam_instantiation.py``), so a device that
stores ``quam_config.my_quam.Quam`` resolves to whatever that module currently says
— which is an uncommitted local edit, not a stable identity.
"""

from typing import Any, Dict, Optional, Union

from quam.core import quam_dataclass
from quam.serialisation import JSONSerialiser  # the return annotation below
from quam_builder.architecture.superconducting.qpu.flux_tunable_quam import (
    FluxTunableQuam,
)
from quam_builder.architecture.superconducting.qubit import AnyTransmon
from quam_builder.architecture.superconducting.qubit_pair import (
    AnyTransmonPair,
    FluxTunableTransmonPair,
)

from scqo_qm.quam_io import INCLUDE_DEFAULTS

__all__ = ["MixedTransmonQuam", "flux_line"]


def flux_line(component: Any) -> Optional[Any]:
    """The component's flux line, or ``None`` when it has none.

    Covers both ways a component can lack one, because they mean the same thing to
    a caller: a fixed-frequency transmon has no ``z`` ATTRIBUTE at all (``z`` is
    added by ``FluxTunableTransmon``, its subclass), and a flux-tunable one carries
    ``z = None`` until wiring gives it a flux line. Qubit PAIRS never have ``z`` —
    they bias their members through ``to_mutual_idle()``.
    """
    return getattr(component, "z", None)


@quam_dataclass
class MixedTransmonQuam(FluxTunableQuam):
    """QUAM root for a chip whose qubits need not all be the same kind.

    Handles the two degenerate cases as well as the mixed one, so an all-fixed or
    all-tunable device can use it unchanged.
    """

    # Widened back to BaseQuam's Union: FluxTunableQuam narrows these to
    # FluxTunableTransmon/Pair purely for type hints, and that narrowing is the only
    # thing standing between this root and a mixed tree.
    qubits: Dict[str, AnyTransmon] = None
    qubit_pairs: Dict[str, AnyTransmonPair] = None

    def __post_init__(self, *args, **kwargs):
        # dataclass fields default to None above rather than field(default_factory=dict)
        # so the widened annotations do not fight quam_dataclass's inherited defaults.
        if self.qubits is None:
            self.qubits = {}
        if self.qubit_pairs is None:
            self.qubit_pairs = {}
        super().__post_init__(*args, **kwargs)

    @classmethod
    def load(cls, *args, **kwargs) -> "MixedTransmonQuam":
        return super().load(*args, **kwargs)

    @classmethod
    def get_serialiser(cls) -> JSONSerialiser:
        """QUAM's serialiser with ``include_defaults`` pinned (``scqo_qm.quam_io``).

        Left unset, QUAM looks it up in ``~/.qualibrate/config.toml`` on every save
        that does not pass it, and raises FileNotFoundError on a machine that has no
        such file (issue #38). scqo's own saves pass it explicitly; this covers the
        ones no scqo call site reaches — the bare ``machine.save()`` inside
        quam_builder's ``build_quam_wiring`` / ``build_quam``, and the register_*
        scripts. It is QUAM's documented extension point, and it applies to every
        lab state, because each one names this class as its root ``__class__``.

        EXTEND the vendor's serialiser, never replace it: quam_builder's
        ``BaseQuam.get_serialiser`` carries the ``content_mapping`` that splits
        ``wiring`` / ``network`` out into ``wiring.json``. A fresh ``JSONSerialiser``
        would write the whole tree into ``state.json`` and leave ``wiring.json``
        stale beside it.
        """
        serialiser = super().get_serialiser()
        serialiser.include_defaults = INCLUDE_DEFAULTS
        return serialiser

    # ------------------------------------------------------------------ flux points
    def apply_all_flux_to_joint_idle(self) -> None:
        """Active qubits to their joint sweet spot, everything else to minimum.

        Qubits with no flux line are skipped in both groups.
        """
        active = self.active_qubits
        for q in active:
            z = flux_line(q)
            if z is not None:
                z.to_joint_idle()
        for q in self.qubits.values():
            if q not in active:
                z = flux_line(q)
                if z is not None:
                    z.to_min()
        self.apply_all_couplers_to_min()

    def apply_all_flux_to_min(self) -> None:
        """Every qubit with a flux line to its minimum-frequency point."""
        for q in self.qubits.values():
            z = flux_line(q)
            if z is not None:
                z.to_min()
        self.apply_all_couplers_to_min()

    def apply_all_flux_to_zero(self) -> None:
        """Every ACTIVE qubit with a flux line to zero bias."""
        for q in self.active_qubits:
            z = flux_line(q)
            if z is not None:
                z.to_zero()

    def set_all_fluxes(
        self,
        flux_point: str,
        target: Union[AnyTransmon, AnyTransmonPair, None] = None,
    ):
        """Set the fluxes for ``target``, returning its bias (``None`` if it has no
        flux line).

        Same control flow as ``FluxTunableQuam``, with three differences:

        * every ``.z`` access goes through :func:`flux_line`, so a fixed-frequency
          qubit (or a PAIR, which never has one) is settled by skipping rather than
          by ``AttributeError``;
        * the ``independent`` / ``pairwise`` guards RAISE instead of ``assert``,
          because an ``assert`` disappears under ``python -O`` and this is a real
          caller error, not an invariant;
        * ``independent`` is gated on having a flux line rather than on being a
          ``FluxTunableTransmon`` — a fixed qubit is refused for the reason that
          actually matters.
        """
        if flux_point == "independent" and flux_line(target) is None:
            raise TypeError(
                f"flux_point='independent' needs a qubit with a flux line; "
                f"{getattr(target, 'name', target)!r} has none"
            )
        if flux_point == "pairwise" and not isinstance(target, FluxTunableTransmonPair):
            raise TypeError(
                f"flux_point='pairwise' needs a flux-tunable qubit PAIR; "
                f"got {type(target).__name__}"
            )

        target_bias = None
        if flux_point == "joint":
            self.apply_all_flux_to_joint_idle()
            if isinstance(target, FluxTunableTransmonPair):
                target_bias = target.mutual_flux_bias
            elif flux_line(target) is not None:
                target_bias = target.z.joint_offset
        else:
            self.apply_all_flux_to_min()

        if flux_point == "independent":
            target.z.to_independent_idle()
            target_bias = target.z.independent_offset
        elif flux_point == "pairwise":
            target.to_mutual_idle()
            target_bias = target.mutual_flux_bias

        # align() is NOT flux-specific and must still run for a fixed qubit: it is
        # what keeps the element timelines together before the sequence starts.
        if target is None:
            for q in self.qubits.values():
                z = flux_line(q)
                if z is not None:
                    z.settle()
                q.align()
        else:
            z = flux_line(target)
            if z is not None:
                z.settle()
            target.align()

        return target_bias
