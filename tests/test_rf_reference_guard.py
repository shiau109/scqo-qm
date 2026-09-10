"""``RF_frequency`` must be WRITABLE, not merely readable.

The frequency mapping's quietest failure, and the reason it needs an audit of its
own rather than a check inside ``drive_frequency_problems``:

``IQChannel`` and ``MWChannel`` both default ``RF_frequency`` to
``"#./inferred_RF_frequency"``. A tree that keeps that default reads correctly —
QUAM resolves the reference and hands back LO + IF — so ``drive_frequency_problems``
passes, the device snapshot looks right, and ``scqo state`` shows a sensible number.
QUAM then refuses to overwrite a reference with a literal, so the failure surfaces
at the FIRST WRITEBACK: mid-run, after the instrument time is already spent.

An audit that reads cannot see this, because the read is exactly what works.
"""

from __future__ import annotations

from types import SimpleNamespace as NS

from scqo_qm.quam_fields import (
    drive_frequency_problems,
    rf_frequency_reference_problems,
)


class RefChannel:
    """A channel storing a QUAM reference: reads resolved, refuses a literal write."""

    def __init__(self, resolved: float, raw: str = "#./inferred_RF_frequency"):
        self.RF_frequency = resolved
        self._raw = raw

    def get_raw_value(self, field: str):
        return self._raw if field == "RF_frequency" else None


def literal_channel(rf: float):
    """The shape ``build_quam`` produces and every live state carries."""
    channel = NS(RF_frequency=rf)
    channel.get_raw_value = lambda field: rf if field == "RF_frequency" else None
    return channel


def test_a_literal_tree_is_silent():
    machine = NS(qubits={"q1": NS(f_01=5e9, xy=literal_channel(5e9),
                                  resonator=literal_channel(6e9))})
    assert rf_frequency_reference_problems(machine) == []


def test_a_reference_tree_is_caught_on_both_lines():
    machine = NS(qubits={"q1": NS(f_01=5e9, xy=RefChannel(5e9),
                                  resonator=RefChannel(6e9))})
    problems = rf_frequency_reference_problems(machine)
    assert len(problems) == 2
    assert any("q1.xy.RF_frequency" in p for p in problems)
    assert any("q1.resonator.RF_frequency" in p for p in problems)


def test_the_message_names_the_conversion_script():
    machine = NS(qubits={"q1": NS(f_01=5e9, xy=RefChannel(5e9), resonator=None)})
    message = rf_frequency_reference_problems(machine)[0]
    assert "convert_state_rf_literal.py" in message
    assert "writeback" in message


def test_the_reading_audit_cannot_see_it_which_is_why_this_one_exists():
    """The load-bearing claim. If ``drive_frequency_problems`` ever did catch the
    reference form, this audit would be redundant — so pin that it does not."""
    machine = NS(qubits={"q1": NS(f_01=5e9, xy=RefChannel(5e9), resonator=None)})
    assert drive_frequency_problems(machine) == []
    assert rf_frequency_reference_problems(machine) != []


def test_a_qubit_with_no_such_line_is_skipped_never_accused():
    machine = NS(qubits={"q1": NS(f_01=5e9, xy=None, resonator=None)})
    assert rf_frequency_reference_problems(machine) == []


def test_a_stub_without_raw_access_is_not_an_accusation():
    """A plain stub exposes no ``get_raw_value``; unknown must not read as guilty."""
    machine = NS(qubits={"q1": NS(f_01=5e9, xy=NS(RF_frequency=5e9), resonator=None)})
    assert rf_frequency_reference_problems(machine) == []
