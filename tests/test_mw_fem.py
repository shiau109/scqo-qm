"""``scqo_qm._mw_fem``: an MW-FEM port's LO in either of QUAM's two spellings.

A port declares ``upconverter_frequency`` (one upconverter) or an ``upconverters``
dict once an adopted borrowed channel rides its second one; qm-qua refuses both at
once, and quam's MWChannel reads the scalar whenever it is set. Every retune goes
through these helpers - pinned here on plain stand-ins, no QUAM needed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scqo_qm._mw_fem import (
    MW_FEM_BANDS,
    band_holds,
    partner_port_id,
    port_info,
    port_lo,
    port_los,
    second_upconverter,
    set_port_lo,
)


def _scalar(lo=4.9e9):
    return SimpleNamespace(upconverter_frequency=lo, upconverters=None, band=1)


def _dict(lo1=4.9e9, lo2=7.1e9):
    return SimpleNamespace(upconverter_frequency=None, band=2,
                           upconverters={1: {"frequency": lo1}, 2: {"frequency": lo2}})


def test_both_spellings_read_the_same_way():
    assert port_los(_scalar()) == {1: 4.9e9}
    assert port_los(_dict()) == {1: 4.9e9, 2: 7.1e9}
    assert port_lo(_dict(), 2) == 7.1e9 and second_upconverter(_dict()) == 7.1e9
    assert second_upconverter(_scalar()) is None
    assert port_los(None) == {} and second_upconverter(None) is None


def test_a_retune_writes_the_spelling_the_port_uses():
    port = _scalar()
    set_port_lo(port, 1, 5.0e9)
    assert port.upconverter_frequency == 5.0e9 and port.upconverters is None
    port = _dict()
    set_port_lo(port, 1, 7.0e9)
    assert port.upconverter_frequency is None
    assert port_los(port) == {1: 7.0e9, 2: 7.1e9}   # upconverter 2 untouched


def test_a_second_upconverter_is_never_created_by_a_retune():
    with pytest.raises(KeyError, match="upconverter 2 does not exist"):
        set_port_lo(_scalar(), 2, 7.1e9)
    with pytest.raises(KeyError, match="not 3"):
        set_port_lo(_dict(), 3, 7.1e9)


def test_bands_and_port_pairs():
    assert band_holds(2, 4.9e9, 7.1e9) and not band_holds(1, 4.9e9, 7.1e9)
    assert MW_FEM_BANDS[1][1] == 5.5e9
    assert [partner_port_id(p) for p in (2, 3, 4, 5, 6, 7)] == [3, 2, 5, 4, 7, 6]
    assert port_info(SimpleNamespace(controller_id="con1", fem_id=6, port_id=3)) == ("con1", 6, 3)
    assert port_info(SimpleNamespace()) is None
