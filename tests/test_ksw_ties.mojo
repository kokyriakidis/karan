# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the
# MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Tests of `karan.ksw_ties`: KSW2's rule for equal alignments over karan's costs.

Over minimap2's scores turned to costs (a substitution `2 (a + b)`, a gap `2 q + (2 e + a) k`), the alignment it
writes must be `ksw_extd2`'s, itself minimap2's C byte for byte (see `test_ksw2.mojo`), whatever path guides it."""

from std.random import random_ui64, seed
from std.testing import assert_equal, assert_true, TestSuite

from karan import Costs, Mode, Ties, align
from karan.cigar import run_length, run_letter, runs_of
from karan.ksw2 import KswBuffers, KswResult, ksw_extd2
from karan.ksw_ties import TieBuffers, ksw_cigar


def _draw(n: Int) -> Int:
    return Int(random_ui64(0, UInt64(n - 1)))


def _piece(n: Int, rate: Int) -> Tuple[List[UInt8], List[UInt8]]:
    """A target with homopolymers and short repeats over ACGT, and a query from it at `rate` edits per thousand."""
    var t = List[UInt8]()
    var letters = "ACGT".as_bytes()
    while len(t) < n:
        var r = _draw(100)
        if r < 15:
            var c = letters[_draw(4)]
            for _ in range(2 + _draw(8)):
                t.append(c)
        elif r < 22:
            var unit = List[UInt8]()
            for _ in range(2 + _draw(4)):
                unit.append(letters[_draw(4)])
            for _ in range(2 + _draw(5)):
                for u in unit:
                    t.append(u)
        else:
            t.append(letters[_draw(4)])
    var q = List[UInt8]()
    for c in t:
        var x = _draw(1000)
        if x < rate // 3:
            q.append(letters[_draw(4)])
        elif x < 2 * rate // 3:
            pass
        elif x < rate:
            q.append(c)
            q.append(letters[_draw(4)])
        else:
            q.append(c)
    if len(q) == 0:
        q.append(letters[0])
    return (t^, q^)


def _text(runs: List[UInt64]) -> String:
    var s = String()
    for r in runs:
        s += String(run_length(r))
        s += chr(Int(run_letter(r)))
    return s


def _codes(s: List[UInt8]) -> List[UInt8]:
    var out = List[UInt8]()
    for c in s:
        out.append(
            UInt8(0) if c
            == UInt8(ord("A")) else (
                UInt8(1) if c == UInt8(ord("C")) else (UInt8(2) if c == UInt8(ord("G")) else UInt8(3))
            )
        )
    return out^


def _ksw2(t: List[UInt8], q: List[UInt8], mut buf: KswBuffers, mut ez: KswResult) -> String:
    """`ksw_extd2`'s global alignment under minimap2's ava-ont scores, its `M` runs spelled `=` and `X`."""
    var mat = List[Int8]()
    for i in range(5):
        for j in range(5):
            mat.append(Int8(-1) if i == 4 or j == 4 else (Int8(2) if i == j else Int8(-4)))
    var tc = _codes(t)
    var qc = _codes(q)
    ksw_extd2[64](Span(qc), Span(tc), 5, Span(mat), 4, 2, 24, 1, -1, -1, -1, 0, buf, ez)
    var runs = List[UInt64]()
    var i = 0
    var j = 0
    for c in ez.cigar:
        var op = Int(c & 0xF)
        var n = Int(c >> 4)
        if op == 1:
            runs.append((UInt64(n) << 8) | UInt64(ord("I")))
            j += n
        elif op == 2:
            runs.append((UInt64(n) << 8) | UInt64(ord("D")))
            i += n
        else:
            for k in range(n):
                var letter = UInt64(ord("=")) if t[i + k] == q[j + k] else UInt64(ord("X"))
                if len(runs) > 0 and (runs[len(runs) - 1] & 0xFF) == letter:
                    runs[len(runs) - 1] += UInt64(1) << 8
                else:
                    runs.append((UInt64(1) << 8) | letter)
            i += n
            j += n
    return _text(runs)


def test_minimap2_scores_give_ksw2s_alignment() raises:
    """Over minimap2's ava-ont scores as costs, guided by karan's own alignment: 600 random pairs, every one
    as KSW2 writes it."""
    seed(17)
    var costs = Costs.two_piece(12, 8, 6, 48, 4)
    var buf = TieBuffers()
    var kb = KswBuffers()
    var ez = KswResult()
    for k in range(600):
        var p = _piece(1 + _draw(500), [0, 20, 40, 80, 200][_draw(5)])
        var t = p[0].copy()
        var q = p[1].copy()
        var guide = align(String(unsafe_from_utf8=t), String(unsafe_from_utf8=q), costs, Mode.GLOBAL)
        var g = runs_of(guide.cigar)
        var got = _text(ksw_cigar(Span(t), Span(q), costs, Span(g), buf))
        assert_equal(got, _ksw2(t, q, kb, ez), String("pair ", k))


def test_rule_under_unit_costs() raises:
    """KSW2's rule under the edit distance: gaps as early as the cost allows, substitutions as late."""
    var buf = TieBuffers()
    var t = "GCTACAGTG".as_bytes()
    var q = "GCTGAGTG".as_bytes()
    # Guided by the substitution-first alignment karan's LEFT rule writes.
    var g = runs_of("3=1X1D4=")
    assert_equal(_text(ksw_cigar(t, q, Costs.edit(), Span(g), buf)), "3=1D1X4=")
    var t2 = "ACGTTTACG".as_bytes()
    var q2 = "ACGTTTTACG".as_bytes()
    var g2 = runs_of("6=1I3=")
    assert_equal(_text(ksw_cigar(t2, q2, Costs.edit(), Span(g2), buf)), "3=1I6=")


def test_api_ties_ksw2() raises:
    """`align` with `Ties.KSW2`: the same cost as `LEFT`, KSW2's CIGAR, in a global, an infix and an extension."""
    var left = align("GCTACAGTG", "GCTGAGTG", Costs.edit(), Mode.GLOBAL, ties=Ties.LEFT)
    var ksw = align("GCTACAGTG", "GCTGAGTG", Costs.edit(), Mode.GLOBAL, ties=Ties.KSW2)
    assert_equal(left.cigar, "3=1X1D4=")
    assert_equal(ksw.cigar, "3=1D1X4=")
    assert_equal(ksw.cost, left.cost)
    var infix = align("TTTTGCTACAGTGTTTT", "GCTGAGTG", Costs.edit(), Mode.INFIX, ties=Ties.KSW2)
    assert_equal(infix.cigar, "3=1D1X4=")
    assert_equal(infix.reference_start, 4)
    var costs = Costs.two_piece(4, 4, 2, 24, 1)
    var ext = align("GCTACAGTGACGTTGCA", "GCTGAGTGACGTTGCA", costs, Mode.extension(2), ties=Ties.KSW2)
    var ext_left = align("GCTACAGTGACGTTGCA", "GCTGAGTGACGTTGCA", costs, Mode.extension(2), ties=Ties.LEFT)
    assert_equal(ext.score, ext_left.score)
    assert_equal(ext.cigar, "3=1D1X12=")


def test_band_gives_the_whole_matrixs_alignment() raises:
    """Searched about any path, widened as its walk nears the edge, KSW2's rule gives the alignment the whole
    matrix gives: 1,200 random pairs with repeats, at unit, linear, affine and two-piece costs, deletions priced
    apart too, guided by either of karan's other rules."""
    seed(29)
    var models = List[Costs]()
    models.append(Costs.edit())
    models.append(Costs.linear(3, 2))
    models.append(Costs.affine(4, 6, 2))
    models.append(Costs.two_piece(4, 4, 2, 24, 1))
    models.append(Costs.two_piece(12, 8, 6, 48, 4))
    models.append(Costs.affine(4, 6, 2).with_deletions(5, 1))
    var buf = TieBuffers()
    for k in range(1200):
        var p = _piece(1 + _draw(400), [0, 20, 60, 150, 300][_draw(5)])
        var t = p[0].copy()
        var q = p[1].copy()
        var costs = models[k % len(models)]
        var guide = align(
            String(unsafe_from_utf8=t),
            String(unsafe_from_utf8=q),
            costs,
            Mode.GLOBAL,
            ties=Ties.RIGHT if k % 2 == 0 else Ties.LEFT,
        )
        var g = runs_of(guide.cigar)
        var banded = _text(ksw_cigar(Span(t), Span(q), costs, Span(g), buf))
        var whole = _text(ksw_cigar(Span(t), Span(q), costs, Span(g), buf, start_width=max(len(t), len(q))))
        assert_equal(banded, whole, String("pair ", k, " costs model ", k % len(models)))


def test_anti_diagonals_match_rows() raises:
    """The vectorized sweep by anti-diagonal and the row sweep it replaces write the same CIGAR: 1,500 random
    pairs at every cost model, narrow and wide bands, guided by either other rule."""
    seed(41)
    var models = List[Costs]()
    models.append(Costs.edit())
    models.append(Costs.linear(3, 2))
    models.append(Costs.affine(4, 6, 2))
    models.append(Costs.two_piece(4, 4, 2, 24, 1))
    models.append(Costs.two_piece(12, 8, 6, 48, 4))
    models.append(Costs.affine(4, 6, 2).with_deletions(5, 1))
    var buf = TieBuffers()
    for k in range(1500):
        var p = _piece(1 + _draw(700), [0, 20, 60, 150, 300][_draw(5)])
        var t = p[0].copy()
        var q = p[1].copy()
        var costs = models[k % len(models)]
        var guide = align(
            String(unsafe_from_utf8=t),
            String(unsafe_from_utf8=q),
            costs,
            Mode.GLOBAL,
            ties=Ties.RIGHT if k % 2 == 0 else Ties.LEFT,
        )
        var g = runs_of(guide.cigar)
        var width = [1, 4, 32, 1000][k % 4]
        var rows = _text(ksw_cigar(Span(t), Span(q), costs, Span(g), buf, start_width=width, by_rows=True))
        var diagonals = _text(ksw_cigar(Span(t), Span(q), costs, Span(g), buf, start_width=width))
        assert_equal(diagonals, rows, String("pair ", k, " model ", k % len(models), " width ", width))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
