# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the
# MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Tests of `karan.ksw_ties`: KSW2's rule for equal alignments over karan's costs.

Over minimap2's scores turned to costs (a substitution `2 (a + b)`, a gap `2 q + (2 e + a) k`), the alignment it
writes must be `ksw_extd2`'s, itself minimap2's C byte for byte (see `test_ksw2.mojo`), whatever path guides it."""

from std.random import random_ui64, seed
from std.testing import assert_equal, assert_true, TestSuite

from karan import Aligner, Band, Costs, Mode, Ties, align
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


def test_edit_engine_walks_ksw2() raises:
    """At unit costs the edit engine walks KSW2's rule itself, through its fronts or about a band's path: the
    whole matrix's alignment, globally and in an infix's span, as text and as runs, for 900 pairs from identical
    to unrelated, short to 1.5 kbp."""
    seed(53)
    var buf = TieBuffers()
    var aligner = Aligner()
    var letters = "ACGT".as_bytes()
    for k in range(900):
        var t: List[UInt8]
        var q: List[UInt8]
        if k % 9 == 8:
            # Unrelated: the searches give up and a band settles it.
            t = List[UInt8]()
            q = List[UInt8]()
            for _ in range(200 + _draw(1300)):
                t.append(letters[_draw(4)])
            for _ in range(200 + _draw(1300)):
                q.append(letters[_draw(4)])
        else:
            var p = _piece(1 + _draw(1500), [0, 5, 20, 60, 150, 300, 450][_draw(7)])
            t = p[0].copy()
            q = p[1].copy()
        var got = align(String(unsafe_from_utf8=t), String(unsafe_from_utf8=q), Costs.edit(), Mode.GLOBAL)
        var g = runs_of(got.cigar)
        var whole = _text(ksw_cigar(Span(t), Span(q), Costs.edit(), Span(g), buf, by_rows=True))
        assert_equal(got.cigar, whole, String("pair ", k))
        var as_runs = aligner.align(Span(t), Span(q), Costs.edit(), Mode.GLOBAL, runs=True)
        assert_equal(_text(as_runs.runs), whole, String("runs, pair ", k))
        if k % 5 == 0 and len(q) > 0:
            var host = List[UInt8]()
            for _ in range(_draw(50)):
                host.append(letters[_draw(4)])
            for c in t:
                host.append(c)
            for _ in range(_draw(50)):
                host.append(letters[_draw(4)])
            var infix = align(String(unsafe_from_utf8=host), String(unsafe_from_utf8=q), Costs.edit(), Mode.INFIX)
            var span = Span(host)[infix.reference_start : infix.reference_end]
            var gi = runs_of(infix.cigar)
            var wi = _text(ksw_cigar(span, Span(q), Costs.edit(), Span(gi), buf, by_rows=True))
            assert_equal(infix.cigar, wi, String("infix, pair ", k))


def test_wavefront_walks_ksw2() raises:
    """At gap-affine costs the wavefront walks KSW2's rule itself through its kept fronts, one search alone or
    two grown on: the whole matrix's alignment, for 600 pairs to 900 bp at affine, two-piece and linear costs,
    deletions priced apart too; and under a memory cap far below its fronts and the cells to write again, the
    same cost by another rule."""
    seed(61)
    var models = List[Costs]()
    models.append(Costs.affine(4, 6, 2))
    models.append(Costs.two_piece(4, 4, 2, 24, 1))
    models.append(Costs.two_piece(12, 8, 6, 48, 4))
    models.append(Costs.linear(3, 2))
    models.append(Costs.affine(4, 6, 2).with_deletions(5, 1))
    models.append(Costs.two_piece(4, 4, 2, 24, 1).with_deletions(3, 3, 20, 1))
    var buf = TieBuffers()
    for k in range(600):
        var p = _piece(1 + _draw(900), [0, 10, 40, 100, 200, 350][_draw(6)])
        var tb = p[0].copy()
        var qb = p[1].copy()
        var t = String(unsafe_from_utf8=tb)
        var q = String(unsafe_from_utf8=qb)
        var costs = models[k % len(models)]
        var got = align(t, q, costs, Mode.GLOBAL)
        var g = runs_of(got.cigar)
        var whole = _text(ksw_cigar(Span(tb), Span(qb), costs, Span(g), buf, by_rows=True))
        assert_equal(got.cigar, whole, String("pair ", k, " model ", k % len(models)))
        if k % 10 == 0:
            # Split for a cap far below its fronts, and too small for the cells written again: the same cost,
            # by another rule, as `max_memory` allows.
            var split = align(t, q, costs, Mode.GLOBAL, max_memory=4096)
            assert_equal(split.cost, got.cost, String("split cost, pair ", k))


def test_bounded_sweep_gives_the_whole_matrixs_alignment() raises:
    """Swept only where an alignment no costlier than the guide can pass, KSW2's rule gives the alignment every
    cell gives: 1,500 random pairs with repeats, from identical to unrelated, at unit, linear, affine and
    two-piece costs, deletions priced apart too, guided by either of karan's other rules or, a looser bound, by
    the guide's gaps placed apart."""
    seed(29)
    var models = List[Costs]()
    models.append(Costs.edit())
    models.append(Costs.linear(3, 2))
    models.append(Costs.affine(4, 6, 2))
    models.append(Costs.two_piece(4, 4, 2, 24, 1))
    models.append(Costs.two_piece(12, 8, 6, 48, 4))
    models.append(Costs.affine(4, 6, 2).with_deletions(5, 1))
    var buf = TieBuffers()
    for k in range(1500):
        var p = _piece(1 + _draw(700), [0, 20, 60, 150, 300, 450, 700][_draw(7)])
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
        if k % 7 == 6:
            # A costlier guide: every reference letter deleted, then every query letter inserted.
            g = List[UInt64]()
            if len(t) > 0:
                g.append((UInt64(len(t)) << 8) | UInt64(ord("D")))
            if len(q) > 0:
                g.append((UInt64(len(q)) << 8) | UInt64(ord("I")))
        var bounded = _text(ksw_cigar(Span(t), Span(q), costs, Span(g), buf))
        var whole = _text(ksw_cigar(Span(t), Span(q), costs, Span(g), buf, by_rows=True))
        assert_equal(bounded, whole, String("pair ", k, " costs model ", k % len(models)))


def test_gaps_along_the_edges() raises:
    """Alignments whose least cost runs one gap along the matrix's first row or column, then on into it, or ends
    in one: the bound counts the letters left at an extension alone, so none is cut; at every cost model."""
    var models = List[Costs]()
    models.append(Costs.edit())
    models.append(Costs.linear(3, 2))
    models.append(Costs.affine(4, 6, 2))
    models.append(Costs.two_piece(4, 4, 2, 24, 1))
    models.append(Costs.two_piece(12, 8, 6, 48, 4))
    models.append(Costs.affine(4, 6, 2).with_deletions(5, 1))
    var pairs = [
        ("A", "CCCCCA"),
        ("CCCCCA", "A"),
        ("ACGT", "TTTTTTTTTTTTTTTTACGT"),
        ("TTTTTTTTTTTTTTTTACGT", "ACGT"),
        ("ACGTTTTTTTTTTTTTTTTT", "ACG"),
        ("ACG", "ACGTTTTTTTTTTTTTTTTT"),
        ("GATTACAGATTACA", "CCCCCCCCCCCCCCCCCCCCCCCCGATTACAGATTACAGGGGGGGGGGGGGGGGGGGG"),
    ]
    var buf = TieBuffers()
    for p in pairs:
        var t = List[UInt8](p[0].as_bytes())
        var q = List[UInt8](p[1].as_bytes())
        for mi in range(len(models)):
            var costs = models[mi]
            var g = runs_of(align(p[0], p[1], costs, Mode.GLOBAL, ties=Ties.LEFT).cigar)
            var bounded = _text(ksw_cigar(Span(t), Span(q), costs, Span(g), buf))
            var whole = _text(ksw_cigar(Span(t), Span(q), costs, Span(g), buf, by_rows=True))
            assert_equal(bounded, whole, String(p[0], " ", p[1], " model ", mi))


def test_anti_diagonals_match_rows() raises:
    """Inside a caller's diagonals the bounded sweep by anti-diagonal and the row sweep of every cell write the
    same CIGAR: 1,500 random pairs at every cost model, the diagonals narrow, wide or none, guided by either other
    rule; diagonals no alignment fits give the guide back from both."""
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
        var low = -(1 << 40)
        var high = 1 << 40
        var reach = [-1, 3, 10, 40][k % 4]
        var shift = len(t) - len(q)
        if reach >= 0:
            low = min(0, shift) - reach
            high = max(0, shift) + reach
        var guide = align(
            String(unsafe_from_utf8=t),
            String(unsafe_from_utf8=q),
            costs,
            Mode.GLOBAL,
            band=Band(low, high) if reach >= 0 else Band(),
            ties=Ties.RIGHT if k % 2 == 0 else Ties.LEFT,
        )
        var g = runs_of(guide.cigar)
        var rows = _text(ksw_cigar(Span(t), Span(q), costs, Span(g), buf, low, high, by_rows=True))
        var diagonals = _text(ksw_cigar(Span(t), Span(q), costs, Span(g), buf, low, high))
        assert_equal(diagonals, rows, String("pair ", k, " model ", k % len(models), " reach ", reach))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
