# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the
# MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Tests of `karan.ksw2` against minimap2 2.31's own `ksw_extd2_sse`.

`tests/data/ksw2_cases.txt` holds 600 cases (`tests/ksw2_oracle/gen.py 600 11 1,2,5,17,40,100,250`: random pairs
with homopolymers, short tandem repeats, substitutions, indels and wildcards, every flag the port takes, bands,
Z-drops, end bonuses, minimap2's ava-ont and asm5 scores), and `tests/data/ksw2_expected.txt` what the C gave for
each (`tests/ksw2_oracle/oracle.c`, built against minimap2's `ksw2_extd2_sse.c` and `kalloc.c`): every field of
`ksw_extz_t` and the CIGAR. The port must give them byte for byte with KSW2's sixteen-lane vectors, and with
sixty-four lanes wherever the band holds the whole matrix."""

from std.testing import assert_equal, assert_true, TestSuite

from std.random import random_ui64, seed

from karan.ksw2 import KSW_EZ_APPROX_MAX, KswBuffers, KswResult, ksw_extd2
from karan.ksw2_lanes import LaneBuffers, ksw_lanes, touches_band


def _code(c: UInt8) -> UInt8:
    if c == UInt8(ord("A")):
        return 0
    if c == UInt8(ord("C")):
        return 1
    if c == UInt8(ord("G")):
        return 2
    if c == UInt8(ord("T")):
        return 3
    return 4


def _line[L: Int](entry: String, mut buf: KswBuffers, mut ez: KswResult) raises -> String:
    """One case through the port, printed as the oracle prints it."""
    var f = entry.split(" ")
    var t = List[UInt8]()
    for c in String(f[11]).as_bytes():
        t.append(_code(c))
    var q = List[UInt8]()
    for c in String(f[12]).as_bytes():
        q.append(_code(c))
    var a = Int(f[8])
    var b = Int(f[9])
    var ambi = Int(f[10])
    var mat = List[Int8]()
    for i in range(5):
        for j in range(5):
            mat.append(Int8(-ambi) if i == 4 or j == 4 else (Int8(a) if i == j else Int8(-b)))
    ksw_extd2[L](
        Span(q),
        Span(t),
        5,
        Span(mat),
        Int(f[4]),
        Int(f[5]),
        Int(f[6]),
        Int(f[7]),
        Int(f[1]),
        Int(f[2]),
        Int(f[3]),
        Int(f[0]),
        buf,
        ez,
    )
    var out = String(ez.score, " ", ez.max, " ", ez.max_q, " ", ez.max_t, " ", ez.mqe, " ", ez.mqe_t, " ")
    out += String(ez.mte, " ", ez.mte_q, " ", Int(ez.zdropped), " ", Int(ez.reach_end), " ", len(ez.cigar), " ")
    var ops = "MIDN"
    for c in ez.cigar:
        out += String(Int(c >> 4))
        out += String(ops[byte=Int(c & 0xF)])
    return out


def _cases() raises -> Tuple[List[String], List[String]]:
    var cases = List[String]()
    for line in open("tests/data/ksw2_cases.txt", "r").read().splitlines():
        if line.byte_length() > 0:
            cases.append(String(line))
    var expected = List[String]()
    for line in open("tests/data/ksw2_expected.txt", "r").read().splitlines():
        if line.byte_length() > 0:
            expected.append(String(line))
    return (cases^, expected^)


def test_sixteen_lanes_match_ksw2() raises:
    var data = _cases()
    var cases = data[0].copy()
    var expected = data[1].copy()
    assert_equal(len(cases), 600)
    assert_equal(len(cases), len(expected))
    var buf = KswBuffers()
    var ez = KswResult()
    for i in range(len(cases)):
        assert_equal(_line[16](cases[i], buf, ez), expected[i], String("case ", i, ": ", cases[i]))


def test_wide_vectors_match_ksw2_unbanded() raises:
    """Past a band's edge the cells a vector holds depend on its width; with no band, every width agrees."""
    var data = _cases()
    var cases = data[0].copy()
    var expected = data[1].copy()
    var buf = KswBuffers()
    var ez = KswResult()
    var checked = 0
    for i in range(len(cases)):
        if String(cases[i].split(" ")[1]) != "-1":
            continue
        assert_equal(_line[32](cases[i], buf, ez), expected[i], String("case ", i))
        assert_equal(_line[64](cases[i], buf, ez), expected[i], String("case ", i))
        checked += 1
    assert_true(checked > 100)


def _draw(n: Int) -> Int:
    return Int(random_ui64(0, UInt64(n - 1)))


def _piece(n: Int, rate: Int) -> Tuple[List[UInt8], List[UInt8]]:
    """A target with homopolymers and short repeats, and a query from it at `rate` edits per thousand."""
    var t = List[UInt8]()
    while len(t) < n:
        var r = _draw(100)
        if r < 15:
            var c = UInt8(_draw(4))
            for _ in range(2 + _draw(8)):
                t.append(c)
        elif r < 22:
            var unit = List[UInt8]()
            for _ in range(2 + _draw(4)):
                unit.append(UInt8(_draw(4)))
            for _ in range(2 + _draw(5)):
                for u in unit:
                    t.append(u)
        else:
            t.append(UInt8(_draw(4)))
    var q = List[UInt8]()
    for c in t:
        var x = _draw(1000)
        if x < rate // 3:
            q.append(UInt8(_draw(4)))
        elif x < 2 * rate // 3:
            pass
        elif x < rate:
            q.append(c)
            q.append(UInt8(_draw(4)))
        else:
            q.append(c)
    if len(q) == 0:
        q.append(0)
    return (t^, q^)


def test_lanes_match_ksw2() raises:
    """`ksw_lanes` against `ksw_extd2` on 512 random pieces in batches of 32: identical with a band holding the
    whole matrix, and in a band of the batch's length difference plus 32 wherever the path stays clear of it."""
    seed(5)
    var mat = List[Int8]()
    for i in range(5):
        for j in range(5):
            mat.append(Int8(-1) if i == 4 or j == 4 else (Int8(2) if i == j else Int8(-4)))
    var single = KswBuffers()
    var ez = KswResult()
    var lanes = LaneBuffers[32]()
    var cigars = List[List[UInt32]]()
    var clear = 0
    for _ in range(16):
        var ts = List[List[UInt8]]()
        var qs = List[List[UInt8]]()
        for _ in range(32):
            var p = _piece(20 + _draw(600), [0, 20, 40, 80][_draw(4)])
            ts.append(p[0].copy())
            qs.append(p[1].copy())
        var tp = List[ImmPointer[UInt8, ImmUntrackedOrigin]]()
        var qp = List[ImmPointer[UInt8, ImmUntrackedOrigin]]()
        var tn = List[Int]()
        var qn = List[Int]()
        var widest = 0
        var w = 0
        for k in range(32):
            tp.append(ts[k].unsafe_ptr().as_imm().unsafe_origin_cast[ImmUntrackedOrigin]())
            qp.append(qs[k].unsafe_ptr().as_imm().unsafe_origin_cast[ImmUntrackedOrigin]())
            tn.append(len(ts[k]))
            qn.append(len(qs[k]))
            widest = max(widest, max(len(ts[k]), len(qs[k])))
            w = max(w, abs(len(ts[k]) - len(qs[k])) + 32)
        var want = List[List[UInt32]]()
        for k in range(32):
            ksw_extd2[64](
                Span(qs[k]), Span(ts[k]), 5, Span(mat), 4, 2, 24, 1, -1, -1, -1, KSW_EZ_APPROX_MAX, single, ez
            )
            want.append(ez.cigar.copy())
        ksw_lanes[32](tp, tn, qp, qn, 2, 4, 1, 4, 2, 24, 1, widest + 1, lanes, cigars)
        for k in range(32):
            if cigars[k] != want[k]:
                var a = String()
                for c in want[k]:
                    a += String(Int(c >> 4), "MIDN"[byte=Int(c & 0xF)])
                var b = String()
                for c in cigars[k]:
                    b += String(Int(c >> 4), "MIDN"[byte=Int(c & 0xF)])
                var tt = String()
                for c in ts[k]:
                    tt += "ACGTN"[byte=Int(c)]
                var qq = String()
                for c in qs[k]:
                    qq += "ACGTN"[byte=Int(c)]
                assert_true(False, String("unbanded piece ", k, " t ", tt, " q ", qq, " ksw2 ", a, " lanes ", b))
        ksw_lanes[32](tp, tn, qp, qn, 2, 4, 1, 4, 2, 24, 1, w, lanes, cigars)
        for k in range(32):
            if not touches_band(cigars[k], w, 8):
                clear += 1
                if cigars[k] != want[k]:
                    var a = String()
                    for c in want[k]:
                        a += String(Int(c >> 4), "MIDN"[byte=Int(c & 0xF)])
                    var b = String()
                    for c in cigars[k]:
                        b += String(Int(c >> 4), "MIDN"[byte=Int(c & 0xF)])
                    assert_true(
                        False,
                        String(
                            "banded piece ",
                            k,
                            " w ",
                            w,
                            " tlen ",
                            len(ts[k]),
                            " qlen ",
                            len(qs[k]),
                            " ksw2 ",
                            a,
                            " lanes ",
                            b,
                        ),
                    )
        # The kernel reads the letters through untracked pointers: keep them alive past it.
        _ = len(ts)
        _ = len(qs)
    assert_true(clear > 400)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
