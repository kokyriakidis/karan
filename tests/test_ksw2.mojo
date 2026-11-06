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

from karan.ksw2 import KswBuffers, KswResult, ksw_extd2


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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
