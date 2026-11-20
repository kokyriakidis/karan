# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the
# MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
Tests of `karan.chaining` and `karan.long_align`, run with `pixi run test-long`.

The parity cases hold karan to minimap2 2.28 itself: `tests/data/long` keeps an 80 kb window of CHM13 chr19 (a
tandem array at 53.1 Mb), four HiFi reads of HG002 that align there and a contig made from the window with 1.7 kb
of the array deleted and 60 substitutions, and minimap2's own PAF for each (`-cx map-hifi`; `-cx asm5 -z200000,10000
-r2k`). Seeded as minimap2 seeds (`mm2_seeds`), every line must come out the same: intervals, matches, block length,
mapping quality, primary or secondary, CIGAR. The synthetic cases check what holds by construction: a colinear
chain's score, a copy-number difference in a tandem array aligned as one deletion of the missing copies, the reverse
strand.
"""

from std.random import random_ui64, seed
from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from karan import ErrorKind, LongAligner, LongAlignment, LongPreset, Seed, encode_nt4, name_hash
from karan.chaining import Packed, chain_dp, chain_rmq, mg_log2
from karan.long_align import CIGAR_DELETION, CIGAR_INSERTION, CIGAR_MATCH
from mm2_seeds import Mm2Index, mm2_seeds

comptime DATA = "tests/data/long/"


def read_fasta(path: String) raises -> Tuple[List[String], List[List[UInt8]]]:
    var names = List[String]()
    var seqs = List[List[UInt8]]()
    with open(path, "r") as f:
        for line in f.read().split("\n"):
            if line.byte_length() == 0:
                continue
            if line.startswith(">"):
                names.append(String(line[byte=1:].split(" ")[0]))
                seqs.append(List[UInt8]())
            else:
                seqs[len(seqs) - 1].extend(line.as_bytes())
    return (names^, seqs^)


def paf_fields(name: String, qlen: Int, hit: LongAlignment) -> String:
    """The PAF columns and tags both write: 1 to 5, 7 to 12, `tp` and `cg` (the reference name left out)."""
    var tp = ("I" if hit.primary else "i") if hit.inversion else ("P" if hit.primary else "S")
    return String(
        t"{name}\t{qlen}\t{hit.query_start}\t{hit.query_end}\t{'-' if hit.reverse else '+'}\t"
        t"{hit.reference_end - hit.reference_start}\t{hit.reference_start}\t{hit.reference_end}\t{hit.matches}\t"
        t"{hit.block_length}\t{hit.mapq}\ttp:A:{tp}\tcg:Z:{hit.cigar_string()}"
    )


def expected_fields(path: String) raises -> List[String]:
    var out = List[String]()
    with open(path, "r") as f:
        for line in f.read().split("\n"):
            if line.byte_length() == 0:
                continue
            var cols = line.split("\t")
            var tp = String()
            var cg = String()
            for index in range(12, len(cols)):
                if cols[index].startswith("tp:A:"):
                    tp = String(cols[index])
                elif cols[index].startswith("cg:Z:"):
                    cg = String(cols[index])
            var span = Int(cols[8]) - Int(cols[7])
            out.append(
                String(
                    t"{cols[0]}\t{cols[1]}\t{cols[2]}\t{cols[3]}\t{cols[4]}\t{span}\t{cols[7]}\t{cols[8]}\t"
                    t"{cols[9]}\t{cols[10]}\t{cols[11]}\t{tp}\t{cg}"
                )
            )
    return out^


def check_parity(preset: LongPreset, queries_path: String, expected_path: String) raises:
    var reference_in = read_fasta(DATA + "ref.fa")
    var queries = read_fasta(queries_path)
    var reference = encode_nt4(Span(reference_in[1][0]))
    var index = Mm2Index(Span(reference_in[1][0]), 19, 19)
    var aligner = LongAligner()
    var got = List[String]()
    for i in range(len(queries[0])):
        var seeded = mm2_seeds(index, Span(queries[1][i]))
        var query = encode_nt4(Span(queries[1][i]))
        for hit in aligner.align(
            Span(reference), Span(query), Span(seeded[0]), preset, name_hash(queries[0][i]), seeded[1]
        ):
            got.append(paf_fields(queries[0][i], len(query), hit))
    var expected = expected_fields(expected_path)
    assert_equal(len(got), len(expected))
    for i in range(len(got)):
        assert_equal(got[i], expected[i])


def test_reads_match_minimap2() raises:
    """Four HiFi reads in a tandem array, each with its secondaries on other copies: minimap2's lines exactly."""
    check_parity(LongPreset.map_hifi(), DATA + "reads.fa", DATA + "reads.paf")


def test_contig_matches_minimap2() raises:
    """A contig missing 1.7 kb of a tandem array, aligned as dipcall aligns: the deletion where minimap2 puts it."""
    check_parity(LongPreset.dipcall(), DATA + "contig.fa", DATA + "contig.paf")


def test_mg_log2() raises:
    """The fast log2 of minimap2 is exact at powers of two and within 0.01 between."""
    assert_true(abs(mg_log2(2.0) - 1.0) < 0.01)
    assert_true(abs(mg_log2(1024.0) - 10.0) < 0.01)
    assert_true(abs(mg_log2(3.0) - 1.585) < 0.01)


def test_name_hash() raises:
    """The X31 string hash of khash: h = 31 h + c."""
    assert_equal(name_hash("ab"), UInt32(97 * 31 + 98))
    assert_equal(name_hash(""), UInt32(0))


def colinear(count: Int, step: Int) -> List[Packed]:
    """`count` 19-letter anchors on one diagonal, `step` apart."""
    var a = List[Packed]()
    for index in range(count):
        var pos = UInt64(100 + index * step + 18)
        a.append(Packed(pos, UInt64(19) << 32 | (pos - 50)))
    return a^


def test_colinear_chain_score() raises:
    """Anchors on one diagonal chain into one, each adding the smaller of its gap and its span: 19 + 10 (n - 1)
    at a step of 10, by both chainers."""
    var a = colinear(50, 10)
    var dp = chain_dp(5000, 5000, 500, 25, 5000, 3, 40, 0.152, 0.0, a)
    assert_equal(len(dp.u), 1)
    assert_equal(Int(dp.u[0] >> 32), 19 + 10 * 49)
    assert_equal(Int(dp.u[0] & 0xFFFFFFFF), 50)
    var rmq = chain_rmq(5000, 1000, 500, 25, 100000, 3, 40, 0.152, 0.0, a)
    assert_equal(len(rmq.u), 1)
    assert_equal(Int(rmq.u[0] >> 32), 19 + 10 * 49)


def test_gap_breaks_chain_beyond_band() raises:
    """A gap difference above the band is never chained: two chains, one on each diagonal."""
    var a = colinear(20, 10)
    for index in range(20):
        var pos = UInt64(5000 + index * 10 + 18)
        a.append(Packed(pos, UInt64(19) << 32 | (pos - 50 - 800)))
    var dp = chain_dp(5000, 5000, 500, 25, 5000, 3, 40, 0.152, 0.0, a)
    assert_equal(len(dp.u), 2)


def random_dna(length: Int) -> List[UInt8]:
    comptime LETTERS = "ACGT"
    var out = List[UInt8](capacity=length)
    for _ in range(length):
        out.append(LETTERS.as_bytes()[Int(random_ui64(0, 3))])
    return out^


def test_tandem_copy_number_difference() raises:
    """A reference with twelve copies of a 300-letter unit (each copy 1% its own), a query with two copies fewer:
    one alignment over the whole query, the missing copies one deletion of 600 letters, nothing else."""
    seed(11)
    var unit = random_dna(300)
    var reference = random_dna(8000)
    for _ in range(12):
        var each = unit.copy()
        for _ in range(3):
            var at = Int(random_ui64(0, 299))
            each[at] = UInt8(ord("A")) if each[at] != UInt8(ord("A")) else UInt8(ord("C"))
        reference.extend(Span(each))
    reference.extend(Span(random_dna(8000)))
    var query = List[UInt8]()
    query.extend(Span(reference)[0 : 8000 + 5 * 300])
    query.extend(Span(reference)[8000 + 7 * 300 :])
    var index = Mm2Index(Span(reference), 19, 19)
    var codes = encode_nt4(Span(reference))
    var aligner = LongAligner()
    var qcodes = encode_nt4(Span(query))
    var seeded = mm2_seeds(index, Span(query))
    for preset in [LongPreset.asm5(), LongPreset.dipcall(), LongPreset.map_hifi()]:
        var hits = aligner.align(Span(codes), Span(qcodes), Span(seeded[0]), preset)
        assert_true(len(hits) >= 1)
        ref hit = hits[0]
        assert_true(hit.primary and not hit.reverse)
        assert_equal(hit.query_start, 0)
        assert_equal(hit.query_end, len(query))
        assert_equal(hit.reference_start, 0)
        assert_equal(hit.reference_end, len(reference))
        var deleted = 0
        var inserted = 0
        var gaps = 0
        for run in hit.cigar:
            var op = Int(run & 0xF)
            if op == CIGAR_DELETION:
                deleted += Int(run >> 4)
                gaps += 1
            elif op == CIGAR_INSERTION:
                inserted += Int(run >> 4)
                gaps += 1
        assert_equal(deleted, 600)
        assert_equal(inserted, 0)
        assert_equal(gaps, 1)
        assert_equal(hit.matches, len(query))


def test_reverse_strand() raises:
    """A query that is a stretch of the reference reverse-complemented aligns on the reverse strand, end to end."""
    seed(5)
    var reference = random_dna(20000)
    var query = List[UInt8]()
    for index in range(15000, 3000, -1):
        var c = reference[index - 1]
        query.append(
            UInt8(ord("T")) if c
            == UInt8(ord("A")) else (
                UInt8(ord("G")) if c
                == UInt8(ord("C")) else (UInt8(ord("C")) if c == UInt8(ord("G")) else UInt8(ord("A")))
            )
        )
    var index = Mm2Index(Span(reference), 19, 19)
    var seeded = mm2_seeds(index, Span(query))
    var aligner = LongAligner()
    var codes = encode_nt4(Span(reference))
    var qcodes = encode_nt4(Span(query))
    var hits = aligner.align(Span(codes), Span(qcodes), Span(seeded[0]), LongPreset.map_hifi())
    assert_equal(len(hits), 1)
    assert_true(hits[0].reverse)
    assert_equal(hits[0].reference_start, 3000)
    assert_equal(hits[0].reference_end, 15000)
    assert_equal(hits[0].query_start, 0)
    assert_equal(hits[0].query_end, 12000)
    assert_equal(hits[0].cigar_string(), "12000M")
    var chains = aligner.chains(20000, len(query), Span(seeded[0]), LongPreset.map_hifi())
    assert_equal(len(chains), 1)
    assert_true(chains[0].reverse and chains[0].primary)


def test_presets_pass_their_own_check() raises:
    """Every named preset is one minimap2 takes."""
    for preset in [
        LongPreset.map_ont(),
        LongPreset.map_hifi(),
        LongPreset.asm5(),
        LongPreset.asm10(),
        LongPreset.asm20(),
        LongPreset.dipcall(),
    ]:
        preset.check()


def test_rejects_options_minimap2_refuses() raises:
    """The rules of mm_check_opt: positive gap costs, the two pieces crossing, at most 127 together, --score-N below -B,
    the Z-drop at least the inversion's, and a match that scores."""
    var aligner = LongAligner()
    var codes = encode_nt4("ACGTACGTACGTACGTACGTACGT".as_bytes())
    var seeds: List[Seed] = [Seed(0, 0, 19, False, False)]
    var cases = List[LongPreset]()
    var p = LongPreset.map_hifi()
    p.gap_extend = 0
    cases.append(p)
    p = LongPreset.map_hifi()
    p.long_gap_extend = 3  # not below gap_extend
    cases.append(p)
    p = LongPreset.asm5()
    p.long_gap_open = 90  # (39 + 3) + (90 + 1) > 127
    cases.append(p)
    p = LongPreset.map_hifi()
    p.ambiguous = 4  # not below mismatch
    cases.append(p)
    p = LongPreset.map_hifi()
    p.zdrop_inv = p.zdrop + 1
    cases.append(p)
    p = LongPreset.map_hifi()
    p.match_score = 0
    cases.append(p)
    for preset in cases:
        var raised = False
        try:
            _ = aligner.align(Span(codes), Span(codes), Span(seeds), preset)
        except e:
            raised = e.kind == ErrorKind.INVALID_SCORING or e.kind == ErrorKind.INVALID_ARGUMENT
        assert_true(raised)


def test_rejects_seeds_outside_the_sequences() raises:
    """A seed past either sequence's end, before its start or of a span outside 1 to 255 raises rather than reading
    past the sequence, before any chaining, in `chains` as in `align`."""
    seed(3)
    var text = random_dna(3000)
    var reference = encode_nt4(Span(text)[0:2000])
    var query = encode_nt4(Span(text)[1000:3000])
    var aligner = LongAligner()
    for bad in [
        Seed(1990, 0, 19, False, False),
        Seed(0, 1990, 19, False, False),
        Seed(-1, 0, 19, False, False),
        Seed(0, 0, 0, False, False),
        Seed(0, 0, 256, False, False),
    ]:
        var seeds = List[Seed]()
        for index in range(0, 900, 20):
            seeds.append(Seed(1000 + index, index, 19, False, False))
        seeds.append(bad)
        var raised = 0
        try:
            _ = aligner.align(Span(reference), Span(query), Span(seeds), LongPreset.map_hifi())
        except e:
            raised += Int(e.kind == ErrorKind.INVALID_ARGUMENT)
        try:
            _ = aligner.chains(len(reference), len(query), Span(seeds), LongPreset.map_hifi())
        except e:
            raised += Int(e.kind == ErrorKind.INVALID_ARGUMENT)
        assert_equal(raised, 2)


def test_rejects_letters_not_encoded() raises:
    """Letters rather than minimap2's codes raise: the score table holds five codes."""
    seed(3)
    var text = random_dna(3000)
    var reference = encode_nt4(Span(text))
    var seeds = List[Seed]()
    for index in range(0, 1900, 20):
        seeds.append(Seed(1000 + index, index, 19, False, False))
    var aligner = LongAligner()
    with assert_raises(contains="encode_nt4"):
        _ = aligner.align(Span(reference), Span(text)[1000:3000], Span(seeds), LongPreset.map_hifi())
    with assert_raises(contains="encode_nt4"):
        _ = aligner.align(Span(text), Span(reference)[1000:3000], Span(seeds), LongPreset.map_hifi())


def test_degenerate_inputs() raises:
    """No seeds, one seed, an empty query, a query of Ns and seeds that are all tandem: no alignment, or one that
    holds the whole query, and nothing read out of range."""
    seed(3)
    var text = random_dna(5000)
    var reference = encode_nt4(Span(text))
    var query = encode_nt4(Span(text)[1000:3000])
    var aligner = LongAligner()
    var preset = LongPreset.map_hifi()
    var none = List[Seed]()
    assert_equal(len(aligner.align(Span(reference), Span(query), Span(none), preset)), 0)
    assert_equal(len(aligner.chains(len(reference), len(query), Span(none), preset)), 0)
    var one: List[Seed] = [Seed(1000, 0, 19, False, False)]
    assert_equal(len(aligner.align(Span(reference), Span(query), Span(one), preset)), 0)
    var empty = List[UInt8]()
    assert_equal(len(aligner.align(Span(reference), Span(empty), Span(none), preset)), 0)
    var tandem = List[Seed]()
    for index in range(0, 1900, 20):
        tandem.append(Seed(1000 + index, index, 19, False, True))
    var hits = aligner.align(Span(reference), Span(query), Span(tandem), preset)
    assert_equal(len(hits), 1)
    assert_equal(hits[0].cigar_string(), "2000M")
    var ns = List[UInt8](length=2000, fill=4)
    assert_equal(len(aligner.align(Span(reference), Span(ns), Span(tandem), preset)), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
