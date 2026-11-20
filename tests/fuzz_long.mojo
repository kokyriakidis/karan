# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the
# MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
`pixi run fuzz-long [cases] [seed]`: `LongAligner` on random references (a tandem array in random flanks) and queries
cut from them (substitutions, indels, inverted stretches, either strand), seeded with minimap2's minimizers and with
random valid seeds of every span from 1 to 255 on both strands, under every preset. Built with `-D ASSERT=all`, so a
read outside any sequence stops it. Each case is aligned again with one byte of the reference made no code: the call
must raise, or the byte lay outside every stretch it reads and the alignments must be the same. Seed 2 at 500 cases
(CI's) catches each of the three reads past a sequence that unequal spans once caused.
"""

from std.random import random_ui64, seed
from std.sys import argv
from karan import ErrorKind, LongAligner, LongAlignment, LongPreset, Seed, encode_nt4
from mm2_seeds import Mm2Index, mm2_seeds


def r(lo: Int, hi: Int) -> Int:
    return Int(random_ui64(UInt64(lo), UInt64(hi)))


def rnd(n: Int) -> List[UInt8]:
    var out = List[UInt8](capacity=n)
    for _ in range(n):
        out.append("ACGT".as_bytes()[r(0, 3)])
    return out^


def comp(c: UInt8) -> UInt8:
    if c == UInt8(ord("A")):
        return UInt8(ord("T"))
    if c == UInt8(ord("C")):
        return UInt8(ord("G"))
    if c == UInt8(ord("G")):
        return UInt8(ord("C"))
    return UInt8(ord("A"))


def same(left: List[LongAlignment], right: List[LongAlignment]) -> Bool:
    """Whether two runs gave the same alignments, field by field."""
    if len(left) != len(right):
        return False
    for index in range(len(left)):
        ref x = left[index]
        ref y = right[index]
        if (
            x.query_start != y.query_start
            or x.query_end != y.query_end
            or x.reference_start != y.reference_start
            or x.reference_end != y.reference_end
            or x.reverse != y.reverse
            or x.cigar != y.cigar
            or x.chain_score != y.chain_score
            or x.secondary_score != y.secondary_score
            or x.score != y.score
            or x.dp_max != y.dp_max
            or x.mapq != y.mapq
            or x.matches != y.matches
            or x.block_length != y.block_length
            or x.primary != y.primary
            or x.supplementary != y.supplementary
            or x.inversion != y.inversion
        ):
            return False
    return True


def main() raises:
    var args = argv()
    var cases = Int(args[1]) if len(args) > 1 else 1000
    seed(Int(args[2]) if len(args) > 2 else 1)
    var presets = [
        LongPreset.map_hifi(),
        LongPreset.map_ont(),
        LongPreset.asm5(),
        LongPreset.asm20(),
        LongPreset.dipcall(),
    ]
    var aligner = LongAligner()
    var total = 0
    var caught = 0
    var unread = 0
    for ci in range(cases):
        # A reference with a tandem array, a query from a piece of it: mutated, with indels, an inverted stretch.
        var unit = rnd(r(20, 400))
        var ref_text = rnd(r(500, 6000))
        for _ in range(r(0, 12)):
            var each = unit.copy()
            for _ in range(r(0, 4)):
                each[r(0, len(each) - 1)] = "ACGT".as_bytes()[r(0, 3)]
            ref_text.extend(Span(each))
        ref_text.extend(Span(rnd(r(0, 6000))))
        var lo = r(0, len(ref_text) // 2)
        var hi = r(lo + 1, len(ref_text))
        var q_text = List[UInt8]()
        var i = lo
        while i < hi:
            var roll = r(0, 999)
            if roll < 8:
                i += r(1, 60)
            elif roll < 16:
                q_text.extend(Span(rnd(r(1, 60))))
            elif roll < 18 and i + 300 < hi:
                for j in range(i + 299, i - 1, -1):
                    q_text.append(comp(ref_text[j]))
                i += 300
            else:
                q_text.append(ref_text[i] if roll > 30 else "ACGT".as_bytes()[r(0, 3)])
                i += 1
        if r(0, 3) == 0:
            var rc = List[UInt8]()
            for j in range(len(q_text) - 1, -1, -1):
                rc.append(comp(q_text[j]))
            q_text = rc^
        if len(q_text) == 0:
            continue
        var index = Mm2Index(Span(ref_text), 15, 10)
        var seeded = mm2_seeds(index, Span(q_text))
        var seeds = seeded[0].copy()
        # Random valid seeds of every span, both strands, some tandem.
        for _ in range(r(0, 200)):
            var span = r(1, min(255, min(len(ref_text), len(q_text))))
            seeds.append(Seed(r(0, len(ref_text) - span), r(0, len(q_text) - span), span, r(0, 1) == 1, r(0, 9) == 0))
        var reference = encode_nt4(Span(ref_text))
        var query = encode_nt4(Span(q_text))
        var preset = presets[ci % len(presets)]
        var found = aligner.align(Span(reference), Span(query), Span(seeds), preset, UInt32(ci))
        total += len(found)
        _ = aligner.chains(len(reference), len(query), Span(seeds), preset)
        # One byte that is no code, anywhere in the reference: the call raises, or the byte lay outside every
        # stretch it reads and the alignments are the same.
        var poisoned = reference.copy()
        poisoned[r(0, len(poisoned) - 1)] = 9
        var raised = False
        var changed = False
        try:
            changed = not same(found, aligner.align(Span(poisoned), Span(query), Span(seeds), preset, UInt32(ci)))
        except e:
            if e.kind != ErrorKind.UNKNOWN_SYMBOL:
                raise Error(String(e))
            raised = True
        if changed:
            raise Error(String(t"case {ci}: a byte outside the checked stretches changed the alignments"))
        if raised:
            caught += 1
        else:
            unread += 1
    print("cases", cases, "alignments", total, "poisoned bytes caught", caught, "unread", unread)
