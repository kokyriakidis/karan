# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the
# MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
minimap2 2.28's seeds (`sketch.c`, `seed.c`, `index.c`, `map.c`; MIT, Heng Li), for the tests and the parity
harness only: karan takes seeds from its caller and indexes nothing. The (w, k) minimizers of a reference (one
sequence) and of a query, with minimap2's hash and its robust winnowing; the reference's occurrence cutoff
(`mm_idx_cal_max_occ` at `-f 2e-4`, clamped to the preset's 50 to 500); the query's own frequent minimizers left out
(`mm_seed_mz_flt`); in a streak of frequent ones a few kept every 500 letters (`mm_seed_select`); and the tandem flag.
Fed to `LongAligner`, they are the anchors minimap2 chains (checked against its `--print-seeds`).
"""

from std.collections import Dict

from karan import Seed


@always_inline
def _nt4(byte: UInt8) -> Int:
    var c = byte | 0x20
    if c == UInt8(ord("a")):
        return 0
    if c == UInt8(ord("c")):
        return 1
    if c == UInt8(ord("g")):
        return 2
    if c == UInt8(ord("t")):
        return 3
    return 4


@always_inline
def _hash64(key_in: UInt64, mask: UInt64) -> UInt64:
    """`hash64` of `sketch.c`, invertible within `mask`."""
    var key = key_in
    key = (~key + (key << 21)) & mask
    key = key ^ (key >> 24)
    key = ((key + (key << 3)) + (key << 8)) & mask
    key = key ^ (key >> 14)
    key = ((key + (key << 2)) + (key << 4)) & mask
    key = key ^ (key >> 28)
    key = (key + (key << 31)) & mask
    return key


def sketch(seq: Span[UInt8, _], w: Int, k: Int) -> List[Tuple[UInt64, UInt64]]:
    """`mm_sketch` without homopolymer compression: each minimizer's `hash << 8 | span` and `last << 1 | strand`."""
    var out = List[Tuple[UInt64, UInt64]]()
    var shift1 = UInt64(2 * (k - 1))
    var mask = (UInt64(1) << UInt64(2 * k)) - 1
    var kmer0 = UInt64(0)
    var kmer1 = UInt64(0)
    var buf = List[Tuple[UInt64, UInt64]](length=w, fill=(UInt64.MAX, UInt64.MAX))
    var best = (UInt64.MAX, UInt64.MAX)
    var l = 0
    var buf_pos = 0
    var min_pos = 0
    for i in range(len(seq)):
        var c = _nt4(seq[i])
        var info = (UInt64.MAX, UInt64.MAX)
        if c < 4:
            var kmer_span = min(l + 1, k)
            kmer0 = ((kmer0 << 2) | UInt64(c)) & mask
            kmer1 = (kmer1 >> 2) | (UInt64(3 ^ c) << shift1)
            if kmer0 == kmer1:
                continue
            var z = 0 if kmer0 < kmer1 else 1
            l += 1
            if l >= k and kmer_span < 256:
                info = (
                    _hash64(kmer0 if z == 0 else kmer1, mask) << 8 | UInt64(kmer_span),
                    UInt64(i) << 1 | UInt64(z),
                )
        else:
            l = 0
        buf[buf_pos] = info
        if l == w + k - 1 and best[0] != UInt64.MAX:
            for j in range(buf_pos + 1, w):
                if best[0] == buf[j][0] and buf[j][1] != best[1]:
                    out.append(buf[j])
            for j in range(buf_pos):
                if best[0] == buf[j][0] and buf[j][1] != best[1]:
                    out.append(buf[j])
        if info[0] <= best[0]:
            if l >= w + k and best[0] != UInt64.MAX:
                out.append(best)
            best = info
            min_pos = buf_pos
        elif buf_pos == min_pos:
            if l >= w + k - 1 and best[0] != UInt64.MAX:
                out.append(best)
            best = (UInt64.MAX, best[1])
            for j in range(buf_pos + 1, w):
                if best[0] >= buf[j][0]:
                    best = buf[j]
                    min_pos = j
            for j in range(buf_pos + 1):
                if best[0] >= buf[j][0]:
                    best = buf[j]
                    min_pos = j
            if l >= w + k - 1 and best[0] != UInt64.MAX:
                for j in range(buf_pos + 1, w):
                    if best[0] == buf[j][0] and best[1] != buf[j][1]:
                        out.append(buf[j])
                for j in range(buf_pos + 1):
                    if best[0] == buf[j][0] and best[1] != buf[j][1]:
                        out.append(buf[j])
        buf_pos += 1
        if buf_pos == w:
            buf_pos = 0
    if best[0] != UInt64.MAX:
        out.append(best)
    return out^


struct Mm2Index(Movable):
    """A reference's minimizers: each hash's `position << 1 | strand`, by position, and the occurrence cutoff."""

    var w: Int
    var k: Int
    var table: Dict[UInt64, List[UInt64]]
    var mid_occ: Int

    def __init__(out self, reference: Span[UInt8, _], w: Int, k: Int, min_mid_occ: Int = 50, max_mid_occ: Int = 500):
        self.w = w
        self.k = k
        self.table = Dict[UInt64, List[UInt64]]()
        for entry in sketch(reference, w, k):
            var h = entry[0] >> 8
            if h in self.table:
                try:
                    self.table[h].append(entry[1])
                except:
                    pass
            else:
                self.table[h] = [entry[1]]
        var counts = List[Int]()
        var keys = List[UInt64]()
        for key in self.table.keys():
            keys.append(key)
        for key in keys:
            try:
                ref positions = self.table[key]
                sort(Span(positions))
                counts.append(len(positions))
            except:
                pass
        sort(Span(counts))
        # mm_idx_cal_max_occ at 2e-4: the count at that quantile, plus one.
        var n = len(counts)
        var occ = counts[Int((1.0 - Float64(Float32(2e-4))) * Float64(n))] + 1 if n > 0 else Int(Int32.MAX)
        if occ < min_mid_occ:
            occ = min_mid_occ
        if max_mid_occ > min_mid_occ and occ > max_mid_occ:
            occ = max_mid_occ
        self.mid_occ = occ


struct _Match(Copyable, Movable):
    var q_pos: UInt64
    var q_span: Int
    var hits: List[UInt64]
    var tandem: Bool
    var flt: Bool

    def __init__(out self, q_pos: UInt64, q_span: Int, var hits: List[UInt64], tandem: Bool):
        self.q_pos = q_pos
        self.q_span = q_span
        self.hits = hits^
        self.tandem = tandem
        self.flt = False


def mm2_seeds(
    index: Mm2Index,
    query: Span[UInt8, _],
    q_occ_frac: Float32 = 0.01,
    max_max_occ: Int = 4095,
    occ_dist: Int = 500,
) -> Tuple[List[Seed], Int]:
    """The query's seeds as minimap2 collects them (`collect_seed_hits`), and the letters under minimizers left out
    as too frequent (`rep_len`)."""
    var qlen = len(query)
    var mv = sketch(query, index.w, index.k)
    var max_occ = index.mid_occ
    # mm_seed_mz_flt: minimizers frequent in the query itself.
    if len(mv) > max_occ and q_occ_frac > 0.0:
        var counts = Dict[UInt64, Int]()
        for entry in mv:
            counts[entry[0]] = counts.get(entry[0], 0) + 1
        var kept = List[Tuple[UInt64, UInt64]]()
        for entry in mv:
            var cnt = counts.get(entry[0], 0)
            if not (cnt > max_occ and Float32(cnt) > Float32(len(mv)) * q_occ_frac):
                kept.append(entry)
        mv = kept^
    # mm_seed_collect_all.
    var m = List[_Match]()
    for i in range(len(mv)):
        var h = mv[i][0] >> 8
        var hits = index.table.get(h, List[UInt64]())
        if len(hits) == 0:
            continue
        var tandem = (i > 0 and mv[i - 1][0] >> 8 == h) or (i < len(mv) - 1 and mv[i + 1][0] >> 8 == h)
        m.append(_Match(mv[i][1], Int(mv[i][0] & 0xFF), hits^, tandem))
    # mm_seed_select: in a streak of frequent minimizers, the least frequent kept, one every occ_dist letters.
    var n = len(m)
    if occ_dist > 0 and max_max_occ > max_occ:
        var any_high = False
        for i in range(n):
            if len(m[i].hits) > max_occ:
                any_high = True
        if any_high and n > 1:
            var last0 = -1
            for i in range(n + 1):
                if i == n or len(m[i].hits) <= max_occ:
                    if i - last0 > 1:
                        var ps = 0 if last0 < 0 else Int(m[last0].q_pos >> 1)
                        var pe = qlen if i == n else Int(m[i].q_pos >> 1)
                        var st = last0 + 1
                        var en = i
                        var max_high_occ = Int(Float64(pe - ps) / Float64(occ_dist) + 0.499)
                        if max_high_occ > 0:
                            max_high_occ = min(max_high_occ, 128)
                            # A max-heap of (occurrences, index): a later one replaces the top when rarer.
                            var chosen = List[Tuple[Int, Int]]()
                            var j = st
                            while j < en and len(chosen) < max_high_occ:
                                chosen.append((len(m[j].hits), j))
                                j += 1
                            while j < en:
                                var top = 0
                                for c in range(1, len(chosen)):
                                    if chosen[c][0] > chosen[top][0] or (
                                        chosen[c][0] == chosen[top][0] and chosen[c][1] > chosen[top][1]
                                    ):
                                        top = c
                                if len(m[j].hits) < chosen[top][0]:
                                    chosen[top] = (len(m[j].hits), j)
                                j += 1
                            for c in chosen:
                                m[c[1]].flt = True
                        for j in range(st, en):
                            m[j].flt = not m[j].flt
                        for j in range(st, en):
                            if len(m[j].hits) > max_max_occ:
                                m[j].flt = True
                    last0 = i
    else:
        for i in range(n):
            if len(m[i].hits) > max_occ:
                m[i].flt = True
    var seeds = List[Seed]()
    var rep_len = 0
    var rep_st = 0
    var rep_en = 0
    for i in range(n):
        ref q = m[i]
        if q.flt:
            var en = Int(q.q_pos >> 1) + 1
            var st = en - q.q_span
            if st > rep_en:
                rep_len += rep_en - rep_st
                rep_st = st
                rep_en = en
            else:
                rep_en = en
            continue
        var qend = Int(q.q_pos >> 1)
        for r in q.hits:
            var rpos = Int(r >> 1)
            var reverse = (r & 1) != (q.q_pos & 1)
            seeds.append(Seed(rpos - q.q_span + 1, qend - q.q_span + 1, q.q_span, reverse, q.tandem))
    rep_len += rep_en - rep_st
    return (seeds^, rep_len)
