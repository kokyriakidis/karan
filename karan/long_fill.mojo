# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the
# MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
minimap2's base-level alignment of a chain (`align.c`, `ksw2_ll_sse.c`; MIT, Heng Li), ported for `long_align`: bad
ends and seeds left out, the ends extended and the stretches between anchors filled by `karan.ksw2.ksw_extd2`, the
Z-drop tested on each fill and the chain split where it drops, an inversion between two pieces aligned, and each CIGAR
fixed and scored. Internal: the bodies keep the C's shape and names so that each function reads against its namesake.
"""

from .chaining import SEED_IGNORE, SEED_LONG_JOIN, SEED_TANDEM, Packed, mg_log2
from .ksw2 import KSW_EZ_APPROX_MAX, KSW_EZ_EXTZ_ONLY, KSW_EZ_REV_CIGAR, KSW_EZ_RIGHT, KswBuffers, KswResult, ksw_extd2
from .long_regions import PARENT_TMP_PRI, PARENT_UNSET, Region, split_reg
from .long_types import CIGAR_DELETION, CIGAR_INSERTION, CIGAR_MATCH, LongPreset

comptime KSW_L = 16
"""The vector width `ksw_extd2` follows: minimap2's SSE build, so its CIGARs byte for byte."""
comptime KSW_W = 64
"""Cells `ksw_extd2` computes an instruction (AVX-512's 64 bytes), the same results as 16 at a time."""


def ksw_ll(
    query: Span[UInt8, _], target: Span[UInt8, _], mat: List[Int8], gapo: Int, gape: Int
) -> Tuple[Int, Int, Int]:
    """`ksw_ll_i16` (local, gap of `k` costs `gapo + k gape`): the best local score, its query end and target end,
    the ends chosen as the C's striped vectors choose them (the last target column reaching it; within it the last
    query position in striped order, padding letters included)."""
    var qlen = len(query)
    var slen = (qlen + 7) // 8
    var qlen8 = slen * 8
    var h0 = List[Int](length=qlen8, fill=0)
    var h1 = List[Int](length=qlen8, fill=0)
    var e = List[Int](length=qlen8, fill=0)
    var hmax = List[Int](length=qlen8, fill=0)
    var gmax = 0
    var te = -1
    var gapoe = gapo + gape
    for i in range(len(target)):
        var row_base = Int(target[i]) * 5
        var diag = 0
        var f = 0
        var imax = 0
        for row in range(qlen8):
            var s = Int(mat[row_base + Int(query[row])]) if row < qlen else 0
            var h = max(diag + s, e[row], f)
            diag = h0[row]
            h1[row] = h
            imax = max(imax, h)
            var ho = max(h - gapoe, 0)
            e[row] = max(max(e[row] - gape, 0), ho)
            f = max(max(f - gape, 0), ho)
        if imax >= gmax:
            gmax = imax
            te = i
            for row in range(qlen8):
                hmax[row] = h1[row]
        var swap = h0^
        h0 = h1^
        h1 = swap^
    var qe = -1
    var best_key = -1
    for row in range(qlen8):
        if hmax[row] == gmax:
            var key = (row % slen) * 8 + row // slen
            if key > best_key:
                best_key = key
                qe = row
    return (gmax, qe, te)


def gen_mat(a: Int, b: Int, ambi: Int) -> List[Int8]:
    """`ksw_gen_ts_mat` with no transition score: `a` on the diagonal, `-b` off it, `-ambi` against a wildcard."""
    var mat = List[Int8](length=25, fill=0)
    for i in range(4):
        for j in range(4):
            mat[i * 5 + j] = Int8(a) if i == j else Int8(-b)
        mat[i * 5 + 4] = Int8(-ambi)
    for j in range(5):
        mat[20 + j] = Int8(-ambi)
    return mat^


def append_cigar(mut r: Region, cigar: List[UInt32]):
    """`mm_append_cigar`: runs joined where the ops meet."""
    if len(cigar) == 0:
        return
    r.has_p = True
    var start = 0
    if len(r.cigar) > 0 and (r.cigar[len(r.cigar) - 1] & 0xF) == (cigar[0] & 0xF):
        r.cigar[len(r.cigar) - 1] += (cigar[0] >> 4) << 4
        start = 1
    for index in range(start, len(cigar)):
        r.cigar.append(cigar[index])


def fix_cigar(mut r: Region, qseq: Span[UInt8, _], tseq: Span[UInt8, _]) -> Tuple[Int, Int]:
    """`mm_fix_cigar`: indels shifted left through equal letters, runs of insertions and deletions merged into one
    of each, empty runs dropped, a leading indel trimmed off the ends. The query and reference shifts of the last."""
    ref cigar = r.cigar
    if len(cigar) <= 1:
        return (0, 0)
    var toff = 0
    var qoff = 0
    var to_shrink = False
    for k in range(len(cigar)):
        var op = Int(cigar[k] & 0xF)
        var length = Int(cigar[k] >> 4)
        if length == 0:
            to_shrink = True
        if op == CIGAR_MATCH:
            toff += length
            qoff += length
        elif op == CIGAR_INSERTION or op == CIGAR_DELETION:
            if k > 0 and k < len(cigar) - 1 and (cigar[k - 1] & 0xF) == 0 and (cigar[k + 1] & 0xF) == 0:
                var prev_len = Int(cigar[k - 1] >> 4)
                var l = 0
                if op == CIGAR_INSERTION:
                    while l < prev_len:
                        if qseq[qoff - 1 - l] != qseq[qoff + length - 1 - l]:
                            break
                        l += 1
                else:
                    while l < prev_len:
                        if tseq[toff - 1 - l] != tseq[toff + length - 1 - l]:
                            break
                        l += 1
                if l > 0:
                    cigar[k - 1] -= UInt32(l << 4)
                    cigar[k + 1] += UInt32(l << 4)
                    qoff -= l
                    toff -= l
                if l == prev_len:
                    to_shrink = True
            if op == CIGAR_INSERTION:
                qoff += length
            else:
                toff += length
    var k = 0
    while k < len(cigar) - 2:
        var op = Int(cigar[k] & 0xF)
        if op > 0 and op + Int(cigar[k + 1] & 0xF) == 3:
            var s1 = 0
            var s2 = 0
            var l = k
            while l < len(cigar):
                var opl = Int(cigar[l] & 0xF)
                if opl == CIGAR_INSERTION or opl == CIGAR_DELETION or (cigar[l] >> 4) == 0:
                    if opl == CIGAR_INSERTION:
                        s1 += Int(cigar[l] >> 4)
                    elif opl == CIGAR_DELETION:
                        s2 += Int(cigar[l] >> 4)
                    l += 1
                else:
                    break
            if s1 > 0 and s2 > 0 and l - k > 2:
                cigar[k] = UInt32(s1 << 4 | CIGAR_INSERTION)
                cigar[k + 1] = UInt32(s2 << 4 | CIGAR_DELETION)
                k += 2
                while k < l:
                    cigar[k] &= 0xF
                    k += 1
                to_shrink = True
            k = l
        k += 1
    if to_shrink:
        var squeezed = List[UInt32](capacity=len(cigar))
        for run in cigar:
            if (run >> 4) != 0:
                squeezed.append(run)
        var merged = List[UInt32](capacity=len(squeezed))
        for index in range(len(squeezed)):
            if index == len(squeezed) - 1 or (squeezed[index] & 0xF) != (squeezed[index + 1] & 0xF):
                merged.append(squeezed[index])
            else:
                squeezed[index + 1] += (squeezed[index] >> 4) << 4
        cigar = merged^
    var qshift = 0
    var tshift = 0
    var first_op = Int(cigar[0] & 0xF)
    if first_op == CIGAR_INSERTION or first_op == CIGAR_DELETION:
        var l = Int(cigar[0] >> 4)
        if first_op == CIGAR_INSERTION:
            if r.rev:
                r.qe -= l
            else:
                r.qs += l
            qshift = l
        else:
            r.rs += l
            tshift = l
        var rest = List[UInt32](capacity=len(cigar) - 1)
        for index in range(1, len(cigar)):
            rest.append(cigar[index])
        cigar = rest^
    return (qshift, tshift)


def update_extra(
    mut r: Region, qseq_in: Span[UInt8, _], tseq_in: Span[UInt8, _], mat: List[Int8], q: Int, e: Int, log_gap: Bool
):
    """`mm_update_extra`: the CIGAR fixed, then its letters counted and its best local score with log gaps."""
    if not r.has_p:
        return
    var shifts = fix_cigar(r, qseq_in, tseq_in)
    var qseq = qseq_in[shifts[0] :]
    var tseq = tseq_in[shifts[1] :]
    r.blen = 0
    r.mlen = 0
    var s = 0.0
    var best = 0.0
    var toff = 0
    var qoff = 0
    for run in r.cigar:
        var op = Int(run & 0xF)
        var length = Int(run >> 4)
        if op == CIGAR_MATCH:
            var n_ambi = 0
            var n_diff = 0
            for l in range(length):
                var cq = Int(qseq[qoff + l])
                var ct = Int(tseq[toff + l])
                if ct > 3 or cq > 3:
                    n_ambi += 1
                elif ct != cq:
                    n_diff += 1
                s += Float64(mat[ct * 5 + cq])
                if s < 0:
                    s = 0
                else:
                    best = max(best, s)
            r.blen += length - n_ambi
            r.mlen += length - (n_ambi + n_diff)
            r.n_ambi += n_ambi
            toff += length
            qoff += length
        elif op == CIGAR_INSERTION or op == CIGAR_DELETION:
            var n_ambi = 0
            for l in range(length):
                var c = qseq[qoff + l] if op == CIGAR_INSERTION else tseq[toff + l]
                if c > 3:
                    n_ambi += 1
            r.blen += length - n_ambi
            r.n_ambi += n_ambi
            if log_gap:
                s -= Float64(q) + Float64(e) * Float64(mg_log2(Float32(1.0 + Float64(length))))
            else:
                s -= Float64(q + e)
            if s < 0:
                s = 0
            if op == CIGAR_INSERTION:
                qoff += length
            else:
                toff += length
    r.dp_max = Int(best + 0.499)
    r.dp_max0 = r.dp_max


def collect_long_gaps(as1: Int, cnt1: Int, a: List[Packed], min_gap: Int) -> List[Int]:
    """`collect_long_gaps`: the anchors after a gap difference above `min_gap`, none unless two or more."""
    var out = List[Int]()
    for i in range(1, cnt1):
        var gap = (Int(a[as1 + i].query_end()) - Int(a[as1 + i - 1].query_end())) - (
            Int(a[as1 + i].reference_end()) - Int(a[as1 + i - 1].reference_end())
        )
        if gap < -min_gap or gap > min_gap:
            out.append(i)
    if len(out) <= 1:
        out.clear()
    return out^


@always_inline
def gap_of(a: List[Packed], i: Int) -> Int:
    """The gap difference (query less reference) before anchor `i`."""
    return (Int(a[i].query_end()) - Int(a[i - 1].query_end())) - Int((a[i].x - a[i - 1].x).cast[DType.int32]())


def filter_bad_seeds(
    as1: Int, cnt1: Int, mut a: List[Packed], min_gap: Int, diff_thres: Int, max_ext_len: Int, max_ext_cnt: Int
):
    """`mm_filter_bad_seeds`: anchors inside a short stretch of insertions and deletions that largely cancel out
    marked to be stepped over."""
    var gaps = collect_long_gaps(as1, cnt1, a, min_gap)
    var n = len(gaps)
    if n == 0:
        return
    var best = 0
    var max_st = -1
    var max_en = -1
    var k = 0
    while True:
        if k == n or k >= max_en:
            if max_en > 0:
                for i in range(gaps[max_st], gaps[max_en]):
                    a[as1 + i].y |= SEED_IGNORE
            best = 0
            max_st = -1
            max_en = -1
            if k == n:
                break
        var i = gaps[k]
        var n_ins = 0
        var n_del = 0
        var gap = gap_of(a, as1 + i)
        if gap > 0:
            n_ins += gap
        else:
            n_del += -gap
        var qs = Int(a[as1 + i - 1].query_end())
        var rs = Int(a[as1 + i - 1].reference_end())
        var max_diff = 0
        var max_diff_l = -1
        var l = k + 1
        while l < n and l <= k + max_ext_cnt:
            var j = gaps[l]
            if Int(a[as1 + j].query_end()) - qs > max_ext_len or Int(a[as1 + j].reference_end()) - rs > max_ext_len:
                break
            gap = gap_of(a, as1 + j)
            if gap > 0:
                n_ins += gap
            else:
                n_del += -gap
            var diff = n_ins + n_del - abs(n_ins - n_del)
            if max_diff < diff:
                max_diff = diff
                max_diff_l = l
            l += 1
        if max_diff > diff_thres and max_diff > best:
            best = max_diff
            max_st = k
            max_en = max_diff_l
        k += 1


def filter_bad_seeds_alt(as1: Int, cnt1: Int, mut a: List[Packed], min_gap: Int, max_ext: Int):
    """`mm_filter_bad_seeds_alt`: consecutive large gaps closer than their sizes marked over and joined, the last
    anchor flagged as a long join."""
    var gaps = collect_long_gaps(as1, cnt1, a, min_gap)
    var n = len(gaps)
    if n == 0:
        return
    var k = 0
    while k < n:
        var i = gaps[k]
        var gap1 = abs(gap_of(a, as1 + i))
        var re1 = Int(a[as1 + i].reference_end())
        var qe1 = Int(a[as1 + i].query_end())
        var l = k + 1
        while l < n:
            var j = gaps[l]
            if Int(a[as1 + j].query_end()) - qe1 > max_ext or Int(a[as1 + j].reference_end()) - re1 > max_ext:
                break
            var gap2 = abs(gap_of(a, as1 + j))
            var q_span_pre = Int(a[as1 + j - 1].span())
            var rs2 = Int(a[as1 + j - 1].reference_end()) + q_span_pre
            var qs2 = Int(a[as1 + j - 1].query_end()) + q_span_pre
            var m = min(rs2 - re1, qs2 - qe1)
            if m > gap1 + gap2:
                break
            re1 = Int(a[as1 + j].reference_end())
            qe1 = Int(a[as1 + j].query_end())
            gap1 = gap2
            l += 1
        if l > k + 1:
            var end = gaps[l - 1]
            for j in range(gaps[k], end):
                a[as1 + j].y |= SEED_IGNORE
            a[as1 + end].y |= SEED_LONG_JOIN
        k = l


def fix_bad_ends(r: Region, a: List[Packed], bw: Int, min_match: Int) -> Tuple[Int, Int]:
    """`mm_fix_bad_ends`: a chain's ends trimmed back past anchors whose gap is large against what precedes it."""
    var as1 = r.as_
    var cnt1 = r.cnt
    if r.cnt < 3:
        return (as1, cnt1)
    var l = Int(a[r.as_].span())
    var m = l
    for i in range(r.as_ + 1, r.as_ + r.cnt - 1):
        var q_span = Int(a[i].span())
        if (a[i].y & SEED_LONG_JOIN) != 0:
            break
        var lr = Int(a[i].reference_end()) - Int(a[i - 1].reference_end())
        var lq = Int(a[i].query_end()) - Int(a[i - 1].query_end())
        var lo = min(lr, lq)
        var hi = max(lr, lq)
        if hi - lo > l >> 1:
            as1 = i
        l += lo
        m += min(lo, q_span)
        if l >= bw << 1 or (m >= min_match and m >= bw) or m >= r.mlen >> 1:
            break
    cnt1 = r.as_ + r.cnt - as1
    l = Int(a[r.as_ + r.cnt - 1].span())
    m = l
    var i = r.as_ + r.cnt - 2
    while i > as1:
        var q_span = Int(a[i + 1].span())
        if (a[i + 1].y & SEED_LONG_JOIN) != 0:
            break
        var lr = Int(a[i + 1].reference_end()) - Int(a[i].reference_end())
        var lq = Int(a[i + 1].query_end()) - Int(a[i].query_end())
        var lo = min(lr, lq)
        var hi = max(lr, lq)
        if hi - lo > l >> 1:
            cnt1 = i + 1 - as1
        l += lo
        m += min(lo, q_span)
        if l >= bw << 1 or (m >= min_match and m >= bw) or m >= r.mlen >> 1:
            break
        i -= 1
    return (as1, cnt1)


@always_inline
def adjust_minier(anchor: Packed) -> Tuple[Int, Int]:
    """`mm_adjust_minier`: an anchor's reference and query point, the middle of its match."""
    var half = Int(anchor.span()) >> 1
    return (Int(anchor.reference_end()) - half, Int(anchor.query_end()) - half)


def align_pair(
    preset: LongPreset,
    query: Span[UInt8, _],
    target: Span[UInt8, _],
    mat: List[Int8],
    w: Int,
    end_bonus: Int,
    zdrop: Int,
    flag: Int,
    mut buf: KswBuffers,
    mut ez: KswResult,
):
    """`mm_align_pair` for genomic alignment: `ksw_extd2`, or a Z-drop with no CIGAR when the matrix would pass
    `max_sw_mat`."""
    if preset.max_sw_mat > 0 and len(target) * len(query) > preset.max_sw_mat:
        ez.reset()
        ez.zdropped = True
        return
    ksw_extd2[KSW_L, KSW_W](
        query,
        target,
        5,
        Span(mat),
        preset.gap_open,
        preset.gap_extend,
        preset.long_gap_open,
        preset.long_gap_extend,
        w,
        zdrop,
        end_bonus,
        flag,
        buf,
        ez,
    )


struct Drop:
    """`update_max_zdrop`'s state: the best score so far and where, and the largest drop below it, less the gap
    extension of the diagonal shift, and its two ends."""

    var best: Int
    var max_i: Int
    var max_j: Int
    var max_zdrop: Int
    var t0: Int
    var t1: Int
    var q0: Int
    var q1: Int

    def __init__(out self):
        self.best = Int(Int32.MIN)
        self.max_i = -1
        self.max_j = -1
        self.max_zdrop = 0
        self.t0 = -1
        self.t1 = -1
        self.q0 = -1
        self.q1 = -1

    @always_inline
    def update(mut self, score: Int, i: Int, j: Int, e: Int):
        if score < self.best:
            var li = i - self.max_i
            var lj = j - self.max_j
            var diff = li - lj if li > lj else lj - li
            var z = self.best - score - diff * e
            if z > self.max_zdrop:
                self.max_zdrop = z
                self.t0 = self.max_i
                self.t1 = i
                self.q0 = self.max_j
                self.q1 = j
        else:
            self.best = score
            self.max_i = i
            self.max_j = j


def test_zdrop(
    preset: LongPreset, qseq: Span[UInt8, _], tseq: Span[UInt8, _], cigar: List[UInt32], mat: List[Int8]
) -> Int:
    """`mm_test_zdrop` on a fill's CIGAR: 2 where its worst drop holds an inversion, 1 where it passes the Z-drop,
    else 0."""
    var drop = Drop()
    var score = 0
    var i = 0
    var j = 0
    for run in cigar:
        var op = Int(run & 0xF)
        var length = Int(run >> 4)
        if op == CIGAR_MATCH:
            for l in range(length):
                score += Int(mat[Int(tseq[i + l]) * 5 + Int(qseq[j + l])])
                drop.update(score, i + l, j + l, preset.gap_extend)
            i += length
            j += length
        elif op == CIGAR_INSERTION or op == CIGAR_DELETION:
            score -= preset.gap_open + preset.gap_extend * length
            if op == CIGAR_INSERTION:
                j += length
            else:
                i += length
            drop.update(score, i, j, preset.gap_extend)
    var q_len = drop.q1 - drop.q0
    var t_len = drop.t1 - drop.t0
    if drop.max_zdrop > preset.zdrop_inv and q_len < preset.max_gap and t_len < preset.max_gap:
        var qseq2 = List[UInt8](capacity=q_len)
        for index in range(q_len):
            var c = qseq[drop.q1 - index - 1]
            qseq2.append(4 if c >= 4 else 3 - c)
        var local = ksw_ll(Span(qseq2), tseq[drop.t0 : drop.t0 + t_len], mat, preset.gap_open, preset.gap_extend)
        if local[0] >= preset.min_chain_score * preset.match_score and local[0] >= preset.min_dp_max:
            return 2
    return 1 if drop.max_zdrop > preset.zdrop else 0


def align1(
    mut buf: KswBuffers,
    mut ez: KswResult,
    preset: LongPreset,
    reference: Span[UInt8, _],
    qseq0: Span[UInt8, _],
    qlen: Int,
    mut r: Region,
    mut a: List[Packed],
    mat: List[Int8],
) -> Region:
    """`mm_align1`, not spliced: the region aligned, and the region split off it where a fill Z-dropped (no
    anchors when none)."""
    var r2 = Region()
    if r.cnt == 0:
        return r2^
    var ref_len = len(reference)
    var rev = a[r.as_].reverse()
    # The query strand that matches: the forward strand, or its reverse complement after it.
    var qseq_all = qseq0[qlen:] if rev else qseq0[:qlen]
    var bw = Int(Float64(preset.bw) * 1.5 + 1.0)
    var bw_long = max(Int(Float64(preset.bw_long) * 1.5 + 1.0), bw)
    var as1 = r.as_
    var cnt1 = r.cnt
    if preset.end_filter:
        var ends = fix_bad_ends(r, a, preset.bw, preset.min_chain_score * 2)
        as1 = ends[0]
        cnt1 = ends[1]
    filter_bad_seeds(as1, cnt1, a, 10, 40, preset.max_gap >> 1, 10)
    filter_bad_seeds_alt(as1, cnt1, a, 30, preset.max_gap >> 1)
    var start = adjust_minier(a[as1])
    var rs = start[0]
    var qs = start[1]
    var end = adjust_minier(a[as1 + cnt1 - 1])
    var re = end[0]
    var qe = end[1]

    # How far the end extensions may reach.
    var rs0 = Int(a[r.as_].reference_end()) + 1 - Int(a[r.as_].span())
    var qs0 = Int(a[r.as_].query_end()) + 1 - Int(a[r.as_].span())
    if rs0 < 0:
        rs0 = 0
    var rs1 = 0
    var qs1 = 0
    var l = 0
    var i = r.as_ - 1
    while i >= 0 and a[i].x >> 32 == a[r.as_].x >> 32:
        var x = Int(a[i].reference_end()) + 1 - Int(a[i].span())
        var y = Int(a[i].query_end()) + 1 - Int(a[i].span())
        if x < rs0 and y < qs0:
            l += 1
            if l > preset.min_cnt:
                l = max(rs0 - x, qs0 - y)
                rs1 = max(rs0 - l, 0)
                qs1 = qs0 - l
                break
        i -= 1
    if qs > 0 and rs > 0:
        l = min(qs, preset.max_gap)
        qs1 = max(qs1, qs - l)
        qs0 = min(qs0, qs1)
        if l * preset.match_score > preset.gap_open:
            l += (l * preset.match_score - preset.gap_open) // preset.gap_extend
        l = min(l, preset.max_gap)
        l = min(l, rs)
        rs1 = max(rs1, rs - l)
        rs0 = min(rs0, rs1)
        rs0 = min(rs0, rs)
    else:
        rs0 = rs
        qs0 = qs
    var re0 = Int(a[r.as_ + r.cnt - 1].reference_end()) + 1
    var qe0 = Int(a[r.as_ + r.cnt - 1].query_end()) + 1
    var re1 = ref_len
    var qe1 = qlen
    l = 0
    i = r.as_ + r.cnt
    while i < len(a) and a[i].x >> 32 == a[r.as_].x >> 32:
        var x = Int(a[i].reference_end()) + 1
        var y = Int(a[i].query_end()) + 1
        if x > re0 and y > qe0:
            l += 1
            if l > preset.min_cnt:
                l = max(x - re0, y - qe0)
                re1 = re0 + l
                qe1 = qe0 + l
                break
        i += 1
    if qe < qlen and re < ref_len:
        l = min(qlen - qe, preset.max_gap)
        qe1 = min(qe1, qe + l)
        qe0 = max(qe0, qe1)
        if l * preset.match_score > preset.gap_open:
            l += (l * preset.match_score - preset.gap_open) // preset.gap_extend
        l = min(l, preset.max_gap)
        l = min(l, ref_len - re)
        re1 = min(re1, re + l)
        re0 = max(re0, re1)
    else:
        re0 = re
        qe0 = qe

    if qs > 0 and rs > 0:
        # The left extension, both pieces reversed, gaps as late as the score allows (early once turned back).
        var qrev = List[UInt8](capacity=qs - qs0)
        for index in range(qs - 1, qs0 - 1, -1):
            qrev.append(qseq_all[index])
        var trev = List[UInt8](capacity=rs - rs0)
        for index in range(rs - 1, rs0 - 1, -1):
            trev.append(reference[index])
        align_pair(
            preset,
            Span(qrev),
            Span(trev),
            mat,
            bw,
            preset.end_bonus,
            preset.zdrop_inv if r.split_inv else preset.zdrop,
            KSW_EZ_EXTZ_ONLY | KSW_EZ_RIGHT | KSW_EZ_REV_CIGAR,
            buf,
            ez,
        )
        if len(ez.cigar) > 0:
            append_cigar(r, ez.cigar)
            r.dp_score += ez.max
        rs1 = rs - (ez.mqe_t + 1 if ez.reach_end else ez.max_t + 1)
        qs1 = qs - (qs - qs0 if ez.reach_end else ez.max_q + 1)
    else:
        rs1 = rs
        qs1 = qs
    re1 = rs
    qe1 = qs

    var dropped = False
    for k in range(1, cnt1):
        var flags = a[as1 + k].y
        if (flags & (SEED_IGNORE | SEED_TANDEM)) != 0 and k != cnt1 - 1:
            continue
        var point = adjust_minier(a[as1 + k])
        re = point[0]
        qe = point[1]
        re1 = re
        qe1 = qe
        var long_join = (flags & SEED_LONG_JOIN) != 0
        if k == cnt1 - 1 or long_join or (qe - qs >= preset.min_ksw_len and re - rs >= preset.min_ksw_len):
            var bw1 = max(qe - qs, re - rs) if long_join else bw_long
            var qpiece = qseq_all[qs:qe]
            var tpiece = reference[rs:re]
            # First with the approximate maximum; again exactly where the CIGAR drops.
            align_pair(preset, qpiece, tpiece, mat, bw1, -1, preset.zdrop, KSW_EZ_APPROX_MAX, buf, ez)
            var zdrop_code = test_zdrop(preset, qpiece, tpiece, ez.cigar, mat)
            if zdrop_code != 0:
                var zdrop = preset.zdrop_inv if zdrop_code == 2 else preset.zdrop
                align_pair(preset, qpiece, tpiece, mat, bw1, -1, zdrop, 0, buf, ez)
            if len(ez.cigar) > 0:
                append_cigar(r, ez.cigar)
            r.has_p = True
            if ez.zdropped:
                var j = k - 1
                while j >= 0:
                    if Int(a[as1 + j].reference_end()) <= rs + ez.max_t:
                        break
                    j -= 1
                dropped = True
                if j < 0:
                    j = 0
                r.dp_score += ez.max
                re1 = rs + (ez.max_t + 1)
                qe1 = qs + (ez.max_q + 1)
                if cnt1 - (j + 1) >= preset.min_cnt:
                    r2 = split_reg(r, as1 + j + 1 - r.as_, qlen, a)
                    if zdrop_code == 2:
                        r2.split_inv = True
                break
            r.dp_score += ez.score
            rs = re
            qs = qe

    if not dropped and qe < qe0 and re < re0:
        align_pair(
            preset,
            qseq_all[qe:qe0],
            reference[re:re0],
            mat,
            bw,
            preset.end_bonus,
            preset.zdrop,
            KSW_EZ_EXTZ_ONLY,
            buf,
            ez,
        )
        if len(ez.cigar) > 0:
            append_cigar(r, ez.cigar)
            r.dp_score += ez.max
        re1 = re + (ez.mqe_t + 1 if ez.reach_end else ez.max_t + 1)
        qe1 = qe + (qe0 - qe if ez.reach_end else ez.max_q + 1)

    r.rs = rs1
    r.re = re1
    if not rev:
        r.qs = qs1
        r.qe = qe1
    else:
        r.qs = qlen - qe1
        r.qe = qlen - qs1
    if r.has_p:
        update_extra(r, qseq_all[qs1:], reference[rs1:re1], mat, preset.gap_open, preset.gap_extend, True)
    return r2^


def align1_inv(
    mut buf: KswBuffers,
    mut ez: KswResult,
    preset: LongPreset,
    reference: Span[UInt8, _],
    qseq0: Span[UInt8, _],
    qlen: Int,
    r1: Region,
    r2: Region,
    mat: List[Int8],
) -> Optional[Region]:
    """`mm_align1_inv`: between two pieces of a chain split at an inversion, the other strand's best local
    alignment, extended from its far end."""
    if (r1.split & 1) == 0 or (r2.split & 2) == 0:
        return None
    if r1.id != r1.parent and r1.parent != PARENT_TMP_PRI:
        return None
    if r2.id != r2.parent and r2.parent != PARENT_TMP_PRI:
        return None
    if r1.rev != r2.rev:
        return None
    var ql = r1.qs - r2.qe if r1.rev else r2.qs - r1.qe
    var tl = r2.rs - r1.re
    if ql < preset.min_chain_score or ql > preset.max_gap:
        return None
    if tl < preset.min_chain_score or tl > preset.max_gap:
        return None
    # The query between the pieces on the strand opposite theirs, and the reference between them.
    var qseq = qseq0[r2.qe : r2.qe + ql] if r1.rev else qseq0[qlen + qlen - r2.qs : qlen + qlen - r2.qs + ql]
    var tseq = reference[r1.re : r1.re + tl]
    var qrev = List[UInt8](capacity=ql)
    for index in range(ql - 1, -1, -1):
        qrev.append(qseq[index])
    var trev = List[UInt8](capacity=tl)
    for index in range(tl - 1, -1, -1):
        trev.append(tseq[index])
    var local = ksw_ll(Span(qrev), Span(trev), mat, preset.gap_open, preset.gap_extend)
    if local[0] < preset.min_dp_max:
        return None
    var q_off = ql - (local[1] + 1)
    var t_off = tl - (local[2] + 1)
    align_pair(
        preset,
        qseq[q_off:],
        tseq[t_off:],
        mat,
        Int(Float64(preset.bw) * 1.5),
        -1,
        preset.zdrop,
        KSW_EZ_EXTZ_ONLY,
        buf,
        ez,
    )
    if len(ez.cigar) == 0:
        return None
    var r = Region()
    append_cigar(r, ez.cigar)
    r.dp_score = ez.max
    r.id = -1
    r.parent = PARENT_UNSET
    r.inv = True
    r.rev = not r1.rev
    r.div = -1.0
    if not r.rev:
        r.qs = r2.qe + q_off
        r.qe = r.qs + ez.max_q + 1
    else:
        r.qe = r2.qs - q_off
        r.qs = r.qe - (ez.max_q + 1)
    r.rs = r1.re + t_off
    r.re = r.rs + ez.max_t + 1
    update_extra(r, qseq[q_off:], tseq[t_off:], mat, preset.gap_open, preset.gap_extend, True)
    return r^
