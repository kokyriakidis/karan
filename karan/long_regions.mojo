# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the
# MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
minimap2's regions (`hit.c`, `esterr.c`; MIT, Heng Li), ported for `long_align`: a chain as `mm_reg1_t` holds it, then
its alignment; regions made from chains, primaries and secondaries chosen by query overlap, divergence estimated,
regions split, filtered, re-ranked and given mapping qualities. Internal: the bodies keep the C's shape and names so
that each function reads against its namesake.
"""

from std.math import log, pow

from .chaining import ChainSet, Packed, mg_log2
from .long_types import CIGAR_DELETION, CIGAR_INSERTION, CIGAR_MATCH, LongPreset

comptime PARENT_UNSET = -1
comptime PARENT_TMP_PRI = -2


struct Region(Copyable, Movable):
    """minimap2's `mm_reg1_t` and its `mm_extra_t`: a chain, then its alignment."""

    var id: Int
    var parent: Int
    var score: Int
    var score0: Int
    var hash: UInt32
    var cnt: Int
    var as_: Int
    var rev: Bool
    var rs: Int
    var re: Int
    var qs: Int
    var qe: Int
    var mlen: Int
    var blen: Int
    var subsc: Int
    var n_sub: Int
    var split: Int
    var split_inv: Bool
    var inv: Bool
    var sam_pri: Bool
    var strand_retained: Bool
    var div: Float32
    var mapq: Int
    var has_p: Bool
    var dp_score: Int
    var dp_max: Int
    var dp_max0: Int
    var dp_max2: Int
    var n_ambi: Int
    var cigar: List[UInt32]

    def __init__(out self):
        self.id = -1
        self.parent = PARENT_UNSET
        self.score = 0
        self.score0 = 0
        self.hash = 0
        self.cnt = 0
        self.as_ = 0
        self.rev = False
        self.rs = 0
        self.re = 0
        self.qs = 0
        self.qe = 0
        self.mlen = 0
        self.blen = 0
        self.subsc = 0
        self.n_sub = 0
        self.split = 0
        self.split_inv = False
        self.inv = False
        self.sam_pri = False
        self.strand_retained = False
        self.div = -1.0
        self.mapq = 0
        self.has_p = False
        self.dp_score = 0
        self.dp_max = 0
        self.dp_max0 = 0
        self.dp_max2 = 0
        self.n_ambi = 0
        self.cigar = List[UInt32]()

    def drop_p(mut self):
        """`free(r->p)`."""
        self.has_p = False
        self.dp_score = 0
        self.dp_max = 0
        self.dp_max0 = 0
        self.dp_max2 = 0
        self.n_ambi = 0
        self.cigar.clear()


@always_inline
def hash64(key_in: UInt64) -> UInt64:
    """`hash64` of `hit.c`, Thomas Wang's 64-bit mix."""
    var key = key_in
    key = ~key + (key << 21)
    key = key ^ (key >> 24)
    key = (key + (key << 3)) + (key << 8)
    key = key ^ (key >> 14)
    key = (key + (key << 2)) + (key << 4)
    key = key ^ (key >> 28)
    key = key + (key << 31)
    return key


@always_inline
def wang32(key_in: UInt32) -> UInt32:
    """khash's `__ac_Wang_hash`."""
    var key = key_in
    key += ~(key << 15)
    key ^= key >> 10
    key += key << 3
    key ^= key >> 6
    key += ~(key << 11)
    key ^= key >> 16
    return key


def set_coor(mut r: Region, qlen: Int, a: List[Packed]):
    """`mm_reg_set_coor`: a region's extent from its anchors, and its fuzzy matched and block lengths."""
    var k = r.as_
    var q_span = Int(a[k].span())
    r.rev = a[k].reverse()
    var x0 = Int(a[k].reference_end())
    r.rs = x0 + 1 - q_span if x0 + 1 > q_span else 0
    r.re = Int(a[k + r.cnt - 1].reference_end()) + 1
    if not r.rev:
        r.qs = Int(a[k].query_end()) + 1 - q_span
        r.qe = Int(a[k + r.cnt - 1].query_end()) + 1
    else:
        r.qs = qlen - (Int(a[k + r.cnt - 1].query_end()) + 1)
        r.qe = qlen - (Int(a[k].query_end()) + 1 - q_span)
    r.mlen = 0
    r.blen = 0
    if r.cnt <= 0:
        return
    r.mlen = Int(a[r.as_].span())
    r.blen = r.mlen
    for i in range(r.as_ + 1, r.as_ + r.cnt):
        var span = Int(a[i].span())
        var tl = Int(a[i].reference_end()) - Int(a[i - 1].reference_end())
        var ql = Int(a[i].query_end()) - Int(a[i - 1].query_end())
        r.blen += max(tl, ql)
        r.mlen += span if (tl > span and ql > span) else min(tl, ql)


def gen_regs(hash: UInt32, qlen: Int, chains: ChainSet) -> List[Region]:
    """`mm_gen_regs`: one region a chain, by score, ties by the hash of the query and the chain's first anchor."""
    var z = List[Tuple[UInt64, UInt64]](capacity=len(chains.u))
    var k = 0
    for i in range(len(chains.u)):
        var h = hash64((hash64(chains.a[k].x) + hash64(chains.a[k].y)) ^ UInt64(hash)) & 0xFFFFFFFF
        var cnt = Int(chains.u[i] & 0xFFFFFFFF)
        z.append((chains.u[i] ^ h, UInt64(k) << 32 | UInt64(cnt)))
        k += cnt

    # A stable sort by x, then the order reversed: larger first, among equals the later one first.
    var order = List[Int](capacity=len(z))
    for i in range(len(z)):
        order.append(i)

    def ahead(left: Int, right: Int) {imm z} -> Bool:
        return z[left][0] > z[right][0] or (z[left][0] == z[right][0] and left > right)

    sort(Span(order), ahead)
    var regs = List[Region](capacity=len(z))
    for i in range(len(order)):
        var entry = z[order[i]]
        var r = Region()
        r.id = i
        r.parent = PARENT_UNSET
        r.score = Int(entry[0] >> 32)
        r.score0 = r.score
        r.hash = UInt32(entry[0] & 0xFFFFFFFF)
        r.cnt = Int(entry[1] & 0xFFFFFFFF)
        r.as_ = Int(entry[1] >> 32)
        r.div = -1.0
        set_coor(r, qlen, chains.a)
        regs.append(r^)
    return regs^


def set_parent(mut r: List[Region], mask_level: Float32, mask_len: Int, sub_diff: Int):
    """`mm_set_parent` (no alternate contigs, soft mask level): each region secondary to the first earlier primary it
    overlaps enough on the query, else a primary; each primary's best secondary score and count of near ones."""
    var n = len(r)
    if n <= 0:
        return
    for i in range(n):
        r[i].id = i
    var w = List[Int](capacity=n)
    var cov = List[Tuple[Int, Int]]()
    w.append(0)
    r[0].parent = 0
    for i in range(1, n):
        var si = r[i].qs
        var ei = r[i].qe
        var uncov_len = 0
        cov.clear()
        for j in range(len(w)):
            var sj = r[w[j]].qs
            var ej = r[w[j]].qe
            if ej <= si or sj >= ei:
                continue
            cov.append((max(sj, si), min(ej, ei)))
        var secondary = False
        if len(cov) > 0:
            sort(Span(cov))
            var x = si
            for c in cov:
                if c[0] > x:
                    uncov_len += c[0] - x
                x = max(c[1], x)
            if ei > x:
                uncov_len += ei - x
            for j in range(len(w)):
                var pj = w[j]
                var sj = r[pj].qs
                var ej = r[pj].qe
                if ej <= si or sj >= ei:
                    continue
                var lo = min(ej - sj, ei - si)
                var hi = max(ej - sj, ei - si)
                var ol: Int
                if si < sj:
                    ol = 0 if ei < sj else (ei - sj if ei < ej else ej - sj)
                else:
                    ol = 0 if ej < si else (ej - si if ej < ei else ei - si)
                if Float32(ol) / Float32(lo) - Float32(uncov_len) / Float32(hi) > mask_level and uncov_len <= mask_len:
                    var cnt_sub = False
                    var sci = r[i].score
                    r[i].parent = r[pj].parent
                    r[pj].subsc = max(r[pj].subsc, sci)
                    if r[i].cnt >= r[pj].cnt:
                        cnt_sub = True
                    if r[pj].has_p and r[i].has_p and (r[pj].rs != r[i].rs or r[pj].re != r[i].re or ol != lo):
                        sci = r[i].dp_max
                        r[pj].dp_max2 = max(r[pj].dp_max2, sci)
                        if r[pj].dp_max - r[i].dp_max <= sub_diff:
                            cnt_sub = True
                    if cnt_sub:
                        r[pj].n_sub += 1
                    secondary = True
                    break
        if not secondary:
            w.append(i)
            r[i].parent = i
            r[i].n_sub = 0


def set_sam_pri(mut r: List[Region]) -> Int:
    """`mm_set_sam_pri`: the first primary is SAM's primary."""
    var n_pri = 0
    for i in range(len(r)):
        if r[i].id == r[i].parent:
            n_pri += 1
            r[i].sam_pri = n_pri == 1
        else:
            r[i].sam_pri = False
    return n_pri


def sync_regs(mut r: List[Region]):
    """`mm_sync_regs`: ids renumbered after removals, parents followed."""
    var n = len(r)
    if n <= 0:
        return
    var max_id = -1
    for i in range(n):
        max_id = max(max_id, r[i].id)
    var tmp = List[Int](length=max_id + 1, fill=-1)
    for i in range(n):
        if r[i].id >= 0:
            tmp[r[i].id] = i
    for i in range(n):
        r[i].id = i
        if r[i].parent == PARENT_TMP_PRI:
            r[i].parent = i
        elif r[i].parent >= 0 and tmp[r[i].parent] >= 0:
            r[i].parent = tmp[r[i].parent]
        else:
            r[i].parent = PARENT_UNSET
    _ = set_sam_pri(r)


def select_sub(
    mut r: List[Region], pri_ratio: Float32, min_diff: Int, best_n: Int, check_strand: Bool, min_strand_sc: Int
):
    """`mm_select_sub`: secondaries kept within `pri_ratio` (or `min_diff`) of their primary, `best_n` at most, not
    identical to it; with `check_strand`, a strong one on the other strand kept as well. Compacted in place as the C
    compacts, so a parent index read after removals reads what the C reads."""
    if pri_ratio <= 0.0 or len(r) == 0:
        return
    var n = len(r)
    var k = 0
    var n_2nd = 0
    for i in range(n):
        var p = r[i].parent
        var keep = False
        if p == i or r[i].inv:
            keep = True
        elif (
            Float32(r[i].score) >= Float32(r[p].score) * pri_ratio or r[i].score + min_diff >= r[p].score
        ) and n_2nd < best_n:
            if not (r[i].qs == r[p].qs and r[i].qe == r[p].qe and r[i].rs == r[p].rs and r[i].re == r[p].re):
                keep = True
                n_2nd += 1
        elif check_strand and n_2nd < best_n and r[i].score > min_strand_sc and r[i].rev != r[p].rev:
            r[i].strand_retained = True
            keep = True
            n_2nd += 1
        if keep:
            if k < i:
                r[k] = r[i].copy()
            k += 1
    if k != n:
        while len(r) > k:
            _ = r.pop()
        sync_regs(r)


def est_err(qlen: Int, ref_len: Int, mut regs: List[Region], a: List[Packed], mini_pos: List[UInt64]):
    """`mm_est_err`: each region's divergence from the share of the query's seeded minimizers its chain holds."""
    var n = len(mini_pos)
    if n == 0:
        return
    var sum_k = 0
    for i in range(n):
        sum_k += Int((mini_pos[i] >> 32) & 0xFF)
    var avg_k = Float32(sum_k) / Float32(n)

    @always_inline
    def for_qpos(anchor: Packed) {imm qlen} -> Int:
        var x = Int(anchor.query_end())
        if anchor.reverse():
            x = qlen - 1 - (x + 1 - Int(anchor.span()))
        return x

    for i in range(len(regs)):
        regs[i].div = -1.0
        if regs[i].cnt == 0:
            continue
        var first = a[regs[i].as_ + regs[i].cnt - 1] if regs[i].rev else a[regs[i].as_]
        var x0 = for_qpos(first)
        var lo = 0
        var hi = n - 1
        var st = -1
        while lo <= hi:
            var m = (lo + hi) >> 1
            var y = Int(mini_pos[m].cast[DType.int32]())
            if y < x0:
                lo = m + 1
            elif y > x0:
                hi = m - 1
            else:
                st = m
                break
        if st < 0:
            continue
        var en = st
        var k = 1
        var n_match = 1
        var j = st + 1
        while j < n and k < regs[i].cnt:
            var anchor = a[regs[i].as_ + regs[i].cnt - 1 - k] if regs[i].rev else a[regs[i].as_ + k]
            if for_qpos(anchor) == Int(mini_pos[j].cast[DType.int32]()):
                k += 1
                en = j
                n_match += 1
            j += 1
        var n_tot = en - st + 1
        if Float32(regs[i].qs) > avg_k and Float32(regs[i].rs) > avg_k:
            n_tot += 1
        if Float32(qlen - regs[i].qs) > avg_k and Float32(ref_len - regs[i].re) > avg_k:
            n_tot += 1
        regs[i].div = Float32(0.0) if n_match >= n_tot else Float32(
            1.0 - pow(Float64(n_match) / Float64(n_tot), 1.0 / Float64(avg_k))
        )


def filter_strand_retained(mut regs: List[Region]):
    """`mm_filter_strand_retained`: a secondary kept for its strand dropped when far more divergent than its
    primary; compacted in place as the C compacts."""
    var k = 0
    var n = len(regs)
    for i in range(n):
        var p = regs[i].parent
        if not regs[i].strand_retained or regs[i].div < regs[p].div * 5.0 or regs[i].div < 0.01:
            if k < i:
                regs[k] = regs[i].copy()
            k += 1
    while len(regs) > k:
        _ = regs.pop()


def split_reg(mut r: Region, n: Int, qlen: Int, a: List[Packed]) -> Region:
    """`mm_split_reg`: the anchors from the `n`th on as a region of their own, the score shared by anchors."""
    if n <= 0 or n >= r.cnt:
        return Region()
    var r2 = r.copy()
    r2.drop_p()
    r2.id = -1
    r2.sam_pri = False
    r2.split_inv = False
    r2.cnt = r.cnt - n
    r2.score = Int(Float64(Float32(r.score) * (Float32(r2.cnt) / Float32(r.cnt))) + 0.499)
    r2.as_ = r.as_ + n
    if r.parent == r.id:
        r2.parent = PARENT_TMP_PRI
    set_coor(r2, qlen, a)
    r.cnt -= r2.cnt
    r.score -= r2.score
    set_coor(r, qlen, a)
    r.split |= 1
    r2.split |= 2
    return r2^


def filter_regs(mut regs: List[Region], preset: LongPreset, qlen: Int):
    """`mm_filter_regs`: too few anchors, too few matches, too low a score or clipped at both ends."""
    var kept = List[Region](capacity=len(regs))
    for i in range(len(regs)):
        ref r = regs[i]
        var flt = False
        if not r.inv and r.cnt < preset.min_cnt:
            flt = True
        if r.has_p:
            if r.mlen < preset.min_chain_score:
                flt = True
            elif r.dp_max < preset.min_dp_max:
                flt = True
            elif (
                Float32(r.qs) > Float32(qlen) * preset.max_clip_ratio
                and Float32(qlen - r.qe) > Float32(qlen) * preset.max_clip_ratio
            ):
                flt = True
        if not flt:
            kept.append(r.copy())
    regs = kept^


def event_identity(r: Region) -> Float64:
    """`mm_event_identity`."""
    if not r.has_p:
        return -1.0
    var n_gap = 0
    var n_gapo = 0
    for run in r.cigar:
        var op = Int(run & 0xF)
        if op == CIGAR_INSERTION or op == CIGAR_DELETION:
            n_gapo += 1
            n_gap += Int(run >> 4)
    return Float64(r.mlen) / Float64(r.blen + r.n_ambi - n_gap + n_gapo)


def update_dp_max(qlen: Int, mut regs: List[Region], frac: Float32, a: Int, b: Int):
    """`mm_update_dp_max`: when the best two alignments nearly tie, each rescored by its own divergence."""
    if len(regs) < 2:
        return
    var best = -1
    var best2 = -1
    var max_i = -1
    for i in range(len(regs)):
        if not regs[i].has_p:
            continue
        if regs[i].dp_max > best:
            best2 = best
            best = regs[i].dp_max
            max_i = i
        elif regs[i].dp_max > best2:
            best2 = regs[i].dp_max
    if max_i < 0 or best < 0 or best2 < 0:
        return
    if Float64(regs[max_i].qe - regs[max_i].qs) < Float64(qlen) * Float64(frac):
        return
    if Float64(best2) < Float64(best) * Float64(frac):
        return
    var div = 1.0 - event_identity(regs[max_i])
    if div < 0.02:
        div = 0.02
    var b2 = 0.5 / div
    if b2 * Float64(a) < Float64(b):
        b2 = Float64(a) / Float64(b)
    for i in range(len(regs)):
        if not regs[i].has_p:
            continue
        var n_gap = 0
        var gap_cost = 0.0
        for run in regs[i].cigar:
            var op = Int(run & 0xF)
            if op == CIGAR_INSERTION or op == CIGAR_DELETION:
                var length = Int(run >> 4)
                gap_cost += b2 + Float64(mg_log2(Float32(1.0 + Float64(length))))
                n_gap += length
        var n_mis = regs[i].blen + regs[i].n_ambi - regs[i].mlen - n_gap
        var dp = Int(Float64(a) * (Float64(regs[i].mlen) - b2 * Float64(n_mis) - gap_cost) + 0.499)
        regs[i].dp_max = max(dp, 0)


def hit_sort(mut regs: List[Region]):
    """`mm_hit_sort`: empty regions dropped, the rest by alignment score (chain score if unaligned), ties by hash,
    larger first and among equals the later first."""
    var keys = List[Tuple[UInt64, Int]](capacity=len(regs))
    for i in range(len(regs)):
        if regs[i].inv or regs[i].cnt > 0:
            var score = regs[i].dp_max if regs[i].has_p else regs[i].score
            keys.append((UInt64(score) << 32 | UInt64(regs[i].hash), i))

    def ahead(left: Tuple[UInt64, Int], right: Tuple[UInt64, Int]) -> Bool:
        return left[0] > right[0] or (left[0] == right[0] and left[1] > right[1])

    sort(Span(keys), ahead)
    var sorted = List[Region](capacity=len(keys))
    for key in keys:
        sorted.append(regs[key[1]].copy())
    regs = sorted^


def set_mapq(mut regs: List[Region], min_chain_sc: Int, match_sc: Int, rep_len: Int):
    """`mm_set_mapq` for long queries."""
    comptime q_coef = Float32(40.0)
    if len(regs) == 0:
        return
    var sum_sc = 0
    for i in range(len(regs)):
        if regs[i].parent == regs[i].id:
            sum_sc += regs[i].score
    var uniq_ratio = Float32(sum_sc) / Float32(sum_sc + rep_len)
    for i in range(len(regs)):
        ref r = regs[i]
        if r.inv:
            r.mapq = 0
        elif r.parent == r.id:
            var pen_s1 = (Float32(1.0) if r.score > 100 else Float32(0.01) * Float32(r.score)) * uniq_ratio
            var pen_cm = Float32(1.0) if r.cnt > 10 else Float32(0.1) * Float32(r.cnt)
            pen_cm = min(pen_s1, pen_cm)
            var subsc = max(r.subsc, min_chain_sc)
            var mapq: Int
            if r.has_p and r.dp_max2 > 0 and r.dp_max > 0:
                var identity = Float32(r.mlen) / Float32(r.blen)
                var x = Float32(r.dp_max2) * Float32(subsc) / Float32(r.dp_max) / Float32(r.score0)
                mapq = Int(
                    identity * pen_cm * q_coef * (Float32(1.0) - x * x) * log(Float32(r.dp_max) / Float32(match_sc))
                )
                var mapq_alt = Int(
                    Float32(6.02) * identity * identity * Float32(r.dp_max - r.dp_max2) / Float32(match_sc)
                    + Float32(0.499)
                )
                mapq = min(mapq, mapq_alt)
            else:
                var x = Float32(subsc) / Float32(r.score0)
                if r.has_p:
                    var identity = Float32(r.mlen) / Float32(r.blen)
                    mapq = Int(
                        identity * pen_cm * q_coef * (Float32(1.0) - x) * log(Float32(r.dp_max) / Float32(match_sc))
                    )
                else:
                    mapq = Int(pen_cm * q_coef * (Float32(1.0) - x) * log(Float32(r.score)))
            mapq -= Int(Float32(4.343) * log(Float32(r.n_sub + 1)) + Float32(0.499))
            mapq = max(mapq, 0)
            r.mapq = min(mapq, 60)
            if r.has_p and r.dp_max > r.dp_max2 and r.mapq == 0:
                r.mapq = 1
        else:
            r.mapq = 0
    # mm_set_inv_mapq: an inversion takes the lesser quality of its two neighbours on the reference.
    if len(regs) < 3:
        return
    var any_inv = False
    for i in range(len(regs)):
        if regs[i].inv:
            any_inv = True
    if not any_inv:
        return
    var aux = List[Tuple[Int, Int]]()
    for i in range(len(regs)):
        if regs[i].parent == i or regs[i].parent < 0:
            aux.append((regs[i].rs, i))

    sort(Span(aux))
    for i in range(1, len(aux) - 1):
        if regs[aux[i][1]].inv:
            regs[aux[i][1]].mapq = min(regs[aux[i - 1][1]].mapq, regs[aux[i + 1][1]].mapq)
