# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the
# MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
A long query, an assembled contig or a long read, aligned to a reference as minimap2 2.28 aligns it (`map.c`,
`hit.c`, `align.c`, `esterr.c`; MIT, Heng Li), from seeds the caller found: chains (see `chaining`), the primary and
secondary chains chosen, each chain aligned base by base and the alignments ranked again.

What minimap2 does after seeding is ported for its results, choices among equals included, so that where a gap
lands, a copy-number difference in a tandem array above all, is where minimap2 puts it:

- chaining by `mg_lchain_dp` (reads) or `mg_lchain_rmq` (assemblies), then the chains re-chained through gaps up
  to `bw_long` (minimap2's long join); chains ranked by score, ties by minimap2's hash of the query's name, length
  and first anchor (`mm_gen_regs`); primaries and secondaries by query overlap (`mm_set_parent`, `mm_select_sub`),
  secondaries on the other strand kept unless their estimated divergence is far above their primary's
  (`mm_est_err`);
- each chain aligned as `mm_align1` aligns it: bad ends and bad seeds left out (`mm_fix_bad_ends`,
  `mm_filter_bad_seeds`, `mm_filter_bad_seeds_alt`), the left end extended with Z-drop and end bonus, the stretches
  between anchors filled globally in a band of `1.5 bw_long + 1` (or the gap itself across a long join) by
  `karan.ksw2.ksw_extd2`, minimap2's own `ksw_extd2_sse` byte for byte, the Z-drop tested on each fill's CIGAR and
  the fill redone exactly where it drops, a fill that drops splitting the chain into two alignments, and the right
  end extended; an inversion between two pieces of a split aligned too (`mm_align1_inv`);
- each CIGAR's indels shifted left and adjacent insertions and deletions merged (`mm_fix_cigar`), the alignment
  scored with log gaps (`mm_update_extra`), filtered, re-ranked (`mm_update_dp_max`, `mm_hit_sort`), primaries
  chosen again and mapping qualities given (`mm_set_mapq`).

Seeds come from the caller (a k-mer index, minimizers, a graph walk); minimap2's own come from its minimizer index,
and a caller after minimap2's alignments exactly gives the same: its (w, k) minimizers, those above its occurrence
cutoffs left out, and the tandem flag (a query minimizer equal to its neighbour's). Unlike minimap2, one call
aligns to one reference sequence, and no splicing, short-read pairing or homopolymer-compressed seeds.
`ksw_extz2_sse`, which minimap2 runs when the two gap pieces are equal, is not ported: such costs run
`ksw_extd2` with equal pieces, the same scores, its own choices among equal alignments.
"""

from std.math import log, pow

from .chaining import (
    SEED_IGNORE,
    SEED_LONG_JOIN,
    SEED_TANDEM,
    ChainSet,
    Packed,
    chain_dp,
    chain_rmq,
    mg_log2,
    stable_sort_by_x,
)
from .ksw2 import KSW_EZ_APPROX_MAX, KSW_EZ_EXTZ_ONLY, KSW_EZ_REV_CIGAR, KSW_EZ_RIGHT, KswBuffers, KswResult, ksw_extd2

comptime CIGAR_MATCH = 0
"""A packed CIGAR run's op for aligned letters, match or mismatch (`M`)."""
comptime CIGAR_INSERTION = 1
"""A packed CIGAR run's op for query letters the reference lacks (`I`)."""
comptime CIGAR_DELETION = 2
"""A packed CIGAR run's op for reference letters the query lacks (`D`)."""
comptime _PARENT_UNSET = -1
comptime _PARENT_TMP_PRI = -2
comptime _KSW_L = 16
"""The vector width `ksw_extd2` follows: minimap2's SSE build, so its CIGARs byte for byte."""
comptime _KSW_W = 64
"""Cells `ksw_extd2` computes an instruction (AVX-512's 64 bytes), the same results as 16 at a time."""


@fieldwise_init
struct Seed(Copyable, ImplicitlyCopyable, Movable, TrivialRegisterPassable, Writable):
    """An exact match the caller found: `span` letters (1 to 255) of the query from `query` on its forward strand,
    equal to the reference's from `reference` on, or, when `reverse`, to their reverse complement. `tandem` marks a
    seed whose query minimizer repeats next to itself, as minimap2 marks one (`is_tandem`); the gap filling never
    ends a piece on it."""

    var reference: Int
    var query: Int
    var span: Int
    var reverse: Bool
    var tandem: Bool


@fieldwise_init
struct LongPreset(Copyable, ImplicitlyCopyable, Movable, Writable):
    """minimap2's mapping options for a long query, by its own names where a field has one. The named constructors
    are minimap2's presets; a field may be set after, as minimap2's flags set them (`-r` sets `bw` and `bw_long`,
    `-z` `zdrop` and `zdrop_inv`)."""

    var k: Int
    """The seeds' k-mer length: the chaining penalty scales with it and a secondary within `2 k` of its primary's
    chain score is kept (`mi->k`)."""
    var bw: Int
    """Chaining and alignment band (`-r`): no chain has a gap difference above it; a gap is filled in a band of
    `1.5 bw_long + 1`, an end extended in one of `1.5 bw + 1`."""
    var bw_long: Int
    """The band chains are joined across (`-r`'s second value)."""
    var max_gap: Int
    """The farthest two chained anchors lie apart (`-g`)."""
    var max_gap_ref: Int
    """The same on the reference, when positive (`-G`)."""
    var max_chain_skip: Int
    var max_chain_iter: Int
    var rmq: Bool
    """Chain by range-maximum queries (`--rmq`), as the assembly presets do."""
    var rmq_inner_dist: Int
    var rmq_size_cap: Int
    var rmq_rescue_size: Int
    var rmq_rescue_ratio: Float32
    var chain_gap_scale: Float32
    var chain_skip_scale: Float32
    var min_cnt: Int
    """A chain's least anchors (`-n`)."""
    var min_chain_score: Int
    """A chain's least score (`-m`)."""
    var mask_level: Float32
    var mask_len: Int
    var pri_ratio: Float32
    """A secondary's least score against its primary's (`-p`)."""
    var best_n: Int
    """Secondaries kept at most (`-N`)."""
    var match_score: Int
    """`-A`."""
    var mismatch: Int
    """`-B`."""
    var gap_open: Int
    """`-O`: a gap of `k` letters costs the less of `gap_open + k gap_extend` and `long_gap_open + k
    long_gap_extend`."""
    var gap_extend: Int
    """`-E`."""
    var long_gap_open: Int
    var long_gap_extend: Int
    var ambiguous: Int
    """A letter other than A, C, G, T against anything (`--score-N`)."""
    var zdrop: Int
    """`-z`."""
    var zdrop_inv: Int
    """`-z`'s second value: the Z-drop that splits at an inversion."""
    var end_bonus: Int
    var min_dp_max: Int
    """An alignment's least score (`-s`)."""
    var min_ksw_len: Int
    var max_clip_ratio: Float32
    var max_sw_mat: Int
    """A fill of more cells than this is not done: it Z-drops."""
    var rank_min_len: Int
    var rank_frac: Float32
    var end_filter: Bool
    """Leave bad chain ends out (`mm_fix_bad_ends`; minimap2's `--no-end-flt` turns it off)."""
    var long_join: Bool
    """Re-chain through gaps up to `bw_long` (minimap2's `--no-long-join` turns it off)."""
    var inversions: Bool
    """Align an inversion between the pieces of a split chain (minimap2's `--no-inv` turns it off)."""

    @staticmethod
    def map_ont() -> Self:
        """minimap2's defaults (`-x map-ont`, `-x lr`): k 15, gaps 5 kb, bands 500 and 20 kb."""
        return Self(
            k=15,
            bw=500,
            bw_long=20000,
            max_gap=5000,
            max_gap_ref=-1,
            max_chain_skip=25,
            max_chain_iter=5000,
            rmq=False,
            rmq_inner_dist=1000,
            rmq_size_cap=100000,
            rmq_rescue_size=1000,
            rmq_rescue_ratio=0.1,
            chain_gap_scale=0.8,
            chain_skip_scale=0.0,
            min_cnt=3,
            min_chain_score=40,
            mask_level=0.5,
            mask_len=Int(Int32.MAX),
            pri_ratio=0.8,
            best_n=5,
            match_score=2,
            mismatch=4,
            gap_open=4,
            gap_extend=2,
            long_gap_open=24,
            long_gap_extend=1,
            ambiguous=1,
            zdrop=400,
            zdrop_inv=200,
            end_bonus=-1,
            min_dp_max=80,
            min_ksw_len=200,
            max_clip_ratio=1.0,
            max_sw_mat=100000000,
            rank_min_len=500,
            rank_frac=0.9,
            end_filter=True,
            long_join=True,
            inversions=True,
        )

    @staticmethod
    def map_hifi() -> Self:
        """`-x map-hifi`: k 19, gaps 10 kb, `-A1 -B4 -O6,26 -E2,1 -s200`."""
        var preset = Self.map_ont()
        preset.k = 19
        preset.max_gap = 10000
        preset.match_score = 1
        preset.mismatch = 4
        preset.gap_open = 6
        preset.long_gap_open = 26
        preset.gap_extend = 2
        preset.long_gap_extend = 1
        preset.min_dp_max = 200
        return preset

    @staticmethod
    def _asm() -> Self:
        var preset = Self.map_ont()
        preset.k = 19
        preset.bw = 1000
        preset.bw_long = 100000
        preset.max_gap = 10000
        preset.rmq = True
        preset.min_dp_max = 200
        preset.best_n = 50
        preset.match_score = 1
        preset.zdrop = 200
        preset.zdrop_inv = 200
        return preset

    @staticmethod
    def asm5() -> Self:
        """`-x asm5`: k 19, range-maximum chaining, bands 1 kb and 100 kb, `-A1 -B19 -O39,81 -E3,1 -z200`."""
        var preset = Self._asm()
        preset.mismatch = 19
        preset.gap_open = 39
        preset.long_gap_open = 81
        preset.gap_extend = 3
        preset.long_gap_extend = 1
        return preset

    @staticmethod
    def asm10() -> Self:
        """`-x asm10`: `asm5`'s chaining, `-A1 -B9 -O16,41 -E2,1 -z200`."""
        var preset = Self._asm()
        preset.mismatch = 9
        preset.gap_open = 16
        preset.long_gap_open = 41
        preset.gap_extend = 2
        preset.long_gap_extend = 1
        return preset

    @staticmethod
    def asm20() -> Self:
        """`-x asm20`: `asm5`'s chaining, `-A1 -B4 -O6,26 -E2,1 -z200` (its window of 10 is the seeds')."""
        var preset = Self._asm()
        preset.mismatch = 4
        preset.gap_open = 6
        preset.long_gap_open = 26
        preset.gap_extend = 2
        preset.long_gap_extend = 1
        return preset

    @staticmethod
    def dipcall() -> Self:
        """dipcall's contig alignment, `-x asm5 -z200000,10000 -r2k`: Z-drop only at an inversion, a 2 kb band."""
        var preset = Self.asm5()
        preset.zdrop = 200000
        preset.zdrop_inv = 10000
        preset.bw = 2000
        return preset


@fieldwise_init
struct LongAlignment(Copyable, Movable):
    """One alignment of the query, as a line of minimap2's PAF holds it: the query's interval on its forward strand
    and the reference's, the strand, the CIGAR as KSW2 packs it (`length << 4 | op`, `CIGAR_MATCH`,
    `CIGAR_INSERTION`, `CIGAR_DELETION`, reference forward, the query reverse-complemented when `reverse`), the
    chain's score (`s1`) and, for a primary, its best secondary's (`s2`), the alignment's score (`AS`) and its best
    local score (`ms`), the mapping quality, letters
    aligned equal (`mlen`) and in all (`blen`). `primary` is minimap2's `tp:A:P`, the best of its query stretch;
    `supplementary` a primary other than the first (SAM's flag 2048); `inversion` an inversion between two pieces of
    a split chain (`tp:A:I`)."""

    var query_start: Int
    var query_end: Int
    var reference_start: Int
    var reference_end: Int
    var reverse: Bool
    var cigar: List[UInt32]
    var chain_score: Int
    var secondary_score: Int
    var score: Int
    var dp_max: Int
    var mapq: Int
    var matches: Int
    var block_length: Int
    var primary: Bool
    var supplementary: Bool
    var inversion: Bool

    def cigar_string(self) -> String:
        """The CIGAR as text, minimap2's `M`, `I`, `D`."""
        var out = String()
        for run in self.cigar:
            var op = Int(run & 0xF)
            out.write(Int(run >> 4), "M" if op == CIGAR_MATCH else ("I" if op == CIGAR_INSERTION else "D"))
        return out^


@fieldwise_init
struct LongChain(Copyable, Movable, Writable):
    """A chain as minimap2 keeps it before aligning: its anchors' extent on the query (forward strand) and the
    reference, the strand, its score and anchors, and whether it is a primary."""

    var query_start: Int
    var query_end: Int
    var reference_start: Int
    var reference_end: Int
    var reverse: Bool
    var score: Int
    var anchors: Int
    var primary: Bool


struct _Region(Copyable, Movable):
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
        self.parent = _PARENT_UNSET
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
def _hash64(key_in: UInt64) -> UInt64:
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
def _wang32(key_in: UInt32) -> UInt32:
    """khash's `__ac_Wang_hash`."""
    var key = key_in
    key += ~(key << 15)
    key ^= key >> 10
    key += key << 3
    key ^= key >> 6
    key += ~(key << 11)
    key ^= key >> 16
    return key


def name_hash(name: StringSlice) -> UInt32:
    """minimap2's hash of a query's name (khash's `__ac_X31_hash_string`); passed to `LongAligner.align`, it breaks
    ties among equally scored chains as minimap2 breaks them for a query of that name."""
    var bytes = name.as_bytes()
    if len(bytes) == 0:
        return 0
    var h = UInt32(bytes[0])
    for index in range(1, len(bytes)):
        h = (h << 5) - h + UInt32(bytes[index])
    return h


def encode_nt4(text: Span[UInt8, _]) -> List[UInt8]:
    """Letters as minimap2 codes them (`seq_nt4_table`): A 0, C 1, G 2, T 3 in either case, anything else 4."""
    var out = List[UInt8](capacity=len(text))
    for byte in text:
        var c = byte | 0x20
        var code = UInt8(4)
        if c == UInt8(ord("a")):
            code = 0
        elif c == UInt8(ord("c")):
            code = 1
        elif c == UInt8(ord("g")):
            code = 2
        elif c == UInt8(ord("t")) or c == UInt8(ord("u")):
            code = 3
        out.append(code)
    return out^


def _set_coor(mut r: _Region, qlen: Int, a: List[Packed]):
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


def _gen_regs(hash: UInt32, qlen: Int, chains: ChainSet) -> List[_Region]:
    """`mm_gen_regs`: one region a chain, by score, ties by the hash of the query and the chain's first anchor."""
    var z = List[Tuple[UInt64, UInt64]](capacity=len(chains.u))
    var k = 0
    for i in range(len(chains.u)):
        var h = _hash64((_hash64(chains.a[k].x) + _hash64(chains.a[k].y)) ^ UInt64(hash)) & 0xFFFFFFFF
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
    var regs = List[_Region](capacity=len(z))
    for i in range(len(order)):
        var entry = z[order[i]]
        var r = _Region()
        r.id = i
        r.parent = _PARENT_UNSET
        r.score = Int(entry[0] >> 32)
        r.score0 = r.score
        r.hash = UInt32(entry[0] & 0xFFFFFFFF)
        r.cnt = Int(entry[1] & 0xFFFFFFFF)
        r.as_ = Int(entry[1] >> 32)
        r.div = -1.0
        _set_coor(r, qlen, chains.a)
        regs.append(r^)
    return regs^


def _set_parent(mut r: List[_Region], mask_level: Float32, mask_len: Int, sub_diff: Int):
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

            def less(left: Tuple[Int, Int], right: Tuple[Int, Int]) -> Bool:
                return left[0] < right[0] or (left[0] == right[0] and left[1] < right[1])

            sort(Span(cov), less)
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


def _set_sam_pri(mut r: List[_Region]) -> Int:
    """`mm_set_sam_pri`: the first primary is SAM's primary."""
    var n_pri = 0
    for i in range(len(r)):
        if r[i].id == r[i].parent:
            n_pri += 1
            r[i].sam_pri = n_pri == 1
        else:
            r[i].sam_pri = False
    return n_pri


def _sync_regs(mut r: List[_Region]):
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
        if r[i].parent == _PARENT_TMP_PRI:
            r[i].parent = i
        elif r[i].parent >= 0 and tmp[r[i].parent] >= 0:
            r[i].parent = tmp[r[i].parent]
        else:
            r[i].parent = _PARENT_UNSET
    _ = _set_sam_pri(r)


def _select_sub(
    mut r: List[_Region], pri_ratio: Float32, min_diff: Int, best_n: Int, check_strand: Bool, min_strand_sc: Int
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
        _sync_regs(r)


def _est_err(qlen: Int, ref_len: Int, mut regs: List[_Region], a: List[Packed], mini_pos: List[UInt64]):
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


def _filter_strand_retained(mut regs: List[_Region]):
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


def _ksw_ll(
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


def _gen_mat(a: Int, b: Int, ambi: Int) -> List[Int8]:
    """`ksw_gen_ts_mat` with no transition score: `a` on the diagonal, `-b` off it, `-ambi` against a wildcard."""
    var mat = List[Int8](length=25, fill=0)
    for i in range(4):
        for j in range(4):
            mat[i * 5 + j] = Int8(a) if i == j else Int8(-b)
        mat[i * 5 + 4] = Int8(-ambi)
    for j in range(5):
        mat[20 + j] = Int8(-ambi)
    return mat^


def _append_cigar(mut r: _Region, cigar: List[UInt32]):
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


def _fix_cigar(mut r: _Region, qseq: Span[UInt8, _], tseq: Span[UInt8, _]) -> Tuple[Int, Int]:
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


def _update_extra(
    mut r: _Region, qseq_in: Span[UInt8, _], tseq_in: Span[UInt8, _], mat: List[Int8], q: Int, e: Int, log_gap: Bool
):
    """`mm_update_extra`: the CIGAR fixed, then its letters counted and its best local score with log gaps."""
    if not r.has_p:
        return
    var shifts = _fix_cigar(r, qseq_in, tseq_in)
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


def _split_reg(mut r: _Region, n: Int, qlen: Int, a: List[Packed]) -> _Region:
    """`mm_split_reg`: the anchors from the `n`th on as a region of their own, the score shared by anchors."""
    if n <= 0 or n >= r.cnt:
        return _Region()
    var r2 = r.copy()
    r2.drop_p()
    r2.id = -1
    r2.sam_pri = False
    r2.split_inv = False
    r2.cnt = r.cnt - n
    r2.score = Int(Float64(Float32(r.score) * (Float32(r2.cnt) / Float32(r.cnt))) + 0.499)
    r2.as_ = r.as_ + n
    if r.parent == r.id:
        r2.parent = _PARENT_TMP_PRI
    _set_coor(r2, qlen, a)
    r.cnt -= r2.cnt
    r.score -= r2.score
    _set_coor(r, qlen, a)
    r.split |= 1
    r2.split |= 2
    return r2^


def _filter_regs(mut regs: List[_Region], preset: LongPreset, qlen: Int):
    """`mm_filter_regs`: too few anchors, too few matches, too low a score or clipped at both ends."""
    var kept = List[_Region](capacity=len(regs))
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


def _event_identity(r: _Region) -> Float64:
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


def _update_dp_max(qlen: Int, mut regs: List[_Region], frac: Float32, a: Int, b: Int):
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
    var div = 1.0 - _event_identity(regs[max_i])
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


def _hit_sort(mut regs: List[_Region]):
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
    var sorted = List[_Region](capacity=len(keys))
    for key in keys:
        sorted.append(regs[key[1]].copy())
    regs = sorted^


def _set_mapq(mut regs: List[_Region], min_chain_sc: Int, match_sc: Int, rep_len: Int):
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

    def less(left: Tuple[Int, Int], right: Tuple[Int, Int]) -> Bool:
        return left[0] < right[0] or (left[0] == right[0] and left[1] < right[1])

    sort(Span(aux), less)
    for i in range(1, len(aux) - 1):
        if regs[aux[i][1]].inv:
            regs[aux[i][1]].mapq = min(regs[aux[i - 1][1]].mapq, regs[aux[i + 1][1]].mapq)


def _collect_long_gaps(as1: Int, cnt1: Int, a: List[Packed], min_gap: Int) -> List[Int]:
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
def _gap_of(a: List[Packed], i: Int) -> Int:
    """The gap difference (query less reference) before anchor `i`."""
    return (Int(a[i].query_end()) - Int(a[i - 1].query_end())) - Int((a[i].x - a[i - 1].x).cast[DType.int32]())


def _filter_bad_seeds(
    as1: Int, cnt1: Int, mut a: List[Packed], min_gap: Int, diff_thres: Int, max_ext_len: Int, max_ext_cnt: Int
):
    """`mm_filter_bad_seeds`: anchors inside a short stretch of insertions and deletions that largely cancel out
    marked to be stepped over."""
    var gaps = _collect_long_gaps(as1, cnt1, a, min_gap)
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
        var gap = _gap_of(a, as1 + i)
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
            gap = _gap_of(a, as1 + j)
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


def _filter_bad_seeds_alt(as1: Int, cnt1: Int, mut a: List[Packed], min_gap: Int, max_ext: Int):
    """`mm_filter_bad_seeds_alt`: consecutive large gaps closer than their sizes marked over and joined, the last
    anchor flagged as a long join."""
    var gaps = _collect_long_gaps(as1, cnt1, a, min_gap)
    var n = len(gaps)
    if n == 0:
        return
    var k = 0
    while k < n:
        var i = gaps[k]
        var gap1 = abs(_gap_of(a, as1 + i))
        var re1 = Int(a[as1 + i].reference_end())
        var qe1 = Int(a[as1 + i].query_end())
        var l = k + 1
        while l < n:
            var j = gaps[l]
            if Int(a[as1 + j].query_end()) - qe1 > max_ext or Int(a[as1 + j].reference_end()) - re1 > max_ext:
                break
            var gap2 = abs(_gap_of(a, as1 + j))
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


def _fix_bad_ends(r: _Region, a: List[Packed], bw: Int, min_match: Int) -> Tuple[Int, Int]:
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
def _adjust_minier(anchor: Packed) -> Tuple[Int, Int]:
    """`mm_adjust_minier`: an anchor's reference and query point, the middle of its match."""
    var half = Int(anchor.span()) >> 1
    return (Int(anchor.reference_end()) - half, Int(anchor.query_end()) - half)


def _align_pair(
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
    ksw_extd2[_KSW_L, _KSW_W](
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


struct _Drop:
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


def _test_zdrop(
    preset: LongPreset, qseq: Span[UInt8, _], tseq: Span[UInt8, _], cigar: List[UInt32], mat: List[Int8]
) -> Int:
    """`mm_test_zdrop` on a fill's CIGAR: 2 where its worst drop holds an inversion, 1 where it passes the Z-drop,
    else 0."""
    var drop = _Drop()
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
        var local = _ksw_ll(Span(qseq2), tseq[drop.t0 : drop.t0 + t_len], mat, preset.gap_open, preset.gap_extend)
        if local[0] >= preset.min_chain_score * preset.match_score and local[0] >= preset.min_dp_max:
            return 2
    return 1 if drop.max_zdrop > preset.zdrop else 0


struct LongAligner(Movable):
    """One thread's long-query aligner: KSW2's memory, kept from call to call.

    ```mojo
    from karan import LongAligner, LongPreset, Seed, encode_nt4

    var reference = encode_nt4(reference_text.as_bytes())
    var query = encode_nt4(contig_text.as_bytes())
    var seeds: List[Seed] = ...  # the caller's exact matches, e.g. shared (w, k) minimizers
    var aligner = LongAligner()
    for hit in aligner.align(Span(reference), Span(query), Span(seeds), LongPreset.dipcall()):
        print(hit.reference_start, hit.reference_end, hit.cigar_string())
    ```"""

    var buf: KswBuffers
    var ez: KswResult

    def __init__(out self):
        self.buf = KswBuffers()
        self.ez = KswResult()

    def _align1(
        mut self,
        preset: LongPreset,
        reference: Span[UInt8, _],
        qseq0: Span[UInt8, _],
        qlen: Int,
        mut r: _Region,
        mut a: List[Packed],
        mat: List[Int8],
    ) -> _Region:
        """`mm_align1`, not spliced: the region aligned, and the region split off it where a fill Z-dropped (no
        anchors when none)."""
        var r2 = _Region()
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
            var ends = _fix_bad_ends(r, a, preset.bw, preset.min_chain_score * 2)
            as1 = ends[0]
            cnt1 = ends[1]
        _filter_bad_seeds(as1, cnt1, a, 10, 40, preset.max_gap >> 1, 10)
        _filter_bad_seeds_alt(as1, cnt1, a, 30, preset.max_gap >> 1)
        var start = _adjust_minier(a[as1])
        var rs = start[0]
        var qs = start[1]
        var end = _adjust_minier(a[as1 + cnt1 - 1])
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
            _align_pair(
                preset,
                Span(qrev),
                Span(trev),
                mat,
                bw,
                preset.end_bonus,
                preset.zdrop_inv if r.split_inv else preset.zdrop,
                KSW_EZ_EXTZ_ONLY | KSW_EZ_RIGHT | KSW_EZ_REV_CIGAR,
                self.buf,
                self.ez,
            )
            if len(self.ez.cigar) > 0:
                _append_cigar(r, self.ez.cigar)
                r.dp_score += self.ez.max
            rs1 = rs - (self.ez.mqe_t + 1 if self.ez.reach_end else self.ez.max_t + 1)
            qs1 = qs - (qs - qs0 if self.ez.reach_end else self.ez.max_q + 1)
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
            var point = _adjust_minier(a[as1 + k])
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
                _align_pair(preset, qpiece, tpiece, mat, bw1, -1, preset.zdrop, KSW_EZ_APPROX_MAX, self.buf, self.ez)
                var zdrop_code = _test_zdrop(preset, qpiece, tpiece, self.ez.cigar, mat)
                if zdrop_code != 0:
                    var zdrop = preset.zdrop_inv if zdrop_code == 2 else preset.zdrop
                    _align_pair(preset, qpiece, tpiece, mat, bw1, -1, zdrop, 0, self.buf, self.ez)
                if len(self.ez.cigar) > 0:
                    _append_cigar(r, self.ez.cigar)
                r.has_p = True
                if self.ez.zdropped:
                    var j = k - 1
                    while j >= 0:
                        if Int(a[as1 + j].reference_end()) <= rs + self.ez.max_t:
                            break
                        j -= 1
                    dropped = True
                    if j < 0:
                        j = 0
                    r.dp_score += self.ez.max
                    re1 = rs + (self.ez.max_t + 1)
                    qe1 = qs + (self.ez.max_q + 1)
                    if cnt1 - (j + 1) >= preset.min_cnt:
                        r2 = _split_reg(r, as1 + j + 1 - r.as_, qlen, a)
                        if zdrop_code == 2:
                            r2.split_inv = True
                    break
                r.dp_score += self.ez.score
                rs = re
                qs = qe

        if not dropped and qe < qe0 and re < re0:
            _align_pair(
                preset,
                qseq_all[qe:qe0],
                reference[re:re0],
                mat,
                bw,
                preset.end_bonus,
                preset.zdrop,
                KSW_EZ_EXTZ_ONLY,
                self.buf,
                self.ez,
            )
            if len(self.ez.cigar) > 0:
                _append_cigar(r, self.ez.cigar)
                r.dp_score += self.ez.max
            re1 = re + (self.ez.mqe_t + 1 if self.ez.reach_end else self.ez.max_t + 1)
            qe1 = qe + (qe0 - qe if self.ez.reach_end else self.ez.max_q + 1)

        r.rs = rs1
        r.re = re1
        if not rev:
            r.qs = qs1
            r.qe = qe1
        else:
            r.qs = qlen - qe1
            r.qe = qlen - qs1
        if r.has_p:
            _update_extra(r, qseq_all[qs1:], reference[rs1:re1], mat, preset.gap_open, preset.gap_extend, True)
        return r2^

    def _align1_inv(
        mut self,
        preset: LongPreset,
        reference: Span[UInt8, _],
        qseq0: Span[UInt8, _],
        qlen: Int,
        r1: _Region,
        r2: _Region,
        mat: List[Int8],
    ) -> Optional[_Region]:
        """`mm_align1_inv`: between two pieces of a chain split at an inversion, the other strand's best local
        alignment, extended from its far end."""
        if (r1.split & 1) == 0 or (r2.split & 2) == 0:
            return None
        if r1.id != r1.parent and r1.parent != _PARENT_TMP_PRI:
            return None
        if r2.id != r2.parent and r2.parent != _PARENT_TMP_PRI:
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
        var local = _ksw_ll(Span(qrev), Span(trev), mat, preset.gap_open, preset.gap_extend)
        if local[0] < preset.min_dp_max:
            return None
        var q_off = ql - (local[1] + 1)
        var t_off = tl - (local[2] + 1)
        _align_pair(
            preset,
            qseq[q_off:],
            tseq[t_off:],
            mat,
            Int(Float64(preset.bw) * 1.5),
            -1,
            preset.zdrop,
            KSW_EZ_EXTZ_ONLY,
            self.buf,
            self.ez,
        )
        if len(self.ez.cigar) == 0:
            return None
        var r = _Region()
        _append_cigar(r, self.ez.cigar)
        r.dp_score = self.ez.max
        r.id = -1
        r.parent = _PARENT_UNSET
        r.inv = True
        r.rev = not r1.rev
        r.div = -1.0
        if not r.rev:
            r.qs = r2.qe + q_off
            r.qe = r.qs + self.ez.max_q + 1
        else:
            r.qe = r2.qs - q_off
            r.qs = r.qe - (self.ez.max_q + 1)
        r.rs = r1.re + t_off
        r.re = r.rs + self.ez.max_t + 1
        _update_extra(r, qseq[q_off:], tseq[t_off:], mat, preset.gap_open, preset.gap_extend, True)
        return r^

    def _chain(
        self,
        reference_length: Int,
        qlen: Int,
        seeds: Span[Seed, _],
        preset: LongPreset,
        name_hash: UInt32,
        mut regs: List[_Region],
        mut a: List[Packed],
    ):
        """minimap2's `mm_map_frag` up to its alignment: the seeds chained, re-chained across long gaps, the chains
        ranked, primaries chosen, and secondaries on the other strand filtered by divergence."""
        var keys = List[Tuple[UInt64, Int, Int]](capacity=len(seeds))
        var packed = List[Packed](capacity=len(seeds))
        var mini = List[UInt64](capacity=len(seeds))
        for index in range(len(seeds)):
            var s = seeds[index]
            var span = UInt64(s.span & 0xFF)
            var x = UInt64(s.reference + s.span - 1)
            var y = span << 32
            if s.reverse:
                x |= UInt64(1) << 63
                y |= UInt64(qlen - s.query - 1)
            else:
                y |= UInt64(s.query + s.span - 1)
            if s.tandem:
                y |= SEED_TANDEM
            keys.append((x, s.query, index))
            packed.append(Packed(x, y))
            mini.append(span << 32 | UInt64(s.query + s.span - 1))

        # collect_seed_hits' order: by x, then the query minimizer's order.
        def less(left: Tuple[UInt64, Int, Int], right: Tuple[UInt64, Int, Int]) -> Bool:
            return left[0] < right[0] or (
                left[0] == right[0] and (left[1] < right[1] or (left[1] == right[1] and left[2] < right[2]))
            )

        sort(Span(keys), less)
        a.clear()
        for key in keys:
            a.append(packed[key[2]])

        # The query's seeded minimizers, by position, once each (mm_collect_matches' mini_pos).
        def by_pos(left: UInt64, right: UInt64) -> Bool:
            return (left & 0xFFFFFFFF) < (right & 0xFFFFFFFF) or (
                (left & 0xFFFFFFFF) == (right & 0xFFFFFFFF) and left < right
            )

        sort(Span(mini), by_pos)
        var mini_pos = List[UInt64](capacity=len(mini))
        for value in mini:
            if len(mini_pos) == 0 or (mini_pos[len(mini_pos) - 1] & 0xFFFFFFFF) != (value & 0xFFFFFFFF):
                mini_pos.append(value)

        var pen_gap = Float32(Float64(preset.chain_gap_scale) * 0.01 * Float64(preset.k))
        var pen_skip = Float32(Float64(preset.chain_skip_scale) * 0.01 * Float64(preset.k))
        var max_chain_gap_ref = preset.max_gap_ref if preset.max_gap_ref > 0 else preset.max_gap
        var chains: ChainSet
        if preset.rmq:
            chains = chain_rmq(
                Int32(preset.max_gap),
                Int32(preset.rmq_inner_dist),
                Int32(preset.bw),
                preset.max_chain_skip,
                preset.rmq_size_cap,
                preset.min_cnt,
                Int32(preset.min_chain_score),
                pen_gap,
                pen_skip,
                a,
            )
        else:
            chains = chain_dp(
                Int32(max_chain_gap_ref),
                Int32(preset.max_gap),
                Int32(preset.bw),
                preset.max_chain_skip,
                preset.max_chain_iter,
                preset.min_cnt,
                Int32(preset.min_chain_score),
                pen_gap,
                pen_skip,
                a,
            )
        if preset.bw_long > preset.bw and preset.long_join and len(chains.u) > 1:
            var st = Int(chains.a[0].query_end())
            var en = Int(chains.a[Int(chains.u[0] & 0xFFFFFFFF) - 1].query_end())
            if qlen - (en - st) > preset.rmq_rescue_size or Float32(en - st) > Float32(qlen) * preset.rmq_rescue_ratio:
                stable_sort_by_x(chains.a)
                chains = chain_rmq(
                    Int32(preset.max_gap),
                    Int32(preset.rmq_inner_dist),
                    Int32(preset.bw_long),
                    preset.max_chain_skip,
                    preset.rmq_size_cap,
                    preset.min_cnt,
                    Int32(preset.min_chain_score),
                    pen_gap,
                    pen_skip,
                    chains.a,
                )
        var hash = _wang32(name_hash ^ (_wang32(UInt32(qlen)) + _wang32(11)))
        regs = _gen_regs(hash, qlen, chains)
        _set_parent(regs, preset.mask_level, preset.mask_len, preset.match_score * 2 + preset.mismatch)
        _select_sub(regs, preset.pri_ratio, preset.k * 2, preset.best_n, True, Int(Float64(preset.max_gap) * 0.8))
        _est_err(qlen, reference_length, regs, chains.a, mini_pos)
        _filter_strand_retained(regs)
        a = chains^.anchors()

    def chains(
        self,
        reference_length: Int,
        query_length: Int,
        seeds: Span[Seed, _],
        preset: LongPreset,
        name_hash: UInt32 = 0,
    ) -> List[LongChain]:
        """The chains minimap2 would go on to align, best first, without aligning them: the seeds chained and
        re-chained, primaries and secondaries chosen."""
        var regs = List[_Region]()
        var a = List[Packed]()
        self._chain(reference_length, query_length, seeds, preset, name_hash, regs, a)
        var out = List[LongChain](capacity=len(regs))
        for r in regs:
            out.append(LongChain(r.qs, r.qe, r.rs, r.re, r.rev, r.score, r.cnt, r.id == r.parent))
        return out^

    def align(
        mut self,
        reference: Span[UInt8, _],
        query: Span[UInt8, _],
        seeds: Span[Seed, _],
        preset: LongPreset,
        name_hash: UInt32 = 0,
        repeat_length: Int = 0,
    ) -> List[LongAlignment]:
        """The query's alignments to the reference as minimap2 makes them from these seeds, best first: primaries,
        then secondaries within `pri_ratio` of theirs. The sequences are minimap2's codes (`encode_nt4`); seeds
        hold forward-strand query positions. `name_hash` (`name_hash(query_name)`) breaks ties among equal chains
        as minimap2 breaks them for that name; `repeat_length`, the query letters under seeds left out as too
        frequent (minimap2's `rep_len`), lowers mapping qualities as minimap2's does."""
        var qlen = len(query)
        var out = List[LongAlignment]()
        if qlen == 0 or len(reference) == 0 or len(seeds) == 0:
            return out^
        var regs = List[_Region]()
        var a = List[Packed]()
        self._chain(len(reference), qlen, seeds, preset, name_hash, regs, a)
        if len(regs) == 0:
            return out^
        # The query's two strands, the reverse complement after the forward.
        var qseq0 = List[UInt8](capacity=2 * qlen)
        for index in range(qlen):
            qseq0.append(query[index])
        for index in range(qlen - 1, -1, -1):
            var c = query[index]
            qseq0.append(3 - c if c < 4 else 4)
        var mat = _gen_mat(preset.match_score, preset.mismatch, preset.ambiguous)

        # mm_squeeze_a: the regions' anchors laid one after another in the order of their starts.
        var by_start = List[Tuple[Int, Int]](capacity=len(regs))
        for index in range(len(regs)):
            by_start.append((regs[index].as_, index))

        def less(left: Tuple[Int, Int], right: Tuple[Int, Int]) -> Bool:
            return left[0] < right[0] or (left[0] == right[0] and left[1] < right[1])

        sort(Span(by_start), less)
        var squeezed = List[Packed](capacity=len(a))
        for entry in by_start:
            var start = len(squeezed)
            for index in range(regs[entry[1]].as_, regs[entry[1]].as_ + regs[entry[1]].cnt):
                squeezed.append(a[index])
            regs[entry[1]].as_ = start
        a = squeezed^

        var i = 0
        while i < len(regs):
            var r2 = self._align1(preset, reference, Span(qseq0), qlen, regs[i], a, mat)
            if r2.cnt > 0:
                regs.insert(i + 1, r2^)
            if i > 0 and regs[i].split_inv and preset.inversions:
                var inv = self._align1_inv(preset, reference, Span(qseq0), qlen, regs[i - 1], regs[i], mat)
                if inv:
                    regs.insert(i + 1, inv.take())
                    i += 1
            i += 1
        _filter_regs(regs, preset, qlen)
        if qlen >= preset.rank_min_len:
            _update_dp_max(qlen, regs, preset.rank_frac, preset.match_score, preset.mismatch)
            _filter_regs(regs, preset, qlen)
        _hit_sort(regs)
        _set_parent(regs, preset.mask_level, preset.mask_len, preset.match_score * 2 + preset.mismatch)
        _select_sub(regs, preset.pri_ratio, preset.k * 2, preset.best_n, False, Int(Float64(preset.max_gap) * 0.8))
        _ = _set_sam_pri(regs)
        _set_mapq(regs, preset.min_chain_score, preset.match_score, repeat_length)
        for r in regs:
            if not r.has_p:
                continue
            out.append(
                LongAlignment(
                    r.qs,
                    r.qe,
                    r.rs,
                    r.re,
                    r.rev,
                    r.cigar.copy(),
                    r.score,
                    r.subsc,
                    r.dp_score,
                    r.dp_max,
                    r.mapq,
                    r.mlen,
                    r.blen,
                    r.id == r.parent,
                    r.id == r.parent and not r.sam_pri,
                    r.inv,
                )
            )
        return out^
