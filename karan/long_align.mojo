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

The port is laid out as minimap2's files are, each function named after the one it ports, so a difference from
minimap2 is looked up by name:

| module | minimap2 | holds |
| :-- | :-- | :-- |
| `long_types` | `minimap.h` | `Seed`, `LongPreset`, `LongAlignment`, `LongChain`, letter codes, the name hash |
| `chaining` | `lchain.c` | `mg_lchain_dp`, `mg_lchain_rmq`, the backtrack |
| `long_regions` | `hit.c`, `esterr.c` | regions from chains, parents and secondaries, divergence, filters, mapping quality |
| `long_fill` | `align.c` | `mm_align1` and `mm_align1_inv`: end extensions, gap filling, Z-drop, CIGAR fixing |
| `long_align` | `map.c` | `LongAligner`: the stages in order |
"""

from .chaining import SEED_TANDEM, ChainSet, Packed, chain_dp, chain_rmq, stable_sort_by_x
from .ksw2 import KswBuffers, KswResult
from .long_fill import align1, align1_inv, gen_mat
from .errors import AlignmentError
from .long_regions import (
    Region,
    est_err,
    filter_regs,
    filter_strand_retained,
    gen_regs,
    hit_sort,
    select_sub,
    set_mapq,
    set_parent,
    set_sam_pri,
    trace_regions,
    update_dp_max,
    wang32,
)
from .long_types import (
    CIGAR_DELETION,
    CIGAR_INSERTION,
    CIGAR_MATCH,
    LONG_TRACE,
    LongAlignment,
    LongChain,
    LongPreset,
    Seed,
    check_inputs,
    check_seeds,
    encode_nt4,
    name_hash,
    trace,
)


def trace_chains(stage: StaticString, chains: ChainSet):
    """Each chain's trace line: its score, anchors, strand and its first and last anchors' ends (reference, query on
    the chain's strand)."""
    comptime if LONG_TRACE:
        var k = 0
        for index in range(len(chains.u)):
            var count = Int(chains.u[index] & 0xFFFFFFFF)
            var first = chains.a[k]
            var last = chains.a[k + count - 1]
            trace(
                String(
                    t"{stage}\tchain={index}\tscore={Int(chains.u[index] >> 32)}\tcnt={count}\t"
                    t"{'-' if first.reverse() else '+'}\tfirst={first.reference_end()},{first.query_end()}\t"
                    t"last={last.reference_end()},{last.query_end()}"
                )
            )
            k += count


def chain_seeds(
    reference_length: Int,
    qlen: Int,
    seeds: Span[Seed, _],
    preset: LongPreset,
    name_hash: UInt32,
    mut regs: List[Region],
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

    # collect_seed_hits' order: by x, then the query minimizer's order (tuples sort field by field).

    sort(Span(keys))
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
    if preset.rmq:
        trace_chains("mg_lchain_rmq", chains)
    else:
        trace_chains("mg_lchain_dp", chains)
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
            trace_chains("long_join", chains)
    var hash = wang32(name_hash ^ (wang32(UInt32(qlen)) + wang32(11)))
    regs = gen_regs(hash, qlen, chains)
    trace_regions("mm_gen_regs", regs)
    set_parent(regs, preset.mask_level, preset.mask_len, preset.match_score * 2 + preset.mismatch)
    select_sub(regs, preset.pri_ratio, preset.k * 2, preset.best_n, True, Int(Float64(preset.max_gap) * 0.8))
    est_err(qlen, reference_length, regs, chains.a, mini_pos)
    filter_strand_retained(regs)
    trace_regions("mm_filter_strand_retained", regs)
    a = chains^.anchors()


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

    def chains(
        self,
        reference_length: Int,
        query_length: Int,
        seeds: ImmSpan[Seed, _],
        preset: LongPreset,
        name_hash: UInt32 = 0,
    ) raises AlignmentError -> List[LongChain]:
        """The chains minimap2 would go on to align, best first, without aligning them: the seeds chained and
        re-chained, primaries and secondaries chosen. Raises as `align` does, on the preset or a seed past either
        length."""
        preset.check()
        check_seeds(reference_length, query_length, seeds)
        var regs = List[Region]()
        var a = List[Packed]()
        if query_length == 0 or reference_length == 0 or len(seeds) == 0:
            return List[LongChain]()
        chain_seeds(reference_length, query_length, seeds, preset, name_hash, regs, a)
        var out = List[LongChain](capacity=len(regs))
        for r in regs:
            out.append(LongChain(r.qs, r.qe, r.rs, r.re, r.rev, r.score, r.cnt, r.id == r.parent))
        return out^

    def align(
        mut self,
        reference: ImmSpan[UInt8, _],
        query: ImmSpan[UInt8, _],
        seeds: ImmSpan[Seed, _],
        preset: LongPreset,
        name_hash: UInt32 = 0,
        repeat_length: Int = 0,
    ) raises AlignmentError -> List[LongAlignment]:
        """The query's alignments to the reference as minimap2 makes them from these seeds, best first: primaries,
        then secondaries within `pri_ratio` of theirs. The sequences are minimap2's codes (`encode_nt4`); seeds
        hold forward-strand query positions. `name_hash` (`name_hash(query_name)`) breaks ties among equal chains
        as minimap2 breaks them for that name; `repeat_length`, the query letters under seeds left out as too
        frequent (minimap2's `rep_len`), lowers mapping qualities as minimap2's does.

        Raises an `AlignmentError` on a preset minimap2 refuses (`LongPreset.check`), a sequence longer than
        2^31 - 1 letters, a seed outside 1 to 255 letters inside both sequences, or a byte other than the codes 0 to 4
        in the query or in the stretch of the reference an alignment reads."""
        preset.check()
        check_inputs(len(reference), query, seeds)
        var qlen = len(query)
        var out = List[LongAlignment]()
        comptime if LONG_TRACE:
            trace(String(t"query\tqlen={qlen}\treference={len(reference)}\tseeds={len(seeds)}\tname_hash={name_hash}"))
        if qlen == 0 or len(reference) == 0 or len(seeds) == 0:
            return out^
        var regs = List[Region]()
        var a = List[Packed]()
        chain_seeds(len(reference), qlen, seeds, preset, name_hash, regs, a)
        if len(regs) == 0:
            return out^
        # The query's two strands, the reverse complement after the forward.
        var qseq0 = List[UInt8](capacity=2 * qlen)
        for index in range(qlen):
            qseq0.append(query[index])
        for index in range(qlen - 1, -1, -1):
            var c = query[index]
            qseq0.append(3 - c if c < 4 else 4)
        var mat = gen_mat(preset.match_score, preset.mismatch, preset.ambiguous)

        # mm_squeeze_a: the regions' anchors laid one after another in the order of their starts.
        var by_start = List[Tuple[Int, Int]](capacity=len(regs))
        for index in range(len(regs)):
            by_start.append((regs[index].as_, index))

        sort(Span(by_start))
        var squeezed = List[Packed](capacity=len(a))
        for entry in by_start:
            var start = len(squeezed)
            for index in range(regs[entry[1]].as_, regs[entry[1]].as_ + regs[entry[1]].cnt):
                squeezed.append(a[index])
            regs[entry[1]].as_ = start
        a = squeezed^

        var i = 0
        while i < len(regs):
            var r2 = align1(self.buf, self.ez, preset, reference, Span(qseq0), qlen, regs[i], a, mat)
            if r2.cnt > 0:
                regs.insert(i + 1, r2^)
            if i > 0 and regs[i].split_inv and preset.inversions:
                var inv = align1_inv(self.buf, self.ez, preset, reference, Span(qseq0), qlen, regs[i - 1], regs[i], mat)
                if inv:
                    regs.insert(i + 1, inv.take())
                    i += 1
            i += 1
        trace_regions("mm_align1", regs)
        filter_regs(regs, preset, qlen)
        if qlen >= preset.rank_min_len:
            update_dp_max(qlen, regs, preset.rank_frac, preset.match_score, preset.mismatch)
            trace_regions("mm_update_dp_max", regs)
            filter_regs(regs, preset, qlen)
        hit_sort(regs)
        set_parent(regs, preset.mask_level, preset.mask_len, preset.match_score * 2 + preset.mismatch)
        select_sub(regs, preset.pri_ratio, preset.k * 2, preset.best_n, False, Int(Float64(preset.max_gap) * 0.8))
        _ = set_sam_pri(regs)
        set_mapq(regs, preset.min_chain_score, preset.match_score, repeat_length)
        trace_regions("mm_set_mapq", regs)
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
