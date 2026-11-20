# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the
# MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
One pair's search, chosen for its costs and mode: which engine finds its cost (`cost_within`) and its alignment
(`aligned_within`), the bit-parallel edit distance, the gap-affine wavefront, the sweeps or the lanes, and the
KSW2 rule written over the result where asked. The public calls (`api`) and the batches (`batches`) call these;
every engine below answers the same, so the choice is speed alone.
"""

from .cigar import cigar_matches, reversed_list, runs_of
from .common import FIRST_SENTINEL
from .edit_distance import edit_cigar, edit_distance
from .edit_search import EditHit, edit_search
from .errors import AlignmentError, ErrorKind
from .gap_affine import (
    AffineCigar,
    DEFAULT_MAX_MEMORY,
    EndsFree,
    KEPT_BYTES,
    Spanned,
    cigar_of,
    cigar_within,
    extension_of,
    free_ends_alignment,
    kept_bytes,
    rings_fit,
    rings_within,
    wavefront_distance,
)
from .ksw_ties import TieBuffers, ksw_cigar
from .lanes import LaneCosts, SpanTexts, Texts, lane_alignments, lane_distances, lane_free_alignments
from .modes import ANY_LENGTH, Alignment, Anchor, Band, Costs, Mode, Ties
from .penalties import Penalties, penalties_of, rewarded_penalties
from .probes import ENGINE, ENGINE_LANES, ENGINE_SCORED, ENGINE_SEARCH, SCORED_NS, SEARCH_NS
from .scored import global_rewarded, local_alignment, rewarded_alignment
from .search_space import SearchSpace


def holds_sentinel(text: ImmSpan[UInt8, _]) -> Bool:
    """Whether `text` holds 0xFE or 0xFF, the bytes the wavefront marks the sequences' ends with: thirty-two at a
    time, the bytes past the last whole chunk in one chunk that ends the text, as `encoded_bases` takes them."""
    comptime CHUNK = 32
    comptime HALF = CHUNK // 2
    var bytes = text.unsafe_ptr()
    var count = len(text)
    if count >= CHUNK:
        var found = SIMD[DType.bool, CHUNK](fill=False)
        var index = 0
        while index + CHUNK <= count:
            found |= bytes.unsafe_offset(index).unsafe_load[width=CHUNK]().ge(FIRST_SENTINEL)
            index += CHUNK
        if index < count:
            found |= bytes.unsafe_offset(count - CHUNK).unsafe_load[width=CHUNK]().ge(FIRST_SENTINEL)
        return found.reduce_or()
    if count >= HALF:
        var found = bytes.unsafe_load[width=HALF]().ge(FIRST_SENTINEL)
        found |= bytes.unsafe_offset(count - HALF).unsafe_load[width=HALF]().ge(FIRST_SENTINEL)
        return found.reduce_or()
    for index in range(count):
        if bytes[unsafe_offset=index] >= FIRST_SENTINEL:
            return True
    return False


def refuse_sentinels(reference: ImmSpan[UInt8, _], query: ImmSpan[UInt8, _]) raises AlignmentError:
    """Raises on a 0xFE or 0xFF byte, which UTF-8 never holds and the wavefront marks the sequences' ends with: one
    in a sequence would match its end, as the C API refuses it too."""
    if holds_sentinel(reference) or holds_sentinel(query):
        raise AlignmentError(ErrorKind.UNKNOWN_SYMBOL, "a 0xFE or 0xFF byte, the wavefront's end markers")


def distance_in_lanes(
    reference: ImmSpan[UInt8, _], query: ImmSpan[UInt8, _], costs: Costs, mode: Mode, band: Band, max_cost: Int
) -> Optional[Optional[Int]]:
    """The pair's capped cost as a batch of it alone settles it in the lanes (see `distances_in_lanes`), or
    None when they leave it: a call whose searches' rings could pass the memory allowed takes it, so it
    answers as its batch does, and is refused only where the batch's own search is."""
    var found = List[Optional[Int]](length=1, fill=None)
    var settled = List[Bool](length=1, fill=False)
    _ = distances_in_lanes(
        1,
        SpanTexts.of(reference),
        SpanTexts.of(query),
        costs,
        mode,
        band,
        max_cost,
        1,
        found.unsafe_ptr(),
        settled.unsafe_ptr(),
    )
    if not settled[0]:
        return None
    return Optional[Optional[Int]](found[0])


def alignment_in_lanes(
    reference: ImmSpan[UInt8, _],
    query: ImmSpan[UInt8, _],
    costs: Costs,
    mode: Mode,
    band: Band,
    max_cost: Int,
    ties: Ties,
    eqx: Bool,
    limit: Int,
) -> Optional[Optional[Alignment]]:
    """The pair's capped alignment as a batch of it alone settles it in the lanes (see `alignments_in_lanes`),
    or None when they leave it (see `distance_in_lanes`)."""
    var laned = List[Optional[Int]](length=1, fill=None)
    var paths = List[List[UInt8]](capacity=1)
    paths.append(List[UInt8]())
    var spans = List[Int](length=4, fill=0)
    var settled = List[Bool](length=1, fill=False)
    var penalties = alignments_in_lanes(
        1,
        SpanTexts.of(reference),
        SpanTexts.of(query),
        costs,
        mode,
        band,
        max_cost,
        ties,
        1,
        limit,
        laned.unsafe_ptr(),
        paths.unsafe_ptr(),
        spans.unsafe_ptr(),
        settled.unsafe_ptr(),
    )
    if not settled[0]:
        return None
    if not laned[0]:
        return Optional[Optional[Alignment]](Optional[Alignment]())
    var moves = List[UInt8]()
    swap(moves, paths[0])
    var spelled = alignment_from_lanes(
        reference,
        query,
        laned[0].value(),
        moves^,
        spans.unsafe_ptr(),
        penalties.value(),
        eqx,
        limit,
    )
    if not spelled:
        return None
    return Optional[Optional[Alignment]](spelled^)


def cost_within(
    reference: ImmSpan[UInt8, _], query: ImmSpan[UInt8, _], costs: Costs, mode: Mode, band: Band, max_cost: Int
) raises AlignmentError -> Optional[Int]:
    """The least cost, or None when it would pass `max_cost` (`Int.MAX` for no cap) or none fits `band`."""
    var space = SearchSpace()
    return cost_within(reference, query, costs, mode, band, max_cost, space)


def cost_within(
    reference: ImmSpan[UInt8, _],
    query: ImmSpan[UInt8, _],
    costs: Costs,
    mode: Mode,
    band: Band,
    max_cost: Int,
    mut space: SearchSpace,
) raises AlignmentError -> Optional[Int]:
    """`cost_within` through `space`'s searches, which a batch's worker keeps from pair to pair (see
    `gap_affine.SearchSpace`)."""
    if mode.is_scored():
        raise AlignmentError(
            ErrorKind.INVALID_ARGUMENT, "a mode with a match score maximizes a score, which `align` finds"
        )
    refuse_sentinels(reference, query)
    var columns = len(reference)
    var rows = len(query)
    var ends = EndsFree.of(mode, columns, rows)
    if swept_by_bits(costs, ends, band, max_cost, columns, rows):
        try:
            var scale = costs.unit_scale()
            if ends.first_begin == 0 and ends.first_end == 0:
                return edit_distance(reference, query, space.edit) * scale
            var started = space.edit.tally.start()
            var found: Int
            if ends.first_end == 0:
                # A suffix: a prefix of both reversed.
                var query_back = reversed_list(query)
                var reference_back = reversed_list(reference)
                found = edit_search(Span(query_back), Span(reference_back), True).distance
            else:
                found = edit_search(query, reference, ends.first_begin == 0).distance
            space.edit.tally.since(SEARCH_NS, started)
            space.edit.tally.set(ENGINE, ENGINE_SEARCH)
            return found * scale
        except error:
            # More symbols than the sweep takes: the wavefront takes any.
            if error.kind != ErrorKind.UNKNOWN_SYMBOL:
                raise error
    var started = space.edit.tally.start()
    var penalties = space.penalties_for(costs)
    if max_cost < 0:
        return None
    if not rings_fit(penalties, len(reference), len(query), DEFAULT_MAX_MEMORY, ends):
        # The lanes first, as a batch of this pair alone takes it; else refused.
        var laned = distance_in_lanes(reference, query, costs, mode, band, max_cost)
        if laned:
            space.edit.tally.set(ENGINE, ENGINE_LANES)
            return laned.value()
        rings_within(penalties, len(reference), len(query), DEFAULT_MAX_MEMORY, ends)
    # No cap is the usual case, and a 64-bit division per pair counts when short reads take a microsecond.
    var ceiling = Int.MAX if max_cost == Int.MAX else max_cost // penalties.scale
    var cost: Int
    if costs.pieces() == 2:
        cost = wavefront_distance[2](reference, query, penalties, ceiling, space.wave, ends, band)
    else:
        cost = wavefront_distance[1](reference, query, penalties, ceiling, space.wave, ends, band)
    space.tally_wavefronts(started)
    if cost < 0:
        return None
    return cost * penalties.scale


comptime NOT_FOUND = -1
"""The start an alignment that is not there has: the internal searches give one rather than an `Optional`,
whose moves of a value this large the compiler copies a byte at a time: through the three calls between
`Aligner.align` and the search, a 250 bp Illumina read's alignment took 0.585 µs on the Skylake-X that
way and takes 0.420 so, at gap-affine costs 0.995 and 0.840, built with jumps clear of 32-byte
boundaries."""


def not_found() -> Alignment:
    """No alignment: none within the cap or the band."""
    return Alignment(0, 0, String(), NOT_FOUND, NOT_FOUND, NOT_FOUND, NOT_FOUND)


@always_inline
def aligned(alignment: Alignment) -> Bool:
    """Whether `alignment` is one, not `not_found()`."""
    return alignment.reference_start != NOT_FOUND


def optional(var alignment: Alignment) -> Optional[Alignment]:
    """`alignment` as the public functions give one under a cap: None for `not_found()`."""
    if not aligned(alignment):
        return None
    return alignment^


def aligned_within(
    reference: ImmSpan[UInt8, _],
    query: ImmSpan[UInt8, _],
    costs: Costs,
    mode: Mode,
    band: Band,
    max_cost: Int,
    ties: Ties,
    eqx: Bool,
    limit: Int,
) raises AlignmentError -> Alignment:
    """An optimal alignment, or `not_found()` when its cost would pass `max_cost` (`Int.MAX` for no cap)
    or none fits `band`: the least costly one, or for a mode that maximizes a score the best-scoring
    one, its kept fronts within `limit` diagonals."""
    var space = SearchSpace()
    return aligned_within(reference, query, costs, mode, band, max_cost, ties, eqx, limit, space)


def aligned_within(
    reference: ImmSpan[UInt8, _],
    query: ImmSpan[UInt8, _],
    costs: Costs,
    mode: Mode,
    band: Band,
    max_cost: Int,
    ties: Ties,
    eqx: Bool,
    limit: Int,
    mut space: SearchSpace,
    runs: Bool = False,
) raises AlignmentError -> Alignment:
    """`aligned_within` through `space`'s searches, which a batch's worker keeps from pair to pair; with `runs`
    the CIGAR as runs (`Alignment.runs`) whichever engine finds it."""
    refuse_sentinels(reference, query)
    # The engines find the cost and the span by `LEFT`; `KSW2` then writes the letters between again, unless the
    # edit engine or the global wavefront walked them by it already.
    var engine_ties = Ties.LEFT if ties == Ties.KSW2 else ties
    if not mode.is_scored():
        var walked = False
        var found = least_costly(reference, query, costs, mode, band, max_cost, ties, eqx, limit, space, walked, runs)
        if ties == Ties.KSW2 and not walked:
            ksw_rewrite(found, reference, query, costs, eqx, runs, space.ties, band, max_bytes=limit * KEPT_BYTES)
        if runs:
            as_runs(found)
        return found^
    if max_cost != Int.MAX:
        raise AlignmentError(ErrorKind.INVALID_ARGUMENT, "a mode with a match score takes no cost cap")
    _ = penalties_of(costs)
    var started = space.edit.tally.start()
    var best = best_scoring(reference, query, costs, mode, band, engine_ties, eqx, limit)
    space.edit.tally.since(SCORED_NS, started)
    space.edit.tally.set(ENGINE, ENGINE_SCORED)
    if ties == Ties.KSW2:
        ksw_rewrite(
            best,
            reference,
            query,
            rewarded_costs(costs, mode.match_score),
            eqx,
            runs,
            space.ties,
            band,
            mode.match_score,
            mode.kind == Mode.EXTENSION and mode.anchor == Anchor.END,
            limit * KEPT_BYTES,
        )
    if runs:
        as_runs(best)
    return best^


def rewarded_costs(costs: Costs, reward: Int) -> Costs:
    """Costs whose least-cost alignment between two fixed ends is the best-scoring one with a match earning
    `reward`: a substitution `2 (x + a)`, an opening `2 o`, an extension `2 e + a` (the score at a cell is the
    reward's `(i + j) / 2` less half the cost)."""
    if reward == 0:
        return costs
    var out = Costs(
        2 * (costs.mismatch + reward),
        2 * costs.opening,
        2 * costs.extension + reward,
        2 * costs.opening2 if costs.opening2 >= 0 else -1,
        2 * costs.extension2 + reward if costs.opening2 >= 0 else 0,
    )
    out.deletion_opening = 2 * costs.deletion_opening
    out.deletion_extension = 2 * costs.deletion_extension + reward
    out.deletion_opening2 = 2 * costs.deletion_opening2 if costs.deletion_opening2 >= 0 else -1
    out.deletion_extension2 = 2 * costs.deletion_extension2 + reward if costs.deletion_opening2 >= 0 else 0
    return out


def ksw_rewrite(
    mut found: Alignment,
    reference: ImmSpan[UInt8, _],
    query: ImmSpan[UInt8, _],
    costs: Costs,
    eqx: Bool,
    runs: Bool,
    mut buf: TieBuffers,
    band: Band = Band(),
    reward: Int = 0,
    from_end: Bool = False,
    max_bytes: Int = DEFAULT_MAX_MEMORY,
):
    """An alignment's CIGAR written again by KSW2's rule (`Ties.KSW2`, `ksw_ties.ksw_cigar`) over the same span,
    guided by the path it holds and inside `band`: the same cost (or with a match `reward` the same score, its
    cost then the reward's matches less it), the one equal alignment KSW2 picks; left as it is where the cells
    to sweep would take more than `max_bytes`."""
    if found.reference_end <= found.reference_start and found.query_end <= found.query_start:
        return
    var guide = found.runs.copy() if len(found.runs) > 0 else runs_of(found.cigar)
    # The band's diagonals as the span's own cells read them, `i - j` at reference letter `i` and query letter
    # `j` of the span: counted from the start of both sequences, or for an extension fixed at their ends, from
    # there (`Band.mirrored`).
    var n = found.reference_end - found.reference_start
    var m = found.query_end - found.query_start
    var low: Int
    var high: Int
    if from_end:
        low = (n - m) - band.high
        high = (n - m) - band.low
    else:
        var shift = found.reference_start - found.query_start
        low = band.low - shift
        high = band.high - shift
    var written = ksw_cigar(
        reference[found.reference_start : found.reference_end],
        query[found.query_start : found.query_end],
        costs,
        Span(guide),
        buf,
        low,
        high,
        max_bytes=max_bytes,
    )
    if reward > 0:
        var matches = 0
        for r in written:
            if UInt8(r & 0xFF) == UInt8(ord("=")):
                matches += Int(r >> 8)
        found.cost = reward * matches - found.score
    if not eqx:
        # `=` and `X` as one `M`.
        var merged = List[UInt64](capacity=len(written))
        for r in written:
            var letter = UInt8(r & 0xFF)
            if letter == UInt8(ord("=")) or letter == UInt8(ord("X")):
                letter = UInt8(ord("M"))
            if len(merged) > 0 and UInt8(merged[len(merged) - 1] & 0xFF) == letter:
                merged[len(merged) - 1] += r & ~UInt64(0xFF)
            else:
                merged.append((r & ~UInt64(0xFF)) | UInt64(letter))
        written = merged^
    if len(found.runs) > 0 or runs:
        found.runs = written^
        found.cigar = String()
    else:
        var text = String()
        for r in written:
            text += String(Int(r >> 8))
            text += chr(Int(r & 0xFF))
        found.cigar = text^


def as_runs(mut found: Alignment):
    """An alignment's CIGAR as runs, if it is still text."""
    if len(found.runs) == 0 and found.cigar.byte_length() > 0:
        found.runs = runs_of(found.cigar)
        found.cigar = String()


def swept_by_bits(costs: Costs, ends: EndsFree, band: Band, max_cost: Int, columns: Int, rows: Int) -> Bool:
    """Whether the bit-parallel sweep serves a pair: unit costs, no cap and no band, both sequences
    holding a letter, and free ends of none, the query inside the reference, or at its start or end."""
    if costs.unit_scale() == 0 or max_cost != Int.MAX or not band.covers(columns, rows) or columns == 0 or rows == 0:
        return False
    if ends.second_begin != 0 or ends.second_end != 0:
        return False
    if ends.first_begin == 0:
        return ends.first_end == 0 or ends.first_end == columns
    return ends.first_begin == columns and (ends.first_end == columns or ends.first_end == 0)


def bits_serve(costs: Costs, mode: Mode, band: Band, max_cost: Int) -> Bool:
    """Whether the bit-parallel sweep serves a batch's pairs (see `swept_by_bits`), as it does every pair
    of a batch of unit costs with no cap and no band, globally or with the query found inside, at the
    start or at the end of the reference. Such a batch's alignments take the sweep pair by pair, which beat the lanes on
    short reads and long: 1.26 against 1.44 us a 150 bp read, 21.9 against 23.5 a 1 kbp read at 10%, on
    the Skylake-X. Its distances take the lanes, which beat it on short reads, 0.23 against 0.40 us."""
    if costs.unit_scale() == 0 or max_cost != Int.MAX or not band.covers_any():
        return False
    if mode.is_scored() or mode.query_start != 0 or mode.query_end != 0:
        return False
    if mode.reference_start == 0:
        return mode.reference_end == 0 or mode.reference_end >= ANY_LENGTH
    return mode.reference_start >= ANY_LENGTH and (mode.reference_end >= ANY_LENGTH or mode.reference_end == 0)


def least_costly(
    reference: ImmSpan[UInt8, _],
    query: ImmSpan[UInt8, _],
    costs: Costs,
    mode: Mode,
    band: Band,
    max_cost: Int,
    rule: Ties,
    eqx: Bool,
    limit: Int,
    mut space: SearchSpace,
    mut walked: Bool,
    runs: Bool = False,
) raises AlignmentError -> Alignment:
    """The least costly alignment with `mode`'s free ends, or `not_found()`: by the bit-parallel sweep
    where it serves the pair (see `swept_by_bits`), else by the wavefront, the same alignment at the same
    costs, through `space`'s searches. With `runs` the sweep writes its CIGAR as runs (`Alignment.runs`);
    the wavefront's stays text, which `aligned_within` turns. `Ties.KSW2` the sweep and a global wavefront walk
    themselves where they can, setting `walked`; elsewhere the span and the CIGAR are `Ties.LEFT`'s, for
    `aligned_within` to write again."""
    var ties = Ties.LEFT if rule == Ties.KSW2 else rule
    walked = False
    var columns = len(reference)
    var rows = len(query)
    var ends = EndsFree.of(mode, columns, rows)
    if swept_by_bits(costs, ends, band, max_cost, columns, rows):
        try:
            var scale = costs.unit_scale()
            if ends.first_begin == 0 and ends.first_end == 0:
                var whole = edit_cigar(reference, query, space.edit, eqx, rule, kept_bytes(limit), runs)
                walked = rule == Ties.KSW2
                var cost = whole.distance * scale
                var whole_runs = whole.runs^
                whole.runs = List[UInt64]()
                return Alignment(cost, -cost, whole^.take_cigar(), 0, columns, 0, rows, whole_runs^)
            var started = space.edit.tally.start()
            var hit: EditHit
            if ends.first_end == 0:
                # A suffix: a prefix of both reversed, the rule's span mirrored, as `Ties.RIGHT` is `Ties.LEFT`
                # over both reversed.
                var query_back = reversed_list(query)
                var reference_back = reversed_list(reference)
                var mirrored = edit_search(
                    Span(query_back),
                    Span(reference_back),
                    True,
                    Ties.RIGHT if ties == Ties.LEFT else Ties.LEFT,
                )
                hit = EditHit(mirrored.distance, columns - mirrored.end, columns - mirrored.start)
            else:
                hit = edit_search(query, reference, ends.first_begin == 0, ties)
            space.edit.tally.since(SEARCH_NS, started)
            var found = edit_cigar(
                reference[hit.start : hit.end], query, space.edit, eqx, rule, kept_bytes(limit), runs
            )
            walked = rule == Ties.KSW2
            var cost = found.distance * scale
            var found_runs = found.runs^
            found.runs = List[UInt64]()
            return Alignment(cost, -cost, found^.take_cigar(), hit.start, hit.end, 0, rows, found_runs^)
        except error:
            if error.kind != ErrorKind.UNKNOWN_SYMBOL:
                raise error
    var started = space.edit.tally.start()
    var penalties = space.penalties_for(costs)
    if max_cost < 0:
        return not_found()
    if not rings_fit(penalties, len(reference), len(query), kept_bytes(limit), ends):
        # The lanes first, as a batch of this pair alone takes it; else refused.
        var laned = alignment_in_lanes(reference, query, costs, mode, band, max_cost, ties, eqx, limit)
        if laned:
            space.edit.tally.set(ENGINE, ENGINE_LANES)
            var spelled = laned.take()
            if not spelled:
                return not_found()
            return spelled.take()
        rings_within(penalties, len(reference), len(query), kept_bytes(limit), ends)
    var ceiling = Int.MAX if max_cost == Int.MAX else max_cost // penalties.scale
    if mode.is_global():
        var found: Optional[AffineCigar]
        if costs.pieces() == 2:
            found = cigar_within[2](reference, query, penalties, eqx, ceiling, space.wave, band, rule, limit)
        else:
            found = cigar_within[1](reference, query, penalties, eqx, ceiling, space.wave, band, rule, limit)
        space.tally_wavefronts(started)
        if not found:
            return not_found()
        walked = found.value().walked
        var cost = found.value().cost
        return Alignment(cost, -cost, found.take().cigar, 0, columns, 0, rows)
    var spanned: Optional[Spanned]
    if costs.pieces() == 2:
        spanned = free_ends_alignment[2](
            space.wave.forward2,
            space.wave.backward2,
            reference,
            query,
            penalties,
            eqx,
            ceiling,
            ends,
            band,
            ties,
            limit,
        )
    else:
        spanned = free_ends_alignment[1](
            space.wave.forward, space.wave.backward, reference, query, penalties, eqx, ceiling, ends, band, ties, limit
        )
    space.tally_wavefronts(started)
    if not spanned:
        return not_found()
    ref found = spanned.value()
    return Alignment(
        found.cost,
        -found.cost,
        found.cigar,
        found.first_start,
        found.first_end,
        found.second_start,
        found.second_end,
    )


def best_scoring(
    reference: ImmSpan[UInt8, _],
    query: ImmSpan[UInt8, _],
    costs: Costs,
    mode: Mode,
    band: Band,
    ties: Ties,
    eqx: Bool,
    limit: Int,
) raises AlignmentError -> Alignment:
    """The best-scoring alignment for a mode with a match score: an extension by the wavefront from its
    anchor; a global alignment, its letters fixed so the reward folds into the costs, by the wavefront
    too; a local alignment or other free ends by sweep (see `scored`)."""
    var columns = len(reference)
    var rows = len(query)
    if mode.kind == Mode.EXTENSION:
        return extended_alignment(reference, query, costs, mode, band, ties, eqx, limit)
    if mode.kind == Mode.ENDS and mode.is_global():
        var whole = global_rewarded(reference, query, costs, mode.match_score, band, ties, eqx, limit)
        var score = mode.match_score * cigar_matches(reference, query, whole[1]) - whole[0]
        return Alignment(whole[0], score, whole[1], 0, columns, 0, rows)
    if mode.match_score <= 0:
        raise AlignmentError(
            ErrorKind.INVALID_SCORING, "a local alignment under Costs needs a match that earns: Mode.local"
        )
    if not band.covers(columns, rows):
        raise AlignmentError(ErrorKind.INVALID_ARGUMENT, "a sweep takes no band: a local alignment or free ends")
    if mode.kind == Mode.SMITH_WATERMAN:
        return local_alignment(reference, query, costs, mode.match_score, ties, eqx, limit)
    return rewarded_alignment(
        reference, query, costs, mode.match_score, EndsFree.of(mode, columns, rows), ties, eqx, limit
    )


def unrewarded_reaching(mode: Mode) raises AlignmentError -> Mode:
    """`mode.reaching_end()` with no match score, for an extension the search does not run, one side empty or no
    reward: no letter earns anything there, so the alignment through the query that scores best is the least
    costly one, its score minus its cost, and the wavefront finds it inside a band, which the sweep a reward takes
    does not."""
    var from_end = mode.anchor == Anchor.END
    return Mode.ends_free(reference_start=Int.MAX if from_end else 0, reference_end=0 if from_end else Int.MAX)


def reaching_band(mode: Mode, band: Band, columns: Int, rows: Int) -> Band:
    """An extension's band as the free-ends alignment through the query counts it: that one's diagonals run from
    the origin, an extension's from the end back from the corner, so from the end it is mirrored there."""
    return band.mirrored(columns - rows) if mode.anchor == Anchor.END else band


def extended_alignment(
    reference: ImmSpan[UInt8, _],
    query: ImmSpan[UInt8, _],
    costs: Costs,
    mode: Mode,
    band: Band,
    ties: Ties,
    eqx: Bool,
    limit: Int,
) raises AlignmentError -> Alignment:
    """The best extension from `mode`'s anchor (see `Mode.extension`)."""
    var two = costs.pieces() == 2
    var penalties = rewarded_penalties(mode.match_score, costs)
    rings_within(penalties, len(reference), len(query), kept_bytes(limit))
    # The Z-drop's slack a diagonal is the cheapest extension, as KSW2 charges a long gap.
    var drop_extension = costs.cheapest_extension()
    var found = extension_of[2](
        reference, query, penalties, eqx, mode.anchor, band, ties, -1, limit, mode.zdrop, drop_extension, mode.end_bonus
    ) if two else extension_of[1](
        reference, query, penalties, eqx, mode.anchor, band, ties, -1, limit, mode.zdrop, drop_extension, mode.end_bonus
    )
    # The search weighs the end bonus itself (see `extension_of`), but for an empty side or no reward,
    # where it does not run: there the free ends' own alignment through the query is weighed against it.
    var searched = len(reference) > 0 and len(query) > 0 and penalties.reward > 0
    if mode.end_bonus > 0 and not found.dropped and not searched:
        var reaching = aligned_within(
            reference,
            query,
            costs,
            unrewarded_reaching(mode),
            reaching_band(mode, band, len(reference), len(query)),
            Int.MAX,
            ties,
            eqx,
            limit,
        )
        if aligned(reaching) and reaching.score + mode.end_bonus > found.score:
            return reaching^
    var columns = len(reference)
    var rows = len(query)
    var cost = mode.match_score * found.matches - found.score
    if mode.anchor == Anchor.END:
        return Alignment(
            cost,
            found.score,
            found.cigar,
            columns - found.first_length,
            columns,
            rows - found.second_length,
            rows,
        )
    return Alignment(cost, found.score, found.cigar, 0, found.first_length, 0, found.second_length)


# endregion Costs

# region Scoring


def distances_in_lanes[
    T: Texts
](
    pairs: Int,
    references: T,
    queries: T,
    costs: Costs,
    mode: Mode,
    band: Band,
    max_cost: Int,
    workers: Int,
    found: MutPointer[Optional[Int], _],
    settled: MutPointer[Bool, _],
) -> Int:
    """A batch's distances as many pairs at once as a register holds lanes (see `lanes`), every pair `settled`
    does not mark that they take: costs with no reward, globally or with free ends, which the searches would
    take too. The pairs settled are counted; none for costs the searches refuse, which a pair's own search
    then reports."""
    var lane_costs = LaneCosts.of(costs, mode, free_ends=True)
    if not lane_costs:
        return 0
    try:
        _ = penalties_of(costs)
    except:
        return 0
    return lane_distances(pairs, references, queries, lane_costs.value(), band, max_cost, workers, found, settled, mode)


def alignments_in_lanes[
    T: Texts
](
    pairs: Int,
    references: T,
    queries: T,
    costs: Costs,
    mode: Mode,
    band: Band,
    max_cost: Int,
    ties: Ties,
    workers: Int,
    limit: Int,
    found: MutPointer[Optional[Int], _],
    paths: MutPointer[List[UInt8], _],
    spans: MutPointer[Int, _],
    settled: MutPointer[Bool, _],
) -> Optional[Penalties]:
    """A batch's alignments as many pairs at once as a register holds lanes, each traced from the flags its band
    kept (see `lanes`), with free ends the span the tie rule picks first: every pair `settled` does not mark that
    they take, its cost, its path's moves and its span, four numbers, into `found`, `paths` and `spans`. The
    costs' penalties, for spelling each, or None where the lanes took none: unit costs the bit-parallel sweep
    serves, which it beats the lanes on, free ends under a band, which would bound the span's search too, and
    costs the searches refuse."""
    var unbanded = band.covers_any()
    if bits_serve(costs, mode, band, max_cost):
        return None
    var lane_costs = LaneCosts.of(costs, mode, unbanded)
    if not lane_costs:
        return None
    var penalties: Penalties
    try:
        penalties = penalties_of(costs)
    except:
        return None
    if mode.is_global():
        for index in range(pairs):
            spans[unsafe_offset=4 * index] = 0
            spans[unsafe_offset=4 * index + 1] = references.length(index)
            spans[unsafe_offset=4 * index + 2] = 0
            spans[unsafe_offset=4 * index + 3] = queries.length(index)
        _ = lane_alignments(
            pairs,
            references,
            queries,
            lane_costs.value(),
            band,
            max_cost,
            ties == Ties.LEFT,
            workers,
            limit * KEPT_BYTES,
            found,
            paths,
            settled,
        )
    else:
        lane_free_alignments(
            pairs,
            references,
            queries,
            lane_costs.value(),
            mode,
            max_cost,
            ties == Ties.RIGHT,
            workers,
            limit * KEPT_BYTES,
            found,
            paths,
            spans,
            settled,
        )
    return penalties


def alignment_from_lanes(
    reference: ImmSpan[UInt8, _],
    query: ImmSpan[UInt8, _],
    cost: Int,
    var moves: List[UInt8],
    span: ImmPointer[Int, _],
    penalties: Penalties,
    eqx: Bool,
    limit: Int,
) -> Optional[Alignment]:
    """The alignment the lanes traced for a pair (see `alignments_in_lanes`), its CIGAR spelled from their moves over
    its span, `span[0]` to `span[1]` of the reference and `span[2]` to `span[3]` of the query; None where the
    searches might have split the span to keep their fronts within `limit` entries. The lanes follow the tie rule
    across the whole span, as the searches do unless they split it, which they never do while both sides'
    fronts, each at most a diagonal of the matrix at every cost, stay within it; such a pair takes them."""
    var first_start = span[unsafe_offset=0]
    var first_end = span[unsafe_offset=1]
    var second_start = span[unsafe_offset=2]
    var second_end = span[unsafe_offset=3]
    var units = cost // penalties.scale
    if 2 * (units + 1) * (first_end - first_start + second_end - second_start + 1) > limit:
        return None
    var cigar = cigar_of(
        reference[first_start:first_end], query[second_start:second_end], moves^, units, penalties, eqx
    )
    return Alignment(cost, -cost, cigar^, first_start, first_end, second_start, second_end)
