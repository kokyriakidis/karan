# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the
# MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
The entry points a caller uses, each deciding which search serves a pair so the caller never has to.

Under `Costs`, which an alignment minimizes, `distance` finds the least cost and `align` an optimal
alignment as a CIGAR, for any `Mode`; `distances` and `alignments` take a batch, many pairs at once.
A single pair comes as two `String`s or as two byte spans, read where they lie.
Every call runs on the caller's own thread unless asked for more (see `distances`), and keeps no state
between calls, so an application may call it from as many threads as it likes; an `Aligner` keeps one
thread's memory from call to call.
Unit costs, global or with the query found inside, at the start or at the end of the reference, take the
bit-parallel band doubling of A*PA2 (see `edit_distance` and `edit_search`), the end as the start of both
sequences reversed; every other case, and a
pair holding more symbols than it takes, the gap-affine wavefront from both ends (see `gap_affine`),
which gives the same alignment at the same costs. The modes with a match score maximize a score: an
extension, and a global alignment with a reward, by the wavefront; a local alignment and other free
ends with a reward by sweep (see `scored`).

Under a `Scoring`, an alphabet's substitution table and gap scores, which an alignment maximizes,
`score` and `align` take a global or a local alignment, on the host or the device (see `scoring`).

This module holds the public calls alone. Which engine takes one pair is `engines`' choice (`cost_within`,
`aligned_within`), many pairs at once `batches`' (the lanes, the threads, the GPU), the costs each engine reads
`penalties`', and a thread's kept memory `search_space`'s.
"""

from std.atomic import Atomic

from .batches import capped_alignments, capped_distances, gpu_distances
from .common import Device, Placement, next_share, spread, thread_count
from .engines import (
    aligned,
    aligned_within,
    cost_within,
    optional,
    reaching_band,
    refuse_sentinels,
    unrewarded_reaching,
)
from .errors import AlignmentError, ErrorKind
from .gap_affine import DEFAULT_MAX_MEMORY, EndsFree, KEPT_BYTES, extend, outside, rings_within, wavefront_distance
from .gotoh import AlignmentMode, GappedAlignment
from .modes import Alignment, Anchor, Band, Costs, Mode, Ties
from .penalties import penalties_of, rewarded_penalties
from .scored import FROM_EDGE, LocalScores, end_of, local_scores as local_scores_of, swept
from .scoring import (
    ScoreReach,
    Scoring,
    as_alignment,
    batch_scores,
    cells_within,
    device_batch_alignments,
    first_past_32_bits,
    laned_alignments,
    pair_count,
    scoring_alignment,
    scoring_score,
)
from .search import Hit, local_scores_by_lane
from .search_space import SearchSpace


def distance(
    reference: String, query: String, costs: Costs = Costs.edit(), mode: Mode = Mode.GLOBAL, *, band: Band = Band()
) raises AlignmentError -> Int:
    """The least cost of aligning `query` to `reference` as `mode` asks, every move inside `band` (see
    `Band`), with no alignment; raises when no alignment fits the band.

    Unit costs, globally, take A*PA2's band doubling: guess a bound, sweep only the cells a path within
    it could cross, raise the guess until the answer fits under it. Other costs take the wavefront from
    both ends, keeping only a few costs' fronts, so a few rows of memory however long the pair.
    """
    var found = cost_within(reference.as_bytes(), query.as_bytes(), costs, mode, band, Int.MAX)
    if not found:
        raise outside(band)
    return found.value()


def distance(
    reference: ImmSpan[UInt8, _],
    query: ImmSpan[UInt8, _],
    costs: Costs = Costs.edit(),
    mode: Mode = Mode.GLOBAL,
    *,
    band: Band = Band(),
) raises AlignmentError -> Int:
    """`distance` over the two sequences' bytes where they lie, a `List[UInt8]`'s or any buffer's, no
    `String` made of them: the same cost. Each single-pair call under `Costs` has such an overload, and
    every search beneath it reads the bytes alone, so the `String` overloads pass theirs down."""
    var found = cost_within(reference, query, costs, mode, band, Int.MAX)
    if not found:
        raise outside(band)
    return found.value()


def distance(
    reference: String,
    query: String,
    costs: Costs = Costs.edit(),
    mode: Mode = Mode.GLOBAL,
    *,
    max_cost: Int,
    band: Band = Band(),
) raises AlignmentError -> Optional[Int]:
    """`distance`, or None when it would pass `max_cost` or no alignment fits `band`: the searches stop
    once each has grown to about half of `max_cost` without the two meeting within it, so a pair far
    over costs a fraction of its full search."""
    return cost_within(reference.as_bytes(), query.as_bytes(), costs, mode, band, max_cost)


def distance(
    reference: ImmSpan[UInt8, _],
    query: ImmSpan[UInt8, _],
    costs: Costs = Costs.edit(),
    mode: Mode = Mode.GLOBAL,
    *,
    max_cost: Int,
    band: Band = Band(),
) raises AlignmentError -> Optional[Int]:
    """`distance` under a cap, over the two sequences' bytes."""
    return cost_within(reference, query, costs, mode, band, max_cost)


def align(
    reference: String,
    query: String,
    costs: Costs = Costs.edit(),
    mode: Mode = Mode.GLOBAL,
    *,
    band: Band = Band(),
    ties: Ties = Ties.KSW2,
    eqx: Bool = True,
    max_memory: Int = DEFAULT_MAX_MEMORY,
) raises AlignmentError -> Alignment:
    """An optimal alignment of `query` to `reference` as `mode` asks, every move inside `band`, as a
    CIGAR with `=` and `X`, or with `eqx` false `M` for both (see `Alignment`); raises when no
    alignment fits the band.

    Of several equally good alignments the CIGAR is always the one `ties` names (see `Ties`): by
    default every edit as far left as it goes, indels placed as minimap2 places them, or with
    `Ties.RIGHT` as far right, WFA2-lib's CIGAR byte for byte, whichever search found the cost.

    Every byte is a symbol matching only itself, so DNA in either case, or any other text, needs no
    alphabet; only 0xFE and 0xFF, which UTF-8 never holds, are refused (`ErrorKind.UNKNOWN_SYMBOL`). The wavefront's work grows with the square of the cost rather than with the matrix, and
    its memory stays bounded: the fronts it keeps for the traceback, the bulk of it, stay within
    `max_memory` bytes, past which the pair is split where an optimal path crosses and each piece
    aligned alone (see `gap_affine.solve`), the cost still the least, the tie rule followed within each
    piece. At unit costs the bit-parallel band keeps its tiles' edges for the traceback within
    `max_memory` too, past which it sweeps stretches of the band again from the edges it kept, the same
    alignment byte for byte (see `band.retrace`). The sweeps keep a few rows whatever the cap.
    """
    var found = aligned_within(
        reference.as_bytes(), query.as_bytes(), costs, mode, band, Int.MAX, ties, eqx, max_memory // KEPT_BYTES
    )
    if not aligned(found):
        raise outside(band)
    return found^


def align(
    reference: ImmSpan[UInt8, _],
    query: ImmSpan[UInt8, _],
    costs: Costs = Costs.edit(),
    mode: Mode = Mode.GLOBAL,
    *,
    band: Band = Band(),
    ties: Ties = Ties.KSW2,
    eqx: Bool = True,
    max_memory: Int = DEFAULT_MAX_MEMORY,
) raises AlignmentError -> Alignment:
    """`align` over the two sequences' bytes where they lie, no `String` made of them (see `distance`):
    the same alignment, the CIGAR its `ties` picks byte for byte."""
    var found = aligned_within(reference, query, costs, mode, band, Int.MAX, ties, eqx, max_memory // KEPT_BYTES)
    if not aligned(found):
        raise outside(band)
    return found^


def align(
    reference: String,
    query: String,
    costs: Costs = Costs.edit(),
    mode: Mode = Mode.GLOBAL,
    *,
    max_cost: Int,
    band: Band = Band(),
    ties: Ties = Ties.KSW2,
    eqx: Bool = True,
    max_memory: Int = DEFAULT_MAX_MEMORY,
) raises AlignmentError -> Optional[Alignment]:
    """`align`, or None when the cost would pass `max_cost` or no alignment fits `band`, found as
    `distance` finds that, with no fronts traced. A mode with a match score, which maximizes a score,
    takes no cap."""
    return optional(
        aligned_within(
            reference.as_bytes(), query.as_bytes(), costs, mode, band, max_cost, ties, eqx, max_memory // KEPT_BYTES
        )
    )


def align(
    reference: ImmSpan[UInt8, _],
    query: ImmSpan[UInt8, _],
    costs: Costs = Costs.edit(),
    mode: Mode = Mode.GLOBAL,
    *,
    max_cost: Int,
    band: Band = Band(),
    ties: Ties = Ties.KSW2,
    eqx: Bool = True,
    max_memory: Int = DEFAULT_MAX_MEMORY,
) raises AlignmentError -> Optional[Alignment]:
    """`align` under a cap, over the two sequences' bytes."""
    return optional(aligned_within(reference, query, costs, mode, band, max_cost, ties, eqx, max_memory // KEPT_BYTES))


struct Aligner(Movable):
    """One thread's aligner: `distance` and `align` as the functions of those names give them, the memory
    their searches take kept from call to call, so a loop of calls takes none once it is warm, and it is
    warm from the start for short reads (see `warm`), so it is made once and kept rather than made a call. An
    application calling from many threads keeps one a thread; one never crosses threads, and the library
    keeps no state of its own, so any number of them work at once."""

    var space: SearchSpace

    def __init__(out self, *, warm: Bool = True):
        """An aligner whose memory a short divergent pair has already grown (see `warm`), some 80 µs on the
        Skylake-X; with `warm` false, one holding no memory yet, for a single call."""
        self.space = SearchSpace()
        if warm:
            self.warm()

    def warm(mut self):
        """Aligns one built-in pair of 256 letters a third apart, at gap-affine and at unit costs, so the
        searches, their rings and kept fronts, and the profile hold what a short read takes, their pages
        touched: grown by the first reads that needed them instead, a few of those took up to sixteen
        times a read's time, 2.9% of 250 bp Illumina reads' at (4, 6, 2)."""
        var first = String()
        var second = String()
        var state: UInt64 = 0x9E3779B97F4A7C15
        comptime LETTERS = "ACGT"
        for index in range(256):
            state = state * 6364136223846793005 + 1442695040888963407
            var letter = Int(state >> 62)
            first += LETTERS[byte=letter]
            # A third of the letters changed, and every fortieth dropped or doubled.
            if index % 40 == 7:
                continue
            if index % 40 == 23:
                second += LETTERS[byte=letter]
            var changed = (letter + 1) % 4 if index % 3 == 0 else letter
            second += LETTERS[byte=changed]
        try:
            _ = self.align(first, second, Costs.affine(4, 6, 2))
            _ = self.align(first, second)
        except:
            pass

    def distance(
        mut self,
        reference: String,
        query: String,
        costs: Costs = Costs.edit(),
        mode: Mode = Mode.GLOBAL,
        *,
        band: Band = Band(),
    ) raises AlignmentError -> Int:
        """`distance`, through this aligner's memory."""
        var found = cost_within(reference.as_bytes(), query.as_bytes(), costs, mode, band, Int.MAX, self.space)
        if not found:
            raise outside(band)
        return found.value()

    def distance(
        mut self,
        reference: ImmSpan[UInt8, _],
        query: ImmSpan[UInt8, _],
        costs: Costs = Costs.edit(),
        mode: Mode = Mode.GLOBAL,
        *,
        band: Band = Band(),
    ) raises AlignmentError -> Int:
        """`distance` over the two sequences' bytes, through this aligner's memory."""
        var found = cost_within(reference, query, costs, mode, band, Int.MAX, self.space)
        if not found:
            raise outside(band)
        return found.value()

    def distance(
        mut self,
        reference: String,
        query: String,
        costs: Costs = Costs.edit(),
        mode: Mode = Mode.GLOBAL,
        *,
        max_cost: Int,
        band: Band = Band(),
    ) raises AlignmentError -> Optional[Int]:
        """`distance` under a cap, through this aligner's memory."""
        return cost_within(reference.as_bytes(), query.as_bytes(), costs, mode, band, max_cost, self.space)

    def distance(
        mut self,
        reference: ImmSpan[UInt8, _],
        query: ImmSpan[UInt8, _],
        costs: Costs = Costs.edit(),
        mode: Mode = Mode.GLOBAL,
        *,
        max_cost: Int,
        band: Band = Band(),
    ) raises AlignmentError -> Optional[Int]:
        """`distance` under a cap over the two sequences' bytes, through this aligner's memory."""
        return cost_within(reference, query, costs, mode, band, max_cost, self.space)

    def align(
        mut self,
        reference: String,
        query: String,
        costs: Costs = Costs.edit(),
        mode: Mode = Mode.GLOBAL,
        *,
        band: Band = Band(),
        ties: Ties = Ties.KSW2,
        eqx: Bool = True,
        max_memory: Int = DEFAULT_MAX_MEMORY,
    ) raises AlignmentError -> Alignment:
        """`align`, through this aligner's memory: the same alignment, the CIGAR its `ties` picks."""
        var found = aligned_within(
            reference.as_bytes(),
            query.as_bytes(),
            costs,
            mode,
            band,
            Int.MAX,
            ties,
            eqx,
            max_memory // KEPT_BYTES,
            self.space,
        )
        if not aligned(found):
            raise outside(band)
        return found^

    def align(
        mut self,
        reference: ImmSpan[UInt8, _],
        query: ImmSpan[UInt8, _],
        costs: Costs = Costs.edit(),
        mode: Mode = Mode.GLOBAL,
        *,
        band: Band = Band(),
        ties: Ties = Ties.KSW2,
        eqx: Bool = True,
        max_memory: Int = DEFAULT_MAX_MEMORY,
        runs: Bool = False,
    ) raises AlignmentError -> Alignment:
        """`align` over the two sequences' bytes, through this aligner's memory: a caller holding its
        sequences as bytes, as a POA graph's or a FASTQ reader's are, makes no `String` a call. The bytes may be
        letters or the bases' 2-bit codes (0 `A`, 1 `C`, 2 `G`, 3 `T`; any other byte a symbol of its own), the
        same alignment either way, both sequences the same way: a code matches only itself, never its letter. With `runs`, the CIGAR comes as `Alignment.runs`, numbers (`cigar.run_of`),
        `cigar` left empty: no text written and parsed back."""
        var found = aligned_within(
            reference, query, costs, mode, band, Int.MAX, ties, eqx, max_memory // KEPT_BYTES, self.space, runs
        )
        if not aligned(found):
            raise outside(band)
        return found^

    def align(
        mut self,
        reference: String,
        query: String,
        costs: Costs = Costs.edit(),
        mode: Mode = Mode.GLOBAL,
        *,
        max_cost: Int,
        band: Band = Band(),
        ties: Ties = Ties.KSW2,
        eqx: Bool = True,
        max_memory: Int = DEFAULT_MAX_MEMORY,
    ) raises AlignmentError -> Optional[Alignment]:
        """`align` under a cap, through this aligner's memory."""
        return optional(
            aligned_within(
                reference.as_bytes(),
                query.as_bytes(),
                costs,
                mode,
                band,
                max_cost,
                ties,
                eqx,
                max_memory // KEPT_BYTES,
                self.space,
            )
        )

    def align(
        mut self,
        reference: ImmSpan[UInt8, _],
        query: ImmSpan[UInt8, _],
        costs: Costs = Costs.edit(),
        mode: Mode = Mode.GLOBAL,
        *,
        max_cost: Int,
        band: Band = Band(),
        ties: Ties = Ties.KSW2,
        eqx: Bool = True,
        max_memory: Int = DEFAULT_MAX_MEMORY,
    ) raises AlignmentError -> Optional[Alignment]:
        """`align` under a cap over the two sequences' bytes, through this aligner's memory."""
        return optional(
            aligned_within(
                reference, query, costs, mode, band, max_cost, ties, eqx, max_memory // KEPT_BYTES, self.space
            )
        )


def score(
    reference: String, query: String, costs: Costs, mode: Mode = Mode.GLOBAL, *, band: Band = Band()
) raises AlignmentError -> Int:
    """The best score `align` would return, with no alignment traced: for a mode with a match score its
    matches' reward less its costs, else minus the least cost, `distance`'s. A local alignment or free
    ends with a reward take the sweep alone, an extension its search alone, and a global alignment
    with a reward the wavefront's cost with the reward folded in, so each skips the traceback."""
    return score(reference.as_bytes(), query.as_bytes(), costs, mode, band=band)


def score(
    reference: ImmSpan[UInt8, _],
    query: ImmSpan[UInt8, _],
    costs: Costs,
    mode: Mode = Mode.GLOBAL,
    *,
    band: Band = Band(),
) raises AlignmentError -> Int:
    """`score` over the two sequences' bytes, no `String` made of them."""
    if not mode.is_scored():
        return -distance(reference, query, costs, mode, band=band)
    refuse_sentinels(reference, query)
    var columns = len(reference)
    var rows = len(query)
    var two = costs.pieces() == 2
    _ = penalties_of(costs)
    var penalties = rewarded_penalties(mode.match_score, costs)
    if mode.kind == Mode.EXTENSION:
        if not band.holds(0):
            raise outside(band)
        rings_within(penalties, columns, rows, DEFAULT_MAX_MEMORY)
        var drop_extension = costs.cheapest_extension()
        var at_end = mode.anchor == Anchor.END
        var found = extend[2](
            reference,
            query,
            penalties,
            band,
            at_end,
            -1,
            mode.zdrop,
            drop_extension,
            mode.end_bonus,
        ) if two else extend[1](
            reference,
            query,
            penalties,
            band,
            at_end,
            -1,
            mode.zdrop,
            drop_extension,
            mode.end_bonus,
        )
        var best = penalties.score(found[0], found[1] + found[2])
        # The end bonus prefers the best extension reaching the query's far end, unless the Z-drop gave up.
        if mode.end_bonus > 0 and not found[3]:
            var reaching: Int
            if found[5] >= 0:
                reaching = penalties.score(found[4], found[5] + rows)
            elif columns > 0 and rows > 0 and penalties.reward > 0:
                # No front point reached the query's end before the search could stop: none comes close.
                return best
            else:
                # An empty side or no reward, where the search does not run: the free ends' own, none inside the
                # band leaving the best stop.
                try:
                    reaching = score(
                        reference,
                        query,
                        costs,
                        unrewarded_reaching(mode),
                        band=reaching_band(mode, band, columns, rows),
                    )
                except:
                    return best
            if reaching + mode.end_bonus > best:
                return reaching
        return best
    if mode.kind == Mode.ENDS and mode.is_global():
        rings_within(penalties, columns, rows, DEFAULT_MAX_MEMORY)
        var cost = wavefront_distance[2](
            reference, query, penalties, Int.MAX, EndsFree(), band
        ) if two else wavefront_distance[1](reference, query, penalties, Int.MAX, EndsFree(), band)
        if cost < 0:
            raise outside(band)
        return penalties.score(cost, columns + rows)
    if mode.match_score <= 0:
        raise AlignmentError(
            ErrorKind.INVALID_SCORING, "a local alignment under Costs needs a match that earns: Mode.local"
        )
    if not band.covers(columns, rows):
        raise AlignmentError(ErrorKind.INVALID_ARGUMENT, "a sweep takes no band: a local alignment or free ends")
    if mode.kind == Mode.SMITH_WATERMAN:
        return end_of(reference, query, costs, mode.match_score)[0]
    return swept[FROM_EDGE](reference, query, costs, mode.match_score, EndsFree.of(mode, columns, rows))[0]


def local_scores(
    reference: String, query: String, costs: Costs, mode: Mode, *, window: Optional[Int] = None
) raises AlignmentError -> LocalScores:
    """A local alignment's best score and where it ends, with the best score of an alignment ending
    more than `window` reference letters away, as SSW's `score2` and `ref_end2` report it for a mapping
    quality (see `LocalScores`): one sweep, with no alignment traced. The window is half the query, and
    at least 15, by default, as SSW suggests."""
    return local_scores(reference.as_bytes(), query.as_bytes(), costs, mode, window=window)


def local_scores(
    reference: ImmSpan[UInt8, _], query: ImmSpan[UInt8, _], costs: Costs, mode: Mode, *, window: Optional[Int] = None
) raises AlignmentError -> LocalScores:
    """`local_scores` over the two sequences' bytes, no `String` made of them."""
    if mode.kind != Mode.SMITH_WATERMAN or mode.match_score <= 0:
        raise AlignmentError(ErrorKind.INVALID_ARGUMENT, "local scores for a local alignment: Mode.local")
    _ = penalties_of(costs)
    refuse_sentinels(reference, query)
    var span = window.or_else(max(len(query) // 2, 15))
    return local_scores_of(reference, query, costs, mode.match_score, span)


def score(
    reference: String,
    query: String,
    scoring: Scoring,
    mode: Mode = Mode.GLOBAL,
    *,
    placement: Optional[Placement] = None,
) raises -> Int:
    """The optimal score under `scoring`, with no alignment traced: `Mode.GLOBAL` and `Mode.local()` in two
    rows of memory on either device (see `scoring.pair_score`), free ends and extensions by sweep on the
    host. The table holds what a match earns, so a mode's own match score must be zero:
    `Mode.extension(0)` for an extension."""
    return scoring_score(reference, query, scoring, mode, placement)


def align(
    reference: String,
    query: String,
    scoring: Scoring,
    mode: Mode = Mode.GLOBAL,
    *,
    placement: Optional[Placement] = None,
    max_memory: Int = DEFAULT_MAX_MEMORY,
    eqx: Bool = True,
) raises -> Alignment:
    """An optimal alignment under `scoring`, as `Costs` give one (see `Alignment`), its `cost` minus its
    score: both sequences whole for `Mode.GLOBAL`, Needleman-Wunsch, the best-scoring window of each
    for `Mode.local()`, Smith-Waterman, on either device; free ends and extensions, with Z-drop as KSW2
    gauges it, on the host, their span by sweep and the letters between aligned globally (see
    `scoring.scoring_alignment`). Its rows come back with `Alignment.gapped`. Of equally good
    alignments, Gotoh's walk picks the CIGAR (see `gotoh.reconstruct`), not `Ties`. A traceback whose
    stored matrix would pass `max_memory` bytes recurses in linear space instead (see
    `scoring.cells_within`)."""
    return scoring_alignment(reference, query, scoring, mode, placement, cells_within(max_memory), eqx)


def within_32_bits(references: List[String], queries: List[String], scoring: Scoring, mode: Mode, aligned: Bool) raises:
    """Refuses a batch holding a pair whose scores could pass 32 bits (see `ScoreReach`) with the error a
    serial loop of `align`, or with `aligned` false of `score`, would raise first: an earlier pair's own,
    else that pair's. Checked up front, it raised a later pair's over an earlier one's."""
    var floored = mode.kind == Mode.SMITH_WATERMAN
    var past = first_past_32_bits(scoring, references, queries, floored)
    if past < 0:
        return
    var single = Placement.on_cpu(1)
    for index in range(past):
        if aligned:
            _ = align(references[index], queries[index], scoring, mode, placement=single)
        else:
            _ = score(references[index], queries[index], scoring, mode, placement=single)
    ScoreReach.of(scoring).check(references[past].byte_length(), queries[past].byte_length(), floored)


def scores(
    references: List[String],
    queries: List[String],
    scoring: Scoring,
    mode: Mode = Mode.GLOBAL,
    *,
    placement: Optional[Placement] = None,
) raises -> List[Int]:
    """`score` for every pair; on the device, every pair one block can carry goes out in one launch."""
    var found: List[Int32]
    if mode.kind == Mode.SMITH_WATERMAN and mode.match_score == 0:
        _ = pair_count(references, queries)
        within_32_bits(references, queries, scoring, mode, False)
        found = batch_scores[AlignmentMode.LOCAL](references, queries, scoring, placement)
    elif mode.is_global() and mode.match_score == 0:
        _ = pair_count(references, queries)
        within_32_bits(references, queries, scoring, mode, False)
        found = batch_scores[AlignmentMode.GLOBAL](references, queries, scoring, placement)
    else:
        # Free ends and extensions, and modes `score` refuses: each pair as `score` takes it, over the threads
        # asked for, a pair that raises raising here as a serial loop would have raised it first.
        var pairs = pair_count(references, queries)
        var results = List[Int](length=pairs, fill=0)
        var failed = List[Bool](length=pairs, fill=False)
        var out = results.unsafe_ptr()
        var flags = failed.unsafe_ptr()
        var resolved = placement.or_else(Placement.default())
        var workers = thread_count(resolved.threads, pairs)
        var single = Placement(resolved.device, resolved.gpu_id, 1)
        var chunks = max(min(pairs, workers * 8), 1)

        def score_chunk(
            chunk: Int,
        ) {imm references, imm queries, imm scoring, imm mode, imm single, imm out, imm flags, imm pairs, imm chunks}:
            """Scores chunk `chunk` of the pairs, flagging any pair that raised for the serial retry below."""
            for index in range(pairs * chunk // chunks, pairs * (chunk + 1) // chunks):
                try:
                    out[unsafe_offset=index] = score(references[index], queries[index], scoring, mode, placement=single)
                except:
                    flags[unsafe_offset=index] = True

        spread(score_chunk, chunks, workers)
        for index in range(pairs):
            if failed[index]:
                results[index] = score(references[index], queries[index], scoring, mode, placement=single)
        return results^
    var results = List[Int](capacity=len(found))
    for value in found:
        results.append(Int(value))
    return results^


def alignments(
    references: List[String],
    queries: List[String],
    scoring: Scoring,
    mode: Mode = Mode.GLOBAL,
    *,
    placement: Optional[Placement] = None,
    max_memory: Int = DEFAULT_MAX_MEMORY,
    eqx: Bool = True,
) raises -> List[Alignment]:
    """`align` for every pair; on the device, every pair both bounds admit goes out in one launch."""
    var pairs = pair_count(references, queries)
    within_32_bits(references, queries, scoring, mode, True)
    var stored_cells = cells_within(max_memory)
    var resolved = placement.or_else(Placement.default())
    if resolved.device != Device.GPU:
        # On the host each pair as `align` takes it, the pairs over the threads asked for.
        var results = List[Alignment](capacity=pairs)
        for _ in range(pairs):
            results.append(Alignment(0, 0, String(), 0, 0, 0, 0))
        var out = results.unsafe_ptr()
        var failed = List[Bool](length=pairs, fill=False)
        var flags = failed.unsafe_ptr()
        var single = Placement.on_cpu(1)
        var workers = thread_count(resolved.threads, pairs)
        var chunks = max(min(pairs, workers * 8), 1)
        # Global alignments under a table of one match and one mismatch score: many pairs at once in the
        # lanes (see `scoring.laned_alignments`); the rest one at a time.
        var settled = List[Bool](length=pairs, fill=False)
        if mode.is_global() and mode.match_score == 0:
            settled = laned_alignments(references, queries, scoring, eqx, workers, max_memory, out)
        var settled_ptr = settled.unsafe_ptr()

        def align_chunk(
            chunk: Int,
        ) {
            imm references,
            imm queries,
            imm scoring,
            imm mode,
            imm single,
            imm stored_cells,
            imm eqx,
            imm out,
            imm flags,
            imm pairs,
            imm chunks,
            imm settled_ptr,
        }:
            """Aligns chunk `chunk` of the pairs, flagging any pair that raised for the serial retry below."""
            var space = SearchSpace()
            for index in range(pairs * chunk // chunks, pairs * (chunk + 1) // chunks):
                if settled_ptr[unsafe_offset=index]:
                    continue
                try:
                    out[unsafe_offset=index] = scoring_alignment(
                        references[index], queries[index], scoring, mode, single, stored_cells, eqx, space
                    )
                except:
                    flags[unsafe_offset=index] = True

        spread(align_chunk, chunks, workers)
        # A pair that failed raises here, the same error a serial loop would have raised first.
        for index in range(pairs):
            if failed[index]:
                results[index] = scoring_alignment(
                    references[index], queries[index], scoring, mode, single, stored_cells, eqx
                )
        return results^
    var gapped: List[GappedAlignment]
    if mode.kind == Mode.SMITH_WATERMAN and mode.match_score == 0:
        gapped = device_batch_alignments[AlignmentMode.LOCAL](references, queries, scoring, resolved, stored_cells)
    elif mode.is_global() and mode.match_score == 0:
        gapped = device_batch_alignments[AlignmentMode.GLOBAL](references, queries, scoring, resolved, stored_cells)
    else:
        var results = List[Alignment](capacity=pairs)
        for index in range(pairs):
            results.append(
                align(
                    references[index],
                    queries[index],
                    scoring,
                    mode,
                    placement=placement,
                    max_memory=max_memory,
                    eqx=eqx,
                )
            )
        return results^
    # Each pair's rows read as its CIGAR, on the host threads the placement asks for.
    var results = List[Alignment](capacity=pairs)
    for _ in range(pairs):
        results.append(Alignment(0, 0, String(), 0, 0, 0, 0))
    var out = results.unsafe_ptr()
    var whole = mode.is_global()
    var workers = thread_count(resolved.threads, pairs)
    var chunks = max(min(pairs, workers * 8), 1)

    def convert(
        chunk: Int,
    ) {imm gapped, imm references, imm queries, imm out, imm whole, imm eqx, imm pairs, imm chunks}:
        """Reads chunk `chunk` of the gapped pairs as `Alignment`s, each with its CIGAR and spans."""
        for index in range(pairs * chunk // chunks, pairs * (chunk + 1) // chunks):
            out[unsafe_offset=index] = as_alignment(gapped[index], references[index], queries[index], whole, eqx)

    spread(convert, chunks, workers)
    return results^


# endregion Scoring

# region Batches


def distances(
    references: List[String],
    queries: List[String],
    costs: Costs = Costs.edit(),
    mode: Mode = Mode.GLOBAL,
    *,
    band: Band = Band(),
    threads: Optional[Int] = None,
    placement: Optional[Placement] = None,
) raises -> List[Int]:
    """Every pair's `distance`, on the caller's own thread by default, or spread over `threads` threads
    when asked: an application that calls the library from threads of its own spreads its work itself,
    and it alone knows how many its machine can spare.

    The pairs are independent, so each runs on one thread start to finish, each thread taking the
    next pair of the batch, longest first, as soon as it is free (see `longest_first`). A pair that
    fails raises, after the rest, the same error a serial loop would have raised first.

    On the GPU, `placement`, unit costs align globally, a thread a pair by Myers' bit-vectors (see
    `device_edit`), every pair whose shorter sequence fits a thread's 4,096 letters; the rest, and
    every other cost or mode, on the host.
    """
    if placement and placement.value().device == Device.GPU:
        if mode.is_scored():
            raise AlignmentError(
                ErrorKind.INVALID_ARGUMENT, "a mode with a match score maximizes a score, which `align` finds"
            )
        if costs.unit_scale() == 0 or not mode.is_global() or not band.covers_any():
            raise AlignmentError(
                ErrorKind.INVALID_ARGUMENT, "on the GPU, distances at unit costs, globally, with no band"
            )
        return gpu_distances(references, queries, costs.unit_scale(), placement.value())
    var found = capped_distances(references, queries, costs, mode, band, Int.MAX, threads, raise_outside=True)
    var results = List[Int](capacity=len(found))
    for index in range(len(found)):
        results.append(found[index].value())
    return results^


def distances(
    references: List[String],
    queries: List[String],
    costs: Costs = Costs.edit(),
    mode: Mode = Mode.GLOBAL,
    *,
    max_cost: Int,
    band: Band = Band(),
    threads: Optional[Int] = None,
) raises AlignmentError -> List[Optional[Int]]:
    """Every pair's `distance` under `max_cost`, None for a pair past it or with no alignment inside
    `band`, the pairs spread over threads as the uncapped `distances` spreads them: a batch of
    candidates filtered by cost, the far ones costing a fraction of their full search."""
    return capped_distances(references, queries, costs, mode, band, max_cost, threads)


def alignments(
    references: List[String],
    queries: List[String],
    costs: Costs = Costs.edit(),
    mode: Mode = Mode.GLOBAL,
    *,
    band: Band = Band(),
    ties: Ties = Ties.KSW2,
    eqx: Bool = True,
    threads: Optional[Int] = None,
    max_memory: Int = DEFAULT_MAX_MEMORY,
) raises AlignmentError -> List[Alignment]:
    """Every pair's `align`, the pairs spread over threads as `distances` spreads them, each thread's
    kept fronts within `max_memory` bytes."""
    var found = capped_alignments(
        references,
        queries,
        costs,
        mode,
        band,
        Int.MAX,
        ties,
        eqx,
        threads,
        max_memory // KEPT_BYTES,
        raise_outside=True,
    )
    var results = List[Alignment](capacity=len(found))
    for index in range(len(found)):
        results.append(found[index].take())
    return results^


def alignments(
    references: List[String],
    queries: List[String],
    costs: Costs = Costs.edit(),
    mode: Mode = Mode.GLOBAL,
    *,
    max_cost: Int,
    band: Band = Band(),
    ties: Ties = Ties.KSW2,
    eqx: Bool = True,
    threads: Optional[Int] = None,
    max_memory: Int = DEFAULT_MAX_MEMORY,
) raises AlignmentError -> List[Optional[Alignment]]:
    """Every pair's `align` under `max_cost`, None for a pair past it or with no alignment inside
    `band`, the pairs spread over threads as `distances` spreads them."""
    return capped_alignments(
        references, queries, costs, mode, band, max_cost, ties, eqx, threads, max_memory // KEPT_BYTES
    )


def search(
    references: List[String],
    query: String,
    costs: Costs = Costs.edit(),
    mode: Mode = Mode.GLOBAL,
    *,
    best: Optional[Int] = None,
    max_cost: Optional[Int] = None,
    aligned: Bool = False,
    ties: Ties = Ties.KSW2,
    threads: Optional[Int] = None,
) raises AlignmentError -> List[Hit]:
    """The query against every reference, a database search: each reference's `Hit`, its score, the
    best first, ties by the references' order; with `best` that many alone, with `max_cost` (a mode
    with no reward) those within it alone, and with `aligned` each kept hit's alignment too.

    A local alignment scores a group of references at once, one to a SIMD lane, as SWIPE does (see
    `local_scores_by_lane`); a mode with no reward takes `distances`, under the cap when there is one;
    any other mode each pair's `score`. Every kept hit is then aligned on its own, when asked for, by
    `align`."""
    var count = len(references)
    if best and best.value() < 0:
        raise AlignmentError(ErrorKind.INVALID_ARGUMENT, String("a best of ", best.value(), " hits"))
    if max_cost and mode.is_scored():
        # Every mode with a reward alike: a local alignment's lanes would drop the cap without a word.
        raise AlignmentError(ErrorKind.INVALID_ARGUMENT, "a mode with a match score takes no cost cap")
    var workers = thread_count(threads.or_else(1), count)
    var scores = List[Int](length=count, fill=Int.MIN)
    if mode.kind == Mode.SMITH_WATERMAN and mode.match_score > 0:
        _ = penalties_of(costs)
        var laned = local_scores_by_lane(query, references, costs, mode.match_score, workers)
        for index in range(count):
            # A reference the lanes leave, costs under which padding could earn, scored alone.
            scores[index] = laned[index].value() if laned[index] else score(references[index], query, costs, mode)
    elif not mode.is_scored():
        var queries = List[String](length=count, fill=query)
        var found = capped_distances(
            references, queries, costs, mode, Band(), max_cost.or_else(Int.MAX), Optional[Int](workers)
        )
        for index in range(count):
            if found[index]:
                scores[index] = -found[index].value()
    else:
        var out = scores.unsafe_ptr()
        var failed = List[Bool](length=count, fill=False)
        var flags = failed.unsafe_ptr()
        var taken = Atomic[Int64](0)

        def work(slot: Int) {mut taken, imm}:
            """Scores the next unclaimed references against the query until none is left, flagging any that
            raised."""
            var last = 0
            while True:
                var share = next_share(taken, count, workers, last)
                if share[0] >= count:
                    return
                for index in range(share[0], share[1]):
                    try:
                        out[unsafe_offset=index] = score(references[index], query, costs, mode)
                    except:
                        flags[unsafe_offset=index] = True

        spread(work, min(workers, max(count, 1)), min(workers, max(count, 1)))
        for index in range(count):
            if failed[index]:
                scores[index] = score(references[index], query, costs, mode)
    # The best first, ties by the references' order; a reference past the cap is no hit.
    var order = List[Int](capacity=count)
    for index in range(count):
        if scores[index] != Int.MIN:
            order.append(index)

    def ahead(left: Int, right: Int) {imm scores} -> Bool:
        """Whether reference `left` ranks before `right`: the higher score, then the earlier reference."""
        if scores[left] != scores[right]:
            return scores[left] > scores[right]
        return left < right

    sort(order, ahead)
    var kept = min(len(order), best.or_else(len(order)))
    var hits = List[Hit](capacity=kept)
    for rank in range(kept):
        var index = order[rank]
        var alignment = Optional[Alignment]()
        if aligned:
            alignment = align(references[index], query, costs, mode, ties=ties)
        hits.append(Hit(index, scores[index], alignment^))
    return hits^


# endregion Batches
