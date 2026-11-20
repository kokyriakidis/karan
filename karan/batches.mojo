# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the
# MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
Many pairs at once: the lanes for short pairs (`distances_in_lanes`, `alignments_in_lanes`), the rest one at a time
over the threads asked for, longest first (`longest_first`, `each_pair`), each through `engines`, and the GPU's
unit-cost distances (`gpu_distances`). A pair that fails raises, after the rest, as a serial loop would first.
"""

from std.atomic import Atomic
from std.bit import count_leading_zeros

from .common import DeviceScope, Placement, next_share, spread, thread_count
from .device_edit import MAX_PATTERN_WORDS, device_edit_distances
from .engines import (
    aligned_within,
    alignment_from_lanes,
    alignments_in_lanes,
    cost_within,
    distances_in_lanes,
    ksw_rewrite,
    optional,
    rewarded_costs,
)
from .errors import AlignmentError
from .gap_affine import KEPT_BYTES, outside
from .lanes import StringTexts, Texts
from .modes import Alignment, Anchor, Band, Costs, Mode, Ties
from .scoring import pair_count
from .search_space import SearchSpace


def longest_first[T: Texts](pairs: Int, references: T, queries: T, workers: Int) -> List[Int]:
    """The pairs' indices, the longest pair first: taken in that order by whichever thread is free, the
    long pairs start first and the short ones fill in around them, so none is left alone at the end
    holding up the rest.

    Balancing a batch needs no finer order than lengths within an eighth of each other, so the pairs
    are dealt into such classes, no sort, within a class in the batch's own order. Each of `workers`
    counts its stretch of the batch's classes and then places its stretch's pairs: on one thread the
    order of short reads took as long as a fifteenth of the work ten threads did on them."""
    comptime CLASSES = 8 * 62
    # A stretch holds at least as many pairs as there are classes, so summing its counts costs no more
    # than counting them did.
    var stretches = max(min(workers, pairs // CLASSES), 1)
    var classes = List[Int](capacity=pairs)
    classes.resize(unsafe_uninit_length=pairs)
    var order = List[Int](capacity=pairs)
    order.resize(unsafe_uninit_length=pairs)
    var starts = List[Int](length=stretches * CLASSES, fill=0)
    var class_ptr = classes.unsafe_ptr()
    var order_ptr = order.unsafe_ptr()
    var start_ptr = starts.unsafe_ptr()

    def count(stretch: Int) {imm references, imm queries, imm pairs, imm stretches, imm class_ptr, imm start_ptr}:
        """Each pair of stretch `stretch` its class, and the stretch's count of each."""
        var counts = start_ptr.unsafe_offset(stretch * CLASSES)
        for index in range(pairs * stretch // stretches, pairs * (stretch + 1) // stretches):
            var length = references.length(index) + queries.length(index)
            # Lengths below 16 are classes of their own; past that, a power of two and its top three
            # bits under the leading one, longest the lowest class.
            var shift = max(60 - Int(count_leading_zeros(UInt64(length))), 0)
            var found = CLASSES - 1 - (8 * shift + (length >> shift))
            class_ptr[unsafe_offset=index] = found
            counts[unsafe_offset=found] += 1

    def place(stretch: Int) {imm pairs, imm stretches, imm class_ptr, imm order_ptr, imm start_ptr}:
        """Each pair of stretch `stretch` at the next place its class has for the stretch."""
        var next = start_ptr.unsafe_offset(stretch * CLASSES)
        for index in range(pairs * stretch // stretches, pairs * (stretch + 1) // stretches):
            var found = class_ptr[unsafe_offset=index]
            order_ptr[unsafe_offset=next[unsafe_offset=found]] = index
            next[unsafe_offset=found] += 1

    spread(count, stretches, stretches)
    # Each stretch's first place in each class: the classes in turn, each stretch's pairs in turn.
    var placed = 0
    for found in range(CLASSES):
        for stretch in range(stretches):
            var counted = starts[stretch * CLASSES + found]
            starts[stretch * CLASSES + found] = placed
            placed += counted
    spread(place, stretches, stretches)
    return order^


def each_pair[F: def(Int, mut SearchSpace) -> None](work: F, order: List[Int], workers: Int):
    """`work` for every pair in `order` on `workers` threads, each thread taking the next pairs as soon as it is
    free (see `next_share`), its searches kept from pair to pair."""
    var pairs = len(order)
    if pairs == 0:
        return
    var taken = Atomic[Int64](0)
    var threads = min(workers, pairs)

    def worker(slot: Int) {mut taken, imm work, imm order, imm pairs, imm threads}:
        """Takes the next pairs in `order` until none is left."""
        var space = SearchSpace()
        var last = 0
        while True:
            var share = next_share(taken, pairs, threads, last)
            if share[0] >= pairs:
                return
            for dealt in range(share[0], share[1]):
                work(order[dealt], space)

    spread(worker, threads, threads)


def gpu_distances(
    references: List[String], queries: List[String], scale: Int, placement: Placement
) raises -> List[Int]:
    """Every pair's unit-cost distance times `scale`, on the device where its shorter sequence fits a
    thread (see `device_edit`), on the host otherwise."""
    var pairs = pair_count(references, queries)
    var results = List[Int](length=pairs, fill=0)
    var patterns = List[String]()
    var texts = List[String]()
    var on_device = List[Int]()
    var on_host = List[Int]()
    comptime LIMIT = 64 * MAX_PATTERN_WORDS
    for index in range(pairs):
        var reference = references[index].byte_length()
        var query = queries[index].byte_length()
        if min(reference, query) == 0:
            results[index] = max(reference, query) * scale
        elif min(reference, query) <= LIMIT and max(reference, query) <= Int(Int32.MAX):
            # The device keeps a distance in 32 bits, which a longer text could pass.
            var shorter_query = query <= reference
            patterns.append(queries[index] if shorter_query else references[index])
            texts.append(references[index] if shorter_query else queries[index])
            on_device.append(index)
        else:
            on_host.append(index)
    if len(on_device) > 0:
        var found = device_edit_distances(DeviceScope(placement.gpu_id), patterns, texts, placement.threads)
        for slot in range(len(on_device)):
            results[on_device[slot]] = found[slot] * scale
    if len(on_host) > 0:
        # Pairs too long for a thread's letters, on the host over the threads asked for.
        var host_references = List[String](capacity=len(on_host))
        var host_queries = List[String](capacity=len(on_host))
        for index in on_host:
            host_references.append(references[index])
            host_queries.append(queries[index])
        var found = capped_distances(
            host_references, host_queries, Costs.edit(), Mode.GLOBAL, Band(), Int.MAX, Optional[Int](placement.threads)
        )
        for slot in range(len(on_host)):
            results[on_host[slot]] = found[slot].value() * scale
    return results^


def capped_distances(
    references: List[String],
    queries: List[String],
    costs: Costs,
    mode: Mode,
    band: Band,
    max_cost: Int,
    threads: Optional[Int],
    raise_outside: Bool = False,
) raises AlignmentError -> List[Optional[Int]]:
    """Every pair's `cost_within`, on every thread asked for, longest first. With `raise_outside`, the uncapped
    `distances`', a pair no alignment inside `band` fits raises as a failed pair does, in the batch's order; else
    it is None, as one past `max_cost` is, whatever the cap (`Int.MAX` included)."""
    var pairs = pair_count(references, queries)
    var results = List[Optional[Int]](length=pairs, fill=None)
    if pairs == 0:
        return results^
    var workers = thread_count(threads.or_else(1), pairs)
    var out = results.unsafe_ptr()
    var failed = List[Bool](length=pairs, fill=False)
    var flags = failed.unsafe_ptr()
    # The lanes first (see `distances_in_lanes`); the pairs too long for their 16 bits, a pair with an empty side
    # and free ends, and every pair of a mode with a reward, one at a time.
    var settled = List[Bool](length=pairs, fill=False)
    var settled_ptr = settled.unsafe_ptr()
    var reference_texts = StringTexts.of(references)
    var query_texts = StringTexts.of(queries)
    if (
        distances_in_lanes(pairs, reference_texts, query_texts, costs, mode, band, max_cost, workers, out, settled_ptr)
        == pairs
    ):
        for index in range(pairs):
            if not results[index] and raise_outside:
                raise outside(band)
        return results^

    def distance_one(
        index: Int, mut space: SearchSpace
    ) {imm references, imm queries, imm out, imm flags, imm settled_ptr, imm costs, imm mode, imm band, imm max_cost}:
        """Pair `index`'s capped cost, or a flag that it raised; a pair the lanes settled it leaves."""
        if settled_ptr[unsafe_offset=index]:
            return
        try:
            out[unsafe_offset=index] = cost_within(
                references[index].as_bytes(), queries[index].as_bytes(), costs, mode, band, max_cost, space
            )
        except:
            flags[unsafe_offset=index] = True

    each_pair(distance_one, longest_first(pairs, reference_texts, query_texts, workers), workers)

    for index in range(pairs):
        if failed[index]:
            results[index] = cost_within(
                references[index].as_bytes(), queries[index].as_bytes(), costs, mode, band, max_cost
            )
        if not results[index] and raise_outside:
            raise outside(band)
    return results^


def capped_alignments(
    references: List[String],
    queries: List[String],
    costs: Costs,
    mode: Mode,
    band: Band,
    max_cost: Int,
    ties: Ties,
    eqx: Bool,
    threads: Optional[Int],
    limit: Int,
    raise_outside: Bool = False,
) raises AlignmentError -> List[Optional[Alignment]]:
    """Every pair's `aligned_within`, on every thread asked for, longest first. With `raise_outside`, the
    uncapped `alignments`', a pair no alignment inside `band` fits raises as a failed pair does, in the batch's
    order; else it is None, whatever the cap."""
    var pairs = pair_count(references, queries)
    var results = List[Optional[Alignment]](capacity=pairs)
    for _ in range(pairs):
        results.append(None)
    if pairs == 0:
        return results^
    var workers = thread_count(threads.or_else(1), pairs)
    var out = results.unsafe_ptr()
    var failed = List[Bool](length=pairs, fill=False)
    var flags = failed.unsafe_ptr()
    # The lanes first (see `alignments_in_lanes`); every other pair, and every pair of a mode that scores, one at
    # a time.
    var settled = List[Bool](length=pairs, fill=False)
    var settled_ptr = settled.unsafe_ptr()
    var laned = List[Optional[Int]](length=pairs, fill=None)
    var laned_ptr = laned.unsafe_ptr()
    var paths = List[List[UInt8]](capacity=pairs)
    for _ in range(pairs):
        paths.append(List[UInt8]())
    var path_ptr = paths.unsafe_ptr()
    var spans = List[Int](length=4 * pairs, fill=0)
    var span_ptr = spans.unsafe_ptr()
    var reference_texts = StringTexts.of(references)
    var query_texts = StringTexts.of(queries)
    # The lanes spell `LEFT`'s CIGAR; `KSW2` writes a settled pair's again below.
    var lane_ties = Ties.LEFT if ties == Ties.KSW2 else ties
    var penalties = alignments_in_lanes(
        pairs,
        reference_texts,
        query_texts,
        costs,
        mode,
        band,
        max_cost,
        lane_ties,
        workers,
        limit,
        laned_ptr,
        path_ptr,
        span_ptr,
        settled_ptr,
    )

    def alignment_one(
        index: Int, mut space: SearchSpace
    ) {
        imm references,
        imm queries,
        imm out,
        imm flags,
        imm costs,
        imm mode,
        imm band,
        imm max_cost,
        imm ties,
        imm eqx,
        imm limit,
        imm settled_ptr,
        imm laned_ptr,
        imm path_ptr,
        imm span_ptr,
        imm penalties,
    }:
        """Pair `index`'s capped alignment, or a flag that it raised; a pair the lanes settled has its CIGAR
        spelled from their path, unless the searches might split it (see `alignment_from_lanes`)."""
        if settled_ptr[unsafe_offset=index]:
            if not laned_ptr[unsafe_offset=index]:
                return
            var moves = List[UInt8]()
            swap(moves, path_ptr[unsafe_offset=index])
            var spelled = alignment_from_lanes(
                references[index].as_bytes(),
                queries[index].as_bytes(),
                laned_ptr[unsafe_offset=index].value(),
                moves^,
                span_ptr.unsafe_offset(4 * index),
                penalties.value(),
                eqx,
                limit,
            )
            if spelled:
                var found = spelled.take()
                if ties == Ties.KSW2:
                    ksw_rewrite(
                        found,
                        references[index].as_bytes(),
                        queries[index].as_bytes(),
                        rewarded_costs(costs, mode.match_score) if mode.is_scored() else costs,
                        eqx,
                        False,
                        space.ties,
                        band,
                        mode.match_score if mode.is_scored() else 0,
                        mode.kind == Mode.EXTENSION and mode.anchor == Anchor.END,
                        limit * KEPT_BYTES,
                    )
                out[unsafe_offset=index] = Optional(found^)
                return
        try:
            out[unsafe_offset=index] = optional(
                aligned_within(
                    references[index].as_bytes(),
                    queries[index].as_bytes(),
                    costs,
                    mode,
                    band,
                    max_cost,
                    ties,
                    eqx,
                    limit,
                    space,
                )
            )
        except:
            flags[unsafe_offset=index] = True

    each_pair(alignment_one, longest_first(pairs, reference_texts, query_texts, workers), workers)

    for index in range(pairs):
        if failed[index]:
            results[index] = optional(
                aligned_within(
                    references[index].as_bytes(),
                    queries[index].as_bytes(),
                    costs,
                    mode,
                    band,
                    max_cost,
                    ties,
                    eqx,
                    limit,
                )
            )
        if not results[index] and raise_outside:
            raise outside(band)
    return results^
