# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the
# MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
The wavefront's costs, and every conversion from the public `Costs` (or a uniform `Scoring` table) to them: the
fronts' layers and their flag bits, `Penalties`, and the checks that refuse costs the searches cannot count
(`MAX_WINDOW`). Below every engine that reads them (`gap_affine`, `differences`, `lanes`, `vector_score`), so none
imports another for its costs.
"""

from std.math import gcd

from .bit_parallel import DIAGONAL, LEFT, UP
from .errors import AlignmentError, ErrorKind
from .modes import Costs
from .substitutions import uniform_pair


comptime ALIGNED = Int(DIAGONAL)
"""The front ending in two letters aligned, a match or a substitution; as a move, the bit-parallel
traceback's `DIAGONAL`, so either's moves spell a CIGAR alike (see `traceback.cigar_string`)."""
comptime FIRST_GAP = Int(LEFT)
"""The front ending in a letter of the first sequence against a gap of the first piece; a move that
consumes a letter of the first sequence alone, `LEFT`."""
comptime SECOND_GAP = Int(UP)
"""The front ending in a letter of the second sequence against a gap of the first piece; a move that
consumes a letter of the second sequence alone, `UP`."""
comptime MAX_PIECES = 2
"""Gap pieces a cost may have: a gap of `k` letters costs the least of `opening + k extension` over them."""


@always_inline
def layers_of[pieces: Int]() -> Int:
    """Fronts a cost keeps: the alignment front and two gap layers a piece, one each way."""
    return 1 + 2 * pieces


@always_inline
def gap_layer(piece: Int, along_first: Bool) -> Int:
    """The layer of `piece`'s gaps of the first sequence's letters, or of the second's: `FIRST_GAP` and
    `SECOND_GAP` for the first piece, the next two for the second."""
    return 1 + 2 * piece + (0 if along_first else 1)


@always_inline
def along_first(layer: Int) -> Bool:
    """Whether a gap layer's letters are the first sequence's, its moves `FIRST_GAP`."""
    return (layer - 1) % 2 == 0


struct Penalties(ImplicitlyCopyable, TrivialRegisterPassable):
    """The wavefront's costs for one scoring, and what turns a cost back into a score. A second gap
    piece, `opening2` and `extension2`, counts only where a search runs two (see `MAX_PIECES`). A gap
    of the first sequence's letters, a deletion, has its own costs, the `deletion_` ones, which are an
    insertion's unless set otherwise; each gap layer reads its own (see `opening_of`)."""

    var mismatch: Int
    var opening: Int
    """Charged once a gap run, on top of `extension` for its first letter."""
    var extension: Int
    var scale: Int
    """The common factor the costs were divided by."""
    var reward: Int
    """The match score, which every letter of both sequences earns half of before the costs."""
    var opening2: Int
    var extension2: Int
    var deletion_opening: Int
    var deletion_extension: Int
    var deletion_opening2: Int
    var deletion_extension2: Int
    var layer_openings: SIMD[DType.int32, 8]
    """Each gap layer's opening, at the layer's index, so a walk over layers reads one lane."""
    var layer_extensions: SIMD[DType.int32, 8]

    def __init__(
        out self,
        mismatch: Int,
        opening: Int,
        extension: Int,
        scale: Int,
        reward: Int,
        opening2: Int,
        extension2: Int,
    ):
        """Costs whose deletions cost what insertions do."""
        self.mismatch = mismatch
        self.opening = opening
        self.extension = extension
        self.scale = scale
        self.reward = reward
        self.opening2 = opening2
        self.extension2 = extension2
        self.deletion_opening = opening
        self.deletion_extension = extension
        self.deletion_opening2 = opening2
        self.deletion_extension2 = extension2
        self.layer_openings = SIMD[DType.int32, 8](0)
        self.layer_extensions = SIMD[DType.int32, 8](0)
        self.tabulate()

    def set_deletions(mut self, opening: Int, extension: Int, opening2: Int, extension2: Int):
        """A deletion's own costs, a gap of the first sequence's letters."""
        self.deletion_opening = opening
        self.deletion_extension = extension
        self.deletion_opening2 = opening2
        self.deletion_extension2 = extension2
        self.tabulate()

    def tabulate(mut self):
        """Each gap layer's opening and extension at its index (see `gap_layer`)."""
        self.layer_openings[gap_layer(0, True)] = Int32(self.deletion_opening)
        self.layer_openings[gap_layer(0, False)] = Int32(self.opening)
        self.layer_openings[gap_layer(1, True)] = Int32(self.deletion_opening2)
        self.layer_openings[gap_layer(1, False)] = Int32(self.opening2)
        self.layer_extensions[gap_layer(0, True)] = Int32(self.deletion_extension)
        self.layer_extensions[gap_layer(0, False)] = Int32(self.extension)
        self.layer_extensions[gap_layer(1, True)] = Int32(self.deletion_extension2)
        self.layer_extensions[gap_layer(1, False)] = Int32(self.extension2)

    def score(self, cost: Int, letters: Int) -> Int:
        """The Gotoh score of an alignment over `letters` letters in all, costing `cost`."""
        return (self.reward * letters - cost * self.scale) // 2

    @always_inline
    def opening_of(self, layer: Int) -> Int:
        """A gap layer's opening: its piece's, for its direction."""
        return Int(self.layer_openings[layer])

    @always_inline
    def extension_of(self, layer: Int) -> Int:
        """A gap layer's extension: its piece's, for its direction."""
        return Int(self.layer_extensions[layer])

    def window[pieces: Int](self) -> Int:
        """The dearest single move: a substitution, or a gap's first letter in any piece either way."""
        var dearest = self.mismatch
        comptime for layer in range(1, layers_of[pieces]()):
            dearest = max(dearest, self.opening_of(layer) + self.extension_of(layer))
        return dearest

    def widest_opening[pieces: Int](self) -> Int:
        """The dearest opening, what a meeting inside a gap pays once for both halves at most."""
        var widest = 0
        comptime for layer in range(1, layers_of[pieces]()):
            widest = max(widest, self.opening_of(layer))
        return widest

    def cheapest_extension[pieces: Int](self) -> Int:
        """The cheapest extension, what each letter a path strays off a diagonal pays at least."""
        var cheapest = Int.MAX
        comptime for layer in range(1, layers_of[pieces]()):
            cheapest = min(cheapest, self.extension_of(layer))
        return cheapest


comptime MAX_WINDOW = 1 << 16
"""The dearest single move a wavefront takes, in its costs' common factor (see `Penalties.window`): its
rings hold a front for each of the last that many costs, a kilobyte or two each, so past this even a
pair of a few letters would take hundreds of megabytes, and the layers' costs could pass their 32 bits.
Dear gaps far past any aligner's, an opening of 100,000 among them, stay within it."""


def scaled_penalties(costs: Costs, reward: Int, folded: Bool) raises AlignmentError -> Penalties:
    """The wavefront's costs for `costs`, deletions and insertions each their own, with `folded` a match
    earning `reward` folded in as for a global alignment or an extension (see the module's notes), all
    divided by their common factor."""
    if reward < 0 or costs.mismatch <= 0:
        raise AlignmentError(
            ErrorKind.INVALID_SCORING, "a match earns at least nothing, and a mismatch and an extension must cost"
        )
    var two = costs.pieces() == 2
    var values: List[Int] = [
        2 * (reward + costs.mismatch) if folded else costs.mismatch,
        costs.opening,
        costs.extension,
        costs.deletion_opening,
        costs.deletion_extension,
    ]
    if two:
        values.append(costs.opening2)
        values.append(costs.extension2)
        values.append(costs.deletion_opening2)
        values.append(costs.deletion_extension2)
    for index in range(1, len(values)):
        if values[index] < 0 or (index % 2 == 0 and values[index] <= 0):
            raise AlignmentError(
                ErrorKind.INVALID_SCORING,
                String("costs ", costs, ": an extension must cost, an opening no less than nothing"),
            )
        if folded:
            # A gap's opening doubles, its extension doubles and pays half the reward a letter.
            values[index] = 2 * values[index] + (reward if index % 2 == 0 else 0)
    var scale = 0
    for value in values:
        scale = gcd(scale, value)
    var out = Penalties(
        values[0] // scale,
        values[1] // scale,
        values[2] // scale,
        scale,
        reward,
        values[5] // scale if two else 0,
        values[6] // scale if two else 0,
    )
    out.set_deletions(
        values[3] // scale, values[4] // scale, values[7] // scale if two else 0, values[8] // scale if two else 0
    )
    # From the costs themselves, not the layers' 32 bits they may not fit.
    var dearest = values[0] // scale
    for index in range(1, len(values), 2):
        dearest = max(dearest, (values[index] + values[index + 1]) // scale)
    if dearest > MAX_WINDOW:
        raise AlignmentError(
            ErrorKind.INVALID_SCORING,
            String("costs ", costs, ": a move dearer than ", MAX_WINDOW, " in their common factor"),
        )
    return out


def wavefront_penalties(
    substitutions: List[Scalar[DType.int8]], alphabet_size: Int, open: Int, extend: Int
) -> Optional[Penalties]:
    """The wavefront costs for a table of one match and one mismatch score, if it is one.

    `open` and `extend` are the Gotoh gap scores, a run of `k` scoring `open + (k - 1) extend`.
    None when the table is not uniform, or when the costs would not all be positive: a free
    mismatch or extension would let a front grow without its cost growing.
    """
    var uniform = uniform_pair(substitutions, alphabet_size)
    if not uniform:
        return None
    var reward = uniform.value()[0]
    var mismatch = uniform.value()[1]
    # A run of `k` scores `open + (k - 1) extend`, so it costs `(extend - open) - k extend`.
    var gap_opening = extend - open
    var gap_extension = -extend
    var x = 2 * (reward - mismatch)
    var o = 2 * gap_opening
    var e = 2 * gap_extension + reward
    if x <= 0 or e <= 0 or o < 0:
        return None
    var scale = gcd(gcd(x, e), o)
    # Past the wavefront's rings, the table's own sweeps, whose memory the scores do not set.
    if max(x, o + e) // scale > MAX_WINDOW:
        return None
    return Penalties(x // scale, o // scale, e // scale, scale, reward, 0, 0)


comptime ENTRY_MASK = UInt8(7)
"""A flag's low three bits: the layer the alignment front's entry came from, zero for a substitution."""


@always_inline
def layer_bit(layer: Int) -> UInt8:
    """A flag's bit for gap layer `layer`, above `ENTRY_MASK`'s three."""
    return UInt8(1) << UInt8(2 + layer)


def penalties_of(costs: Costs) raises AlignmentError -> Penalties:
    """The wavefront's costs for `costs`, one gap piece or two, either way, divided by their common factor."""
    return scaled_penalties(costs, 0, False)


def rewarded_penalties(match_score: Int, costs: Costs) raises AlignmentError -> Penalties:
    """The wavefront's costs for a match earning `match_score` under `costs`, the reward folded in (see
    the module's notes), deletions and insertions each their own."""
    return scaled_penalties(costs, match_score, True)


def reduced_costs(costs: Costs) -> Costs:
    """`costs` divided by their common factor, which picks the same alignments by the same rule; a second piece's
    absent `-1` kept. `Ties.KSW2`'s rewrite counts these (see `ksw_ties.ksw_cigar`); the wavefront's own reduction,
    over its folded values, is `scaled_penalties`."""
    var two = costs.pieces() == 2
    var factor = gcd(gcd(costs.mismatch, costs.opening), gcd(costs.extension, costs.deletion_opening))
    factor = gcd(factor, costs.deletion_extension)
    if two:
        factor = gcd(gcd(factor, costs.opening2), costs.extension2)
        factor = gcd(gcd(factor, costs.deletion_opening2), costs.deletion_extension2)
    if factor <= 1:
        return costs
    var out = Costs(
        costs.mismatch // factor,
        costs.opening // factor,
        costs.extension // factor,
        costs.opening2 // factor if two else costs.opening2,
        costs.extension2 // factor if two else costs.extension2,
    )
    out.deletion_opening = costs.deletion_opening // factor
    out.deletion_extension = costs.deletion_extension // factor
    out.deletion_opening2 = costs.deletion_opening2 // factor if two else costs.deletion_opening2
    out.deletion_extension2 = costs.deletion_extension2 // factor if two else costs.deletion_extension2
    return out
