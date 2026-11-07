# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the
# MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
KSW2's global dual-affine alignment (`ksw_extd2` without a flag, as minimap2 fills the gaps between a chain's
anchors) for many pairs at once, a pair a lane of one SIMD register: the same cells, the same choice among equal
alignments, so the same CIGARs as `karan.ksw2` wherever the band holds the alignment.

`karan.ksw2` runs one pair along its anti-diagonals, a vector of cells a step, so every step pays its bounds,
boundary cells and score vector for the few dozen cells a narrow band holds. Here a batch of pairs runs row by
row (a target letter a row, a query letter a column) and each step is one cell of every pair: the per-step work is
shared by the whole batch, and every lane holds a cell.

Each cell keeps Gotoh's five scores as KSW2's difference recurrence implies them, absolute and in sixteen bits:
`H` the best, `E1`, `E2` ending in a deletion (target letters alone) of each gap piece, `F1`, `F2` in an insertion.
A gap of `k` letters costs `q + k e`, or `q2 + k e2`; a match earns `a`, a mismatch costs `b`, the wildcard (code 4)
`ambi`. Its code byte is KSW2's: which source gave `H`, the diagonal before a deletion before an insertion, the
first piece before the second, a later source only when strictly better; and in bits 3 to 6 whether the next cell's
gap of each layer continues this one, only when strictly better than one opened here. A cell keeps it as the eight
comparisons that make it, each a mask of the batch's lanes (`N` bits): building the bytes lane by lane took more
than the cell's scores. The walk back is `ksw_backtrack` over rows, each step reading its lane's code from the
masks.

The band is KSW2's, the diagonals `-w + 1 ..= w` of target letters less query letters, one `w` for the batch, or
each pair its own (`bands`, none wider than `w`): a pair's cells outside its own band are held at minus infinity,
as KSW2 run in that band alone holds them, so its alignment is the one its band gives whatever pairs share the
batch. Cells outside the band are lost, as KSW2's are when its band is the whole matrix: the two agree whenever the
alignment stays inside, which the caller checks (`touches_band`). Scores must stay within sixteen bits: pairs of up to
`MAX_LANE_LENGTH` letters.
"""

from std.memory import pack_bits

comptime MAX_LANE_LENGTH = 6000
"""The longest side a pair may have: its scores, a match or a gap letter at a time, stay inside sixteen bits."""
comptime NEG = Int16(-30000)


struct LaneBuffers[N: Int](Movable):
    """One thread's memory for batches of `N` pairs: the previous row's `H`, `E1`, `E2` by column, the letters
    by column (each pair's query letter in its lane), and the code bytes, a row's band at a time."""

    var h: List[SIMD[DType.int16, Self.N]]
    var e1: List[SIMD[DType.int16, Self.N]]
    var e2: List[SIMD[DType.int16, Self.N]]
    var query: List[SIMD[DType.uint8, Self.N]]
    var codes: List[SIMD[DType.uint8, Self.N]]

    def __init__(out self):
        self.h = List[SIMD[DType.int16, Self.N]]()
        self.e1 = List[SIMD[DType.int16, Self.N]]()
        self.e2 = List[SIMD[DType.int16, Self.N]]()
        self.query = List[SIMD[DType.uint8, Self.N]]()
        self.codes = List[SIMD[DType.uint8, Self.N]]()


@always_inline
def _mask_type[N: Int]() -> DType:
    """The unsigned integer of `N` bits, one a lane: a cell's eight comparison masks fill its `N` code bytes."""
    comptime assert N == 8 or N == 16 or N == 32 or N == 64, "a batch of 8, 16, 32 or 64 lanes"
    comptime if N == 8:
        return DType.uint8
    elif N == 16:
        return DType.uint16
    elif N == 32:
        return DType.uint32
    else:
        return DType.uint64


def _has_wildcard(letters: ImmPointer[UInt8, ImmUntrackedOrigin], length: Int) -> Bool:
    """Whether the letters hold the wildcard, code 4."""
    comptime W = 64
    var k = 0
    while k + W <= length:
        if letters.unsafe_offset(k).unsafe_load[width=W]().eq(4).reduce_or():
            return True
        k += W
    while k < length:
        if letters[unsafe_offset=k] == 4:
            return True
        k += 1
    return False


@always_inline
def _push(mut cigar: List[UInt32], op: Int, length: Int):
    if len(cigar) == 0 or Int(cigar[len(cigar) - 1] & 0xF) != op:
        cigar.append(UInt32(length << 4 | op))
    else:
        cigar[len(cigar) - 1] += UInt32(length << 4)


def ksw_lanes[
    N: Int
](
    targets: List[ImmPointer[UInt8, ImmUntrackedOrigin]],
    target_lengths: List[Int],
    queries: List[ImmPointer[UInt8, ImmUntrackedOrigin]],
    query_lengths: List[Int],
    a: Int,
    b: Int,
    ambi: Int,
    q_in: Int,
    e_in: Int,
    q2_in: Int,
    e2_in: Int,
    w: Int,
    mut buf: LaneBuffers[N],
    mut cigars: List[List[UInt32]],
    bands: List[Int] = List[Int](),
):
    """Aligns up to `N` pairs (the `target_lengths[l]` letters at `targets[l]` against the `query_lengths[l]` at
    `queries[l]`, codes 0 to 3 and the wildcard 4; the pointers are untracked, so the caller keeps the letters
    alive past the call) end to end,
    each its CIGAR into `cigars[l]` as KSW2's (`length << 4 | op`, `M` `I` `D`), all in the band `w`, or each in
    its own band `bands[l]` (at most `w`) when `bands` are given."""
    comptime V = SIMD[DType.int16, N]
    comptime B = SIMD[DType.uint8, N]
    var lanes = len(targets)
    var q = q_in
    var e = e_in
    var q2 = q2_in
    var e2 = e2_in
    if q2 + e2 < q + e:
        var tq = q
        q = q2
        q2 = tq
        var te = e
        e = e2
        e2 = te
    var n = 0
    var m = 0
    for l in range(lanes):
        n = max(n, target_lengths[l])
        m = max(m, query_lengths[l])
    # The query letters by column, a pair a lane; past a pair's end, and in lanes no pair holds, a letter that
    # never matters (no cell a pair's alignment reads lies there).
    buf.query.resize(m + 1, B(0))
    for j in range(1, m + 1):
        var col = B(4)
        for l in range(lanes):
            if j - 1 < query_lengths[l]:
                col[l] = queries[l][unsafe_offset=j - 1]
        buf.query[j] = col
    buf.h.resize(m + 2, V(NEG))
    buf.e1.resize(m + 2, V(NEG))
    buf.e2.resize(m + 2, V(NEG))
    var width = 2 * w + 1
    # Every code the walk back reads, this call writes first (a lane's band lies inside the batch's): the buffer
    # grows uninitialized, its old codes not copied.
    var cells = (n + 1) * width
    if buf.codes.capacity() < cells:
        buf.codes = List[B](capacity=cells)
    else:
        buf.codes.clear()
    buf.codes.resize(unsafe_uninit_length=cells)
    # Each lane's band, and the narrowest: between the narrowest's edge and the batch's, cells are masked per lane.
    var lane_w = V(Int16(w))
    var narrowest = w
    for l in range(min(lanes, len(bands))):
        lane_w[l] = Int16(bands[l])
        narrowest = min(narrowest, bands[l])
    var neg = V(NEG)
    var qe = V(Int16(q + e))
    var qe2 = V(Int16(q2 + e2))
    var ev = V(Int16(e))
    var e2v = V(Int16(e2))
    var qv = V(Int16(q))
    var q2v = V(Int16(q2))
    var match_ = V(Int16(a))
    var mismatch = V(Int16(-b))
    var wild = V(Int16(-ambi))
    # Row 0: insertions alone, the cheaper gap piece's.
    for j in range(m + 2):
        buf.h[j] = neg
        buf.e1[j] = neg
        buf.e2[j] = neg
    for j in range(0, min(m, w - 1) + 1):
        var h0 = V(Int16(0)) if j == 0 else V(Int16(-min(q + j * e, q2 + j * e2)))
        buf.h[j] = V(Int16(j)).le(lane_w - 1).select(h0, neg) if j > narrowest - 1 else h0
    var h = buf.h.unsafe_ptr()
    var e1 = buf.e1.unsafe_ptr()
    var e2p = buf.e2.unsafe_ptr()
    var qc = buf.query.unsafe_ptr()
    # Wildcards are rare in reads: a batch without one scores its cells without the check. (Past a pair's end its
    # lane holds the wildcard, but no cell of the pair's own matrix reads those cells.)
    var wildcards = False
    for l in range(lanes):
        wildcards = (
            wildcards or _has_wildcard(targets[l], target_lengths[l]) or _has_wildcard(queries[l], query_lengths[l])
        )
    comptime M = _mask_type[N]()
    # A cell's code, as eight masks of `N` bits (one a lane): which comparisons made `H` (deletion, insertion, second
    # deletion, second insertion) and which gaps go on, in the `N` bytes the cell has.
    var codes = buf.codes.unsafe_ptr().unsafe_bitcast[Scalar[M]]()
    for i in range(1, n + 1):
        var tl = B(4)
        for l in range(lanes):
            if i - 1 < target_lengths[l]:
                tl[l] = targets[l][unsafe_offset=i - 1]
        var lo = max(0, i - w)
        var hi = min(m, i + w - 1)
        var diag = h[unsafe_offset=lo - 1] if lo > 0 else neg
        var left = neg
        var f1 = neg
        var f2 = neg
        if lo == 0:
            # Column 0: deletions alone.
            var hv = V(Int16(-min(q + i * e, q2 + i * e2)))
            var d1 = max(h[unsafe_offset=0] - qe, e1[unsafe_offset=0] - ev)
            var d2 = max(h[unsafe_offset=0] - qe2, e2p[unsafe_offset=0] - e2v)
            diag = h[unsafe_offset=0]
            d1 = max(d1, neg)
            d2 = max(d2, neg)
            if i > narrowest:
                # Column 0 lies outside the bands narrower than `i`.
                var inside = V(Int16(i)).le(lane_w)
                hv = inside.select(hv, neg)
                d1 = inside.select(d1, neg)
                d2 = inside.select(d2, neg)
            h[unsafe_offset=0] = hv
            e1[unsafe_offset=0] = d1
            e2p[unsafe_offset=0] = d2
            left = hv
        var row = codes.unsafe_offset(8 * i * width)
        comptime for checked in range(2):
            # The row's cells, with the wildcard check (`checked`) or without; one of the two runs.
            if (checked == 1) != wildcards:
                continue
            for j in range(max(lo, 1), hi + 1):
                var up = h[unsafe_offset=j]
                var qj = qc[unsafe_offset=j]
                var sc = tl.eq(qj).select(match_, mismatch)
                comptime if checked == 1:
                    sc = (tl.eq(B(4)) | qj.eq(B(4))).select(wild, sc)
                var z = diag + sc
                var de1 = max(max(up - qe, e1[unsafe_offset=j] - ev), neg)
                var de2 = max(max(up - qe2, e2p[unsafe_offset=j] - e2v), neg)
                f1 = max(max(left - qe, f1 - ev), neg)
                f2 = max(max(left - qe2, f2 - e2v), neg)
                var from_e1 = de1.gt(z)
                z = max(z, de1)
                var from_f1 = f1.gt(z)
                z = max(z, f1)
                var from_e2 = de2.gt(z)
                z = max(z, de2)
                var from_f2 = f2.gt(z)
                z = max(z, f2)
                # Whether the next cell's gap of each layer continues this one.
                var cell = row.unsafe_offset(8 * (j - i + w))
                cell[unsafe_offset=0] = pack_bits[M](from_e1)
                cell[unsafe_offset=1] = pack_bits[M](from_f1)
                cell[unsafe_offset=2] = pack_bits[M](from_e2)
                cell[unsafe_offset=3] = pack_bits[M](from_f2)
                cell[unsafe_offset=4] = pack_bits[M](de1.gt(z - qv))
                cell[unsafe_offset=5] = pack_bits[M](f1.gt(z - qv))
                cell[unsafe_offset=6] = pack_bits[M](de2.gt(z - q2v))
                cell[unsafe_offset=7] = pack_bits[M](f2.gt(z - q2v))
                if j < i - narrowest or j > i + narrowest - 1:
                    # Outside some lane's own band: there the cell is minus infinity.
                    var inside = V(Int16(j)).ge(V(Int16(i)) - lane_w) & V(Int16(j)).le(V(Int16(i)) + lane_w - 1)
                    z = inside.select(z, neg)
                    de1 = inside.select(de1, neg)
                    de2 = inside.select(de2, neg)
                    f1 = inside.select(f1, neg)
                    f2 = inside.select(f2, neg)
                diag = up
                h[unsafe_offset=j] = z
                e1[unsafe_offset=j] = de1
                e2p[unsafe_offset=j] = de2
                left = z
        # The cell just past the band holds nothing for the next row.
        if hi + 1 <= m:
            h[unsafe_offset=hi + 1] = neg
            e1[unsafe_offset=hi + 1] = neg
            e2p[unsafe_offset=hi + 1] = neg
    # Each pair's walk back, `ksw_backtrack` by rows, from its last cell.
    cigars.resize(lanes, List[UInt32]())
    for l in range(lanes):
        ref cigar = cigars[l]
        cigar.clear()
        var i = target_lengths[l]
        var j = query_lengths[l]
        var band = Int(lane_w[l])
        var bit = Scalar[M](1) << Scalar[M](l)
        var state = 0
        while i > 0 and j > 0:
            var force_state = -1
            if j < max(1, i - band):
                force_state = 2
            if j > i + band - 1:
                force_state = 1
            if force_state >= 0:
                state = force_state
            else:
                # The lane's code, read from the cell's masks as far as the step needs it: whether the gap being
                # walked goes on (its flag), and only off a gap the last source that won.
                var cell = codes.unsafe_offset(8 * (i * width + j - i + w))
                if state != 0 and (cell[unsafe_offset=3 + state] & bit) == 0:
                    state = 0
                if state == 0:
                    if (cell[unsafe_offset=3] & bit) != 0:
                        state = 4
                    elif (cell[unsafe_offset=2] & bit) != 0:
                        state = 3
                    elif (cell[unsafe_offset=1] & bit) != 0:
                        state = 2
                    elif (cell[unsafe_offset=0] & bit) != 0:
                        state = 1
            if state == 0:
                _push(cigar, 0, 1)
                i -= 1
                j -= 1
            elif state == 1 or state == 3:
                _push(cigar, 2, 1)
                i -= 1
            else:
                _push(cigar, 1, 1)
                j -= 1
        if i > 0:
            _push(cigar, 2, i)
        if j > 0:
            _push(cigar, 1, j)
        cigar.reverse()


def touches_band(cigar: List[UInt32], w: Int, margin: Int) -> Bool:
    """Whether an alignment (KSW2's CIGAR) comes within `margin` diagonals of the band `w`'s edge."""
    var i = 0
    var j = 0
    for c in cigar:
        var op = Int(c & 0xF)
        var k = Int(c >> 4)
        if op != 1:
            i += k
        if op != 2:
            j += k
        if i - j >= w - margin or j - i >= w - 1 - margin:
            return True
    return False
