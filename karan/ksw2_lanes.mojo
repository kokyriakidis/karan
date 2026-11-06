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
gap of each layer continues this one, only when strictly better than one opened here. The walk back is
`ksw_backtrack` over rows.

The band is KSW2's, the diagonals `-w + 1 ..= w` of target letters less query letters, one `w` for the batch;
cells outside it are lost, as KSW2's are when its band is the whole matrix: the two agree whenever the alignment
stays inside, which the caller checks (`touches_band`). Scores must stay within sixteen bits: pairs of up to
`MAX_LANE_LENGTH` letters.
"""

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
):
    """Aligns up to `N` pairs (the `target_lengths[l]` letters at `targets[l]` against the `query_lengths[l]` at
    `queries[l]`, codes 0 to 3 and the wildcard 4; the pointers are untracked, so the caller keeps the letters
    alive past the call) end to end,
    each its CIGAR into `cigars[l]` as KSW2's (`length << 4 | op`, `M` `I` `D`), all in the band `w`."""
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
    buf.codes.resize((n + 1) * width, B(0))
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
        buf.h[j] = V(Int16(0)) if j == 0 else V(Int16(-min(q + j * e, q2 + j * e2)))
    var h = buf.h.unsafe_ptr()
    var e1 = buf.e1.unsafe_ptr()
    var e2p = buf.e2.unsafe_ptr()
    var qc = buf.query.unsafe_ptr()
    var codes = buf.codes.unsafe_ptr()
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
            h[unsafe_offset=0] = hv
            e1[unsafe_offset=0] = max(d1, neg)
            e2p[unsafe_offset=0] = max(d2, neg)
            left = hv
        var row = codes.unsafe_offset(i * width)
        for j in range(max(lo, 1), hi + 1):
            var up = h[unsafe_offset=j]
            var qj = qc[unsafe_offset=j]
            var sc = tl.eq(qj).select(match_, mismatch)
            sc = (tl.eq(B(4)) | qj.eq(B(4))).select(wild, sc)
            var z = diag + sc
            var de1 = max(max(up - qe, e1[unsafe_offset=j] - ev), neg)
            var de2 = max(max(up - qe2, e2p[unsafe_offset=j] - e2v), neg)
            f1 = max(max(left - qe, f1 - ev), neg)
            f2 = max(max(left - qe2, f2 - e2v), neg)
            var d = B(0)
            d = de1.gt(z).select(B(1), d)
            z = max(z, de1)
            d = f1.gt(z).select(B(2), d)
            z = max(z, f1)
            d = de2.gt(z).select(B(3), d)
            z = max(z, de2)
            d = f2.gt(z).select(B(4), d)
            z = max(z, f2)
            # Whether the next cell's gap of each layer continues this one.
            d = d | de1.gt(z - qv).select(B(0x08), B(0))
            d = d | f1.gt(z - qv).select(B(0x10), B(0))
            d = d | de2.gt(z - q2v).select(B(0x20), B(0))
            d = d | f2.gt(z - q2v).select(B(0x40), B(0))
            row[unsafe_offset=j - i + w] = d
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
        var state = 0
        while i > 0 and j > 0:
            var force_state = -1
            if j < max(1, i - w):
                force_state = 2
            if j > i + w - 1:
                force_state = 1
            var tmp = 0
            if force_state < 0:
                tmp = Int(codes[unsafe_offset=i * width + j - i + w][l])
            if state == 0:
                state = tmp & 7
            elif ((tmp >> (state + 2)) & 1) == 0:
                state = 0
            if state == 0:
                state = tmp & 7
            if force_state >= 0:
                state = force_state
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
