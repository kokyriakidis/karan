# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the
# MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
KSW2's rule for equal alignments over karan's costs (`Ties.KSW2`): of the alignments of a stretch that cost the
least, the one minimap2's KSW2 writes, the rule plegma's graph alignment follows too.

KSW2 walks back from the end of the stretch and at every cell takes, of the moves into it that give its best
score, the diagonal (a match or a substitution), then a deletion of the first gap piece (a reference letter
alone), an insertion of it (a query letter alone), a deletion of the second piece, an insertion of it: gaps come
as early as the cost allows and substitutions as late. Inside a gap it keeps to the gap while the gap's run into
the cell costs strictly less than one opened there. A rule of cells, it needs every cell's costs, which karan's
wavefront and bit-parallel searches never hold; so the CIGAR is written again here, by Gotoh's recurrence over
the cells about the path a search found (`ksw_cigar`).

Costs and scores choose alike. A score with a match reward, as KSW2 maximizes, is at a cell `(i, j)` the reward's
`(i + j) / 2` less half a cost (a substitution `2 (b + a)`, a gap of `k` letters `2 q + (2 e + a) k`), the same
for every move into that cell: so its moves tie, and its gaps run on, exactly where these costs' do. The first
gap piece is the one cheaper to open and extend once, as KSW2 orders them.

The cells searched lie within `width` columns either side of the path given; cells outside cost without bound.
A walk that comes within `MARGIN` of that edge, where the edge is not the matrix's, is done again four times
wider, so the CIGAR is the one the whole matrix gives.
"""


from .modes import Costs

comptime INF = Int32(1 << 29)
comptime MARGIN = 8
comptime START_WIDTH = 32

comptime _MATCH = UInt8(ord("="))
comptime _MISMATCH = UInt8(ord("X"))
comptime _INSERTION = UInt8(ord("I"))
comptime _DELETION = UInt8(ord("D"))


@fieldwise_init
struct _Piece(ImplicitlyCopyable, TrivialRegisterPassable):
    var opening: Int32
    var extension: Int32


@always_inline
def _ordered(a: _Piece, b: _Piece) -> Tuple[_Piece, _Piece]:
    """The two pieces with the cheaper first letter first, as KSW2 orders them (`q2 + e2 < q + e` swaps)."""
    if b.opening + b.extension < a.opening + a.extension:
        return (b, a)
    return (a, b)


struct TieBuffers(Movable):
    """One thread's memory for `ksw_cigar`: the band's rows, two rows of costs and the cells' code bytes."""

    var lo: List[Int]
    var hi: List[Int]
    var lo_cap: List[Int]
    var hi_cap: List[Int]
    var offset: List[Int]
    var h: List[Int32]
    var e1: List[Int32]
    var e2: List[Int32]
    var codes: List[UInt8]
    # The anti-diagonal sweep's: three rows of H, two of each gap layer, by reference letter; each
    # anti-diagonal's reference range and code offset; the letters padded for whole vectors.
    var hd: List[Int32]
    var gl: List[Int32]
    var ilo: List[Int]
    var ihi: List[Int]
    var vstart: List[Int]
    var coff: List[Int]
    var rpad: List[UInt8]
    var qpad: List[UInt8]

    def __init__(out self):
        self.lo = List[Int]()
        self.hi = List[Int]()
        self.lo_cap = List[Int]()
        self.hi_cap = List[Int]()
        self.offset = List[Int]()
        self.h = List[Int32]()
        self.e1 = List[Int32]()
        self.e2 = List[Int32]()
        self.codes = List[UInt8]()
        self.hd = List[Int32]()
        self.gl = List[Int32]()
        self.ilo = List[Int]()
        self.ihi = List[Int]()
        self.vstart = List[Int]()
        self.coff = List[Int]()
        self.rpad = List[UInt8]()
        self.qpad = List[UInt8]()


@always_inline
def _push(mut runs: List[UInt64], letter: UInt8, length: Int):
    if len(runs) > 0 and UInt8(runs[len(runs) - 1] & 0xFF) == letter:
        runs[len(runs) - 1] += UInt64(length) << 8
    else:
        runs.append((UInt64(length) << 8) | UInt64(letter))


def _band(guide: Span[UInt64, _], n: Int, m: Int, width: Int, low: Int, high: Int, mut buf: TieBuffers) -> Bool:
    """For each reference row `0 ..= n`, the query columns within `width` of those the guide (runs, `=` `X` `M`
    `I` `D`) crosses there, moving only forward row to row, and inside the diagonals `low ..= high` (reference
    letters less query letters) a caller's band allows; `buf.lo_cap`, `buf.hi_cap` keep the columns that band
    and the matrix alone allow, so a walk is widened again only against the guide's edge."""
    buf.lo.resize(n + 1, 0)
    buf.hi.resize(n + 1, 0)
    for i in range(n + 1):
        buf.lo[i] = m + 1
        buf.hi[i] = -1
    var i = 0
    var j = 0
    buf.lo[0] = 0
    buf.hi[0] = 0
    for run in guide:
        var letter = UInt8(run & 0xFF)
        var k = Int(run >> 8)
        if letter == _INSERTION:
            j += k
            if i <= n:
                buf.hi[i] = max(buf.hi[i], min(j, m))
        else:
            for _ in range(k):
                i += 1
                if letter != _DELETION:
                    j += 1
                if i <= n:
                    buf.lo[i] = min(buf.lo[i], min(j, m))
                    buf.hi[i] = max(buf.hi[i], min(j, m))
    for r in range(1, n + 1):
        if buf.hi[r] < 0:
            buf.lo[r] = buf.lo[r - 1]
            buf.hi[r] = buf.hi[r - 1]
    buf.lo_cap.resize(n + 1, 0)
    buf.hi_cap.resize(n + 1, 0)
    for r in range(n + 1):
        buf.lo_cap[r] = max(0, r - high)
        buf.hi_cap[r] = min(m, r - low)
        buf.lo[r] = max(buf.lo_cap[r], buf.lo[r] - width)
        buf.hi[r] = min(buf.hi_cap[r], buf.hi[r] + width)
    for r in range(1, n + 1):
        buf.lo[r] = max(buf.lo[r], buf.lo[r - 1])
    for r in range(n - 1, -1, -1):
        buf.hi[r] = min(buf.hi[r], buf.hi[r + 1])
    # The end cell inside, whatever the guide, if the caller's band allows it.
    buf.hi[n] = min(m, buf.hi_cap[n])
    # Every row must hold a cell, joined to the row above, the first one the start and the last one the end.
    if buf.lo[0] != 0 or buf.hi[n] != m:
        return False
    for r in range(n + 1):
        if buf.lo[r] > buf.hi[r]:
            return False
        if r > 0 and buf.lo[r] > buf.hi[r - 1] + 1:
            return False
    return True


def _gap(k: Int32, a: _Piece, b: _Piece, two: Bool) -> Int32:
    var one = a.opening + a.extension * k
    if two:
        return min(one, b.opening + b.extension * k)
    return one


def _fill_and_walk_rows(
    reference: Span[UInt8, _], query: Span[UInt8, _], costs: Costs, mut buf: TieBuffers, mut runs: List[UInt64]
) -> Bool:
    """Gotoh's recurrence over the band with KSW2's code bytes, then KSW2's walk back from the last cell.
    Returns whether the walk came near an edge of the band that is not the matrix's."""
    var n = len(reference)
    var m = len(query)
    var two = costs.pieces() == 2
    var x = Int32(costs.mismatch)
    var ins = _ordered(
        _Piece(Int32(costs.opening), Int32(costs.extension)), _Piece(Int32(costs.opening2), Int32(costs.extension2))
    ) if two else (_Piece(Int32(costs.opening), Int32(costs.extension)), _Piece(INF, 0))
    var dels = _ordered(
        _Piece(Int32(costs.deletion_opening), Int32(costs.deletion_extension)),
        _Piece(Int32(costs.deletion_opening2), Int32(costs.deletion_extension2)),
    ) if two else (_Piece(Int32(costs.deletion_opening), Int32(costs.deletion_extension)), _Piece(INF, 0))
    var i1 = ins[0]
    var i2 = ins[1]
    var d1 = dels[0]
    var d2 = dels[1]
    buf.offset.resize(n + 1, 0)
    var total = 0
    for r in range(n + 1):
        buf.offset[r] = total
        total += buf.hi[r] - buf.lo[r] + 1
    buf.codes.resize(total, 0)
    buf.h.resize(m + 2, INF)
    buf.e1.resize(m + 2, INF)
    buf.e2.resize(m + 2, INF)
    var h = buf.h.unsafe_ptr()
    var e1 = buf.e1.unsafe_ptr()
    var e2 = buf.e2.unsafe_ptr()
    var codes = buf.codes.unsafe_ptr()
    for j in range(m + 2):
        h[unsafe_offset=j] = INF
        e1[unsafe_offset=j] = INF
        e2[unsafe_offset=j] = INF
    for j in range(buf.lo[0], buf.hi[0] + 1):
        h[unsafe_offset=j] = Int32(0) if j == 0 else _gap(Int32(j), i1, i2, two)
    var prev_hi = buf.hi[0]
    for i in range(1, n + 1):
        var lo = buf.lo[i]
        var hi = buf.hi[i]
        # Columns new to the band hold nothing from the row above.
        for j in range(prev_hi + 1, hi + 1):
            h[unsafe_offset=j] = INF
            e1[unsafe_offset=j] = INF
            e2[unsafe_offset=j] = INF
        var diag = h[unsafe_offset=lo - 1] if lo > 0 else INF
        var left = INF
        var f1 = INF
        var f2 = INF
        var start = lo
        if lo == 0:
            diag = h[unsafe_offset=0]
            var u = h[unsafe_offset=0]
            e1[unsafe_offset=0] = min(u + d1.opening + d1.extension, e1[unsafe_offset=0] + d1.extension)
            if two:
                e2[unsafe_offset=0] = min(u + d2.opening + d2.extension, e2[unsafe_offset=0] + d2.extension)
            h[unsafe_offset=0] = _gap(Int32(i), d1, d2, two)
            left = h[unsafe_offset=0]
            start = 1
        var base = buf.offset[i] - lo
        var t = reference[i - 1]
        for j in range(start, hi + 1):
            var up = h[unsafe_offset=j]
            var z = diag + (Int32(0) if t == query[j - 1] else x)
            var de1 = min(up + d1.opening + d1.extension, e1[unsafe_offset=j] + d1.extension)
            f1 = min(left + i1.opening + i1.extension, f1 + i1.extension)
            var de2 = INF
            if two:
                de2 = min(up + d2.opening + d2.extension, e2[unsafe_offset=j] + d2.extension)
                f2 = min(left + i2.opening + i2.extension, f2 + i2.extension)
            var d = UInt8(0)
            if de1 < z:
                z = de1
                d = 1
            if f1 < z:
                z = f1
                d = 2
            if two:
                if de2 < z:
                    z = de2
                    d = 3
                if f2 < z:
                    z = f2
                    d = 4
            # Whether the next cell's gap of each layer runs on from this one, strictly cheaper than opening.
            if de1 < z + d1.opening:
                d |= 0x08
            if f1 < z + i1.opening:
                d |= 0x10
            if two:
                if de2 < z + d2.opening:
                    d |= 0x20
                if f2 < z + i2.opening:
                    d |= 0x40
            codes[unsafe_offset=base + j] = d
            diag = up
            h[unsafe_offset=j] = min(z, INF)
            e1[unsafe_offset=j] = min(de1, INF)
            e2[unsafe_offset=j] = min(de2, INF)
            left = z
        if lo > 0:
            h[unsafe_offset=lo - 1] = INF
            e1[unsafe_offset=lo - 1] = INF
            e2[unsafe_offset=lo - 1] = INF
        prev_hi = hi
    # `ksw_backtrack` by rows from the last cell.
    var back = List[UInt64]()
    var i = n
    var j = m
    var state = 0
    var near = False
    while i > 0 and j > 0:
        var lo = buf.lo[i]
        var hi = buf.hi[i]
        if (lo > buf.lo_cap[i] and j - lo < MARGIN) or (hi < buf.hi_cap[i] and hi - j < MARGIN):
            near = True
        var force = -1
        if j < max(lo, 1):
            force = 2
        if j > hi:
            force = 1
        var tmp = 0
        if force < 0:
            tmp = Int(codes[unsafe_offset=buf.offset[i] - lo + j])
        if state == 0:
            state = tmp & 7
        elif ((tmp >> (state + 2)) & 1) == 0:
            state = 0
        if state == 0:
            state = tmp & 7
        if force >= 0:
            state = force
        if state == 0:
            _push(back, _MATCH if reference[i - 1] == query[j - 1] else _MISMATCH, 1)
            i -= 1
            j -= 1
        elif state == 1 or state == 3:
            _push(back, _DELETION, 1)
            i -= 1
        else:
            _push(back, _INSERTION, 1)
            j -= 1
    if i > 0:
        _push(back, _DELETION, i)
    if j > 0:
        _push(back, _INSERTION, j)
    runs.clear()
    for k in range(len(back) - 1, -1, -1):
        _push(runs, UInt8(back[k] & 0xFF), Int(back[k] >> 8))
    return near


comptime LANES = 16
"""Cells of an anti-diagonal a vector holds: 32-bit costs in a 512-bit register."""


@always_inline
def _fill(p: MutPointer[Int32, _], n: Int, value: Int32):
    """`n` values at `p` set to `value`, a vector at a time."""
    comptime V = SIMD[DType.int32, LANES]
    var k = 0
    while k + LANES <= n:
        p.unsafe_offset(k).unsafe_store(V(value))
        k += LANES
    while k < n:
        p[unsafe_offset=k] = value
        k += 1


def _fill_and_walk_shaped[
    shape: Int
](
    reference: Span[UInt8, _], query: Span[UInt8, _], costs: Costs, mut buf: TieBuffers, mut runs: List[UInt64]
) -> Bool:
    """`_fill_and_walk_rows` swept by anti-diagonal, `LANES` cells at once, for one shape of costs: `shape` 0 linear
    (no opening: a gap's run into a cell is the cell above or left plus an extension, and never runs on strictly
    cheaper than opening, so only the cells' costs are kept), 1 one gap piece, 2 two: a cell `(i, j)` reads the anti-diagonal
    two back at `i - 1` (the diagonal) and the one before at `i - 1` (above) and `i` (left), so a whole
    anti-diagonal's cells, `i` running over the band, are independent. Each anti-diagonal's cells outside the band
    hold no finite cost, so the next ones read past its edge safely; the same codes, so the same walk."""
    comptime V = SIMD[DType.int32, LANES]
    comptime B = SIMD[DType.uint8, LANES]
    var n = len(reference)
    var m = len(query)
    var two = shape == 2
    var x = Int32(costs.mismatch)
    var ins = _ordered(_Piece(Int32(costs.opening), Int32(costs.extension)), _Piece(
        Int32(costs.opening2), Int32(costs.extension2)
    )) if two else (_Piece(Int32(costs.opening), Int32(costs.extension)), _Piece(INF, 0))
    var dels = _ordered(_Piece(Int32(costs.deletion_opening), Int32(costs.deletion_extension)), _Piece(
        Int32(costs.deletion_opening2), Int32(costs.deletion_extension2)
    )) if two else (_Piece(Int32(costs.deletion_opening), Int32(costs.deletion_extension)), _Piece(INF, 0))
    var i1 = ins[0]
    var i2 = ins[1]
    var d1 = dels[0]
    var d2 = dels[1]
    var nd = n + m + 1
    # Each anti-diagonal's reference range: rows `i` whose band holds column `d - i`.
    buf.ilo.resize(nd, 0)
    buf.ihi.resize(nd, 0)
    buf.vstart.resize(nd, 0)
    buf.coff.resize(nd + 1, 0)
    var a = 0
    var b = 0
    for d in range(nd):
        while a <= n and a + buf.hi[a] < d:
            a += 1
        while b + 1 <= n and b + 1 + buf.lo[b + 1] <= d:
            b += 1
        buf.ilo[d] = a
        buf.ihi[d] = b
    var total = 0
    for d in range(nd):
        var vs = buf.ilo[d] // LANES * LANES
        buf.vstart[d] = vs
        buf.coff[d] = total
        if buf.ilo[d] <= buf.ihi[d]:
            total += (buf.ihi[d] - vs) // LANES * LANES + LANES
    buf.coff[nd] = total
    buf.codes.resize(total + LANES, 0)
    # Three anti-diagonals of H, two of each gap layer, offset by `LANES` so `i - 1` reads at `i = 0`.
    var stride = n + 3 * LANES
    buf.hd.resize(3 * stride, INF)
    buf.gl.resize(8 * stride, INF)
    _fill(buf.hd.unsafe_ptr(), 3 * stride, INF)
    comptime if shape > 0:
        _fill(buf.gl.unsafe_ptr(), 8 * stride, INF)
    # The letters: the reference by `i - 1`, the query reversed so `query[d - i - 1]` runs forward with `i`.
    buf.rpad.resize(n + 3 * LANES, 0)
    buf.qpad.resize(m + 3 * LANES, 0)
    var rw = buf.rpad.unsafe_ptr()
    var qw = buf.qpad.unsafe_ptr()
    for k in range(n):
        rw[unsafe_offset=LANES + k] = reference[k]
    for k in range(m):
        qw[unsafe_offset=LANES + k] = query[m - 1 - k]
    # Letters past an end differ from any real one, so they never pair as a match.
    for k in range(LANES):
        buf.rpad[k] = 254
        buf.rpad[LANES + n + k] = 254
        buf.qpad[k] = 255
        buf.qpad[LANES + m + k] = 255
    var hp = buf.hd.unsafe_ptr()
    var gp = buf.gl.unsafe_ptr()
    var rp = buf.rpad.unsafe_ptr()
    var qp = buf.qpad.unsafe_ptr()
    var cp = buf.codes.unsafe_ptr()
    var lane = V(0)
    for k in range(LANES):
        lane[k] = Int32(k)
    var inf = V(INF)
    var xv = V(x)
    var d1oe = V(d1.opening + d1.extension)
    var d1e = V(d1.extension)
    var d1o = V(d1.opening)
    var i1oe = V(i1.opening + i1.extension)
    var i1e = V(i1.extension)
    var i1o = V(i1.opening)
    var d2oe = V(d2.opening + d2.extension)
    var d2e = V(d2.extension)
    var d2o = V(d2.opening)
    var i2oe = V(i2.opening + i2.extension)
    var i2e = V(i2.extension)
    var i2o = V(i2.opening)
    # The rows each anti-diagonal writes and reads, turned round by swapping pointers: H three deep, each gap
    # layer two.
    var h0 = hp.unsafe_offset(LANES)
    var h1 = hp.unsafe_offset(2 * stride + LANES)
    var h2 = hp.unsafe_offset(stride + LANES)
    var e1c = gp.unsafe_offset(LANES)
    var e2c = gp.unsafe_offset(stride + LANES)
    var f1c = gp.unsafe_offset(2 * stride + LANES)
    var f2c = gp.unsafe_offset(3 * stride + LANES)
    var e1p = gp.unsafe_offset(4 * stride + LANES)
    var e2p = gp.unsafe_offset(5 * stride + LANES)
    var f1p = gp.unsafe_offset(6 * stride + LANES)
    var f2p = gp.unsafe_offset(7 * stride + LANES)
    for d in range(nd):
        if d > 0:
            var oldest = h2
            h2 = h1
            h1 = h0
            h0 = oldest
            comptime if shape > 0:
                var t_e1c = e1c
                e1c = e1p
                e1p = t_e1c
                var t_f1c = f1c
                f1c = f1p
                f1p = t_f1c
            comptime if shape == 2:
                var t_e2c = e2c
                e2c = e2p
                e2p = t_e2c
                var t_f2c = f2c
                f2c = f2p
                f2p = t_f2c
        var lo = buf.ilo[d]
        var hi = buf.ihi[d]
        var vs = buf.vstart[d]
        if lo > hi:
            continue
        # The cells inside the matrix's interior (both letters consumed): `1 ..= d - 1` rows of the band.
        var first = max(lo, 1)
        var last = min(hi, d - 1)
        var ve = hi
        var at = vs
        while at <= ve:
            var iv = V(Int32(at)) + lane
            var inside = iv.ge(V(Int32(first))) & iv.le(V(Int32(last)))
            var diag = h2.unsafe_offset(at - 1).unsafe_load[width=LANES]()
            var up = h1.unsafe_offset(at - 1).unsafe_load[width=LANES]()
            var left = h1.unsafe_offset(at).unsafe_load[width=LANES]()
            var rl = rp.unsafe_offset(LANES + at - 1).unsafe_load[width=LANES]()
            var ql = qp.unsafe_offset(LANES + m - d + at).unsafe_load[width=LANES]()
            var z = diag + rl.eq(ql).select(V(0), xv)
            var de1: V
            var f1: V
            comptime if shape == 0:
                de1 = up + d1e
                f1 = left + i1e
            else:
                de1 = min(up + d1oe, e1p.unsafe_offset(at - 1).unsafe_load[width=LANES]() + d1e)
                f1 = min(left + i1oe, f1p.unsafe_offset(at).unsafe_load[width=LANES]() + i1e)
            var dcode = B(0)
            var t1 = de1.lt(z)
            z = t1.select(de1, z)
            dcode = t1.select(B(1), dcode)
            var t2 = f1.lt(z)
            z = t2.select(f1, z)
            dcode = t2.select(B(2), dcode)
            var de2 = inf
            var f2 = inf
            comptime if shape == 2:
                de2 = min(up + d2oe, e2p.unsafe_offset(at - 1).unsafe_load[width=LANES]() + d2e)
                f2 = min(left + i2oe, f2p.unsafe_offset(at).unsafe_load[width=LANES]() + i2e)
                var t3 = de2.lt(z)
                z = t3.select(de2, z)
                dcode = t3.select(B(3), dcode)
                var t4 = f2.lt(z)
                z = t4.select(f2, z)
                dcode = t4.select(B(4), dcode)
            # Whether the next cell's gap of each layer runs on from this one, strictly cheaper than opening.
            comptime if shape > 0:
                dcode = dcode | de1.lt(z + d1o).select(B(0x08), B(0))
                dcode = dcode | f1.lt(z + i1o).select(B(0x10), B(0))
            comptime if shape == 2:
                dcode = dcode | de2.lt(z + d2o).select(B(0x20), B(0))
                dcode = dcode | f2.lt(z + i2o).select(B(0x40), B(0))
            h0.unsafe_offset(at).unsafe_store(inside.select(min(z, inf), inf))
            comptime if shape > 0:
                e1c.unsafe_offset(at).unsafe_store(inside.select(min(de1, inf), inf))
                f1c.unsafe_offset(at).unsafe_store(inside.select(min(f1, inf), inf))
            comptime if shape == 2:
                e2c.unsafe_offset(at).unsafe_store(inside.select(min(de2, inf), inf))
                f2c.unsafe_offset(at).unsafe_store(inside.select(min(f2, inf), inf))
            cp.unsafe_offset(buf.coff[d] + at - vs).unsafe_store(dcode)
            at += LANES
        # The edges: row 0 (insertions alone) and column 0 (deletions alone), when the band holds them.
        if lo == 0 and d <= m:
            h0[unsafe_offset=0] = Int32(0) if d == 0 else _gap(Int32(d), i1, i2, two)
            e1c[unsafe_offset=0] = INF
            e2c[unsafe_offset=0] = INF
            f1c[unsafe_offset=0] = i1.opening + i1.extension * Int32(d) if d > 0 else INF
            f2c[unsafe_offset=0] = (i2.opening + i2.extension * Int32(d)) if two and d > 0 else INF
        if hi == d and d <= n and d > 0:
            h0[unsafe_offset=d] = _gap(Int32(d), d1, d2, two)
            e1c[unsafe_offset=d] = d1.opening + d1.extension * Int32(d)
            e2c[unsafe_offset=d] = (d2.opening + d2.extension * Int32(d)) if two else INF
            f1c[unsafe_offset=d] = INF
            f2c[unsafe_offset=d] = INF
        # The cells just outside this anti-diagonal's range hold nothing for the next two.
        if lo >= 1:
            h0[unsafe_offset=lo - 1] = INF
            comptime if shape > 0:
                e1c[unsafe_offset=lo - 1] = INF
                f1c[unsafe_offset=lo - 1] = INF
            comptime if shape == 2:
                e2c[unsafe_offset=lo - 1] = INF
                f2c[unsafe_offset=lo - 1] = INF
        # Past the last vector's last cell; the lanes between `hi` and it were stored as nothing already.
        h0[unsafe_offset=at] = INF
        comptime if shape > 0:
            e1c[unsafe_offset=at] = INF
            f1c[unsafe_offset=at] = INF
        comptime if shape == 2:
            e2c[unsafe_offset=at] = INF
            f2c[unsafe_offset=at] = INF
    # `ksw_backtrack` by anti-diagonal from the last cell.
    var back = List[UInt64]()
    var i = n
    var j = m
    var state = 0
    var near = False
    while i > 0 and j > 0:
        if (buf.lo[i] > buf.lo_cap[i] and j - buf.lo[i] < MARGIN) or (
            buf.hi[i] < buf.hi_cap[i] and buf.hi[i] - j < MARGIN
        ):
            near = True
        var dd = i + j
        var force = -1
        if i < max(buf.ilo[dd], 1):
            force = 2
        if i > min(buf.ihi[dd], dd - 1):
            force = 1
        var tmp = 0
        if force < 0:
            tmp = Int(cp[unsafe_offset=buf.coff[dd] + i - buf.vstart[dd]])
        if state == 0:
            state = tmp & 7
        elif ((tmp >> (state + 2)) & 1) == 0:
            state = 0
        if state == 0:
            state = tmp & 7
        if force >= 0:
            state = force
        if state == 0:
            _push(back, _MATCH if reference[i - 1] == query[j - 1] else _MISMATCH, 1)
            i -= 1
            j -= 1
        elif state == 1 or state == 3:
            _push(back, _DELETION, 1)
            i -= 1
        else:
            _push(back, _INSERTION, 1)
            j -= 1
    if i > 0:
        _push(back, _DELETION, i)
    if j > 0:
        _push(back, _INSERTION, j)
    runs.clear()
    for k in range(len(back) - 1, -1, -1):
        _push(runs, UInt8(back[k] & 0xFF), Int(back[k] >> 8))
    return near


def _fill_and_walk(
    reference: Span[UInt8, _], query: Span[UInt8, _], costs: Costs, mut buf: TieBuffers, mut runs: List[UInt64]
) -> Bool:
    """`_fill_and_walk_shaped` for these costs' shape."""
    if costs.pieces() == 2:
        return _fill_and_walk_shaped[2](reference, query, costs, buf, runs)
    if costs.opening == 0 and costs.deletion_opening == 0:
        return _fill_and_walk_shaped[0](reference, query, costs, buf, runs)
    return _fill_and_walk_shaped[1](reference, query, costs, buf, runs)


def _spelled(guide: Span[UInt64, _], reference: Span[UInt8, _], query: Span[UInt8, _]) -> List[UInt64]:
    """The guide's runs with `M` told into `=` and `X`."""
    var out = List[UInt64]()
    var i = 0
    var j = 0
    for run in guide:
        var letter = UInt8(run & 0xFF)
        var k = Int(run >> 8)
        if letter == _INSERTION:
            _push(out, _INSERTION, k)
            j += k
        elif letter == _DELETION:
            _push(out, _DELETION, k)
            i += k
        else:
            for x in range(k):
                _push(out, _MATCH if reference[i + x] == query[j + x] else _MISMATCH, 1)
            i += k
            j += k
    return out^


def ksw_cigar(
    reference: Span[UInt8, _],
    query: Span[UInt8, _],
    costs: Costs,
    guide: Span[UInt64, _],
    mut buf: TieBuffers,
    low: Int = -(1 << 40),
    high: Int = 1 << 40,
    start_width: Int = START_WIDTH,
    by_rows: Bool = False,
) -> List[UInt64]:
    """The alignment of all of `reference` to all of `query` at least cost that KSW2's rule picks, as runs
    (`cigar.run_of`, `=` `X` `I` `D`), searched about the path `guide` (any alignment of the two inside the
    diagonals `low ..= high`, reference letters less query letters, as runs) and widened until its walk keeps
    clear of the edge the guide sets; never past those diagonals."""
    var runs = List[UInt64]()
    var n = len(reference)
    var m = len(query)
    var width = start_width
    while True:
        if not _band(guide, n, m, width, low, high, buf):
            # No band joins the ends (the caller's diagonals and the guide disagree): the guide as it stands.
            return _spelled(guide, reference, query)
        var near = _fill_and_walk_rows(reference, query, costs, buf, runs) if by_rows else _fill_and_walk(
            reference, query, costs, buf, runs
        )
        if not near or width >= max(n, m):
            return runs^
        width = min(max(n, m), width * 4)
