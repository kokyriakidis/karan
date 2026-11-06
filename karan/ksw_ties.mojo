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

The cells swept are those that can lie on an alignment of least cost: a cell is kept while its cost so far, in
any layer, and the least the letters left can cost (their gap, from the difference in letters left) come to no
more than the path given costs. Every cell of every alignment of least cost passes, and its cost is exact, as the
cells it is reached from pass too; a cell dropped costs without bound, never less than it does. KSW2's walk reads
only the cells of alignments of least cost, so it is the whole matrix's, at about the cost of the cells a
wavefront search visits.
"""


from .modes import Costs

comptime INF = Int32(1 << 29)

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
    """One thread's memory for `ksw_cigar`: the cells' rows of costs and their code bytes."""

    var lo: List[Int]
    var hi: List[Int]
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


def _rows_band(n: Int, m: Int, low: Int, high: Int, mut buf: TieBuffers) -> Bool:
    """For each reference row `0 ..= n`, the query columns inside the diagonals `low ..= high` (reference letters
    less query letters); whether they join the start to the end."""
    buf.lo.resize(n + 1, 0)
    buf.hi.resize(n + 1, 0)
    for r in range(n + 1):
        buf.lo[r] = max(0, r - high)
        buf.hi[r] = min(m, r - low)
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
    """Gotoh's recurrence over every cell of the rows `buf.lo`, `buf.hi` give, with KSW2's code bytes, then KSW2's
    walk back from the last cell: the reference the swept cells are held to. Returns whether the last cell was
    reached."""
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
    if h[unsafe_offset=m] >= INF:
        return False
    # `ksw_backtrack` by rows from the last cell.
    var back = List[UInt64]()
    var i = n
    var j = m
    var state = 0
    while i > 0 and j > 0:
        var lo = buf.lo[i]
        var hi = buf.hi[i]
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
    return True


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
    reference: Span[UInt8, _],
    query: Span[UInt8, _],
    costs: Costs,
    bound: Int32,
    low: Int,
    high: Int,
    mut buf: TieBuffers,
    mut runs: List[UInt64],
) -> Bool:
    """Gotoh's recurrence with KSW2's code bytes swept by anti-diagonal, `LANES` cells at once, then KSW2's walk
    back from the last cell, for one shape of costs: `shape` 0 linear (no opening: a gap's run into a cell is the
    cell above or left plus an extension, and never runs on strictly cheaper than opening, so only the cells'
    costs are kept), 1 one gap piece, 2 two. A cell `(i, j)` reads the anti-diagonal two back at `i - 1` (the
    diagonal) and the one before at `i - 1` (above) and `i` (left), so a whole anti-diagonal's cells are
    independent.

    Each anti-diagonal sweeps the rows its two before kept a cell next to, inside the diagonals `low ..= high`;
    the rows it passes on are those from its first to its last cell whose cost (its least layer's) and the least
    the letters left can cost come to no more than `bound` (see the module's note). Every row outside what an
    anti-diagonal swept costs without bound. Returns whether the last cell was reached within `bound`, and the
    walk with it."""
    comptime V = SIMD[DType.int32, LANES]
    comptime B = SIMD[DType.uint8, LANES]
    var n = len(reference)
    var m = len(query)
    var two = shape == 2
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
    var nd = n + m + 1
    buf.ilo.resize(nd, 0)
    buf.ihi.resize(nd, 0)
    buf.vstart.resize(nd, 0)
    buf.coff.resize(nd + 1, 0)
    if len(buf.codes) < 1 << 16:
        buf.codes.resize(1 << 16, 0)
    # Three anti-diagonals of H, two of each gap layer, offset by `LANES` so `i - 1` reads at `i = 0`; outside
    # the stretch each last wrote, every one holds no finite cost.
    var stride = n + 3 * LANES
    buf.hd.resize(3 * stride, INF)
    _fill(buf.hd.unsafe_ptr(), 3 * stride, INF)
    comptime if shape > 0:
        buf.gl.resize(8 * stride, INF)
        _fill(buf.gl.unsafe_ptr(), 8 * stride, INF)
    var written = Array[Int, 22](fill=0)
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
        rw[unsafe_offset=k] = 254
        rw[unsafe_offset=LANES + n + k] = 254
        qw[unsafe_offset=k] = 255
        qw[unsafe_offset=LANES + m + k] = 255
    var hp = buf.hd.unsafe_ptr()
    var gp = buf.gl.unsafe_ptr()
    var rp = buf.rpad.unsafe_ptr()
    var qp = buf.qpad.unsafe_ptr()
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
    # The least the letters left can cost: their difference as one gap, or inside a gap of that kind already,
    # as its letters at the cheaper extension.
    var del_least = V(min(d1.extension, d2.extension) if two else d1.extension)
    var ins_least = V(min(i1.extension, i2.extension) if two else i1.extension)
    var ceiling = V(bound)
    var zero = V(0)
    # Letters left past this many count as no more, a bound still under their cost, kept clear of overflow.
    var longest = V(1 << 16)
    # Rows each of the last two anti-diagonals kept a cell in, `lo > hi` for none.
    var lo1 = 1
    var hi1 = 0
    var lo2 = 1
    var hi2 = 0
    var total = 0
    for d in range(nd):
        var hs = d % 3
        var gs = (d % 2) * 4
        var ps = ((d + 1) % 2) * 4
        var h0 = hp.unsafe_offset(hs * stride + LANES)
        var h1 = hp.unsafe_offset(((d + 2) % 3) * stride + LANES)
        var h2 = hp.unsafe_offset(((d + 1) % 3) * stride + LANES)
        var e1c = gp.unsafe_offset((gs + 0) * stride + LANES)
        var e2c = gp.unsafe_offset((gs + 1) * stride + LANES)
        var f1c = gp.unsafe_offset((gs + 2) * stride + LANES)
        var f2c = gp.unsafe_offset((gs + 3) * stride + LANES)
        var e1p = gp.unsafe_offset((ps + 0) * stride + LANES)
        var e2p = gp.unsafe_offset((ps + 1) * stride + LANES)
        var f1p = gp.unsafe_offset((ps + 2) * stride + LANES)
        var f2p = gp.unsafe_offset((ps + 3) * stride + LANES)
        # The rows to sweep: next to a cell either anti-diagonal before kept, and an edge cell that may lie on
        # an alignment of least cost; inside the matrix and the caller's diagonals (`2 i - d`).
        var clo = Int.MAX
        var chi = -1
        if lo1 <= hi1:
            clo = min(clo, lo1)
            chi = max(chi, hi1 + 1)
        if lo2 <= hi2:
            clo = min(clo, lo2 + 1)
            chi = max(chi, hi2 + 1)
        var top_cost = Int32(0) if d == 0 else _gap(Int32(min(d, m)), i1, i2, two)
        var top_live = d <= m and top_cost + _least_left(n - m + d, del_least[0], ins_least[0]) <= bound
        var side_cost = _gap(Int32(min(d, n)), d1, d2, two)
        var side_live = d > 0 and d <= n and side_cost + _least_left(n - m - d, del_least[0], ins_least[0]) <= bound
        if top_live:
            clo = 0
            chi = max(chi, 0)
        if side_live:
            clo = min(clo, d)
            chi = d
        clo = max(clo, max(0, d - m), -((-(d + low)) // 2))
        chi = min(chi, min(n, d), (d + high) // 2)
        var vs = 0
        var at = 0
        var nlo = Int.MAX
        var nhi = -1
        buf.coff[d] = total
        if clo <= chi:
            vs = clo // LANES * LANES
            at = vs
            var need = total + (chi - vs) // LANES * LANES + 2 * LANES
            if need > len(buf.codes):
                buf.codes.resize(max(2 * len(buf.codes), need), 0)
            var cp = buf.codes.unsafe_ptr().unsafe_offset(total - vs)
            # The cells inside the matrix's interior (both letters consumed).
            var first = V(Int32(max(clo, 1)))
            var last = V(Int32(min(chi, d - 1)))
            var left_of = V(Int32(n - m + d))
            var live_lo = V(Int32.MAX)
            var live_hi = V(-1)
            while at <= chi:
                var iv = V(Int32(at)) + lane
                var inside = iv.ge(first) & iv.le(last)
                var diag = h2.unsafe_offset(at - 1).unsafe_load[width=LANES]()
                var up = h1.unsafe_offset(at - 1).unsafe_load[width=LANES]()
                var left = h1.unsafe_offset(at).unsafe_load[width=LANES]()
                var rl = rp.unsafe_offset(LANES + at - 1).unsafe_load[width=LANES]()
                var ql = qp.unsafe_offset(LANES + m - d + at).unsafe_load[width=LANES]()
                var z = diag + rl.eq(ql).select(zero, xv)
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
                # A cell is kept while its least layer, H, and the least its letters left can cost in any layer
                # (`over` more reference letters than query ones, each at least the cheaper extension) fit the
                # bound; the cells between kept ones keep their costs, each some path's.
                var over = left_of - iv - iv
                var least_left = (
                    min(max(over, zero), longest) * del_least + min(max(zero - over, zero), longest) * ins_least
                )
                var any = inside & (z + least_left).le(ceiling)
                h0.unsafe_offset(at).unsafe_store(inside.select(min(z, inf), inf))
                comptime if shape > 0:
                    e1c.unsafe_offset(at).unsafe_store(inside.select(min(de1, inf), inf))
                    f1c.unsafe_offset(at).unsafe_store(inside.select(min(f1, inf), inf))
                comptime if shape == 2:
                    e2c.unsafe_offset(at).unsafe_store(inside.select(min(de2, inf), inf))
                    f2c.unsafe_offset(at).unsafe_store(inside.select(min(f2, inf), inf))
                live_lo = min(live_lo, any.select(iv, V(Int32.MAX)))
                live_hi = max(live_hi, any.select(iv, V(-1)))
                cp.unsafe_offset(at).unsafe_store(dcode)
                at += LANES
            nlo = Int(live_lo.reduce_min())
            nhi = Int(live_hi.reduce_max())
            # The edges: row 0 (insertions alone) and column 0 (deletions alone), where kept.
            if top_live and clo == 0:
                h0[unsafe_offset=0] = top_cost
                comptime if shape > 0:
                    f1c[unsafe_offset=0] = i1.opening + i1.extension * Int32(d) if d > 0 else INF
                comptime if shape == 2:
                    f2c[unsafe_offset=0] = i2.opening + i2.extension * Int32(d) if d > 0 else INF
                nlo = 0
                nhi = max(nhi, 0)
            if side_live and chi == d:
                h0[unsafe_offset=d] = side_cost
                comptime if shape > 0:
                    e1c[unsafe_offset=d] = d1.opening + d1.extension * Int32(d)
                comptime if shape == 2:
                    e2c[unsafe_offset=d] = d2.opening + d2.extension * Int32(d)
                nlo = min(nlo, d)
                nhi = d
            total += at - vs
        buf.ilo[d] = clo
        buf.ihi[d] = chi
        buf.vstart[d] = vs
        # What this anti-diagonal's rows held before, past what it wrote now, holds nothing again.
        _forget(hp.unsafe_offset(hs * stride + LANES), written, hs, vs, at)
        comptime if shape > 0:
            _forget(gp.unsafe_offset((gs + 0) * stride + LANES), written, 3 + gs + 0, vs, at)
            _forget(gp.unsafe_offset((gs + 2) * stride + LANES), written, 3 + gs + 2, vs, at)
        comptime if shape == 2:
            _forget(gp.unsafe_offset((gs + 1) * stride + LANES), written, 3 + gs + 1, vs, at)
            _forget(gp.unsafe_offset((gs + 3) * stride + LANES), written, 3 + gs + 3, vs, at)
        lo2 = lo1
        hi2 = hi1
        if nlo <= nhi:
            lo1 = nlo
            hi1 = nhi
        else:
            lo1 = 1
            hi1 = 0
    buf.coff[nd] = total
    var last_d = nd - 1
    # The last cell reached within the bound, which every alignment of least cost meets.
    if n < buf.ilo[last_d] or n > buf.ihi[last_d] or hp[unsafe_offset=(last_d % 3) * stride + LANES + n] > bound:
        return False
    # `ksw_backtrack` by anti-diagonal from the last cell, every cell it passes one an alignment of least cost
    # passes, so swept.
    var cp = buf.codes.unsafe_ptr()
    var back = List[UInt64]()
    var i = n
    var j = m
    var state = 0
    while i > 0 and j > 0:
        var dd = i + j
        if i < buf.ilo[dd] or i > buf.ihi[dd]:
            return False
        var tmp = Int(cp[unsafe_offset=buf.coff[dd] + i - buf.vstart[dd]])
        if state == 0:
            state = tmp & 7
        elif ((tmp >> (state + 2)) & 1) == 0:
            state = 0
        if state == 0:
            state = tmp & 7
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
    return True


@always_inline
def _least_left(over: Int, del_least: Int32, ins_least: Int32) -> Int32:
    """The least the letters left can cost in any layer, `over` more of the reference's than the query's: each of
    those at the cheaper extension of its kind."""
    var k = Int32(min(abs(over), 1 << 16))
    return k * (del_least if over > 0 else ins_least)


@always_inline
def _forget(row: MutPointer[Int32, _], mut written: Array[Int, 22], slot: Int, start: Int, end: Int):
    """A row of costs that held `written`'s stretch for `slot` and now holds `start ..< end`: the rest of the
    old stretch set to no finite cost, and the new one recorded."""
    var old_start = written[2 * slot]
    var old_end = written[2 * slot + 1]
    if old_start < old_end:
        var a = min(old_end, start) if start < end else old_end
        if a > old_start:
            _fill(row.unsafe_offset(old_start), a - old_start, INF)
        if start < end and old_end > end:
            var b = max(old_start, end)
            _fill(row.unsafe_offset(b), old_end - b, INF)
    written[2 * slot] = start
    written[2 * slot + 1] = end


def _fill_and_walk(
    reference: Span[UInt8, _],
    query: Span[UInt8, _],
    costs: Costs,
    bound: Int32,
    low: Int,
    high: Int,
    mut buf: TieBuffers,
    mut runs: List[UInt64],
) -> Bool:
    """`_fill_and_walk_shaped` for these costs' shape."""
    if costs.pieces() == 2:
        return _fill_and_walk_shaped[2](reference, query, costs, bound, low, high, buf, runs)
    if costs.opening == 0 and costs.deletion_opening == 0:
        return _fill_and_walk_shaped[0](reference, query, costs, bound, low, high, buf, runs)
    return _fill_and_walk_shaped[1](reference, query, costs, bound, low, high, buf, runs)


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


def _guide_cost(guide: Span[UInt64, _], reference: Span[UInt8, _], query: Span[UInt8, _], costs: Costs) -> Int32:
    """What the path `guide` costs, as `ksw_cigar` counts costs; past what it can count, or a path that does not
    span both, no bound at all."""
    var two = costs.pieces() == 2
    var ins1 = _Piece(Int32(costs.opening), Int32(costs.extension))
    var ins2 = _Piece(Int32(costs.opening2), Int32(costs.extension2)) if two else _Piece(INF, 0)
    var del1 = _Piece(Int32(costs.deletion_opening), Int32(costs.deletion_extension))
    var del2 = _Piece(Int32(costs.deletion_opening2), Int32(costs.deletion_extension2)) if two else _Piece(INF, 0)
    var n = len(reference)
    var m = len(query)
    var total = 0
    var i = 0
    var j = 0
    for run in guide:
        var letter = UInt8(run & 0xFF)
        var k = Int(run >> 8)
        if letter == _INSERTION:
            if j + k > m:
                return INF - 1
            total += Int(_gap(Int32(k), ins1, ins2, two))
            j += k
        elif letter == _DELETION:
            if i + k > n:
                return INF - 1
            total += Int(_gap(Int32(k), del1, del2, two))
            i += k
        else:
            if i + k > n or j + k > m:
                return INF - 1
            for t in range(k):
                if reference[i + t] != query[j + t]:
                    total += costs.mismatch
            i += k
            j += k
        if total >= Int(INF - 1):
            return INF - 1
    if i != n or j != m:
        return INF - 1
    return Int32(total)


def ksw_cigar(
    reference: Span[UInt8, _],
    query: Span[UInt8, _],
    costs: Costs,
    guide: Span[UInt64, _],
    mut buf: TieBuffers,
    low: Int = -(1 << 40),
    high: Int = 1 << 40,
    by_rows: Bool = False,
) -> List[UInt64]:
    """The alignment of all of `reference` to all of `query` at least cost that KSW2's rule picks, as runs
    (`cigar.run_of`, `=` `X` `I` `D`), inside the diagonals `low ..= high` (reference letters less query letters).
    `guide`, an alignment of the two inside them as runs, bounds the cost: the cells swept are those that can lie
    on an alignment no costlier (see the module's note), so the cheaper `guide`, the fewer; an alignment of least
    cost sweeps about what a wavefront search did. With `by_rows`, every cell inside the diagonals is swept by
    rows instead, the reference the bounded sweep is held to. No alignment inside the diagonals: `guide` as it
    stands."""
    var runs = List[UInt64]()
    var n = len(reference)
    var m = len(query)
    if by_rows:
        if _rows_band(n, m, low, high, buf) and _fill_and_walk_rows(reference, query, costs, buf, runs):
            return runs^
        return _spelled(guide, reference, query)
    var bound = _guide_cost(guide, reference, query, costs)
    if _fill_and_walk(reference, query, costs, bound, low, high, buf, runs):
        return runs^
    # A guide cheaper than any alignment inside the diagonals, which no true path is: every cell they hold.
    if bound < INF - 1 and _fill_and_walk(reference, query, costs, INF - 1, low, high, buf, runs):
        return runs^
    return _spelled(guide, reference, query)
