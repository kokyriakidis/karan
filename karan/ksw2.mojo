# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the
# MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
KSW2's dual-affine alignment, `ksw_extd2_sse` as minimap2 2.31 ships it (MIT, Heng Li), ported line for line:
the same scores, the same cells, the same choices among equal alignments, so the same CIGARs.

minimap2 fills the gaps between a chain's anchors and extends its ends with this one function, and what reads
minimap2's alignments, as dorado's HERRO does, learned its choices: of equally good alignments, the walk back
from the end takes the diagonal at every tie (a match or a mismatch), then a deletion, an insertion, the second
gap piece's deletion, its insertion, so gaps come as early and mismatches as late as the score allows (or, with
`RIGHT`, gaps as late, as minimap2 asks of its reversed left extension). A gap of `k` letters costs the less of
`q + k e` and `q2 + k e2`; a match earns `mat[0]`, a mismatch `mat[1]` (negative), the last symbol `m - 1` is a
wildcard.

The recurrence is Suzuki and Kasahara's, by anti-diagonal: each cell keeps its score less its neighbours' in
eight bits, `u` and `v` for the aligning layer, `x`, `y` and `x2`, `y2` for each gap piece's runs, and a code
byte for the walk back: which source gave the cell (`0` the diagonal, `1` and `3` a deletion of each piece, `2`
and `4` an insertion) and, in bits 3 to 6, whether the next cell's gap of each layer continues this one. As in
the C, a vector of `L` cells runs at a time in place, its first cell's left neighbour carried in from the
vector before, and the cells a vector holds past the band's edge are computed too, from whatever the arrays
hold; KSW2's own results at a band's edge depend on them, so this port keeps them, with `L = 16` matching it
byte for byte. With a band no narrower than the matrix every vector width gives the same.

Its band is KSW2's, `w` diagonals either side of the main one, and nothing else: in this representation a band
whose edge moves otherwise than KSW2's reads cells it never computed, and the scores inside it go wrong.
"""

from std.memory import unsafe_memcpy, unsafe_memset

comptime KSW_NEG_INF = -0x40000000
comptime KSW_EZ_SCORE_ONLY = 0x01
comptime KSW_EZ_RIGHT = 0x02
comptime KSW_EZ_APPROX_MAX = 0x08
comptime KSW_EZ_APPROX_DROP = 0x10
comptime KSW_EZ_EXTZ_ONLY = 0x40
comptime KSW_EZ_REV_CIGAR = 0x80
comptime KSW_CIGAR_MATCH = 0
comptime KSW_CIGAR_INS = 1
comptime KSW_CIGAR_DEL = 2


struct KswResult(Movable):
    """`ksw_extz_t`: the best score `max` and its cell (`max_q`, `max_t`), the best reaching the query's end
    (`mqe` at `mqe_t`) and the target's (`mte` at `mte_q`), the score reaching both ends, whether the Z-drop
    gave up, whether an extension reaches the query's end, and the CIGAR as KSW2's (`length << 4 | op`)."""

    var max: Int
    var zdropped: Bool
    var max_q: Int
    var max_t: Int
    var mqe: Int
    var mqe_t: Int
    var mte: Int
    var mte_q: Int
    var score: Int
    var reach_end: Bool
    var cigar: List[UInt32]

    def __init__(out self):
        self.max = 0
        self.zdropped = False
        self.max_q = -1
        self.max_t = -1
        self.mqe = KSW_NEG_INF
        self.mqe_t = -1
        self.mte = KSW_NEG_INF
        self.mte_q = -1
        self.score = KSW_NEG_INF
        self.reach_end = False
        self.cigar = List[UInt32]()

    def reset(mut self):
        """`ksw_reset_extz`."""
        self.max_q = -1
        self.max_t = -1
        self.mqe_t = -1
        self.mte_q = -1
        self.max = 0
        self.score = KSW_NEG_INF
        self.mqe = KSW_NEG_INF
        self.mte = KSW_NEG_INF
        self.zdropped = False
        self.reach_end = False
        self.cigar.clear()


struct KswBuffers(Movable):
    """One thread's memory, kept from call to call: the difference arrays, the scores, the target and the
    reversed query laid one after the other as the C lays them, the exact maxima and the walk's codes."""

    var u: List[Int8]
    var v: List[Int8]
    var x: List[Int8]
    var y: List[Int8]
    var x2: List[Int8]
    var y2: List[Int8]
    var s: List[Int8]
    var sfqr: List[UInt8]
    var h: List[Int32]
    var p: List[UInt8]
    var off: List[Int]
    var off_end: List[Int]

    def __init__(out self):
        self.u = List[Int8]()
        self.v = List[Int8]()
        self.x = List[Int8]()
        self.y = List[Int8]()
        self.x2 = List[Int8]()
        self.y2 = List[Int8]()
        self.s = List[Int8]()
        self.sfqr = List[UInt8]()
        self.h = List[Int32]()
        self.p = List[UInt8]()
        self.off = List[Int]()
        self.off_end = List[Int]()


@always_inline
def _push_cigar(mut cigar: List[UInt32], op: Int, length: Int):
    """`ksw_push_cigar`."""
    if len(cigar) == 0 or Int(cigar[len(cigar) - 1] & 0xF) != op:
        cigar.append(UInt32(length << 4 | op))
    else:
        cigar[len(cigar) - 1] += UInt32(length << 4)


def _apply_zdrop(mut ez: KswResult, h: Int, r: Int, t: Int, zdrop: Int, e: Int) -> Bool:
    """`ksw_apply_zdrop` by anti-diagonal."""
    if h > ez.max:
        ez.max = h
        ez.max_t = t
        ez.max_q = r - t
    elif t >= ez.max_t and r - t >= ez.max_q:
        var tl = t - ez.max_t
        var ql = (r - t) - ez.max_q
        var l = tl - ql if tl > ql else ql - tl
        if zdrop >= 0 and ez.max - h > zdrop + l * e:
            ez.zdropped = True
            return True
    return False


def _backtrack(mut ez: KswResult, buf: KswBuffers, n_col: Int, i0: Int, j0: Int, is_rev: Bool):
    """`ksw_backtrack` by anti-diagonal (`is_rot`), no introns."""
    var i = i0
    var j = j0
    var state = 0
    ez.cigar.clear()
    while i >= 0 and j >= 0:
        var force_state = -1
        var r = i + j
        if i < buf.off[r]:
            force_state = 2
        if i > buf.off_end[r]:
            force_state = 1
        var tmp = 0
        if force_state < 0:
            tmp = Int(buf.p[r * n_col + i - buf.off[r]])
        if state == 0:
            state = tmp & 7
        elif ((tmp >> (state + 2)) & 1) == 0:
            state = 0
        if state == 0:
            state = tmp & 7
        if force_state >= 0:
            state = force_state
        if state == 0:
            _push_cigar(ez.cigar, KSW_CIGAR_MATCH, 1)
            i -= 1
            j -= 1
        elif state == 1 or state == 3:
            _push_cigar(ez.cigar, KSW_CIGAR_DEL, 1)
            i -= 1
        else:
            _push_cigar(ez.cigar, KSW_CIGAR_INS, 1)
            j -= 1
    if i >= 0:
        _push_cigar(ez.cigar, KSW_CIGAR_DEL, i + 1)
    if j >= 0:
        _push_cigar(ez.cigar, KSW_CIGAR_INS, j + 1)
    if not is_rev:
        ez.cigar.reverse()


@always_inline
def _lanes_below[W: Int](count: Int) -> SIMD[DType.bool, W]:
    """The first `count` lanes of `W`."""
    var lanes = SIMD[DType.int32, W]()
    comptime for k in range(W):
        lanes[k] = Int32(k)
    return lanes.lt(SIMD[DType.int32, W](Int32(count)))


@always_inline
def _keep[L: Int, W: Int](computed: SIMD[DType.int8, W], old: SIMD[DType.int8, W], cells: Int) -> SIMD[DType.int8, W]:
    """A piece's new values in its first `cells` lanes, and past them what the arrays held: the cells an `L`-wide
    vector would not have reached keep their stale values, as in the C."""
    comptime if W == L:
        return computed
    else:
        if cells >= W:
            return computed
        return _lanes_below[W](cells).select(computed, old)


@always_inline
def _resize[T: Copyable & Deinitable](mut buf: List[T], n: Int, value: T):
    if len(buf) < n:
        buf.resize(n, value)


def ksw_extd2[
    L: Int = 16, W: Int = L
](
    query: Span[UInt8, _],
    target: Span[UInt8, _],
    m: Int,
    mat: Span[Int8, _],
    q_in: Int,
    e_in: Int,
    q2_in: Int,
    e2_in: Int,
    w_in: Int,
    zdrop: Int,
    end_bonus: Int,
    flag: Int,
    mut buf: KswBuffers,
    mut ez: KswResult,
):
    """`ksw_extd2_sse(km, qlen, query, tlen, target, m, mat, q, e, q2, e2, w, zdrop, end_bonus, flag, ez)`.

    `L` is the vector width the results follow: the cells an anti-diagonal computes past the band's edge are those of
    whole `L`-wide vectors, as the C's (16, SSE) are. `W`, a multiple of `L`, is how many cells one instruction
    computes: an anti-diagonal's cells depend only on the one before, so its `L`-rounded range runs `W` at a time, the
    last piece storing nothing past it; `ksw_extd2[16, 64]` gives the C's results at AVX-512's width."""
    comptime assert W % L == 0, "W computes whole L-wide vectors"
    comptime V = SIMD[DType.int8, W]
    ez.reset()
    var qlen = len(query)
    var tlen = len(target)
    if m <= 1 or qlen <= 0 or tlen <= 0:
        return
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
    var qe = q + e
    var with_cigar = (flag & KSW_EZ_SCORE_ONLY) == 0
    var approx_max = (flag & KSW_EZ_APPROX_MAX) != 0
    var right = (flag & KSW_EZ_RIGHT) != 0
    var zero_ = V(0)
    var q_ = V(Int8(q))
    var q2_ = V(Int8(q2))
    var qe_ = V(Int8(q + e))
    var qe2_ = V(Int8(q2 + e2))
    var sc_mch = Int8(mat[0])
    var sc_mis = Int8(mat[1])
    var sc_n = Int8(-e2) if mat[m * m - 1] == 0 else mat[m * m - 1]
    var sc_mch_ = V(sc_mch)
    var sc_mis_ = V(sc_mis)
    var sc_n_ = V(sc_n)
    var m1_ = SIMD[DType.uint8, W](UInt8(m - 1))

    var w = w_in
    if w < 0:
        w = tlen if tlen > qlen else qlen
    var wl = w
    var wr = w
    var tlen_ = (tlen + L - 1) // L
    var n_col_ = qlen if qlen < tlen else tlen
    n_col_ = ((n_col_ if n_col_ < w + 1 else w + 1) + L - 1) // L + 1
    var qlen_ = (qlen + L - 1) // L
    var max_sc = Int(mat[0])
    var min_sc = Int(mat[1])
    for k in range(1, m * m):
        max_sc = max(max_sc, Int(mat[k]))
        min_sc = min(min_sc, Int(mat[k]))
    if -min_sc > 2 * (q + e):
        return

    var long_thres = (q2 - q) // (e - e2) - 1 if e != e2 else 0
    if q2 + e2 + long_thres * e2 > q + e + long_thres * e:
        long_thres += 1
    var long_diff = long_thres * (e - e2) - (q2 - q) - e2

    # The arrays, a `W` vector past the last cell so a vector's tail may be loaded, and one before the first so the
    # cell before a range always exists (see the core loop).
    var n = W + tlen_ * L + W
    _resize(buf.u, n, Int8(0))
    _resize(buf.v, n, Int8(0))
    _resize(buf.x, n, Int8(0))
    _resize(buf.y, n, Int8(0))
    _resize(buf.x2, n, Int8(0))
    _resize(buf.y2, n, Int8(0))
    _resize(buf.s, n, Int8(0))
    var u = buf.u.unsafe_ptr().unsafe_offset(W)
    var v = buf.v.unsafe_ptr().unsafe_offset(W)
    var x = buf.x.unsafe_ptr().unsafe_offset(W)
    var y = buf.y.unsafe_ptr().unsafe_offset(W)
    var x2 = buf.x2.unsafe_ptr().unsafe_offset(W)
    var y2 = buf.y2.unsafe_ptr().unsafe_offset(W)
    var s = buf.s.unsafe_ptr().unsafe_offset(W)
    # Each array starts as one byte throughout.
    unsafe_memset(buf.u.unsafe_ptr(), UInt8(Int8(-q - e)), n)
    unsafe_memset(buf.v.unsafe_ptr(), UInt8(Int8(-q - e)), n)
    unsafe_memset(buf.x.unsafe_ptr(), UInt8(Int8(-q - e)), n)
    unsafe_memset(buf.y.unsafe_ptr(), UInt8(Int8(-q - e)), n)
    unsafe_memset(buf.x2.unsafe_ptr(), UInt8(Int8(-q2 - e2)), n)
    unsafe_memset(buf.y2.unsafe_ptr(), UInt8(Int8(-q2 - e2)), n)
    unsafe_memset(buf.s.unsafe_ptr(), 0, n)
    # The target, then the reversed query, zero past each, as the C's one zeroed block lays them.
    var sf_n = tlen_ * L
    var qr_n = (qlen_ + 1) * L
    _resize(buf.sfqr, sf_n + qr_n + W, UInt8(0))
    var sf = buf.sfqr.unsafe_ptr()
    unsafe_memset(sf, 0, sf_n + qr_n + W)
    unsafe_memcpy(dest=sf, src=target.unsafe_ptr(), count=tlen)
    var qr = sf.unsafe_offset(sf_n)
    # The query back to front, 64 letters at a time.
    var qp = query.unsafe_ptr()
    var k = 0
    while k + 64 <= qlen:
        qr.unsafe_offset(qlen - k - 64).unsafe_store(qp.unsafe_offset(k).unsafe_load[width=64]().reversed())
        k += 64
    while k < qlen:
        qr[unsafe_offset=qlen - 1 - k] = qp[unsafe_offset=k]
        k += 1
    var h = buf.h.unsafe_ptr()
    if not approx_max:
        _resize(buf.h, tlen_ * L + 16, Int32(0))
        h = buf.h.unsafe_ptr()
        var neg16 = SIMD[DType.int32, 16](Int32(KSW_NEG_INF))
        var hk = 0
        while hk + 16 <= tlen_ * L:
            h.unsafe_offset(hk).unsafe_store(neg16)
            hk += 16
        while hk < tlen_ * L:
            h[unsafe_offset=hk] = Int32(KSW_NEG_INF)
            hk += 1
    if with_cigar:
        _resize(buf.p, (qlen + tlen - 1) * n_col_ * L + W, UInt8(0))
        _resize(buf.off, qlen + tlen - 1, 0)
        _resize(buf.off_end, qlen + tlen - 1, 0)
    var p = buf.p.unsafe_ptr()
    var u8 = u
    var v8 = v
    var x8 = x
    var x28 = x2

    var h0 = 0
    var last_h0_t = 0
    var last_st = -1
    var last_en = -1
    for r in range(qlen + tlen - 1):
        var st = 0
        var en = tlen - 1
        # The boundaries.
        if st < r - qlen + 1:
            st = r - qlen + 1
        if en > r:
            en = r
        if st < (r - wr + 1) >> 1:
            st = (r - wr + 1) >> 1
        if en > (r + wl) >> 1:
            en = (r + wl) >> 1
        if st > en:
            ez.zdropped = True
            break
        var st0 = st
        var en0 = en
        st = st // L * L
        en = (en + L) // L * L - 1
        # Boundary conditions.
        var x1: Int8
        var x21: Int8
        var v1: Int8
        if st > 0:
            if st - 1 >= last_st and st - 1 <= last_en:
                x1 = x8[unsafe_offset=st - 1]
                x21 = x28[unsafe_offset=st - 1]
                v1 = v8[unsafe_offset=st - 1]
            else:
                x1 = Int8(-q - e)
                x21 = Int8(-q2 - e2)
                v1 = Int8(-q - e)
        else:
            x1 = Int8(-q - e)
            x21 = Int8(-q2 - e2)
            v1 = Int8(-q - e) if r == 0 else (
                Int8(-e) if r < long_thres else (Int8(long_diff) if r == long_thres else Int8(-e2))
            )
        if en >= r:
            y[unsafe_offset=r] = Int8(-q - e)
            y2[unsafe_offset=r] = Int8(-q2 - e2)
            u8[unsafe_offset=r] = Int8(-q - e) if r == 0 else (
                Int8(-e) if r < long_thres else (Int8(long_diff) if r == long_thres else Int8(-e2))
            )
        # The scores, from the first cell on in whole vectors, as the C stores them.
        var qrr = qr.unsafe_offset(qlen - 1 - r)
        var s_end = st0 + ((en0 - st0) // L + 1) * L
        var t = st0
        while t < s_end:
            var sq = sf.unsafe_offset(t).unsafe_load[width=W]()
            var sr = qrr.unsafe_offset(t).unsafe_load[width=W]()
            var wild = sq.eq(m1_) | sr.eq(m1_)
            var sc = wild.select(sc_n_, sq.eq(sr).select(sc_mch_, sc_mis_))
            comptime if W > L:
                if s_end - t < W:
                    sc = _lanes_below[W](s_end - t).select(sc, s.unsafe_offset(t).unsafe_load[width=W]())
            s.unsafe_offset(t).unsafe_store(sc)
            t += W
        # The core loop.
        var pr = p.unsafe_offset(r * n_col_ * L - st)
        if with_cigar:
            buf.off[r] = st
            buf.off_end[r] = en
        # Each cell reads the cell before it (`x`, `v`, `x2`) as it was: the pieces run last to first, so a piece's
        # load one cell early reads cells no piece has written yet; the cell before the range holds the boundary
        # values meanwhile, and gets its own back after (a stale cell the C keeps).
        var kept_x = x[unsafe_offset=st - 1]
        var kept_v = v[unsafe_offset=st - 1]
        var kept_x2 = x2[unsafe_offset=st - 1]
        x[unsafe_offset=st - 1] = x1
        v[unsafe_offset=st - 1] = v1
        x2[unsafe_offset=st - 1] = x21
        var at = st + (en - st) // W * W
        while at >= st:
            # A piece of `W` cells; the first done, the last, may hold cells past `en`, which keep what they held
            # (see `_keep`).
            var z = s.unsafe_offset(at).unsafe_load[width=W]()
            var xo = x.unsafe_offset(at).unsafe_load[width=W]()
            var xt1 = x.unsafe_offset(at - 1).unsafe_load[width=W]()
            var vo = v.unsafe_offset(at).unsafe_load[width=W]()
            var vt1 = v.unsafe_offset(at - 1).unsafe_load[width=W]()
            var a = xt1 + vt1
            var ut = u.unsafe_offset(at).unsafe_load[width=W]()
            var yo = y.unsafe_offset(at).unsafe_load[width=W]()
            var b = yo + ut
            var x2o = x2.unsafe_offset(at).unsafe_load[width=W]()
            var x2t1 = x2.unsafe_offset(at - 1).unsafe_load[width=W]()
            var a2 = x2t1 + vt1
            var y2o = y2.unsafe_offset(at).unsafe_load[width=W]()
            var b2 = y2o + ut
            var cells = en - at + 1
            var d = V(0)
            if not with_cigar:
                z = max(z, a)
                z = max(z, b)
                z = max(z, a2)
                z = max(z, b2)
                z = min(z, sc_mch_)
            elif not right:
                d = a.gt(z).select(V(1), V(0))
                z = max(z, a)
                d = b.gt(z).select(V(2), d)
                z = max(z, b)
                d = a2.gt(z).select(V(3), d)
                z = max(z, a2)
                d = b2.gt(z).select(V(4), d)
                z = max(z, b2)
                z = min(z, sc_mch_)
            else:
                d = z.gt(a).select(V(0), V(1))
                z = max(z, a)
                d = z.gt(b).select(d, V(2))
                z = max(z, b)
                d = z.gt(a2).select(d, V(3))
                z = max(z, a2)
                d = z.gt(b2).select(d, V(4))
                z = max(z, b2)
                z = min(z, sc_mch_)
            # Save u and v; turn a, b, a2, b2 into what a run through each layer costs the next cell.
            u.unsafe_offset(at).unsafe_store(_keep[L](z - vt1, ut, cells))
            v.unsafe_offset(at).unsafe_store(_keep[L](z - ut, vo, cells))
            var tmp = z - q_
            a = a - tmp
            b = b - tmp
            tmp = z - q2_
            a2 = a2 - tmp
            b2 = b2 - tmp
            if not with_cigar or not right:
                var ka = a.gt(zero_)
                var kb = b.gt(zero_)
                var ka2 = a2.gt(zero_)
                var kb2 = b2.gt(zero_)
                x.unsafe_offset(at).unsafe_store(_keep[L](ka.select(a, zero_) - qe_, xo, cells))
                y.unsafe_offset(at).unsafe_store(_keep[L](kb.select(b, zero_) - qe_, yo, cells))
                x2.unsafe_offset(at).unsafe_store(_keep[L](ka2.select(a2, zero_) - qe2_, x2o, cells))
                y2.unsafe_offset(at).unsafe_store(_keep[L](kb2.select(b2, zero_) - qe2_, y2o, cells))
                if with_cigar:
                    d = d | ka.select(V(0x08), zero_)
                    d = d | kb.select(V(0x10), zero_)
                    d = d | ka2.select(V(0x20), zero_)
                    d = d | kb2.select(V(0x40), zero_)
            else:
                var na = zero_.gt(a)
                var nb = zero_.gt(b)
                var na2 = zero_.gt(a2)
                var nb2 = zero_.gt(b2)
                x.unsafe_offset(at).unsafe_store(_keep[L](na.select(zero_, a) - qe_, xo, cells))
                y.unsafe_offset(at).unsafe_store(_keep[L](nb.select(zero_, b) - qe_, yo, cells))
                x2.unsafe_offset(at).unsafe_store(_keep[L](na2.select(zero_, a2) - qe2_, x2o, cells))
                y2.unsafe_offset(at).unsafe_store(_keep[L](nb2.select(zero_, b2) - qe2_, y2o, cells))
                d = d | na.select(zero_, V(0x08))
                d = d | nb.select(zero_, V(0x10))
                d = d | na2.select(zero_, V(0x20))
                d = d | nb2.select(zero_, V(0x40))
            if with_cigar:
                # Codes past `en` land in the row's slack or the next row's, which that row writes before any read.
                pr.unsafe_offset(at).unsafe_store(d.cast[DType.uint8]())
            at -= W
        x[unsafe_offset=st - 1] = kept_x
        v[unsafe_offset=st - 1] = kept_v
        x2[unsafe_offset=st - 1] = kept_x2
        if not approx_max:
            # The exact maximum, with the C's 32-bit array and its four-way reduction.
            var max_h: Int
            var max_t: Int
            if r > 0:
                var en1 = st0 + (en0 - st0) // 4 * 4
                if en0 > 0:
                    h[unsafe_offset=en0] = h[unsafe_offset=en0 - 1] + Int32(u8[unsafe_offset=en0])
                else:
                    h[unsafe_offset=en0] = h[unsafe_offset=en0] + Int32(v8[unsafe_offset=en0])
                max_h = Int(h[unsafe_offset=en0])
                max_t = en0
                # The C reduces four lanes, each the positions `st0 + i (mod 4)`, keeping each lane's first maximum, then
                # takes the lanes in order, a later one only when strictly greater. The same here sixteen at a time:
                # each of the sixteen lanes keeps its first maximum, the lanes of one residue mod 4 then give their
                # class's maximum at its first position.
                comptime M = 16
                var lanes = SIMD[DType.int32, M]()
                comptime for k in range(M):
                    lanes[k] = Int32(k)
                var best = SIMD[DType.int32, M](Int32.MIN)
                var at_best = SIMD[DType.int32, M](0)
                var tk = st0
                while tk + M <= en1:
                    var hv = (
                        h.unsafe_offset(tk).unsafe_load[width=M]()
                        + v8.unsafe_offset(tk).unsafe_load[width=M]().cast[DType.int32]()
                    )
                    h.unsafe_offset(tk).unsafe_store(hv)
                    var gt = hv.gt(best)
                    best = gt.select(hv, best)
                    at_best = gt.select(lanes + Int32(tk), at_best)
                    tk += M
                # Lanes 8 apart, then 4 apart, are of one class: the greater of each pair, at the earlier position on a
                # tie.
                var lo8 = best.slice[8, offset=0]()
                var hi8 = best.slice[8, offset=8]()
                var lo8_at = at_best.slice[8, offset=0]()
                var hi8_at = at_best.slice[8, offset=8]()
                var take8 = hi8.gt(lo8) | (hi8.eq(lo8) & hi8_at.lt(lo8_at))
                var best8 = take8.select(hi8, lo8)
                var at8 = take8.select(hi8_at, lo8_at)
                var lo4 = best8.slice[4, offset=0]()
                var hi4 = best8.slice[4, offset=4]()
                var lo4_at = at8.slice[4, offset=0]()
                var hi4_at = at8.slice[4, offset=4]()
                var take4 = hi4.gt(lo4) | (hi4.eq(lo4) & hi4_at.lt(lo4_at))
                var class_best = take4.select(hi4, lo4)
                var class_at = take4.select(hi4_at, lo4_at)
                # The rest of the whole fours, after every position above.
                var lanes4 = lanes.slice[4, offset=0]()
                while tk < en1:
                    var hv = (
                        h.unsafe_offset(tk).unsafe_load[width=4]()
                        + v8.unsafe_offset(tk).unsafe_load[width=4]().cast[DType.int32]()
                    )
                    h.unsafe_offset(tk).unsafe_store(hv)
                    var gt = hv.gt(class_best)
                    class_best = gt.select(hv, class_best)
                    class_at = gt.select(lanes4 + Int32(tk), class_at)
                    tk += 4
                comptime for i in range(4):
                    if max_h < Int(class_best[i]):
                        max_h = Int(class_best[i])
                        max_t = Int(class_at[i])
                while tk < en0:
                    h[unsafe_offset=tk] += Int32(v8[unsafe_offset=tk])
                    if Int(h[unsafe_offset=tk]) > max_h:
                        max_h = Int(h[unsafe_offset=tk])
                        max_t = tk
                    tk += 1
            else:
                h[unsafe_offset=0] = Int32(Int(v8[unsafe_offset=0]) - qe)
                max_h = Int(h[unsafe_offset=0])
                max_t = 0
            if en0 == tlen - 1 and Int(h[unsafe_offset=en0]) > ez.mte:
                ez.mte = Int(h[unsafe_offset=en0])
                ez.mte_q = r - en0
            if r - st0 == qlen - 1 and Int(h[unsafe_offset=st0]) > ez.mqe:
                ez.mqe = Int(h[unsafe_offset=st0])
                ez.mqe_t = st0
            if _apply_zdrop(ez, max_h, r, max_t, zdrop, e2):
                break
            if r == qlen + tlen - 2 and en0 == tlen - 1:
                ez.score = Int(h[unsafe_offset=tlen - 1])
        else:
            # The approximate maximum along one path; the Z-drop only with `APPROX_DROP`.
            if r > 0:
                if last_h0_t >= st0 and last_h0_t <= en0 and last_h0_t + 1 >= st0 and last_h0_t + 1 <= en0:
                    var d0 = Int(v8[unsafe_offset=last_h0_t])
                    var d1 = Int(u8[unsafe_offset=last_h0_t + 1])
                    if d0 > d1:
                        h0 += d0
                    else:
                        h0 += d1
                        last_h0_t += 1
                elif last_h0_t >= st0 and last_h0_t <= en0:
                    h0 += Int(v8[unsafe_offset=last_h0_t])
                else:
                    last_h0_t += 1
                    h0 += Int(u8[unsafe_offset=last_h0_t])
            else:
                h0 = Int(v8[unsafe_offset=0]) - qe
                last_h0_t = 0
            if (flag & KSW_EZ_APPROX_DROP) != 0 and _apply_zdrop(ez, h0, r, last_h0_t, zdrop, e2):
                break
            if r == qlen + tlen - 2 and en0 == tlen - 1:
                ez.score = h0
        last_st = st
        last_en = en
    if with_cigar:
        var rev_cigar = (flag & KSW_EZ_REV_CIGAR) != 0
        var n_col = n_col_ * L
        if not ez.zdropped and (flag & KSW_EZ_EXTZ_ONLY) == 0:
            _backtrack(ez, buf, n_col, tlen - 1, qlen - 1, rev_cigar)
        elif not ez.zdropped and (flag & KSW_EZ_EXTZ_ONLY) != 0 and ez.mqe + end_bonus > ez.max:
            ez.reach_end = True
            _backtrack(ez, buf, n_col, ez.mqe_t, qlen - 1, rev_cigar)
        elif ez.max_t >= 0 and ez.max_q >= 0:
            _backtrack(ez, buf, n_col, ez.max_t, ez.max_q, rev_cigar)
