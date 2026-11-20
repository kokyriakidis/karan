# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the
# MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
Colinear chaining of seeds as minimap2 2.28 chains them (`lchain.c`, MIT, Heng Li), ported for its results:
`mg_lchain_dp`, the dynamic programming over at most `max_iter` predecessors that minimap2 runs for reads, and
`mg_lchain_rmq`, the range-maximum chaining it runs for assemblies and for the long joins of reads, then
`mg_chain_backtrack` cutting chains from the best ends and `compact_a` laying them out by reference position.

An anchor is minimap2's `mm128_t`: `x` the strand in bit 63 and the last reference position of the match in the
low 32 bits (one reference, so no sequence id), `y` the flags in bits 40 to 43, the match's span in bits 32 to 39 and
the last query position, on the strand that matches the reference, in the low 32 bits. Keeping minimap2's packing
keeps its orders and the hashes that break its ties.

`mg_lchain_rmq` keeps its active anchors in AVL trees (`krmq.h`) ordered by query position; here a segment tree
over the same order answers the same range-minimum queries (an anchor's priority its score plus half the gap penalty
of its diagonal sum) and a tree of 64-bit words walks the same predecessors back, a few word operations a step. They
differ from the AVL tree only where two active anchors have equal priority, the same score on the same
anti-diagonal: the segment tree takes the one earlier in query order, the AVL tree one its shape picks.
"""


from std.bit import count_leading_zeros
from std.memory import bitcast

comptime SEED_LONG_JOIN = UInt64(1) << 40
"""Set by the alignment on an anchor that closes a gap two of minimap2's filters found joined colinear chains."""
comptime SEED_IGNORE = UInt64(1) << 41
"""Set by the alignment on an anchor inside a stretch of disagreeing gaps: the gap filling steps over it."""
comptime SEED_TANDEM = UInt64(1) << 42
"""The anchor's query minimizer repeats next to itself (a tandem seed): the gap filling never ends a piece on it."""
comptime NO_SCORE = Int32.MIN
"""`comput_sc`'s answer for a predecessor that cannot precede."""


@fieldwise_init
struct Packed(Copyable, ImplicitlyCopyable, Movable, TrivialRegisterPassable):
    """minimap2's `mm128_t`: an anchor's two words."""

    var x: UInt64
    var y: UInt64

    @always_inline
    def reference_end(self) -> Int32:
        """The match's last reference position, `(int32_t)x`."""
        return self.x.cast[DType.int32]()

    @always_inline
    def query_end(self) -> Int32:
        """The match's last query position on the matching strand, `(int32_t)y`."""
        return self.y.cast[DType.int32]()

    @always_inline
    def span(self) -> Int32:
        """The match's length."""
        return ((self.y >> 32) & 0xFF).cast[DType.int32]()

    @always_inline
    def reverse(self) -> Bool:
        """Whether the query's reverse complement matches."""
        return (self.x >> 63) != 0


@always_inline
def mg_log2(x: Float32) -> Float32:
    """minimap2's fast `log2` (for `x >= 2`): the exponent's bits and a quadratic in the mantissa, so chain scores
    round as minimap2's do."""
    var bits = bitcast[DType.uint32](x)
    var log_2 = Float32(Int((bits >> 23) & 255) - 128)
    bits &= ~(UInt32(255) << 23)
    bits += UInt32(127) << 23
    var z = bitcast[DType.float32](bits)
    log_2 += (Float32(-0.34484843) * z + Float32(2.02466578)) * z - Float32(0.67487759)
    return log_2


@always_inline
def gap_penalty(dd: Int32, dg: Int32, pen_gap: Float32, pen_skip: Float32) -> Int32:
    """The chaining penalty of a gap difference `dd` over `dg` letters: linear and half a logarithm, truncated."""
    var lin_pen = pen_gap * Float32(dd) + pen_skip * Float32(dg)
    var log_pen = mg_log2(Float32(dd + 1)) if dd >= 1 else Float32(0)
    return Int32(Int(lin_pen + Float32(0.5) * log_pen))


@always_inline
def comput_sc(
    ai: Packed, aj: Packed, max_dist_x: Int32, max_dist_y: Int32, bw: Int32, pen_gap: Float32, pen_skip: Float32
) -> Int32:
    """What anchor `aj` before `ai` adds to a chain (`comput_sc` for one segment, not spliced), `NO_SCORE` where it
    cannot precede it: the smaller of `aj`'s span and the smaller gap, less the gap's penalty."""
    var dq = ai.query_end() - aj.query_end()
    if dq <= 0 or dq > max_dist_x:
        return NO_SCORE
    var dr = (ai.x - aj.x).cast[DType.int32]()
    if dr == 0 or dq > max_dist_y:
        return NO_SCORE
    var dd = abs(dr - dq)
    if dd > bw:
        return NO_SCORE
    var dg = min(dr, dq)
    var q_span = aj.span()
    var sc = min(q_span, dg)
    if dd != 0 or dg > q_span:
        sc -= gap_penalty(dd, dg, pen_gap, pen_skip)
    return sc


@always_inline
def comput_sc_simple(ai: Packed, aj: Packed, pen_gap: Float32, pen_skip: Float32) -> Tuple[Int32, Bool, Int32]:
    """`comput_sc_simple`: the score `aj` adds before `ai` with no range checks, whether the two lie on one diagonal
    with no gap (`exact`), and the gap difference (`width`)."""
    var dq = ai.query_end() - aj.query_end()
    var dr = (ai.x - aj.x).cast[DType.int32]()
    var dd = abs(dr - dq)
    var dg = min(dr, dq)
    var q_span = aj.span()
    var sc = min(q_span, dg)
    var exact = dd == 0 and dg <= q_span
    if dd != 0 or dq > q_span:
        sc -= gap_penalty(dd, dg, pen_gap, pen_skip)
    return (sc, exact, dd)


@fieldwise_init
struct ChainSet(Movable):
    """Chains as `mg_lchain_*` returns them: the anchors of every chain one after another, each chain's in reference
    order and the chains in order of their first anchor, and `u`, each chain's `score << 32 | anchors`."""

    var a: List[Packed]
    var u: List[UInt64]

    def anchors(deinit self) -> List[Packed]:
        """The anchors, the chains given up."""
        return self.a^


def stable_sort_by_x(mut a: List[Packed]):
    """`radix_sort_128x`: the anchors by `x`, those with equal `x` kept in their order."""

    var keys = List[Tuple[UInt64, Int]](capacity=len(a))
    for index in range(len(a)):
        keys.append((a[index].x, index))
    sort(Span(keys))
    var sorted = List[Packed](capacity=len(a))
    for key in keys:
        sorted.append(a[key[1]])
    a = sorted^


def chain_backtrack(
    f: List[Int32], p: List[Int], mut t: List[Int32], min_cnt: Int, min_sc: Int32, max_drop: Int32
) -> Tuple[List[UInt64], List[Int]]:
    """`mg_chain_backtrack`: chains cut from the best-scoring ends down, each anchor in one chain, a chain stopped where
    its score has fallen `max_drop` below its best (`mg_chain_bk_end`) and kept with `min_cnt` anchors and `min_sc`.
    Each chain's `score << 32 | anchors` and its anchors' indices, last first."""
    var n = len(f)
    var u = List[UInt64]()
    var v = List[Int]()
    var z = List[Tuple[Int32, Int]]()
    for i in range(n):
        if f[i] >= min_sc:
            z.append((f[i], i))
    if len(z) == 0:
        return (u^, v^)

    sort(Span(z))
    for i in range(n):
        t[i] = 0
    for k in range(len(z) - 1, -1, -1):
        var top = z[k][1]
        if t[top] != 0:
            continue
        var n_v0 = len(v)
        # mg_chain_bk_end: where the chain from z[k] should stop.
        var end_i: Int
        var max_i = top
        var max_s = Int32(0)
        var i = top
        while True:
            t[i] = 2
            i = p[i]
            end_i = i
            var s = z[k][0] if i < 0 else z[k][0] - f[i]
            if s > max_s:
                max_s = s
                max_i = i
            elif max_s - s > max_drop:
                break
            if i < 0 or t[i] != 0:
                break
        i = top
        while i >= 0 and i != end_i:
            t[i] = 0
            i = p[i]
        end_i = max_i
        i = top
        while i != end_i:
            v.append(i)
            t[i] = 1
            i = p[i]
        var sc = z[k][0] if i < 0 else z[k][0] - f[i]
        var count = len(v) - n_v0
        if sc >= min_sc and count > 0 and count >= min_cnt:
            u.append(UInt64(Int(sc)) << 32 | UInt64(count))
        else:
            v.shrink(n_v0)
    return (u^, v^)


def compact(u: List[UInt64], v: List[Int], a: List[Packed]) -> ChainSet:
    """`compact_a`: each chain's anchors in reference order, the chains ordered by their first anchor's `x` (ties
    in backtrack order)."""
    var b = List[Packed](capacity=len(v))
    var starts = List[Int](capacity=len(u))
    var k = 0
    for i in range(len(u)):
        var ni = Int(u[i] & 0xFFFFFFFF)
        starts.append(k)
        for j in range(ni):
            b.append(a[v[k + (ni - j - 1)]])
        k += ni
    var w = List[Tuple[UInt64, Int]](capacity=len(u))
    for i in range(len(u)):
        w.append((b[starts[i]].x, i))

    sort(Span(w))
    var out = ChainSet(List[Packed](capacity=len(b)), List[UInt64](capacity=len(u)))
    for entry in w:
        var j = entry[1]
        out.u.append(u[j])
        var ni = Int(u[j] & 0xFFFFFFFF)
        for index in range(starts[j], starts[j] + ni):
            out.a.append(b[index])
    return out^


def chain_dp(
    max_dist_x_in: Int32,
    max_dist_y_in: Int32,
    bw: Int32,
    max_skip: Int,
    max_iter: Int,
    min_cnt: Int,
    min_sc: Int32,
    pen_gap: Float32,
    pen_skip: Float32,
    a: List[Packed],
) -> ChainSet:
    """`mg_lchain_dp` for one segment, not spliced: each anchor's best predecessor among the `max_iter` before it
    within `max_dist_x` on the reference, the look stopped after `max_skip` that only rejoin a chain already reaching
    this anchor, and the best-scoring anchor in range tried as well; then the chains backtracked and compacted.
    `a` must be sorted by `x` (`stable_sort_by_x`)."""
    var n = len(a)
    if n == 0:
        return ChainSet(List[Packed](), List[UInt64]())
    var max_dist_x = max(max_dist_x_in, bw)
    var max_dist_y = max(max_dist_y_in, bw)
    var max_drop = bw
    var f = List[Int32](length=n, fill=0)
    var p = List[Int](length=n, fill=-1)
    var t = List[Int32](length=n, fill=0)
    var st = 0
    var max_ii = -1
    for i in range(n):
        var max_j = -1
        var max_f = a[i].span()
        var n_skip = 0
        while st < i and (a[i].x >> 32 != a[st].x >> 32 or a[i].x > a[st].x + UInt64(Int(max_dist_x))):
            st += 1
        if i - st > max_iter:
            st = i - max_iter
        var j = i - 1
        while j >= st:
            var sc = comput_sc(a[i], a[j], max_dist_x, max_dist_y, bw, pen_gap, pen_skip)
            if sc == NO_SCORE:
                j -= 1
                continue
            sc += f[j]
            if sc > max_f:
                max_f = sc
                max_j = j
                if n_skip > 0:
                    n_skip -= 1
            elif t[j] == Int32(i):
                n_skip += 1
                if n_skip > max_skip:
                    break
            if p[j] >= 0:
                t[p[j]] = Int32(i)
            j -= 1
        var end_j = j
        if max_ii < 0 or a[i].x - a[max_ii].x > UInt64(Int(max_dist_x)):
            var best = Int32.MIN
            max_ii = -1
            for k in range(i - 1, st - 1, -1):
                if best < f[k]:
                    best = f[k]
                    max_ii = k
        if max_ii >= 0 and max_ii < end_j:
            var tmp = comput_sc(a[i], a[max_ii], max_dist_x, max_dist_y, bw, pen_gap, pen_skip)
            if tmp != NO_SCORE and max_f < tmp + f[max_ii]:
                max_f = tmp + f[max_ii]
                max_j = max_ii
        f[i] = max_f
        p[i] = max_j
        if max_ii < 0 or (a[i].x - a[max_ii].x <= UInt64(Int(max_dist_x)) and f[max_ii] < f[i]):
            max_ii = i
    var chains = chain_backtrack(f, p, t, min_cnt, min_sc, max_drop)
    if len(chains[0]) == 0:
        return ChainSet(List[Packed](), List[UInt64]())
    return compact(chains[0], chains[1], a)


struct Presence(Movable):
    """Present leaves as a tree of 64-bit words, each bit of a word above saying whether a word below has any:
    insert, erase and the last present leaf at or before one, each a few word operations a level (four levels hold
    16 million leaves)."""

    var levels: List[List[UInt64]]
    var count: Int

    def __init__(out self, leaves: Int):
        self.levels = List[List[UInt64]]()
        self.count = 0
        var words = max((leaves + 63) >> 6, 1)
        while True:
            self.levels.append(List[UInt64](length=words, fill=0))
            if words == 1:
                break
            words = (words + 63) >> 6

    @always_inline
    def contains(self, leaf: Int) -> Bool:
        return (self.levels[0][leaf >> 6] >> UInt64(leaf & 63)) & 1 != 0

    def insert(mut self, leaf: Int):
        """`leaf` present (it must not be)."""
        self.count += 1
        var p = leaf
        for h in range(len(self.levels)):
            var w = p >> 6
            var was = self.levels[h][w]
            self.levels[h][w] = was | (UInt64(1) << UInt64(p & 63))
            if was != 0:
                return
            p = w

    def erase(mut self, leaf: Int):
        """`leaf` absent (it must be present)."""
        self.count -= 1
        var p = leaf
        for h in range(len(self.levels)):
            var w = p >> 6
            var now = self.levels[h][w] & ~(UInt64(1) << UInt64(p & 63))
            self.levels[h][w] = now
            if now != 0:
                return
            p = w

    def last_at_most(self, leaf: Int) -> Int:
        """The last present leaf at or before `leaf`, -1 if none."""
        var h = 0
        var p = leaf
        while True:
            if p < 0:
                return -1
            var w = p >> 6
            var word = self.levels[h][w] & (UInt64.MAX >> UInt64(63 - (p & 63)))
            if word != 0:
                p = (w << 6) + 63 - Int(count_leading_zeros(word))
                break
            if h + 1 == len(self.levels):
                return -1
            p = w - 1
            h += 1
        while h > 0:
            h -= 1
            p = (p << 6) + 63 - Int(count_leading_zeros(self.levels[h][p]))
        return p


struct RangeMin(Movable):
    """The least priority over any range of leaves, of those present: a segment tree, a power of two leaves, each
    node its range's least priority and the leaf holding it, the leftmost of equals. An update climbs only while it
    changes a node."""

    var size: Int
    var best: List[Float64]
    var arg: List[Int32]

    def __init__(out self, leaves: Int):
        self.size = 1
        while self.size < max(leaves, 1):
            self.size <<= 1
        self.best = List[Float64](length=2 * self.size, fill=Float64.MAX)
        self.arg = List[Int32](length=2 * self.size, fill=-1)

    @always_inline
    def _climb(mut self, leaf: Int):
        var node = (leaf + self.size) >> 1
        while node >= 1:
            var left = 2 * node
            var right = left + 1
            var best: Float64
            var arg: Int32
            if self.best[right] < self.best[left]:
                best = self.best[right]
                arg = self.arg[right]
            else:
                best = self.best[left]
                arg = self.arg[left]
            if best == self.best[node] and arg == self.arg[node]:
                return
            self.best[node] = best
            self.arg[node] = arg
            node >>= 1

    def insert(mut self, leaf: Int, priority: Float64):
        """`leaf` present at `priority`."""
        self.best[leaf + self.size] = priority
        self.arg[leaf + self.size] = Int32(leaf)
        self._climb(leaf)

    def erase(mut self, leaf: Int):
        """`leaf` absent."""
        self.best[leaf + self.size] = Float64.MAX
        self.arg[leaf + self.size] = -1
        self._climb(leaf)

    def least(self, first: Int, last: Int) -> Int:
        """The present leaf of least priority in `first..last` (closed), the leftmost of equals; -1 if none."""
        if first > last:
            return -1
        var lo = first + self.size
        var hi = last + self.size + 1
        var left_best = Float64.MAX
        var left_arg = Int32(-1)
        var right_best = Float64.MAX
        var right_arg = Int32(-1)
        while lo < hi:
            if lo & 1 == 1:
                if self.best[lo] < left_best:
                    left_best = self.best[lo]
                    left_arg = self.arg[lo]
                lo += 1
            if hi & 1 == 1:
                hi -= 1
                if self.arg[hi] >= 0 and self.best[hi] <= right_best:
                    right_best = self.best[hi]
                    right_arg = self.arg[hi]
            lo >>= 1
            hi >>= 1
        if right_arg >= 0 and right_best < left_best:
            return Int(right_arg)
        return Int(left_arg)


def chain_rmq(
    max_dist_in: Int32,
    max_dist_inner_in: Int32,
    bw: Int32,
    max_chn_skip: Int,
    cap_rmq_size: Int,
    min_cnt: Int,
    min_sc: Int32,
    pen_gap: Float32,
    pen_skip: Float32,
    a: List[Packed],
) -> ChainSet:
    """`mg_lchain_rmq`: each anchor's predecessor the active anchor (within `max_dist` behind on the reference, at
    most `cap_rmq_size` of them) of best score plus half the gap penalty of its diagonal sum among those up to
    `max_dist` behind on the query, then, unless that one lies on the anchor's own diagonal, the anchors up to
    `max_dist_inner` behind on the query looked at one by one, nearest first, until `max_chn_skip` only rejoin a
    chain already reaching this anchor; a gap difference above `bw` never chains. Then the chains backtracked and
    compacted. `a` must be sorted by `x`."""
    var n = len(a)
    if n == 0:
        return ChainSet(List[Packed](), List[UInt64]())
    var max_dist = max(max_dist_in, bw)
    var max_dist_inner = min(max(max_dist_inner_in, 0), max_dist)
    var max_drop = bw
    var f = List[Int32](length=n, fill=0)
    var p = List[Int](length=n, fill=-1)
    var t = List[Int32](length=n, fill=0)
    # The trees' order: by query position, then index; leaf_of[j] is anchor j's leaf.
    var order = List[Tuple[Int32, Int]](capacity=n)
    for j in range(n):
        order.append((a[j].query_end(), j))

    sort(Span(order))
    # Each anchor's leaf, and in leaf order what the inner look reads: positions, span, score, the predecessor's
    # leaf and minimap2's t[] (which the backtrack clears before its own use).
    var leaf_of = List[Int](length=n, fill=0)
    var leaf_y = List[Int32](capacity=n)
    var leaf_x = List[Int32](capacity=n)
    var leaf_span = List[Int32](capacity=n)
    var leaf_i = List[Int](capacity=n)
    for leaf in range(n):
        var j = order[leaf][1]
        leaf_of[j] = leaf
        leaf_y.append(order[leaf][0])
        leaf_x.append(a[j].reference_end())
        leaf_span.append(a[j].span())
        leaf_i.append(j)
    var leaf_f = List[Int32](length=n, fill=0)
    var leaf_p = List[Int](length=n, fill=-1)
    var leaf_t = List[Int32](length=n, fill=0)

    @always_inline
    def first_after(y: Int32) {imm leaf_y} -> Int:
        """The first leaf whose query position is above `y`."""
        var lo = 0
        var hi = len(leaf_y)
        while lo < hi:
            var mid = (lo + hi) >> 1
            if leaf_y[mid] <= y:
                lo = mid + 1
            else:
                hi = mid
        return lo

    var root = RangeMin(n)
    var in_root = Presence(n)
    var inner = Presence(n if max_dist_inner > 0 else 0)
    var st = 0
    var st_inner = 0
    var i0 = 0
    var half_gap = 0.5 * Float64(pen_gap)
    for i in range(n):
        var max_j = -1
        var max_f = a[i].span()
        var yi = a[i].query_end()
        if i0 < i and a[i0].x != a[i].x:
            for j in range(i0, i):
                var priority = -(Float64(f[j]) + half_gap * Float64(Int(a[j].reference_end()) + Int(a[j].query_end())))
                if not in_root.contains(leaf_of[j]):
                    root.insert(leaf_of[j], priority)
                    in_root.insert(leaf_of[j])
                if max_dist_inner > 0 and not inner.contains(leaf_of[j]):
                    inner.insert(leaf_of[j])
            i0 = i
        while st < i and (
            a[i].x >> 32 != a[st].x >> 32 or a[i].x > a[st].x + UInt64(Int(max_dist)) or in_root.count > cap_rmq_size
        ):
            if in_root.contains(leaf_of[st]):
                root.erase(leaf_of[st])
                in_root.erase(leaf_of[st])
            st += 1
        if max_dist_inner > 0:
            while st_inner < i and (
                a[i].x >> 32 != a[st_inner].x >> 32
                or a[i].x > a[st_inner].x + UInt64(Int(max_dist_inner))
                or inner.count > cap_rmq_size
            ):
                if inner.contains(leaf_of[st_inner]):
                    inner.erase(leaf_of[st_inner])
                st_inner += 1
        # The closed range (y - max_dist, INT32_MAX) .. (y, 0) of krmq's order.
        var first = first_after(yi - max_dist)
        var last = first_after(yi - 1) - 1
        if last + 1 < n and leaf_y[last + 1] == yi and leaf_i[last + 1] == 0:
            last += 1
        var q = root.least(first, last)
        if q >= 0:
            var j = leaf_i[q]
            var simple = comput_sc_simple(a[i], a[j], pen_gap, pen_skip)
            var sc = f[j] + simple[0]
            if simple[2] <= bw and sc > max_f:
                max_f = sc
                max_j = j
            if not simple[1] and inner.count > 0 and yi > 0:
                # The inner anchors nearest first, read in leaf order: what comput_sc_simple would give, its
                # penalty worked out only where the score before it could beat the best.
                var n_skip = 0
                var xi = a[i].reference_end()
                var leaf = inner.last_at_most(first_after(yi - 1) - 1)
                while leaf >= 0:
                    var yj = leaf_y[leaf]
                    if yj < yi - max_dist_inner:
                        break
                    var dq = yi - yj
                    var dr = xi - leaf_x[leaf]
                    var dd = abs(dr - dq)
                    if dd <= bw:
                        var dg = min(dr, dq)
                        var q_span = leaf_span[leaf]
                        sc = leaf_f[leaf] + min(q_span, dg)
                        if sc > max_f and (dd != 0 or dq > q_span):
                            sc -= gap_penalty(dd, dg, pen_gap, pen_skip)
                        if sc > max_f:
                            max_f = sc
                            max_j = leaf_i[leaf]
                            if n_skip > 0:
                                n_skip -= 1
                        elif leaf_t[leaf] == Int32(i):
                            n_skip += 1
                            if n_skip > max_chn_skip:
                                break
                        if leaf_p[leaf] >= 0:
                            leaf_t[leaf_p[leaf]] = Int32(i)
                    leaf = inner.last_at_most(leaf - 1)
        f[i] = max_f
        p[i] = max_j
        leaf_f[leaf_of[i]] = max_f
        leaf_p[leaf_of[i]] = leaf_of[max_j] if max_j >= 0 else -1
    var chains = chain_backtrack(f, p, t, min_cnt, min_sc, max_drop)
    if len(chains[0]) == 0:
        return ChainSet(List[Packed](), List[UInt64]())
    return compact(chains[0], chains[1], a)
