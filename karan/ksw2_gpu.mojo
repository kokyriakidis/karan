# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the
# MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
KSW2's global dual-affine alignment, as `ksw2_lanes` runs it, on a GPU: the same cells, the same choice among
equal alignments and the same walk back, so for every pair the CIGAR `ksw_lanes` gives it alone in its band.

A warp aligns one pair, along its anti-diagonals: each lane holds a block of `GPU_MAX_BAND / WARP_SIZE` of the
pair's own band's diagonals (the band `-w + 1 ..= w` of target letters less query letters, as `ksw2_lanes`), each
diagonal's latest cell in registers: its best score `H`, its deletion scores `E1`, `E2` and its insertion scores
`F1`, `F2`. A cell needs the cell before it on its diagonal, the cell above (the next diagonal) and the cell to its
left (the previous one), both on the previous anti-diagonal; the cells of one anti-diagonal are independent, so a
step updates every other diagonal, a lane reading its neighbours' edge diagonals by warp shuffle. Each cell's code
byte (which source gave `H`, and which gaps go on) goes to device memory, indexed by anti-diagonal; one lane then
walks back, and only the CIGARs come home. Scores are 32-bit, clamped at `NEG` as `ksw2_lanes` clamps its 16-bit
ones, so pairs of up to `MAX_LANE_LENGTH` letters align identically.
"""

from std.atomic import Atomic
from std.math import ceildiv

from max.gpu import WARP_SIZE, block_idx, lane_id, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.primitives import warp
from max.gpu.sync import syncwarp

from .ksw2_lanes import MAX_LANE_LENGTH, NEG

comptime GPU_MAX_BAND = 256
"""The most diagonals (`2 w`) a pair's band may span on the GPU; wider bands stay on the CPU."""
comptime _WARPS = 4
"""Warps of a block."""
comptime PAIR_LANES = 32
"""Lanes a pair takes. 32 measured best on an RTX 2070: with 16 a lane held more diagonals (6 for a typical band of
70 to 90) and the extra registers cost more warps than the busier lanes gained."""
comptime _SLOT_CHOICES = 4
"""Diagonals a lane may hold: 2, 4, 6, 8 (even, so a lane's diagonals start even), for bands of up to 256."""
comptime _SHAPE = 6
"""Each pair's numbers: target letters' offset and count, query letters' offset and count, band `w`, CIGAR offset."""


@always_inline
def _store_codes[K: Int](dest: MutPointer[UInt8, MutAnyOrigin], pending: Array[UInt8, K]):
    """A lane's `K` codes in power-of-two pieces (`K` even), each with the alignment its block has: `K`'s lowest bit."""
    comptime P = 2 if K <= 2 else (4 if K <= 4 else (8 if K <= 8 else 16))
    comptime align = K & -K
    var v = SIMD[DType.uint8, P](0)
    comptime for s in range(K):
        v[s] = pending[s]
    comptime at8 = K & 16
    comptime at4 = at8 + (K & 8)
    comptime at2 = at4 + (K & 4)
    comptime if (K & 16) != 0:
        dest.unsafe_store[alignment=align](v.slice[16 if P >= 16 else 2, offset=0]())
    comptime if (K & 8) != 0:
        dest.unsafe_offset(at8).unsafe_store[alignment=align](v.slice[8 if P >= 8 else 2, offset=at8]())
    comptime if (K & 4) != 0:
        dest.unsafe_offset(at4).unsafe_store[alignment=align](v.slice[4 if P >= 4 else 2, offset=at4]())
    comptime if (K & 2) != 0:
        dest.unsafe_offset(at2).unsafe_store[alignment=align](v.slice[2, offset=at2]())


def _gpu_kernel[
    K: Int
](
    pair_ids: ImmPointer[Int32, MutAnyOrigin],
    group: Int32,
    targets: ImmPointer[UInt8, MutAnyOrigin],
    queries: ImmPointer[UInt8, MutAnyOrigin],
    shapes: ImmPointer[Int32, MutAnyOrigin],
    code_offsets: ImmPointer[Int64, MutAnyOrigin],
    codes: MutPointer[UInt8, MutAnyOrigin],
    scratch: MutPointer[UInt32, MutAnyOrigin],
    cigars: MutPointer[UInt32, MutAnyOrigin],
    cigar_places: MutPointer[Int32, MutAnyOrigin],
    used: MutPointer[Int32, MutAnyOrigin],
    a: Int32,
    b: Int32,
    ambi: Int32,
    q: Int32,
    e: Int32,
    q2: Int32,
    e2: Int32,
):
    """One pair a warp, `K` of its band's diagonals a lane: its cells along anti-diagonals, then its walk back by lane
    0 into `scratch`, copied to the next free place in `cigars` (`used` counts them), its place and length to
    `cigar_places` (see the module's notes). The warps take the pairs `pair_ids[0 ..< group]`."""
    comptime assert K % 2 == 0, "a lane's diagonals start even, so its active ones follow the step's parity"
    comptime V = Array[Int32, K]
    # A warp holds `WARP_SIZE / PAIR_LANES` pairs; a lane past the last pair idles but keeps step for the shuffles.
    var slot = Int(block_idx.x) * (_WARPS * WARP_SIZE // PAIR_LANES) + Int(thread_idx.x) // PAIR_LANES
    var active = slot < Int(group)
    var pair = Int(pair_ids[unsafe_offset=slot]) if active else 0
    var lane = Int(thread_idx.x) % PAIR_LANES
    var shape = shapes.unsafe_offset(pair * _SHAPE)
    var target = targets.unsafe_offset(Int(shape[unsafe_offset=0]))
    var n = Int(shape[unsafe_offset=1]) if active else 0
    var query = queries.unsafe_offset(Int(shape[unsafe_offset=2]))
    var m = Int(shape[unsafe_offset=3]) if active else 0
    var w = Int(shape[unsafe_offset=4]) if active else 1
    var code = codes.unsafe_offset(Int(code_offsets[unsafe_offset=pair]))
    # The warp steps through its longest pair's anti-diagonals, every lane in step for the shuffles.
    var steps = Int(warp.max(Int32(n + m)))
    var stride = code_stride(w, K)
    var neg = Int32(NEG)
    var qe = q + e
    var qe2 = q2 + e2
    # Each diagonal's latest cell: its boundary cell first, in row 0 or column 0.
    var h = V(fill=neg)
    var e1 = V(fill=neg)
    var e2v = V(fill=neg)
    var f1 = V(fill=neg)
    var f2 = V(fill=neg)
    var base = lane * K
    # This lane's codes of the current pair of anti-diagonals (`2 R`, `2 R + 1`: one parity of diagonals each),
    # written as one `K`-byte store once both are done.
    var pending = Array[UInt8, K](fill=0)
    comptime for s in range(K):
        var t = base + s
        if t < 2 * w:
            var d = t - w
            var gap = Int32(abs(d))
            h[s] = Int32(0) if d == 0 else -min(q + gap * e, q2 + gap * e2)
    for r in range(2, steps + 1):
        # The edge diagonals of the lanes either side, from the previous anti-diagonal.
        var one: UInt32 = 1
        var next_h = warp.shuffle_down(h[0], one)
        var next_e1 = warp.shuffle_down(e1[0], one)
        var next_e2 = warp.shuffle_down(e2v[0], one)
        var prev_h = warp.shuffle_up(h[K - 1], one)
        var prev_f1 = warp.shuffle_up(f1[K - 1], one)
        var prev_f2 = warp.shuffle_up(f2[K - 1], one)
        if lane == PAIR_LANES - 1:
            next_h = neg
            next_e1 = neg
            next_e2 = neg
        if lane == 0:
            prev_h = neg
            prev_f1 = neg
            prev_f2 = neg
        var parity = (r + w) & 1
        comptime for s in range(K):
            if (s & 1) == parity:
                var t = base + s
                var d = t - w
                var i = (r - d) >> 1
                var j = r - i
                if t < 2 * w and i >= 1 and i <= n and j >= 1 and j <= m:
                    var up_h: Int32
                    var up_e1: Int32
                    var up_e2: Int32
                    comptime if s + 1 < K:
                        up_h = h[s + 1]
                        up_e1 = e1[s + 1]
                        up_e2 = e2v[s + 1]
                    else:
                        up_h = next_h
                        up_e1 = next_e1
                        up_e2 = next_e2
                    var left_h: Int32
                    var left_f1: Int32
                    var left_f2: Int32
                    comptime if s > 0:
                        left_h = h[s - 1]
                        left_f1 = f1[s - 1]
                        left_f2 = f2[s - 1]
                    else:
                        left_h = prev_h
                        left_f1 = prev_f1
                        left_f2 = prev_f2
                    var tc = target[unsafe_offset=i - 1]
                    var qc = query[unsafe_offset=j - 1]
                    var sc = a if tc == qc else -b
                    if tc == 4 or qc == 4:
                        sc = -ambi
                    var z = h[s] + sc
                    var de1 = max(max(up_h - qe, up_e1 - e), neg)
                    var de2 = max(max(up_h - qe2, up_e2 - e2), neg)
                    var df1 = max(max(left_h - qe, left_f1 - e), neg)
                    var df2 = max(max(left_h - qe2, left_f2 - e2), neg)
                    var c = UInt8(0)
                    if de1 > z:
                        c = 1
                    z = max(z, de1)
                    if df1 > z:
                        c = 2
                    z = max(z, df1)
                    if de2 > z:
                        c = 3
                    z = max(z, de2)
                    if df2 > z:
                        c = 4
                    z = max(z, df2)
                    # Whether the next cell's gap of each layer continues this one.
                    if de1 > z - q:
                        c |= 0x08
                    if df1 > z - q:
                        c |= 0x10
                    if de2 > z - q2:
                        c |= 0x20
                    if df2 > z - q2:
                        c |= 0x40
                    pending[s] = c
                    h[s] = z
                    e1[s] = de1
                    e2v[s] = de2
                    f1[s] = df1
                    f2[s] = df2
        if (r & 1) == 1 and r <= n + m + 1 and base < 2 * w:
            _store_codes[K](code.unsafe_offset((r >> 1) * stride + base), pending)
    # A pair whose last anti-diagonal is even and the warp's last: its last row is not stored yet.
    if active and ((n + m) & 1) == 0 and n + m == steps and base < 2 * w:
        _store_codes[K](code.unsafe_offset(((n + m) >> 1) * stride + base), pending)
    syncwarp()
    if lane != 0 or not active:
        return
    # The walk back, `ksw_backtrack` by rows as `ksw2_lanes` walks it, the runs merged as they come.
    var out = scratch.unsafe_offset(Int(shape[unsafe_offset=5]))
    var count = 0

    @always_inline
    def push(op: Int, length: Int) {mut count, imm out}:
        if count > 0 and Int(out[unsafe_offset=count - 1] & 0xF) == op:
            out[unsafe_offset=count - 1] += UInt32(length << 4)
        else:
            out[unsafe_offset=count] = UInt32(length << 4 | op)
            count += 1

    var i = n
    var j = m
    var state = 0
    while i > 0 and j > 0:
        var force_state = -1
        if j < max(1, i - w):
            force_state = 2
        if j > i + w - 1:
            force_state = 1
        var tmp = 0
        if force_state < 0:
            tmp = Int(code[unsafe_offset=((i + j) >> 1) * stride + (j - i + w)])
        if state == 0:
            state = tmp & 7
        elif ((tmp >> (state + 2)) & 1) == 0:
            state = 0
        if state == 0:
            state = tmp & 7
        if force_state >= 0:
            state = force_state
        if state == 0:
            push(0, 1)
            i -= 1
            j -= 1
        elif state == 1 or state == 3:
            push(2, 1)
            i -= 1
        else:
            push(1, 1)
            j -= 1
    if i > 0:
        push(2, i)
    if j > 0:
        push(1, j)
    # Reversed into the next free place of the compact CIGARs.
    var place = Int(Atomic.fetch_add(used, Int32(count)))
    for k in range(count):
        cigars[unsafe_offset=place + k] = out[unsafe_offset=count - 1 - k]
    cigar_places[unsafe_offset=2 * pair] = Int32(place)
    cigar_places[unsafe_offset=2 * pair + 1] = Int32(count)


@always_inline
def code_stride(w: Int, slots: Int) -> Int:
    """Bytes of codes a pair of anti-diagonals takes: the band's `2 w` diagonals in whole blocks of `slots`, so each
    lane's block is one aligned store."""
    return (2 * w + slots - 1) // slots * slots


@always_inline
def _slots_for(w: Int) -> Int:
    """Diagonals a lane holds for a band `w`: the fewest even number that spans it over `PAIR_LANES` lanes."""
    var needed = (2 * w + PAIR_LANES - 1) // PAIR_LANES
    return max(2, needed + (needed & 1))


def gpu_fits(target_length: Int, query_length: Int, band: Int) -> Bool:
    """Whether `ksw_lanes_gpu` takes a pair: both sides within `MAX_LANE_LENGTH` and its band's `2 w` diagonals within
    `GPU_MAX_BAND`."""
    return (
        target_length <= MAX_LANE_LENGTH and query_length <= MAX_LANE_LENGTH and 1 <= band and 2 * band <= GPU_MAX_BAND
    )


struct LanesGpu(Movable):
    """A GPU for `ksw_lanes_gpu`, its buffers kept and grown from call to call."""

    var context: DeviceContext
    var targets: DeviceBuffer[DType.uint8]
    var queries: DeviceBuffer[DType.uint8]
    var shapes: DeviceBuffer[DType.int32]
    var code_offsets: DeviceBuffer[DType.int64]
    var codes: DeviceBuffer[DType.uint8]
    var scratch: DeviceBuffer[DType.uint32]
    var cigars: DeviceBuffer[DType.uint32]
    var cigar_places: DeviceBuffer[DType.int32]
    var pair_ids: DeviceBuffer[DType.int32]
    var used: DeviceBuffer[DType.int32]

    def __init__(out self, gpu: Int = 0) raises:
        self.context = DeviceContext(device_id=gpu)
        self.targets = self.context.enqueue_create_buffer[DType.uint8](1)
        self.queries = self.context.enqueue_create_buffer[DType.uint8](1)
        self.shapes = self.context.enqueue_create_buffer[DType.int32](1)
        self.code_offsets = self.context.enqueue_create_buffer[DType.int64](1)
        self.codes = self.context.enqueue_create_buffer[DType.uint8](1)
        self.scratch = self.context.enqueue_create_buffer[DType.uint32](1)
        self.cigars = self.context.enqueue_create_buffer[DType.uint32](1)
        self.cigar_places = self.context.enqueue_create_buffer[DType.int32](2)
        self.pair_ids = self.context.enqueue_create_buffer[DType.int32](1)
        self.used = self.context.enqueue_create_buffer[DType.int32](1)


def ksw_lanes_gpu(
    mut gpu: LanesGpu,
    targets: ImmSpan[ImmPointer[UInt8, ImmUntrackedOrigin], _],
    target_lengths: ImmSpan[Int, _],
    queries: ImmSpan[ImmPointer[UInt8, ImmUntrackedOrigin], _],
    query_lengths: ImmSpan[Int, _],
    bands: ImmSpan[Int, _],
    a: Int,
    b: Int,
    ambi: Int,
    q_in: Int,
    e_in: Int,
    q2_in: Int,
    e2_in: Int,
    mut cigars: List[List[UInt32]],
) raises:
    """`ksw_lanes` for any number of pairs, each in its own band (`bands`, each pair `gpu_fits`), on the GPU: pair
    `l`'s CIGAR into `cigars[l]` (`length << 4 | op`, `M` `I` `D`), the one `ksw_lanes` gives it alone in its band."""
    var count = len(targets)
    cigars.resize(count, List[UInt32]())
    if count == 0:
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
    # The letters end to end, each pair's numbers, and where its codes and CIGAR go.
    var t_tape = List[UInt8]()
    var q_tape = List[UInt8]()
    var shapes = List[Int32](capacity=count * _SHAPE)
    var code_offsets = List[Int64](capacity=count)
    var code_total = 0
    var cigar_total = 0
    for l in range(count):
        var n = target_lengths[l]
        var m = query_lengths[l]
        if not gpu_fits(n, m, bands[l]):
            raise Error(t"ksw_lanes_gpu: pair {l} ({n} x {m}, band {bands[l]}) does not fit")
        shapes.append(Int32(len(t_tape)))
        shapes.append(Int32(n))
        shapes.append(Int32(len(q_tape)))
        shapes.append(Int32(m))
        shapes.append(Int32(bands[l]))
        shapes.append(Int32(cigar_total))
        t_tape.extend(Span(unsafe_ptr=targets[l], length=n))
        q_tape.extend(Span(unsafe_ptr=queries[l], length=m))
        # Each pair's codes 16-byte aligned, for its lanes' stores of up to 16.
        code_total = (code_total + 15) // 16 * 16
        code_offsets.append(Int64(code_total))
        code_total += ((n + m) // 2 + 2) * code_stride(bands[l], _slots_for(bands[l]))
        cigar_total += n + m + 1
    if len(gpu.targets) < len(t_tape):
        gpu.targets = gpu.context.enqueue_create_buffer[DType.uint8](len(t_tape))
    if len(gpu.queries) < len(q_tape):
        gpu.queries = gpu.context.enqueue_create_buffer[DType.uint8](len(q_tape))
    if len(gpu.shapes) < len(shapes):
        gpu.shapes = gpu.context.enqueue_create_buffer[DType.int32](len(shapes))
    if len(gpu.code_offsets) < count:
        gpu.code_offsets = gpu.context.enqueue_create_buffer[DType.int64](count)
        gpu.cigar_places = gpu.context.enqueue_create_buffer[DType.int32](2 * count)
        gpu.pair_ids = gpu.context.enqueue_create_buffer[DType.int32](count)
    if len(gpu.codes) < code_total:
        gpu.codes = gpu.context.enqueue_create_buffer[DType.uint8](code_total)
    if len(gpu.scratch) < cigar_total:
        gpu.scratch = gpu.context.enqueue_create_buffer[DType.uint32](cigar_total)
        gpu.cigars = gpu.context.enqueue_create_buffer[DType.uint32](cigar_total)
    # The pairs by how many diagonals a lane holds: the fewest that span its band.
    var by_slots = List[Int32](capacity=count)
    var group_ends = List[Int]()
    comptime for g in range(_SLOT_CHOICES):
        for l in range(count):
            if _slots_for(bands[l]) == 2 * (g + 1):
                by_slots.append(Int32(l))
        group_ends.append(len(by_slots))
    ref ctx = gpu.context
    ctx.enqueue_copy(gpu.targets.create_sub_buffer[DType.uint8](0, max(len(t_tape), 1)), t_tape.unsafe_ptr())
    ctx.enqueue_copy(gpu.queries.create_sub_buffer[DType.uint8](0, max(len(q_tape), 1)), q_tape.unsafe_ptr())
    ctx.enqueue_copy(gpu.shapes.create_sub_buffer[DType.int32](0, len(shapes)), shapes.unsafe_ptr())
    ctx.enqueue_copy(gpu.code_offsets.create_sub_buffer[DType.int64](0, count), code_offsets.unsafe_ptr())
    ctx.enqueue_copy(gpu.pair_ids.create_sub_buffer[DType.int32](0, count), by_slots.unsafe_ptr())
    ctx.enqueue_memset(gpu.used, Int32(0))
    var start = 0
    comptime for g in range(_SLOT_CHOICES):
        comptime kernel = _gpu_kernel[2 * (g + 1)]
        var group = group_ends[g] - start
        if group > 0:
            ctx.enqueue_function[kernel](
                gpu.pair_ids.unsafe_ptr().unsafe_offset(start).unsafe_origin_cast[MutAnyOrigin](),
                Int32(group),
                gpu.targets.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                gpu.queries.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                gpu.shapes.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                gpu.code_offsets.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                gpu.codes.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                gpu.scratch.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                gpu.cigars.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                gpu.cigar_places.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                gpu.used.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                Int32(a),
                Int32(b),
                Int32(ambi),
                Int32(q),
                Int32(e),
                Int32(q2),
                Int32(e2),
                grid_dim=ceildiv(group, _WARPS * WARP_SIZE // PAIR_LANES),
                block_dim=_WARPS * WARP_SIZE,
            )
        start = group_ends[g]
    var total = List[Int32](length=1, fill=0)
    var places = List[Int32](length=2 * count, fill=0)
    ctx.enqueue_copy(total.unsafe_ptr(), gpu.used.create_sub_buffer[DType.int32](0, 1))
    ctx.enqueue_copy(places.unsafe_ptr(), gpu.cigar_places.create_sub_buffer[DType.int32](0, 2 * count))
    ctx.synchronize()
    var ops = List[UInt32](length=max(Int(total[0]), 1), fill=0)
    ctx.enqueue_copy(ops.unsafe_ptr(), gpu.cigars.create_sub_buffer[DType.uint32](0, max(Int(total[0]), 1)))
    ctx.synchronize()
    for l in range(count):
        ref cigar = cigars[l]
        cigar.clear()
        var at = Int(places[2 * l])
        cigar.extend(Span(ops)[at : at + Int(places[2 * l + 1])])
