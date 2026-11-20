# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the
# MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
The public face of `long_align`: the seeds a caller hands in, minimap2's presets, and the alignments and chains that
come back, with the letter codes and query-name hash minimap2 uses.
"""

comptime CIGAR_MATCH = 0
"""A packed CIGAR run's op for aligned letters, match or mismatch (`M`)."""
comptime CIGAR_INSERTION = 1
"""A packed CIGAR run's op for query letters the reference lacks (`I`)."""
comptime CIGAR_DELETION = 2
"""A packed CIGAR run's op for reference letters the query lacks (`D`)."""


@fieldwise_init
struct Seed(Copyable, ImplicitlyCopyable, Movable, TrivialRegisterPassable, Writable):
    """An exact match the caller found: `span` letters (1 to 255) of the query from `query` on its forward strand,
    equal to the reference's from `reference` on, or, when `reverse`, to their reverse complement. `tandem` marks a
    seed whose query minimizer repeats next to itself, as minimap2 marks one (`is_tandem`); the gap filling never
    ends a piece on it."""

    var reference: Int
    var query: Int
    var span: Int
    var reverse: Bool
    var tandem: Bool


@fieldwise_init
struct LongPreset(Copyable, ImplicitlyCopyable, Movable, Writable):
    """minimap2's mapping options for a long query, by its own names where a field has one. The named constructors
    are minimap2's presets; a field may be set after, as minimap2's flags set them (`-r` sets `bw` and `bw_long`,
    `-z` `zdrop` and `zdrop_inv`)."""

    var k: Int
    """The seeds' k-mer length: the chaining penalty scales with it and a secondary within `2 k` of its primary's
    chain score is kept (`mi->k`)."""
    var bw: Int
    """Chaining and alignment band (`-r`): no chain has a gap difference above it; a gap is filled in a band of
    `1.5 bw_long + 1`, an end extended in one of `1.5 bw + 1`."""
    var bw_long: Int
    """The band chains are joined across (`-r`'s second value)."""
    var max_gap: Int
    """The farthest two chained anchors lie apart (`-g`)."""
    var max_gap_ref: Int
    """The same on the reference, when positive (`-G`)."""
    var max_chain_skip: Int
    var max_chain_iter: Int
    var rmq: Bool
    """Chain by range-maximum queries (`--rmq`), as the assembly presets do."""
    var rmq_inner_dist: Int
    var rmq_size_cap: Int
    var rmq_rescue_size: Int
    var rmq_rescue_ratio: Float32
    var chain_gap_scale: Float32
    var chain_skip_scale: Float32
    var min_cnt: Int
    """A chain's least anchors (`-n`)."""
    var min_chain_score: Int
    """A chain's least score (`-m`)."""
    var mask_level: Float32
    var mask_len: Int
    var pri_ratio: Float32
    """A secondary's least score against its primary's (`-p`)."""
    var best_n: Int
    """Secondaries kept at most (`-N`)."""
    var match_score: Int
    """`-A`."""
    var mismatch: Int
    """`-B`."""
    var gap_open: Int
    """`-O`: a gap of `k` letters costs the less of `gap_open + k gap_extend` and `long_gap_open + k
    long_gap_extend`."""
    var gap_extend: Int
    """`-E`."""
    var long_gap_open: Int
    var long_gap_extend: Int
    var ambiguous: Int
    """A letter other than A, C, G, T against anything (`--score-N`)."""
    var zdrop: Int
    """`-z`."""
    var zdrop_inv: Int
    """`-z`'s second value: the Z-drop that splits at an inversion."""
    var end_bonus: Int
    var min_dp_max: Int
    """An alignment's least score (`-s`)."""
    var min_ksw_len: Int
    var max_clip_ratio: Float32
    var max_sw_mat: Int
    """A fill of more cells than this is not done: it Z-drops."""
    var rank_min_len: Int
    var rank_frac: Float32
    var end_filter: Bool
    """Leave bad chain ends out (`mm_fix_bad_ends`; minimap2's `--no-end-flt` turns it off)."""
    var long_join: Bool
    """Re-chain through gaps up to `bw_long` (minimap2's `--no-long-join` turns it off)."""
    var inversions: Bool
    """Align an inversion between the pieces of a split chain (minimap2's `--no-inv` turns it off)."""

    @staticmethod
    def map_ont() -> Self:
        """minimap2's defaults (`-x map-ont`, `-x lr`): k 15, gaps 5 kb, bands 500 and 20 kb."""
        return Self(
            k=15,
            bw=500,
            bw_long=20000,
            max_gap=5000,
            max_gap_ref=-1,
            max_chain_skip=25,
            max_chain_iter=5000,
            rmq=False,
            rmq_inner_dist=1000,
            rmq_size_cap=100000,
            rmq_rescue_size=1000,
            rmq_rescue_ratio=0.1,
            chain_gap_scale=0.8,
            chain_skip_scale=0.0,
            min_cnt=3,
            min_chain_score=40,
            mask_level=0.5,
            mask_len=Int(Int32.MAX),
            pri_ratio=0.8,
            best_n=5,
            match_score=2,
            mismatch=4,
            gap_open=4,
            gap_extend=2,
            long_gap_open=24,
            long_gap_extend=1,
            ambiguous=1,
            zdrop=400,
            zdrop_inv=200,
            end_bonus=-1,
            min_dp_max=80,
            min_ksw_len=200,
            max_clip_ratio=1.0,
            max_sw_mat=100000000,
            rank_min_len=500,
            rank_frac=0.9,
            end_filter=True,
            long_join=True,
            inversions=True,
        )

    @staticmethod
    def map_hifi() -> Self:
        """`-x map-hifi`: k 19, gaps 10 kb, `-A1 -B4 -O6,26 -E2,1 -s200`."""
        var preset = Self.map_ont()
        preset.k = 19
        preset.max_gap = 10000
        preset.match_score = 1
        preset.mismatch = 4
        preset.gap_open = 6
        preset.long_gap_open = 26
        preset.gap_extend = 2
        preset.long_gap_extend = 1
        preset.min_dp_max = 200
        return preset

    @staticmethod
    def _asm() -> Self:
        var preset = Self.map_ont()
        preset.k = 19
        preset.bw = 1000
        preset.bw_long = 100000
        preset.max_gap = 10000
        preset.rmq = True
        preset.min_dp_max = 200
        preset.best_n = 50
        preset.match_score = 1
        preset.zdrop = 200
        preset.zdrop_inv = 200
        return preset

    @staticmethod
    def asm5() -> Self:
        """`-x asm5`: k 19, range-maximum chaining, bands 1 kb and 100 kb, `-A1 -B19 -O39,81 -E3,1 -z200`."""
        var preset = Self._asm()
        preset.mismatch = 19
        preset.gap_open = 39
        preset.long_gap_open = 81
        preset.gap_extend = 3
        preset.long_gap_extend = 1
        return preset

    @staticmethod
    def asm10() -> Self:
        """`-x asm10`: `asm5`'s chaining, `-A1 -B9 -O16,41 -E2,1 -z200`."""
        var preset = Self._asm()
        preset.mismatch = 9
        preset.gap_open = 16
        preset.long_gap_open = 41
        preset.gap_extend = 2
        preset.long_gap_extend = 1
        return preset

    @staticmethod
    def asm20() -> Self:
        """`-x asm20`: `asm5`'s chaining, `-A1 -B4 -O6,26 -E2,1 -z200` (its window of 10 is the seeds')."""
        var preset = Self._asm()
        preset.mismatch = 4
        preset.gap_open = 6
        preset.long_gap_open = 26
        preset.gap_extend = 2
        preset.long_gap_extend = 1
        return preset

    @staticmethod
    def dipcall() -> Self:
        """dipcall's contig alignment, `-x asm5 -z200000,10000 -r2k`: Z-drop only at an inversion, a 2 kb band."""
        var preset = Self.asm5()
        preset.zdrop = 200000
        preset.zdrop_inv = 10000
        preset.bw = 2000
        return preset


@fieldwise_init
struct LongAlignment(Copyable, Movable):
    """One alignment of the query, as a line of minimap2's PAF holds it: the query's interval on its forward strand
    and the reference's, the strand, the CIGAR as KSW2 packs it (`length << 4 | op`, `CIGAR_MATCH`,
    `CIGAR_INSERTION`, `CIGAR_DELETION`, reference forward, the query reverse-complemented when `reverse`), the
    chain's score (`s1`) and, for a primary, its best secondary's (`s2`), the alignment's score (`AS`) and its best
    local score (`ms`), the mapping quality, letters
    aligned equal (`mlen`) and in all (`blen`). `primary` is minimap2's `tp:A:P`, the best of its query stretch;
    `supplementary` a primary other than the first (SAM's flag 2048); `inversion` an inversion between two pieces of
    a split chain (`tp:A:I`)."""

    var query_start: Int
    var query_end: Int
    var reference_start: Int
    var reference_end: Int
    var reverse: Bool
    var cigar: List[UInt32]
    var chain_score: Int
    var secondary_score: Int
    var score: Int
    var dp_max: Int
    var mapq: Int
    var matches: Int
    var block_length: Int
    var primary: Bool
    var supplementary: Bool
    var inversion: Bool

    def cigar_string(self) -> String:
        """The CIGAR as text, minimap2's `M`, `I`, `D`."""
        var out = String()
        for run in self.cigar:
            var op = Int(run & 0xF)
            out.write(Int(run >> 4), "M" if op == CIGAR_MATCH else ("I" if op == CIGAR_INSERTION else "D"))
        return out^


@fieldwise_init
struct LongChain(Copyable, Movable, Writable):
    """A chain as minimap2 keeps it before aligning: its anchors' extent on the query (forward strand) and the
    reference, the strand, its score and anchors, and whether it is a primary."""

    var query_start: Int
    var query_end: Int
    var reference_start: Int
    var reference_end: Int
    var reverse: Bool
    var score: Int
    var anchors: Int
    var primary: Bool


def name_hash(name: StringSlice) -> UInt32:
    """minimap2's hash of a query's name (khash's `__ac_X31_hash_string`); passed to `LongAligner.align`, it breaks
    ties among equally scored chains as minimap2 breaks them for a query of that name."""
    var bytes = name.as_bytes()
    if len(bytes) == 0:
        return 0
    var h = UInt32(bytes[0])
    for index in range(1, len(bytes)):
        h = (h << 5) - h + UInt32(bytes[index])
    return h


def encode_nt4(text: Span[UInt8, _]) -> List[UInt8]:
    """Letters as minimap2 codes them (`seq_nt4_table`): A 0, C 1, G 2, T 3 in either case, anything else 4."""
    var out = List[UInt8](capacity=len(text))
    for byte in text:
        var c = byte | 0x20
        var code = UInt8(4)
        if c == UInt8(ord("a")):
            code = 0
        elif c == UInt8(ord("c")):
            code = 1
        elif c == UInt8(ord("g")):
            code = 2
        elif c == UInt8(ord("t")) or c == UInt8(ord("u")):
            code = 3
        out.append(code)
    return out^
