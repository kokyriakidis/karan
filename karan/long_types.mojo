# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the
# MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
The public face of `long_align`: the seeds a caller hands in, minimap2's presets, and the alignments and chains that
come back, with the letter codes and query-name hash minimap2 uses, and the checks `LongAligner` makes of them.
"""

from std.sys import is_defined, stderr

from .errors import AlignmentError

comptime LONG_TRACE = is_defined["KARAN_LONG_TRACE"]()
"""Whether this build traces `LongAligner`, built with `-D KARAN_LONG_TRACE` (`pixi run build-long-trace`): a line to
standard error for each stage of each query, tab-separated, its first field the minimap2 function the stage ports
(`mm_gen_regs`, `mm_align1:fill`, `mm_filter_regs`, ...), so two runs' traces diff to the first stage that parts. A
normal build folds every trace away."""


def trace(line: String):
    """One trace line to standard error; call it only under `comptime if LONG_TRACE`."""
    print(line, file=stderr)


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

    def check(self) raises AlignmentError:
        """Raises unless minimap2 would take these options (`mm_check_opt`), and unless the scores fit the byte table
        KSW2 reads: positive gap costs, two pieces with `gap_extend > long_gap_extend` and `gap_open + gap_extend <
        long_gap_open + long_gap_extend` (or equal pieces), both pieces' open and extend together at most 127,
        `ambiguous` in `[0, mismatch)`, `zdrop >= zdrop_inv`, `bw <= bw_long`, `k` in 1 to 255, `best_n >= 0`,
        `pri_ratio` in `[0, 1]` and `match_score` in 1 to 127."""
        var q = self.gap_open
        var e = self.gap_extend
        var q2 = self.long_gap_open
        var e2 = self.long_gap_extend
        if q <= 0 or e <= 0 or q2 <= 0 or e2 <= 0:
            raise AlignmentError(
                .INVALID_SCORING, "gap_open, gap_extend, long_gap_open and long_gap_extend must be positive"
            )
        if (q != q2 or e != e2) and not (e > e2 and q + e < q2 + e2):
            raise AlignmentError(
                .INVALID_SCORING,
                String(
                    t"two gap pieces need gap_extend > long_gap_extend and gap_open + gap_extend < long_gap_open +"
                    t" long_gap_extend: {q},{e} and {q2},{e2}"
                ),
            )
        if (q + e) + (q2 + e2) > 127:
            raise AlignmentError(
                .INVALID_SCORING,
                String(t"(gap_open + gap_extend) + (long_gap_open + long_gap_extend) is {q + e + q2 + e2}, past 127"),
            )
        if self.match_score < 1 or self.match_score > 127:
            raise AlignmentError(.INVALID_SCORING, String(t"match_score {self.match_score} is outside 1 to 127"))
        if self.mismatch > 127:
            raise AlignmentError(.INVALID_SCORING, String(t"mismatch {self.mismatch} is past 127"))
        if self.ambiguous < 0 or self.ambiguous >= self.mismatch:
            raise AlignmentError(
                .INVALID_SCORING, String(t"ambiguous {self.ambiguous} is outside [0, mismatch {self.mismatch})")
            )
        if self.zdrop < self.zdrop_inv:
            raise AlignmentError(.INVALID_ARGUMENT, String(t"zdrop {self.zdrop} is below zdrop_inv {self.zdrop_inv}"))
        if self.bw > self.bw_long:
            raise AlignmentError(.INVALID_ARGUMENT, String(t"bw {self.bw} is above bw_long {self.bw_long}"))
        if self.k < 1 or self.k > 255:
            raise AlignmentError(.INVALID_ARGUMENT, String(t"k {self.k} is outside 1 to 255"))
        if self.best_n < 0:
            raise AlignmentError(.INVALID_ARGUMENT, String(t"best_n {self.best_n} is negative"))
        if self.pri_ratio < 0.0 or self.pri_ratio > 1.0:
            raise AlignmentError(.INVALID_ARGUMENT, String(t"pri_ratio {self.pri_ratio} is outside [0, 1]"))


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


comptime MAX_LONG_LENGTH = Int(Int32.MAX)
"""The longest reference or query `LongAligner` takes: minimap2 packs positions in 32 bits."""


def check_inputs(reference: Span[UInt8, _], query: Span[UInt8, _], seeds: Span[Seed, _]) raises AlignmentError:
    """Raises unless both sequences are minimap2's codes (0 to 4, `encode_nt4`) no longer than `MAX_LONG_LENGTH`, and
    unless every seed lies inside both (see `check_seeds`)."""
    check_seeds(len(reference), len(query), seeds)
    check_codes("reference", reference)
    check_codes("query", query)


def check_codes(name: StaticString, sequence: Span[UInt8, _]) raises AlignmentError:
    """Raises unless every byte of `sequence` is one of minimap2's codes, 0 to 4."""
    var worst = UInt8(0)
    for code in sequence:
        worst = max(worst, code)
    if worst <= 4:
        return
    for index in range(len(sequence)):
        if sequence[index] > 4:
            raise AlignmentError(
                .UNKNOWN_SYMBOL, String(t"{name} holds byte {sequence[index]} at {index}; encode it with encode_nt4")
            )


def check_seeds(reference_length: Int, query_length: Int, seeds: Span[Seed, _]) raises AlignmentError:
    """Raises unless both lengths are at most `MAX_LONG_LENGTH` and every seed spans 1 to 255 letters inside both: its
    query letters `[query, query + span)` on the forward strand and its reference letters `[reference, reference +
    span)`."""
    if reference_length > MAX_LONG_LENGTH or query_length > MAX_LONG_LENGTH:
        raise AlignmentError(
            .SEQUENCE_TOO_LONG,
            String(t"reference {reference_length} or query {query_length} letters, past {MAX_LONG_LENGTH}"),
        )
    for index in range(len(seeds)):
        var s = seeds[index]
        if (
            s.span < 1
            or s.span > 255
            or s.reference < 0
            or s.query < 0
            or s.reference + s.span > reference_length
            or s.query + s.span > query_length
        ):
            raise AlignmentError(
                .INVALID_ARGUMENT,
                String(
                    t"seed {index} (reference {s.reference}, query {s.query}, span {s.span}) is not 1 to 255 letters"
                    t" inside the reference's {reference_length} and the query's {query_length}"
                ),
            )


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
