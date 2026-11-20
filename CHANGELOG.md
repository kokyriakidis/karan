# Changelog

Every release's changes, newest first. A tag `vX.Y.Z` publishes the section of that version as the
release's notes (see `.github/workflows/release.yml`).

## [Unreleased]

The first release: exact pairwise alignment in every mode, from Mojo, C, C++, Python and the command
line.

Licensed under the Mozilla Public License 2.0 (see LICENSE and NOTICE).

Formerly dinara-align: the project, the Mojo and Python packages (`karan`), the command line (`karan`)
and the C library (`libkaran`, `karan.h`, `karan_*`, `KARAN_*`, `karan::`) all take the one name.

### Alignment

- `karan.ksw2`: minimap2's `ksw_extd2_sse` (2.31) ported line for line, so the same CIGARs as minimap2: two-piece
  gap costs, KSW2's choices among equal alignments (gaps as early as the score allows, or as late with `RIGHT`), its
  band, Z-drop, end bonus, exact or approximate maximum, extension-only and reversed-CIGAR modes. Byte for byte with
  the C on 23,000 random cases (600 kept in `tests/data`, `pixi run test-ksw2`); with sixty-four-lane vectors it fills
  1 kbp gaps faster than the C's SSE (0.65 s against 1.34 s for 2,000).
  `ksw_extd2[L, W]` computes `W` cells an instruction while following the C's `L`-wide vectors (the cells computed past
  a band's edge, which change results there): `ksw_extd2[16, 64]` is the C byte for byte at AVX-512's width (64-wide
  vectors alone differed from it on 19 of 9,812 of herro2's calls). On those calls it takes 0.75 s against the C's
  1.45 s: the exact maximum found 16 at a time (ties as the C's four-lane reduction breaks them), the arrays set up by
  `memset`, and each anti-diagonal's pieces run last to first so the shifted values come from one load.
- `karan.ksw2_lanes`: the same global gap filling for a batch of pairs at once, a pair a lane: the cells, codes and
  walk back of `ksw2`, so the same CIGARs wherever the band holds the alignment; twice as fast as `ksw2` on batches
  of like pieces. Each pair may have its own band (`bands`): its cells outside it held at minus infinity, so its
  CIGAR is the one its band gives alone, whatever pairs share its batch. A cell's code is kept as the eight
  comparisons that make it, each a mask of the batch's lanes, and the walk back decodes only what each step needs; a
  batch without a wildcard skips its check: 33 cycles a 32-lane step on Skylake-X, from 40.
- `karan.ksw2_gpu`: `ksw2_lanes` on a GPU (`ksw_lanes_gpu`): a warp a pair along its anti-diagonals, each lane a
  block of the pair's own band in registers, the codes stored a pair of anti-diagonals at a time and walked back on
  the GPU, so only CIGARs come home; every pair's CIGAR is the one `ksw_lanes` gives it alone in its band. About
  18 G cells/s on an RTX 2070, against 10 G for `ksw_lanes` on 10 Skylake-X cores.
- `karan.ksw2_gpu` for pipelines: the caller's threads pack pairs into `GpuSegment`s, `submit` starts them and
  `collect` gives their CIGARs, so the GPU aligns while the caller works; a `LanesGpu` runs its pairs in rounds within
  its `code_budget`, so two (each its own stream) can share a GPU; bands up to 384 diagonals. An fp16 kernel (two cells
  an instruction, `LanesGpu.half`) gives the same CIGARs but is slower on Turing and Apple GPUs, so it is off by
  default.

- `LongAligner`: a long query, a contig or a long read, aligned to a reference from the caller's seeds as minimap2
  2.28 aligns it (`karan.chaining`, `karan.long_align`): `mg_lchain_dp` and `mg_lchain_rmq` chaining and the long join
  across `bw_long`, chains ranked with minimap2's hash for ties, primaries and secondaries chosen (`mm_set_parent`,
  `mm_select_sub`, `mm_est_err`), each chain aligned as `mm_align1` aligns it with `karan.ksw2` (bad ends and seeds
  left out, approximate then exact Z-drop, a Z-drop splitting the chain, an inversion between the pieces), CIGARs
  fixed (`mm_fix_cigar`), alignments re-ranked and given mapping qualities. Presets `asm5`, `asm10`, `asm20`,
  `map_hifi`, `map_ont` and `dipcall`. From minimap2's own seeds it gives minimap2's lines byte for byte: on HG002
  contigs of chr19 tandem arrays, kb-scale copy-number gaps included, the same CIGARs and all 3,591 differences
  paftools calls; on 2,657 HiFi reads all 4,812 records the same, mapping qualities too (`pixi run test-long`).
  `mg_lchain_rmq`'s AVL tree is a segment tree and a 64-ary bit tree here, which take the leftmost of two anchors of
  equal priority; minimap2's `ksw_extz2_sse` (equal gap pieces) is not ported, so such costs run `ksw_extd2`.
  Laid out as minimap2's files are (`long_types`, `chaining`, `long_regions` for `hit.c` and `esterr.c`,
  `long_fill` for `align.c`, `long_align` for `map.c`), each function named after the one it ports. `align` and
  `chains` raise an `AlignmentError` on options minimap2's `mm_check_opt` refuses (`LongPreset.check`), on seeds
  outside either sequence, and on bytes other than `encode_nt4` codes in the query or in the reference an alignment
  reads. Seeds of unequal spans are safe: an anchor whose middle does not advance is stepped over, and an inversion
  whose local end falls in `ksw_ll_i16`'s padding starts in the gap's buffer as the C's does; `pixi run fuzz-long`
  aligns random seed sets with every read bounds-checked. A build with `-D KARAN_LONG_TRACE` (`pixi run
  build-long-trace`) traces each stage to standard error. `CIGAR_MATCH`,
  `CIGAR_INSERTION` and `CIGAR_DELETION`, which read `LongAlignment.cigar`, are exported from `karan`.
- One API for every cost model and mode: `distance`, `align` and `score` take `Costs` (unit, linear,
  gap-affine, two-piece gap-affine, deletions priced apart from insertions) and a `Mode` of three
  kinds, as WFA2-lib's ends-free and extension and abPOA's local: any free ends, with the presets
  global, infix, prefix, suffix and overlap, and a reward for every match when asked; an extension
  from either end with an optional Z-drop and end bonus; and local. The Z-drop weighs every alignment
  still growing, those inside a gap too, over the last move's worth of costs, as KSW2 does, rather than one
  cost's aligned front as WFA2-lib does, which gave up far earlier than minimap2 on noisy reads (a HiFi tail
  matching but for one deleted letter: 1,473 of 10,310 letters at Z-drop 400). An end bonus takes a band: the best
  stop and the best alignment through the query are both weighed inside it, as KSW2 weighs them in its band.
- Unit costs by A\*PA2's bit-parallel band doubling with its seed heuristic, near-identical pairs by
  diagonal transition first; every other cost model by a gap-affine wavefront, one search alone while it
  steps fewer diagonals than two would, as it measures them, then from both ends after BiWFA, the
  alignment the same either way; local alignment, overlaps and rewarded free
  ends by an anti-diagonal sweep in 16-bit lanes while the scores fit, one sweep for every score, a
  local alignment's start and a linear-space traceback's halves; all exact, every alignment's memory
  bounded (`max_memory`): costs so dear that a search's fronts could pass it are refused, and under a
  `Scoring` swept instead.
- A `Scoring`, any alphabet's substitution table with affine gap scores, in every mode on the CPU,
  and globally or locally on the GPU, scores and alignments: a batch's global scores there first over
  a band of sixteen diagonals, a thread a pair, each kept where the pair's own cost proves no path off
  the band could score as well, the rest over whole matrices, several pairs a warp; a pair too tall
  for one block's carry tiled over global memory, its alignment traced in linear space.
- Unit-cost distances on the GPU, globally, a thread a pair by Myers' bit-vectors.
- One rule for ties (`Ties`): of equally good alignments, KSW2's by default (`Ties.KSW2`, minimap2's and
  plegma's: the walk back from the end takes the diagonal at every tie, then a deletion, an insertion, the
  second gap piece's, so gaps come as early and substitutions as late as the cost allows), every edit left
  (`Ties.LEFT`), or every edit right, WFA2-lib's CIGAR byte for byte (`Ties.RIGHT`); with free ends the span
  first (`LEFT`'s for `KSW2`), whichever search found the cost, under any band or cap. At unit costs the edit
  engine walks `KSW2` itself through the fronts it searched (`ksw2_diagonal_into`), at no cost over `LEFT`; at
  gap-affine costs so does a global wavefront, its forward search keeping its gap layers and every front an
  optimal path passes (`ksw2_trace`), about a quarter over `RIGHT` on 1 kbp two-piece pieces; any other
  search's CIGAR (a full sweep, a pair split for memory, the lanes, free ends, extensions, scored modes) is
  written again by Gotoh's recurrence (`ksw_ties`), within `max_memory`, swept by anti-diagonal in sixteen
  lanes over only the cells an alignment no costlier than the search's can pass (its cost so far and the letters
  left at the cheaper extension), so the whole matrix's CIGAR for any pair; over minimap2's scores as costs it
  gives `ksw_extd2`'s CIGAR exactly. The C options' `ties` (formerly `right_ties`), Python's and the
  command line's `ties` take it too.
- A reused search keeps its rings as wide as the last pair grew them, clearing only what that pair wrote:
  each pair had grown them from narrow again, copying them at every doubling, an eighth of a 1 kbp pair's
  time at two-piece costs (20,000 pieces: 3.85 s to 3.26 s, any rule for ties).
- Bands of diagonals, cost caps (`max_cost`), and batches (`distances`, `alignments`, `scores`),
  capped too, many pairs at once in the lanes of a SIMD register: costs and alignments, globally and
  with free ends, an alignment's span found first, its end then its start, as the rule for ties picks
  it; local scores; and tables of up to sixteen entries.
- Every call runs on its caller's own thread and starts none unless asked (`threads`), and keeps no
  state between calls, so an application calls it from as many threads as it likes and spreads its
  work itself. An `Aligner`, one a thread, keeps its searches' memory from call to call; in C,
  `karan_aligner_new`, and in C++ and Python, `Aligner`. The command line spreads its pairs over
  every thread itself.
- The single-pair calls under `Costs`, `distance`, `align`, `score`, `local_scores` and `Aligner`'s, take
  the two sequences as byte spans (`Span[UInt8]`, of a `List[UInt8]` or any buffer) as well as `String`s,
  read where they lie down to the searches, the same costs and CIGARs: a caller holding bytes, as a POA
  graph does, makes no `String` a call.
- What a SAM record holds: soft- or hard-clipped CIGARs, `NM`, `MD`, identity
  (`Alignment.clipped_cigar`, `edit_distance`, `mismatch_string`, `identity`), and SSW's second-best
  local score for a mapping quality (`local_scores`).

### Interfaces

- A C API with a C++ wrapper (`c/karan.h`, `pixi run build-c`), batches included: a batch or a
  search asked for threads starts its own, so the caller's threads may each ask at once; a length
  below zero is `KARAN_INVALID_LENGTH`.
- A Python package (`pixi run build-python`, `pixi run build-wheel`), any CPython 3: sequences
  ASCII `str` or `bytes`, anything else refused.
- A command-line aligner, `karan` (`pixi run build-cli`): FASTA, FASTQ or pairs in, a table,
  SAM or PAF out.

### Quality

- Every answer held to a full-matrix oracle by a differential fuzzer (`pixi run fuzz`), over pairs,
  costs, tables, modes, bands, caps, tie rules and memory budgets.
- WFA2-lib's regression set (`pixi run test-wfa`), byte for byte under `Ties.RIGHT`.
- Continuous integration on x86-64 and AArch64 Linux and Apple silicon.
- Benchmarks build every tool for the same CPU, named in each table, and run each through one aligner
  a file, as each tool's own runner keeps one: A\*PA2's and QuickEd's datasets, and real Illumina and
  PacBio HiFi reads of HG002 from Genome in a Bottle, each against the stretch of HG002's own assembly
  it maps to (`benchmarks/giab.py`).
