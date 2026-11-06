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
- `karan.ksw2_lanes`: the same global gap filling for a batch of pairs at once, a pair a lane: the cells, codes and
  walk back of `ksw2`, so the same CIGARs wherever the band holds the alignment; twice as fast as `ksw2` on batches
  of like pieces.

- One API for every cost model and mode: `distance`, `align` and `score` take `Costs` (unit, linear,
  gap-affine, two-piece gap-affine, deletions priced apart from insertions) and a `Mode` of three
  kinds, as WFA2-lib's ends-free and extension and abPOA's local: any free ends, with the presets
  global, infix, prefix, suffix and overlap, and a reward for every match when asked; an extension
  from either end with an optional Z-drop and end bonus; and local. The Z-drop weighs every alignment
  still growing, those inside a gap too, over the last move's worth of costs, as KSW2 does, rather than one
  cost's aligned front as WFA2-lib does, which gave up far earlier than minimap2 on noisy reads.
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
  first (`LEFT`'s for `KSW2`), whichever search found the cost, under any band or cap. `KSW2` is written by
  Gotoh's recurrence over the cells about the path a search found (`ksw_ties`), and over minimap2's scores as
  costs gives `ksw_extd2`'s CIGAR exactly. The C options' `ties` (formerly `right_ties`), Python's and the
  command line's `ties` take it too.
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
