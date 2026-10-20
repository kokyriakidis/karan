<div align="center">

# karan

**Exact pairwise sequence alignment in Mojo, for the CPU and the GPU.**

[![CI](https://github.com/kokyriakidis/karan/actions/workflows/ci.yml/badge.svg)](https://github.com/kokyriakidis/karan/actions/workflows/ci.yml)
[![License: MPL 2.0](https://img.shields.io/badge/license-MPL%202.0-blue.svg)](LICENSE)
![Platforms](https://img.shields.io/badge/platforms-Linux%20x86--64%20%7C%20Linux%20AArch64%20%7C%20macOS%20Apple%20silicon-lightgrey)
![Languages](https://img.shields.io/badge/API-Mojo%20%7C%20Python%20%7C%20C%20%7C%20C%2B%2B%20%7C%20CLI-orange)

[Installation](#installation) · [Quick start](#quick-start) · [Performance](#performance) ·
[Modes and costs](#alignment-modes) · [Documentation](#documentation) · [Benchmarks](benchmarks/README.md)

</div>

karan finds the optimal alignment of two sequences: the least edit distance, the least gap-affine
cost or the best score. It aligns globally, with free ends, as a seed extension or locally, and returns
the CIGAR. **No heuristic decides the result**, and one fixed rule chooses among equally good alignments.
It runs from Mojo, Python, C, C++ and the command line.

```python
>>> import karan
>>> karan.align("ACGTACGTTTGCA", "ACGTCGTTTTGCA").cigar
'4=1D2=1I6='
```

## Highlights

|  |  |
| :-- | :-- |
| ⚡ **Fast** | The fastest exact aligner on every dataset any tool finished in our benchmarks, against A\*PA2, A\*PA, QuickEd, BSAlign, Edlib, WFA2-lib (WFA and BiWFA), KSW2, parasail and SSW: **1.4–7.0×** the next fastest at unit costs and **1.3–3.7×** at gap-affine costs ([details](#performance)). |
| 🎯 **Exact** | Bands, cost caps and memory limits decide *which* alignments are allowed, never whether the best of them is found. A test suite and a fuzzer check every result against a full dynamic-programming matrix, and WFA2-lib's regression set checks scores and CIGARs. |
| 🔁 **Deterministic** | One rule chooses among equally good alignments: indels placed left, as minimap2 places them, or right, matching WFA2-lib's backtrace byte for byte. The CIGAR does not depend on the band, the cap or which algorithm found the cost. |
| 🧩 **Complete** | Unit, linear, gap-affine and two-piece costs, with deletions priced apart from insertions and substitution tables. Global, infix, prefix, suffix, overlap, any free ends, extension with Z-drop and end bonus, and local alignment. |
| 🧮 **Batches and GPUs** | Batches of short pairs are aligned many at a time in SIMD lanes. Global and local scores and alignments under a substitution table, and global unit-cost distances, also run on NVIDIA GPUs and Apple silicon. |
| 🧠 **Bounded memory** | A traceback stays within `max_memory` (80 MB by default) at any length, and the CIGAR is the same as with unlimited memory. |

## Installation

### Prebuilt releases

Each [release](https://github.com/kokyriakidis/karan/releases) has three archives per platform
(Linux x86-64, Linux AArch64, macOS on Apple silicon). Each archive includes the Mojo runtime it needs, so
neither Mojo nor pixi has to be installed:

| archive | contents |
| :-- | :-- |
| `karan-<version>-<platform>-cli.tar.gz` | the command-line aligner, `cli/karan` |
| `karan-<version>-<platform>-python.tar.gz` | the Python package, `python/karan` |
| `karan-<version>-<platform>-c.tar.gz` | the C/C++ library and header, `c/libkaran` and `c/karan.h` |

```bash
tar -xzf karan-v0.1.0-linux-64-cli.tar.gz && ./cli/karan --help
tar -xzf karan-v0.1.0-linux-64-python.tar.gz && PYTHONPATH=$PWD/python python3 -c "import karan"
```

On Linux the archives need glibc 2.35 or later (Ubuntu 22.04, Debian 12, RHEL 10). They are built for
each platform's baseline CPU.

### From source

Building from source needs [pixi](https://pixi.sh), which installs Mojo and every other dependency into
the project.

```bash
git clone https://github.com/kokyriakidis/karan && cd karan
pixi run build-cli       # build/cli/karan
pixi run build-python    # build/python/karan
pixi run build-wheel     # build/dist/*.whl, for pip install
pixi run build-c         # build/c: libkaran and karan.h
```

Each build script takes an optional target CPU, for example `pixi run build-cli x86-64-v3`. The result
is faster on newer machines but runs only on CPUs with that instruction set.

## Quick start

### Command line

```console
$ karan -r ACGTACGTTTGCA -q ACGTCGTTTTGCA
query	reference	strand	cost	score	reference_start	reference_end	query_start	query_end	cigar
query	reference	+	2	-2	0	13	0	13	4=1D2=1I6=

$ karan --costs affine:4,6,2 --mode infix --format sam reference.fa reads.fq   # reads placed in a reference
```

Inputs can be FASTA or FASTQ files, a tab-separated file of pairs (`--pairs`), or two sequences given
on the command line (`-r`, `-q`). Output is a table, SAM or PAF, and `--threads` spreads the pairs over
cores. `karan --help` lists every option.

### Python

```python
import karan

karan.distance("ACGTACGTTTGCA", "ACGTCGTTTTGCA")                          # 2

costs = karan.Costs.affine(4, 6, 2)                                       # mismatch 4, gap of k: 6 + 2k
found = karan.align("ACGTACGTTTGCA", "ACGTCGTTTTGCA", costs)
found.cost, found.cigar                                                   # (12, '4=3X6=')

placed = karan.align("TTTTACGTACGTTTTT", "ACGTACGT", costs, karan.Mode.INFIX)
placed.cigar, placed.reference_start, placed.reference_end                # ('8=', 4, 12)

core = karan.align("GGGGACGTACGTGGGG", "CCCCACGTACGTCCCC", costs, karan.Mode.local(2))
core.score                                                                # 16

karan.distances(["ACGTACGT", "TTGCA"], ["ACGACGT", "TTGGCA"], threads=4)  # [1, 1]

aligner = karan.Aligner()   # keeps its memory between calls: one per thread, reused in a loop
for reference, query in [("ACGTACGTTTGCA", "ACGTCGTTTTGCA"), ("ACGTAC", "ACGAC")]:
    print(aligner.align(reference, query).cigar)                          # 4=1D2=1I6=, then 3=1D2=
```

<details>
<summary><b>C</b></summary>

```c
#include "karan.h"

int64_t edits = karan_distance("ACGTACGTTTGCA", 13, "ACGTCGTTTTGCA", 13, NULL, NULL, NULL);  /* 2 */

karan_costs affine = {4, 6, 2, -1, 0, 0, 0, -1, 0};
karan_alignment found;
if (karan_align("ACGTACGTTTGCA", 13, "ACGTCGTTTTGCA", 13, &affine, NULL, NULL, &found) == 0) {
    /* found.cost == 12, found.cigar == "4=3X6=" */
    karan_free(found.cigar);
}
```

Every function returns a negative `KARAN_` code on failure, and [c/karan.h](c/karan.h) documents
each one. Link with `-L build/c -lkaran` and put `build/c` on the program's library search path.

</details>

<details>
<summary><b>C++</b></summary>

```cpp
#include "karan.h"

auto found = karan::align("ACGTACGTTTGCA", "ACGTCGTTTTGCA", karan::Costs::affine(4, 6, 2));
// found.cost == 12, found.cigar == "4=3X6="

karan::Aligner aligner;  // one per thread, reused across calls
auto edits = aligner.distance("ACGTACGTTTGCA", "ACGTCGTTTTGCA");  // 2
```

The C++ API is defined inline in the same header over the C library, so it builds and links the same way.

</details>

<details>
<summary><b>Mojo</b></summary>

```mojo
from karan import Aligner, Costs, Mode, align, distance

def main() raises:
    print(distance("ACGTACGTTTGCA", "ACGTCGTTTTGCA"))  # 2
    var found = align("ACGTACGTTTGCA", "ACGTCGTTTTGCA", Costs.affine(4, 6, 2))
    print(found.cost, found.cigar)  # 12 4=3X6=
    var core = align("GGGGACGTACGTGGGG", "CCCCACGTACGTCCCC", Costs.affine(4, 6, 2), Mode.local(2))
    print(core.score)  # 16
    var aligner = Aligner()
    for pair in [("ACGTACGTTTGCA", "ACGTCGTTTTGCA"), ("ACGTAC", "ACGAC")]:
        print(aligner.align(pair[0], pair[1]).cigar)  # 4=1D2=1I6=, then 3=1D2=
```

Build against the source tree with `mojo build -I path/to/karan program.mojo`, or from inside
this repository with `pixi run mojo build -I . program.mojo`.

</details>

The library starts no threads of its own; every call runs on the caller's thread. For a loop of single
pairs, give each thread its own `Aligner`. The batch calls (`distances`, `alignments`, `scores`) pack many
pairs at once.

## Performance

Mean time per alignment on one core of an Intel Core i9-7900X (AVX-512) at a fixed 3.3 GHz. Every tool
was built for the machine's instruction set, and every tool's answers were checked against every
other's. Each time is the median of three rounds. This is a selection; the full tables, datasets and
the commands to reproduce them are in [benchmarks/README.md](benchmarks/README.md).

**Edit distance (unit costs)**

| dataset | karan | next fastest exact aligner | speed-up |
| :-- | --: | :-- | --: |
| ONT reads, mean 0.8 kbp (ont-1k) | **15.8 µs** | WFA2-lib, 31.3 µs | 2.0× |
| ONT reads, mean 3.6 kbp (ont-10k) | **142 µs** | A\*PA2-simple, 247 µs | 1.7× |
| ONT ultra-long PromethION reads of HG002, mean 45 kbp | **4.09 ms** | A\*PA2-simple, 6.76 ms | 1.7× |
| ONT reads, mean 640 kbp (ont-500k) | **116 ms** | A\*PA2-full, 284 ms | 2.4× |
| SARS-CoV-2 genomes, 30 kbp | **230 µs** | A\*PA2-simple, 727 µs | 3.2× |
| Illumina 2×250 reads of HG002 against its assembly | **0.435 µs** | WFA2-lib, 1.12 µs | 2.6× |
| PacBio HiFi reads of HG002, mean 14 kbp | **9.37 µs** | WFA2-lib, 66.0 µs | 7.0× |
| simulated 100 kbp pairs at 10% errors (QuickEd's set) | **7.04 ms** | A\*PA2-full, 13.9 ms | 2.0× |
| simulated 1 Mbp pairs at 20% errors (QuickEd's set) | **886 ms** | A\*PA2-full, 2.02 s | 2.3× |

**Gap-affine costs (mismatch 4, gap 6 + 2k)**

| dataset | karan | next fastest exact aligner | speed-up |
| :-- | --: | :-- | --: |
| ONT reads, mean 3.6 kbp (ont-10k) | **1.42 ms** | WFA2-lib, 4.86 ms | 3.4× |
| ONT ultra-long PromethION reads of HG002 | **231 ms** | BiWFA, 576 ms | 2.5× |
| Illumina 2×250 reads of HG002 | **1.17 µs** | WFA2-lib, 1.54 µs | 1.3× |
| PacBio HiFi reads of HG002 | **74.3 µs** | WFA2-lib, 225 µs | 3.0× |
| simulated 100 kbp pairs at 10% errors | **1.02 s** | BiWFA, 2.43 s | 2.4× |

**Other modes and batches**

| workload | karan | next fastest | speed-up |
| :-- | --: | :-- | --: |
| local alignment, a 1 kbp read in a 10 kbp window | **1.32 ms** | SSW, 3.17 ms | 2.4× |
| overlap of two 2 kbp reads | **703 µs** | parasail, 15.3 ms | 22× |
| seed extension with an end bonus | **149 µs** | KSW2, 1.24 ms | 8.3× |
| two-piece gap-affine costs | **7.54 ms** | WFA2-lib, 15.8 ms | 2.1× |
| 500,000 Illumina-like reads, gap-affine scores, one core (whole batch) | **229 ms** | WFA2-lib, 1.06 s | 4.6× |

At gap-affine costs, karan also finishes more of the hardest long pairs within the 30 s limit
than any other tool: 9 of 10 of the 100 kbp pairs at 20%, against BiWFA's 4.

## Alignment modes

A mode sets which ends of the two sequences an alignment must reach. The named free-end modes are
presets of `ends_free`.

| mode | reference | query | also known as |
| :-- | :-- | :-- | :-- |
| `GLOBAL` | whole | whole | end to end, Needleman-Wunsch, Edlib's NW |
| `INFIX` | any part | whole | semi-global, glocal, Edlib's HW |
| `PREFIX`, `SUFFIX` | a prefix, a suffix | whole | Edlib's SHW |
| `overlap(m)` | a prefix or suffix | a suffix or prefix | parasail's `sg` |
| `ends_free(...)` | any number of free letters at each end | the same | WFA2-lib's ends-free |
| `extension(m, anchor)` | from one end, as far as it pays | from the same end | KSW2's extension, with Z-drop and end bonus |
| `local(m)` | any part | any part | Smith-Waterman |

By default, free-end modes minimize the cost alone, as Edlib and WFA2-lib do. With
`mode.with_match_score(m)`, each match earns `m` and the alignment maximizes the score instead, as
parasail and read mappers score a read in a window.

## Cost models

| costs | a gap of `k` letters costs |
| :-- | :-- |
| `Costs.edit()` | `k`, every edit 1: the edit distance (the default) |
| `Costs.linear(x, g)` | `g k`, a mismatch `x` |
| `Costs.affine(x, o, e)` | `o + e k`, as WFA2-lib and minimap2 count it |
| `Costs.two_piece(x, o, e, o2, e2)` | `min(o + e k, o2 + e2 k)`, minimap2's `-O o,o2 -E e,e2` |
| `costs.with_deletions(o, e, ...)` | deletions priced apart from insertions, bwa's `-O del,ins` |
| `Scoring.tabulated(...)`, `Scoring.dna()` | any alphabet's substitution table with affine gap scores |

The result stays exact under a band of diagonals (`band=Band.around(w)`, KSW2's `w`) and under a cost
cap (`max_cost=`): it is the optimum of the alignments those limits allow.

## How it works

| problem | method |
| :-- | :-- |
| unit costs (edit distance) | diagonal transition for near-identical pairs, from one end and then from both; then A\*PA2's bit-parallel band doubling, pruned by its gap-chaining seed heuristic on long pairs |
| gap-affine and two-piece costs | a gap-affine wavefront, from one end and then from both (after WFA and BiWFA); a pair far apart is handed to a full sweep in 8-bit difference recurrences (after Suzuki and Kasahara) |
| local alignment, overlaps, rewarded free ends | an anti-diagonal sweep in 16-bit SIMD lanes while the scores fit, then an extension back from the best end |
| batches of short pairs | inter-sequence SIMD: one pair per lane, over a band of diagonals that each pair's own cost proves sufficient |
| GPU scores | a band of sixteen diagonals per thread, the rest scored whole, several pairs per warp |

The engines hand a pair over by fixed rules, not tuned thresholds: a cheaper method gives up only after
a set share of the work the next method would need. The handover changes only the speed; every engine
returns the same exact cost and the same CIGAR.

## Documentation

- [docs/API.md](docs/API.md): reference for everything the Mojo package exports.
- [c/karan.h](c/karan.h): the C and C++ API, with every function and error code documented.
- The Python package's docstrings, and `karan --help` for the command line.
- [benchmarks/README.md](benchmarks/README.md): benchmark setup, datasets, full results and profiling tools.
- [CHANGELOG.md](CHANGELOG.md): what each release changes.

## Development

```bash
pixi run test          # the Mojo test suite
pixi run fuzz          # random pairs, costs, modes, bands and caps, checked against a full matrix
pixi run test-wfa      # WFA2-lib's regression set, scores and CIGARs
pixi run test-c        # the C and C++ API
pixi run test-python   # the Python package
pixi run test-cli      # the command line
pixi run format        # mojo format, 120 columns
```

## Acknowledgements and citation

The unit-cost edit distance is ported from `pa-bitpacking` in
[A\*PA](https://github.com/RagnarGrootKoerkamp/astar-pairwise-aligner) by Ragnar Groot Koerkamp and
Pesho Ivanov, and follows A\*PA2's band doubling and seed heuristic. The gap-affine search follows WFA
and BiWFA by Santiago Marco-Sola and colleagues. If you use karan, please also cite the work it
builds on:

- R. Groot Koerkamp. *A\*PA2: up to 19× faster exact global alignment.* WABI 2024.
- R. Groot Koerkamp and P. Ivanov. *Exact global alignment using A\* with chaining seed heuristic and
  match pruning.* Bioinformatics, 2024.
- S. Marco-Sola, J. C. Moure, M. Moreto and A. Espinosa. *Fast gap-affine pairwise alignment using the
  wavefront algorithm.* Bioinformatics, 2021.
- S. Marco-Sola, J. M. Eizenga, A. Guarracino, B. Paten, E. Garrison and M. Moreto. *Optimal gap-affine
  alignment in O(s) space.* Bioinformatics, 2023.

## License

karan is licensed under the [Mozilla Public License 2.0](LICENSE). [NOTICE](NOTICE) credits the
work it builds on.
