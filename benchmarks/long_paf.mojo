# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the
# MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
`long_paf PRESET REF.fa QUERIES.fa|fq [THREADS]`: each query aligned to the reference by `karan.LongAligner` from
minimap2's own seeds (`tests/mm2_seeds.mojo`), written as minimap2 writes PAF with `-c --cs` (`tp`, `cg`, `cs`
tags), so the two outputs compare line for line (`benchmarks/long_parity.py`). PRESET is `asm5`, `dipcall`
(`-x asm5 -z200000,10000 -r2k`) or `map-hifi`. Seeding and alignment times go to standard error.
"""

from std.sys import argv, stderr
from std.time import perf_counter_ns

from karan import LongAligner, LongAlignment, LongPreset, Seed, encode_nt4, name_hash
from karan.long_align import CIGAR_DELETION, CIGAR_INSERTION, CIGAR_MATCH
from mm2_seeds import Mm2Index, mm2_seeds


def read_records(path: String) raises -> Tuple[List[String], List[List[UInt8]]]:
    """FASTA or FASTQ records: names (first word) and sequences."""
    var names = List[String]()
    var seqs = List[List[UInt8]]()
    with open(path, "r") as f:
        var text = f.read()
        var lines = text.split("\n")
        var i = 0
        var fastq = len(lines) > 0 and lines[0].startswith("@")
        while i < len(lines):
            var line = lines[i]
            if line.byte_length() == 0:
                i += 1
                continue
            if fastq:
                names.append(String(line[byte=1:].split(" ")[0].split("\t")[0]))
                seqs.append(List[UInt8](lines[i + 1].as_bytes()))
                i += 4
                continue
            if line.startswith(">"):
                names.append(String(line[byte=1:].split(" ")[0].split("\t")[0]))
                seqs.append(List[UInt8]())
            else:
                seqs[len(seqs) - 1].extend(line.as_bytes())
            i += 1
    return (names^, seqs^)


def paf_line(
    qname: String, qlen: Int, tname: String, reference: List[UInt8], query: List[UInt8], hit: LongAlignment
) -> String:
    """One PAF line as minimap2 writes it with `-c --cs`, its letters compared in minimap2's codes."""
    comptime LETTERS = "acgtn"
    var out = String()
    var strand = "-" if hit.reverse else "+"
    var tp = ("I" if hit.primary else "i") if hit.inversion else ("P" if hit.primary else "S")
    out.write(qname, "\t", qlen, "\t", hit.query_start, "\t", hit.query_end, "\t", strand, "\t", tname, "\t")
    out.write(len(reference), "\t", hit.reference_start, "\t", hit.reference_end, "\t", hit.matches, "\t")
    out.write(hit.block_length, "\t", hit.mapq, "\ttp:A:", tp, "\tAS:i:", hit.score, "\tms:i:", hit.dp_max)
    out.write("\ts1:i:", hit.chain_score)
    if hit.primary:
        out.write("\ts2:i:", hit.secondary_score)
    out.write("\tcg:Z:", hit.cigar_string(), "\tcs:Z:")
    var t = hit.reference_start
    var q = 0
    var rc = hit.reverse
    var qbase = qlen - hit.query_end if rc else hit.query_start

    @always_inline
    def qcode(index: Int) {imm query, imm rc, imm qlen} -> Int:
        if rc:
            var c = Int(query[qlen - 1 - index])
            return 3 - c if c < 4 else 4
        return Int(query[index])

    for run in hit.cigar:
        var op = Int(run & 0xF)
        var length = Int(run >> 4)
        if op == CIGAR_MATCH:
            var same = 0
            for l in range(length):
                var tc = Int(reference[t + l])
                var qc = qcode(qbase + q + l)
                if tc == qc:
                    same += 1
                else:
                    if same > 0:
                        out.write(":", same)
                        same = 0
                    out.write("*", LETTERS[byte=tc], LETTERS[byte=qc])
            if same > 0:
                out.write(":", same)
            t += length
            q += length
        elif op == CIGAR_INSERTION:
            out.write("+")
            for l in range(length):
                out.write(LETTERS[byte = qcode(qbase + q + l)])
            q += length
        else:
            out.write("-")
            for l in range(length):
                out.write(LETTERS[byte = Int(reference[t + l])])
            t += length
    return out^


def main() raises:
    var args = argv()
    var preset_name = String(args[1])
    var preset: LongPreset
    if preset_name == "asm5":
        preset = LongPreset.asm5()
    elif preset_name == "dipcall":
        preset = LongPreset.dipcall()
    elif preset_name == "map-hifi":
        preset = LongPreset.map_hifi()
    else:
        raise Error("unknown preset " + preset_name)
    var ref_in = read_records(String(args[2]))
    var queries = read_records(String(args[3]))
    var tname = ref_in[0][0]
    var reference = encode_nt4(Span(ref_in[1][0]))
    var t0 = perf_counter_ns()
    var index = Mm2Index(Span(ref_in[1][0]), 19, 19)
    var t1 = perf_counter_ns()
    var aligner = LongAligner()
    var seed_ns = 0
    var align_ns = 0
    var out = String()
    for i in range(len(queries[0])):
        var s0 = perf_counter_ns()
        var seeded = mm2_seeds(index, Span(queries[1][i]))
        var s1 = perf_counter_ns()
        var query = encode_nt4(Span(queries[1][i]))
        var hits = aligner.align(
            Span(reference),
            Span(query),
            Span(seeded[0]),
            preset,
            name_hash(queries[0][i]),
            seeded[1],
        )
        var s2 = perf_counter_ns()
        seed_ns += Int(s1 - s0)
        align_ns += Int(s2 - s1)
        for hit in hits:
            out.write(paf_line(queries[0][i], len(query), tname, reference, query, hit), "\n")
    print(out, end="")
    print(
        "index",
        Float64(t1 - t0) / 1e9,
        "s, seeds",
        Float64(seed_ns) / 1e9,
        "s, alignment",
        Float64(align_ns) / 1e9,
        "s, mid_occ",
        index.mid_occ,
        file=stderr,
    )
