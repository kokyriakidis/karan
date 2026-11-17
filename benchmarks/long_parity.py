#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the
# MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
`long_parity.py MINIMAP2.paf KARAN.paf [--ref REF.fa --paftools K8,PAFTOOLS.js]`: how closely karan's long
alignments (`benchmarks/long_paf.mojo`) match minimap2's own, both written with `-c --cs`.

- primary placement: each query's first `tp:A:P` record (SAM's primary), same strand and intervals;
- records: every record matched by query, strand and intervals, and of those the identical CIGARs;
- differences: the cs strings' events (substitutions, insertions, deletions at their reference positions) of the
  primaries, shared over all;
- with `--ref` and `--paftools`, `paftools.js call` on both PAFs and the VCF records shared.
"""

import argparse
import re
import subprocess
import sys


def records(path):
    out = []
    with open(path) as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 12:
                continue
            tags = dict((t[:2], t[5:]) for t in fields[12:])
            out.append(
                {
                    "q": fields[0],
                    "key": (fields[0], fields[4], int(fields[2]), int(fields[3]), int(fields[7]), int(fields[8])),
                    "rs": int(fields[7]),
                    "tp": tags.get("tp", "?"),
                    "cg": tags.get("cg", ""),
                    "cs": tags.get("cs", ""),
                    "mapq": int(fields[11]),
                }
            )
    return out


def primaries(recs):
    first = {}
    for r in recs:
        if r["tp"] == "P" and r["q"] not in first:
            first[r["q"]] = r
    return first


CS = re.compile(r"(:[0-9]+|\*[a-z][a-z]|[=\+\-\~][A-Za-z]+)")


def events(rec):
    pos = rec["rs"]
    out = set()
    for op in CS.findall(rec["cs"]):
        if op[0] == ":":
            pos += int(op[1:])
        elif op[0] == "*":
            out.add((pos, "X", op[1:]))
            pos += 1
        elif op[0] == "+":
            out.add((pos, "I", op[1:]))
        elif op[0] == "-":
            out.add((pos, "D", op[1:]))
            pos += len(op) - 1
    return out


def call(paf, ref, paftools, group):
    """paftools call on the PAF's records, each haplotype (the query names' `group` match) on its own, as dipcall
    calls each; the calls as (haplotype, contig, position, reference, alternative)."""
    k8, js = paftools.split(",")
    lines = {}
    with open(paf) as f:
        for line in f:
            m = re.match(group, line) if group else None
            lines.setdefault(m.group(1) if m else "", []).append(line)
    calls = set()
    for hap, hap_lines in sorted(lines.items()):
        hap_lines.sort(key=lambda l: (l.split("\t")[5], int(l.split("\t")[7])))
        vcf = subprocess.run(
            [k8, js, "call", "-f", ref, "-"], input="".join(hap_lines), capture_output=True, text=True, check=True
        ).stdout
        for line in vcf.splitlines():
            if line.startswith("#"):
                continue
            f = line.split("\t")
            calls.add((hap, f[0], f[1], f[3], f[4]))
    return calls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("minimap2")
    parser.add_argument("karan")
    parser.add_argument("--ref")
    parser.add_argument("--paftools")
    parser.add_argument("--group", default=r"^(h\d)", help="the haplotype in a query name [%(default)s]")
    parser.add_argument("--show", type=int, default=0, help="list this many differing primaries")
    args = parser.parse_args()
    mm = records(args.minimap2)
    ka = records(args.karan)
    mp = primaries(mm)
    kp = primaries(ka)
    queries = sorted(set(mp) | set(kp))
    same_primary = sum(1 for q in queries if q in mp and q in kp and mp[q]["key"] == kp[q]["key"])
    same_cigar = sum(1 for q in queries if q in mp and q in kp and mp[q]["key"] == kp[q]["key"] and mp[q]["cg"] == kp[q]["cg"])
    print(f"queries with a primary: minimap2 {len(mp)}, karan {len(kp)}")
    print(f"same primary placement: {same_primary}/{len(queries)} ({100.0 * same_primary / max(1, len(queries)):.2f}%)")
    print(f"same primary and CIGAR: {same_cigar}/{len(queries)} ({100.0 * same_cigar / max(1, len(queries)):.2f}%)")
    mk = {r["key"]: r for r in mm}
    kk = {r["key"]: r for r in ka}
    both = set(mk) & set(kk)
    print(
        f"records: minimap2 {len(mm)}, karan {len(ka)}, same intervals {len(both)}, "
        f"identical CIGAR {sum(1 for k in both if mk[k]['cg'] == kk[k]['cg'])}, same tp "
        f"{sum(1 for k in both if mk[k]['tp'] == kk[k]['tp'])}, same mapq {sum(1 for k in both if mk[k]['mapq'] == kk[k]['mapq'])}"
    )
    shared = 0
    union = 0
    for q in queries:
        a = events(mp[q]) if q in mp else set()
        b = events(kp[q]) if q in kp else set()
        shared += len(a & b)
        union += len(a | b)
    print(f"primary differences (cs events): shared {shared}/{union} ({100.0 * shared / max(1, union):.3f}%)")
    shown = 0
    for q in queries:
        if shown >= args.show:
            break
        a = mp.get(q)
        b = kp.get(q)
        if a is None or b is None or a["key"] != b["key"] or a["cg"] != b["cg"]:
            print("  differs:", q, a["key"] if a else None, b["key"] if b else None)
            shown += 1
    if args.ref and args.paftools:
        a = call(args.minimap2, args.ref, args.paftools, args.group)
        b = call(args.karan, args.ref, args.paftools, args.group)
        print(
            f"paftools call: minimap2 {len(a)}, karan {len(b)}, identical {len(a & b)} "
            f"({100.0 * len(a & b) / max(1, len(a | b)):.3f}% of either)"
        )


if __name__ == "__main__":
    sys.exit(main())
