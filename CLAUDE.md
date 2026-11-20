# karan

Exact DNA alignment in Mojo: affine-gap Gotoh alignment on CPU and GPU, and a bit-parallel unit-cost
edit distance after A*PA2.

## Mojo skills

Modular's Mojo skills live in `.claude/skills` and correct pretrained Mojo, which is out of date.
A session-start hook (`.claude/settings.json`) reminds the agent to load them; follow it:

- `mojo-syntax` before writing, editing or reviewing any Mojo, with `references/idiomatic-mojo.md`.
- `mojo-gpu-fundamentals` before touching GPU code; `mojo-python-interop` before Mojo-Python interop.
- Load a skill again after a context compaction, which truncates loaded skills.

`pixi run sync-skills` refreshes them from upstream; `.claude/skills/UPSTREAM` records the commit.

## Working here

- Build and test: `pixi run test`; format: `pixi run format` (120 columns).
- Toolchain: one Mojo nightly with its `max`, pinned exactly in `pixi.toml` (1.2.0.dev2026102005), the one sporos and
  plegma build with, so the three link into one program; plegma carries karan as a submodule. Move it on purpose: change
  the pin, `pixi lock`, build and test. The paper's timings were taken with Mojo 1.1.0 and say so.
- `pixi run test-wfa` holds every mode WFA2-lib also has to its regression set (scores and CIGARs).
- `pixi run test-ksw2` holds `karan.ksw2` to minimap2's C byte for byte and `ksw2_lanes` and `ksw2_gpu` to it (the GPU
  test runs on the host's GPU, an Apple M2 included); `pixi run test-ksw-ties` holds `Ties.KSW2` to it. hcm and herro2
  rely on those CIGARs being minimap2's: run both after any change to the three modules.
- `pixi run test-long` holds `LongAligner` (`long_*.mojo`, `chaining.mojo`) to minimap2 2.28's PAF on a kept window; the
  full parity check is `build/long_paf` against minimap2 with `benchmarks/long_parity.py` (see `benchmarks/README.md`).
  `pixi run build-long-trace` builds it with each stage traced, to find where a line first differs; `pixi run fuzz-long`
  aligns random seed sets with every read bounds-checked.
- `pixi run profile duel|rivals|regret|probes|peak|perf|modes|scaling|noise|compare|all` finds where time goes,
  what each engine choice cost and where a rival wins (see `benchmarks/README.md`, Profiling); probes compile
  in only with `-D KARAN_PROBES`, forced policies only with `-D KARAN_FORCE`.
- Finished work merges straight into `main` and is pushed; no pull requests.
- The paper's text lives in `tools/paper/` (`main_text.py`, `supplement.py`); `pixi run paper` rewrites `paper/`,
  an Overleaf project, from it. Edit the source, not the generated `.tex`.
