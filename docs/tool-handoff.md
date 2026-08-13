# Starting the tool — a cold-start handoff

**Read this first, then `docs/tool-roadmap.md` §1–§8.** This file exists so a session can begin
without reconstructing anything from chat history. Everything below is either settled or explicitly
open; nothing is left implied.

**This repository stands alone.** It shares no code, no build and no dependency with the study
repository the generator came from — the only thing connecting them is that both target
[Alan Bacellar's DWN implementation](https://github.com/alanbacellar/DWN). References marked
*(study repo)* are **citations to published evidence, not files you can open from here.**

---

## 1. What the tool is

A **generator**: a trained DWN goes in, synthesizable Verilog comes out, plus the means to prove
that Verilog matches the model. Nothing else.

| in scope | out of scope |
|---|---|
| exporter (checkpoint → tables, wiring, thresholds) | the board harness — UART, vector store, benchmark FSM |
| RTL generator (core, encoder, top) | the design-space sweep (`dse/`) |
| the numpy golden model | the controlled comparisons (`cc/`) |
| **self-checking testbenches + golden vectors** | the area model — see roadmap Q7 |
| optional yosys resource estimate | any vendor toolchain dependency |

**The encoder always ships**, and its area is always reported separately from the network's. It is
intrinsic to a DWN, not preprocessing a user supplies — and on the smallest JSC model it is
**fourteen times** the network it feeds. Emitting the network alone would commit exactly the
reporting defect `docs/jsc/report.md` *(study repo)* §5.2 criticises in published work.

## 2. Settled, with the reasoning

| | decision |
|---|---|
| **New repo, not a fork** | The tool is ~2,300 of ~19,600 lines — 12%. A fork's first commit deletes 88%. Knowledge lives in code comments, which travel with a copy; the ledgers should not travel |
| **Generator-only** | Users bring their own harness, as hls4ml does. The harness changes per application |
| **Vendor-neutral** | The emitted RTL instantiates **no vendor primitives** and is Verilog-2001. Verified by inspection across the whole tree |
| **Verification ships** | Self-checking testbenches the *user* runs in *their* simulator. That makes bit-exactness reproducible by them, not a claim they must trust |
| **No area model** | It filtered zero configs across two completed studies. hls4ml ships none either |
| **This repo is the evidence** | "77 configurations, all bit-exact, two datasets, one verified on silicon per dataset" is a claim nothing else in this space can make. Link to it; do not reproduce it |

## 3. Open — decide before writing a CLI

**Q8. There is no upstream checkpoint format.** Upstream trains and discards; the format this
project reads is our own. And a DWN is *two objects* — thermometer thresholds live outside the
`state_dict`, so `torch.save(model.state_dict())` silently loses the encoder. The tool must define
the format, own both ends, and fail loudly on a bare `state_dict`.
`docs/checkpoint-format.md` is most of the specification already.

~~**Q9. Fractional bits.**~~ ✅ **Resolved — see roadmap Q9.** Short version: the tool asks for
`--input-bits` (the *input's* precision), never for fractional bits. When the input has a native
quantum — 8-bit pixels, most sensors — `frac = n` is **provably** lossless, not merely measured.
MNIST is the proof: pixels are `k/255`, `floor(k·256/255)` is strictly increasing over k = 0…255,
so ordering is preserved exactly, which is why 0 of 10,000 samples diverged. Continuous inputs get
a default plus a measured bit-error, labelled a stress test rather than a proof.

**Q3. Which upstream commits to support.** One pin is honest and cheap; a range needs a
compatibility layer.

**Q4. The name.** `dwn2rtl` is a placeholder used in discussion only. The study repository it came
from is **`dwn-fpga-study`**.

## 4. Where this code came from

Provenance only. The import was a one-time event; none of the left column is a live dependency.

| in the study repo | lines | became |
|---|---|---|
| `rtl/{lut_node,popcount,argmax,pipe_reg}.v` | 194 | `src/dwn2rtl/rtl/` — **verbatim** |
| `tb/{dwn_core,dwn_top}_tb.v` | 177 | `src/dwn2rtl/rtl/tb/` — **verbatim**, already self-checking |
| `exporter/extract.py` | 296 | `src/dwn2rtl/extract.py` — it is the golden model too |
| `rtlgen/emit_core.py`, `emit_encoder.py` | 559 | `src/dwn2rtl/emit_*.py`, CLI entry points stripped |
| `tb/gen_vectors.py` | 215 | `src/dwn2rtl/vectors.py` — **rewritten**, random inputs only |
| `datasets/__init__.py` | 352 | `src/dwn2rtl/precision.py` — **~40 lines extracted**, sweep axes left behind |
| `rtlgen/config.py` | 254 | **not copied** — sweep-shaped; a small config object was written fresh |
| `scripts/run_gate1.py` | 282 | `src/dwn2rtl/verify.py` — **rewritten**, simulator-agnostic |

**Why the split was clean:** the study repo's Gate 1 path compiled nine files, **none from the
board harness**, and the testbenches instantiate only `dwn_top` / `dwn_core`.

## 5. Vectors come from the model, not from data

The tool cannot assume a dataset either. Generate testbench vectors by drawing **random quantised
inputs**, running them through the numpy golden model, and emitting those.

This is not a compromise: **Gate 1 is RTL-versus-golden-model, not RTL-versus-dataset.** Random
vectors are arguably better, since they hit tie-break and saturation edges real data does not.
`tb/gen_vectors.py` already mixes 500 random vectors with real ones; the tool keeps the random half.

**The invariant that must not break:** the testbench vectors and the RTL must derive from the *same*
checkpoint. Otherwise you ship a testbench that passes against wrong RTL — worse than shipping none.

## 6. First real gate

**Run the emitted testbench under `iverilog`.** Neither iverilog nor verilator is installed on the
study machine, so "the RTL is portable" is currently an *inspection* result, not a measurement.
Make it the tool's first CI check rather than an assumption.

## 7. Suggested first steps

1. Create the repo; copy the eight verbatim files **unchanged** as the first commit, so the
   rewrites are visible as their own commits afterwards. Then `pyproject.toml` and a CLI entry
   point.
2. Port the exporter and emitters, then the vector generator in its rewritten form.
3. Implement the precision policy from roadmap Q9 — derive integer bits, take `--input-bits`,
   warn on the comparator-merge floor.
4. Get `iverilog` green on one emitted design end to end. **This is the first real gate.**
5. Only then: yosys estimates, multi-version support, docs.

## 8. In this repository

| | |
|---|---|
| `docs/tool-roadmap.md` | the audited work list — defects, generality gaps, packaging, order |
| `docs/checkpoint-format.md` | the checkpoint schema, verified against the pinned upstream commit |
| `CLAUDE.md` | ground rules: the gate, the design invariants, commit conventions |

## 9. External citations — evidence, not dependencies

These were measured in the study repository **`dwn-fpga-study`** and **cannot be opened from
here.** They are worth citing in this project's README, because they are the evidence the generator
works. Nothing in this repo requires them to exist.

| claim worth citing | measured in |
|---|---|
| 77 configurations, all bit-exact against the golden model | both studies' sweeps |
| 166,000/166,000 and 10,000/10,000 correct **on physical silicon** | JSC and MNIST Gate 1b |
| the generator reproduces the original authors' area/accuracy at matched convention | MNIST study §7.3 |
| the encoder can be 14× the network it feeds, and published counts omit it | JSC study §5.2 |
| seven hard-coded dataset assumptions, none findable by inspection | MNIST study §2 |
| noise floors of 0.15 pp (JSC) and 0.24 pp (MNIST) | MNIST reduction study |

**The one thing the two repositories share** is their target: Alan Bacellar's DWN implementation,
<https://github.com/alanbacellar/DWN>. Pin it here independently — do not assume the study repo's
pin, and re-read `docs/checkpoint-format.md` against whatever commit this project pins.