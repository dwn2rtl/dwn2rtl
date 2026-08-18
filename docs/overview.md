# dwn2rtl — what it is, how you install it, how you use it

**Read this first if you have never used the tool.** This file is the plain description: what the
thing is, what it does to what, and how it lands on a computer. **It is the current document.**

The other two plans beside it are **historical** and carry a banner saying so: `tool-handoff.md`
was the cold-start briefing for phase 0, and `tool-roadmap.md` was the work list for building
what is now built. They are kept for their reasoning, not their tasks. Live work is §6 below and
`phaseN-ledger.md`.

---

## 1. In one paragraph

`dwn2rtl` takes a **trained** Differentiable Weightless Neural Network and emits synthesizable
Verilog-2001 that computes the same function — the thermometer encoder, the LUT network, and a
thin top wiring them together — plus self-checking testbenches and golden vectors that let you
prove, in your own simulator, that the emitted hardware is bit-exact against the software model.
It is a **translator**. It does not train, does not choose an architecture, does not target a
board, and does not depend on any vendor toolchain.

---

## 2. It is one package with two front doors

This is the part that causes the most confusion, so it gets its own section.

`dwn2rtl` is **a Python package that is also a terminal command.** Those are not two different
products or two different installs — it is one `pip install`, and afterwards you have both. This
is completely standard: `pytest`, `black`, `ruff`, `jupyter` and `hls4ml` all work exactly this
way.

The mechanism is four lines in `pyproject.toml`:

```toml
[project.scripts]
dwn2rtl = "dwn2rtl.cli:main"
```

When pip installs the package, it sees that block and writes a small launcher executable named
`dwn2rtl` into the same `Scripts/` (Windows) or `bin/` (Linux/macOS) directory that `pip` itself
lives in — a directory already on your PATH. That launcher does nothing but import
`dwn2rtl.cli` and call `main()`. So:

| you type | what actually runs |
|---|---|
| `import dwn2rtl` in a script | the package, directly |
| `dwn2rtl build ...` in a terminal | the launcher, which imports the same package and calls `cli.main()` |

**Same code, same install, same version.** The CLI is a thin argument-parsing wrapper over the
library API — everything it can do, `import dwn2rtl` can do, because it *is* `import dwn2rtl`.

### Which one should you use?

**The CLI is the primary path**, and it is the one the README will lead with:

```
dwn2rtl build model.pt --out rtl/
dwn2rtl verify rtl/
```

You have a trained model sitting in a file. You want Verilog in a directory. There is no reason
to write a Python script to ask for that, and no reason to learn an API to get it.

**The library API is for the notebook case** — you have just finished training, the model is a
live object in memory, and you want to look at the hardware before you commit to it:

```python
import dwn2rtl

report = dwn2rtl.build(dwn2rtl.from_model(model, thermometer), "rtl/")
print(report.comparators, report.nodes)      # encoder vs core, reported separately
```

This is what hls4ml's users mostly do, and it is worth supporting for the same reason: at the
moment training finishes, a file on disk is a detour.

Neither is a wrapper around a subprocess. They are two entry points into the same functions.

---

## 3. Installing it

### The package

```
pip install dwn2rtl
```

or, to hack on it:

```
git clone https://github.com/dwn2rtl/dwn2rtl.git
cd dwn2rtl
pip install -e ".[dev]"
```

Pure Python — no compiler, no build step, no platform-specific wheel. The hand-written Verilog
primitives in `src/dwn2rtl/rtl/` are **package data**, so they install alongside the code and
`verify` can find them without the user cloning anything.

Python dependencies: `numpy` (everything) and `torch` (reading checkpoints only — the golden
model and every emitter are pure numpy).

### The simulator — NOT a pip dependency

`dwn2rtl build` needs nothing but Python. `dwn2rtl verify` needs a Verilog simulator, and that is
an external program the user installs themselves:

- **Icarus Verilog** (`iverilog`) — the default target. Ships as a prebuilt Windows binary in the
  [OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build); `apt install iverilog` or
  `brew install icarus-verilog` elsewhere.
- **Verilator**, later.
- **yosys**, optional, only for `dwn2rtl estimate`.

`verify` searches PATH and reports what it found. No vendor licence is required for any of this,
which is the point — verification a user cannot run is a claim they have to take on trust.

---

## 4. The whole flow, end to end

### On the training side — once, at the end of your training script

A DWN is **two objects**: the model, and the `DistributiveThermometer` that was fitted *before*
training. The thermometer's thresholds are not model parameters and are not in the `state_dict`,
so they have to be saved deliberately. Plain PyTorch, no `dwn2rtl` import needed:

```python
torch.save({'model': model, 'thermometer': thermometer}, 'model.pt')
```

That single line is the only requirement the tool places on anyone's training code.

> **`torch.save(model.state_dict())` is the trap.** It silently drops the encoder, which on the
> smallest studied model is *fourteen times* the network it feeds. `dwn2rtl` refuses such a file
> by name rather than emitting a design that synthesizes cleanly and runs at chance.

### On the hardware side — in a terminal

```
$ dwn2rtl build model.pt --out rtl/

features 784, classes 10, layers [300], n=6, z=3   from checkpoint
integer bits 0                                     derived, exact
frac bits 8 -> Q0.8 signed (9-bit)                 INFERRED from the thresholds' grid, provably lossless

core      300 nodes, 3 cycles
encoder   720 comparators of 2352 thermometer bits
top       4 cycles latency, II=1

vectors   core 504, top 1227, 7/10 classes hit
note      1169 of 2352 thresholds quantise to a duplicate comparison (1183 distinct)
wrote     rtl/ (18 files)

$ dwn2rtl verify rtl/

iverilog 12.0 (/usr/bin/iverilog)
  dwn_core  504 vectors  PASS
  dwn_top   1227 vectors  PASS
RESULT   PASS
```

### What lands in `rtl/`

```
dwn_core.v  thermometer_encoder.v  dwn_top.v      the design
lut_node.v  popcount.v  argmax.v  pipe_reg.v      hand-written primitives, copied in
dwn_core_params.vh  dwn_top_params.vh             widths and pipeline depth
vec_params.vh  top_params.vh                      vector counts for the testbenches
x_binarized.hex  expected.hex                     core-level golden vectors
x_quant.hex  expected_top.hex                     top-level golden vectors
input_scaling.json                                only if the model was trained on scaled features
tb/dwn_core_tb.v  tb/dwn_top_tb.v                 self-checking testbenches
```

Self-contained. Hand the directory to any simulator or synthesis tool.

### Then it is theirs

Instantiate `dwn_top` in whatever harness the application needs:

```verilog
dwn_top u_dwn (.clk(clk), .x_flat(features), .class_idx(prediction));
```

Latency in cycles is in `dwn_top_params.vh`; throughput is one classification per clock (II=1).
The board, the clock, the I/O and the synthesis strategy are the user's, by decision — see
`tool-roadmap.md` Q1.

---

## 5. The boundary

| the tool does | the tool never |
|---|---|
| read a trained model | train, tune, or choose an architecture |
| derive integer width exactly from the thresholds | guess fractional bits, or ask for them |
| emit encoder, core and top | emit a board harness, UART, or vector store |
| generate test vectors from the *model* | require a dataset |
| report encoder and core cost separately | ship an area model, or need a vendor tool |

The only structural generation anywhere in the project is a **CI test fixture** — a synthetic
checkpoint of a chosen shape, used to test the emitter at shapes we have no real model for. It is
test-only and never on a user's path. It earns its place: in the study repo it caught emitter bugs
at MNIST's shape before any MNIST model existed.

---

## 6. Build plan

Ordered so the gate — a simulator printing PASS — comes as early as possible, because until then
nothing is verified.

**Each phase produces two documents, and they are not the same document.**

| | `docs/phaseN-ledger.md` | `docs/phaseN-report.md` |
|---|---|---|
| when | written *during*, as things happen | written *at the end*, once |
| shape | dated entries: built / hit / decided | a retrospective |
| audience | whoever is working, and future-you asking "why is it like that?" | someone who was not there |
| wrong turns | kept, struck through, with the reason | only where they changed a later decision |

The ledger is the evidence; the report is the argument. Writing only the report loses the
reasoning, and writing only the ledger makes anyone catching up read a diary.

| phase | | status |
|---|---|---|
| **0** | **make it runnable** — `pyproject.toml`, package data, the CLI entry point, a venv, a simulator | ✅ closed |
| **1** | **close the loop** — `checkpoint.py`, synthetic fixtures, `build()`, `dwn_top_tb.v`, `verify.py`, and **the gate green on both levels** | ✅ closed |
| **2** | **make it a tool** — `conftest.py`, CI on Linux and Windows, every commit | ✅ closed |
| **3** | **make it usable by someone else** — README, a worked example, the LICENCE, the upstream pin, and cleaning study-repo citations out of the shipped `rtl/*.v` | ✅ closed |
| **4** | **measure, then optimize** — `estimate` via yosys, then changes justified by measurement | ✅ closed |
| **5** | **publish it** — a GitHub org, packaging metadata, a Trusted Publishing workflow, a TestPyPI rehearsal, and `0.1.0` on PyPI | ✅ closed |

**Phase 1 is the milestone.** Everything before it is scaffolding and everything after is
packaging: a trained DWN goes in and a simulator certifies the Verilog bit-exact against the
golden model.

### Phase 4, and the rule it is organised around

**No optimization lands on an argument. It lands on a measurement, or it does not land.** That is
this project's own scar, not a borrowed principle: the study repo built an area model to filter
configurations too large to synthesize, and across two completed studies it filtered **zero**
(`tool-roadmap.md` Q7). So `estimate` comes *before* any RTL change — altering the emitter before
there is a way to tell whether it helped is the same mistake in a different costume.

⚠️ **And the risk that could make the phase harmful:** yosys is not the tool users synthesize
with. Tuning the emitter until *yosys* numbers improve would optimize against the wrong target.
Hence two guards — calibrate against the study's real Vivado figures before changing anything,
and justify every change **structurally** first, using measurement only to confirm it cost
nothing.

**Outcome: `estimate` shipped, and BOTH candidate optimizations were cancelled by measurement.**
Narrowing the encoder register would have saved nothing (synthesis already trims below the driven
bit count) and an explicit popcount tree would have saved nothing (depth already tracks log2 of
the width). Two measurements instead of two rewrites of shipped RTL.

Also shipped: precision is now **inferred from the thresholds' grid**, so `dwn2rtl build model.pt`
needs no `--input-bits` for quantised inputs and derives a 9-bit word for MNIST where it used to
fall back to a wider default.

⚠️ The calibration is the phase's most useful output: yosys lands on Vivado **exactly** for the
LUT core and **2.1x low** for the encoder, so `estimate` prints that with every report rather than
letting one number borrow the other's authority.

Details, including the deferred items and the PyPI prerequisites, are in `phase4-ledger.md`.

**Out of scope, permanently:** the area model, the board harness, the design-space sweep, and
Learnable Reduction. Each has a recorded reason in `tool-roadmap.md` §7 and Q1/Q7.
