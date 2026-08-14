# dwn2rtl — what it is, how you install it, how you use it

**Read this first if you have never used the tool.** `tool-handoff.md` is the cold-start briefing
for someone building it; `tool-roadmap.md` is the audited work list. This file is the plain
description: what the thing is, what it does to what, and how it lands on a computer.

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
dwn2rtl build model.pt --out rtl/ --input-bits 8
dwn2rtl verify rtl/
```

You have a trained model sitting in a file. You want Verilog in a directory. There is no reason
to write a Python script to ask for that, and no reason to learn an API to get it.

**The library API is for the notebook case** — you have just finished training, the model is a
live object in memory, and you want to look at the hardware before you commit to it:

```python
import dwn2rtl

report = dwn2rtl.from_model(model, thermometer).build("rtl/", input_bits=8)
print(report.comparators, report.nodes)      # encoder vs core, reported separately
```

This is what hls4ml's users mostly do, and it is worth supporting for the same reason: at the
moment training finishes, a file on disk is a detour.

Neither is a wrapper around a subprocess. They are two entry points into the same functions.

---

## 3. Installing it

### The package

Once it is published:

```
pip install dwn2rtl
```

Until then, from source:

```
pip install git+https://github.com/<user>/dwn2rtl.git
```

or, to hack on it:

```
git clone https://github.com/<user>/dwn2rtl.git
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
$ dwn2rtl build model.pt --out rtl/ --input-bits 8

features 784, classes 10, layers [300], n=6, z=3      from checkpoint
integer bits 0                                         derived, exact
frac bits 8 -> Q0.8 signed, 9-bit word                 from --input-bits, provably lossless
1169 of 2352 thresholds merge at this width            warning
core     300 nodes                                     
encoder  720 comparators of 2352 thermometer bits      reported separately, always
latency  4 cycles, II=1
wrote rtl/ (9 files)

$ dwn2rtl verify rtl/

iverilog 12.0 found
  dwn_core  504 vectors   PASS
  dwn_top   1227 vectors  PASS
```

### What lands in `rtl/`

```
dwn_core.v  thermometer_encoder.v  dwn_top.v      the design
lut_node.v  popcount.v  argmax.v  pipe_reg.v      hand-written primitives, copied in
dwn_core_params.vh  dwn_top_params.vh             widths and pipeline depth
vec_params.vh  top_params.vh                      vector counts for the testbenches
x_binarized.hex  expected.hex                     core-level golden vectors
x_quant.hex  expected_top.hex                     top-level golden vectors
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

**Phase 0 — make it runnable.** `pyproject.toml` with `rtl/` as package data and the
`[project.scripts]` entry point, a venv, `iverilog` installed.

**Phase 1 — close the loop.** This is the milestone; everything before it is scaffolding.

1. `checkpoint.py` — sniff the input shape, normalize it, fail loudly by name on a bare
   `state_dict`.
2. A synthetic checkpoint generator for tests (small: n=2, 4 features, 2 classes), covering both
   the learnable and the fixed mapping paths.
3. `__init__.py` / `build()` — wires `build_core` -> `build_encoder` -> `generate`, in that
   order. The encoder reads the core's real pipeline depth out of `dwn_core_params.vh` rather
   than being told it twice.
4. Write `tb/dwn_top_tb.v`. **It is currently an empty file** — half the gate is missing.
5. **`iverilog` green end to end, both levels.** Expect friction: vendor-neutrality is so far an
   *inspection* result, never tested under a non-Xilinx simulator, and `argmax.v`'s 2-D wire
   arrays inside `generate` are the likely first casualty.

**Phase 2 — make it a tool.** `verify.py` (find a simulator, compile, run, parse PASS/FAIL,
ASCII-only output), `cli.py` (`build | verify | estimate`), unit tests plus a `sim`-marked gate
test, CI on Linux and Windows both.

**Phase 3 — make it usable by someone else.** README, a worked example, a LICENCE (nothing can
be used or cited without one), pin the upstream DWN commit and re-read `checkpoint-format.md`
against it, `estimate` via yosys, and clean the study-repo doc references out of the `rtl/*.v`
headers.

**Out of scope, permanently:** the area model, the board harness, the design-space sweep, and
Learnable Reduction. Each has a recorded reason in `tool-roadmap.md` §7 and Q1/Q7.
