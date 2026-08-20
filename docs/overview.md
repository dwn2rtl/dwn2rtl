# dwn2rtl — what it is, how it installs, how it was built

Read this if you've never used the tool and want the plain description: what it is, what it does
to what, and how it gets onto a computer. **This one is current.**

The two plans next to it are history and say so at the top: `tool-handoff.md` was the briefing
that started the project, and `tool-roadmap.md` was the work list for building what now exists.
They're kept for the reasoning in them, not the tasks. Live work is §6 below and the phase
ledgers.

---

## 1. In one paragraph

`dwn2rtl` takes a **trained** Differentiable Weightless Neural Network and emits synthesizable
Verilog-2001 that computes the same function — the thermometer encoder, the LUT network, and a
thin top-level wiring the two together — plus self-checking testbenches and the answers your model
gives, so you can prove in your own simulator that the hardware matches the software. It's a
translator. It doesn't train anything, doesn't pick an architecture, doesn't target a board, and
doesn't need a vendor toolchain.

---

## 2. One package, two ways in

This is the thing that confuses people most, so it gets its own section.

`dwn2rtl` is **a Python package that's also a terminal command.** Not two products, not two
installs — one `pip install`, and afterwards you have both. Completely standard: `pytest`,
`black`, `ruff`, `jupyter` and `hls4ml` all work this way.

The mechanism is four lines in `pyproject.toml`:

```toml
[project.scripts]
dwn2rtl = "dwn2rtl.cli:main"
```

When pip installs the package it sees that block and drops a small launcher named `dwn2rtl` into
the same `Scripts/` (Windows) or `bin/` (Linux, macOS) directory pip itself lives in — already on
your PATH. That launcher does nothing except import `dwn2rtl.cli` and call `main()`. So:

| you type | what runs |
|---|---|
| `import dwn2rtl` in a script | the package, directly |
| `dwn2rtl build ...` in a terminal | the launcher, which imports the same package and calls `cli.main()` |

**Same code, same install, same version.** The CLI is a thin layer of argument parsing over the
library, so anything it can do you can do from Python — because it *is* the Python.

### Which should you use?

**The CLI, most of the time:**

```
dwn2rtl build model.pt --out rtl/
dwn2rtl verify rtl/
```

You have a trained model in a file and you want Verilog in a directory. There's no reason to write
a script to ask for that, and no reason to learn an API to get it.

**The library, when you're in a notebook** — training has just finished, the model is a live object
in memory, and you want to look at the hardware before committing to it:

```python
import dwn2rtl

report = dwn2rtl.build(dwn2rtl.from_model(model, thermometer), "rtl/")
print(report.comparators, report.nodes)      # encoder and network, counted separately
```

That's what most hls4ml users do, and it's worth supporting for the same reason: right after
training finishes, a file on disk is a detour.

Neither one wraps a subprocess. They're two doors into the same functions.

---

## 3. Installing it

### The package

```
pip install dwn2rtl
```

or, to work on it:

```
git clone https://github.com/dwn2rtl/dwn2rtl.git
cd dwn2rtl
pip install -e ".[dev]"
```

Pure Python — no compiler, no build step, no platform-specific wheel. The hand-written Verilog in
`src/dwn2rtl/rtl/` ships as package data, so it installs alongside the code and the tool can find
it without you cloning anything.

It needs numpy for everything, and torch only for reading checkpoints — the software model and
every emitter are pure numpy.

### The simulator, which is not a pip dependency

`dwn2rtl build` needs nothing but Python. `dwn2rtl verify` needs a Verilog simulator, and that's a
separate program you install yourself:

- **Icarus Verilog** (`iverilog`) is the default. It ships as a prebuilt Windows binary in the
  [OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build), and it's `apt install iverilog`
  or `brew install icarus-verilog` everywhere else.
- **Verilator** works on Linux and macOS: `dwn2rtl verify rtl/ --simulator verilator`. iverilog
  stays the default because it's far faster here — 0.38 s against 14.7 s end to end, since
  Verilator compiles to C++ before it runs anything.
- **yosys**, optional, only for `dwn2rtl estimate`.

`verify` searches your PATH and tells you what it found. None of this needs a vendor licence,
which is the point — checking you can't run yourself is just a claim you have to believe.

---

## 4. The whole flow

### While training — one line, at the end of your script

A DWN is **two objects**: the model, and the `DistributiveThermometer` that was fitted *before*
training. The thermometer's thresholds aren't model parameters and aren't in the `state_dict`, so
you have to save them deliberately. Plain PyTorch, no dwn2rtl import needed:

```python
torch.save({'model': model, 'thermometer': thermometer}, 'model.pt')
```

That single line is the only thing this tool asks of your training code.

> **`torch.save(model.state_dict())` is the trap.** It quietly drops the encoder, which on the
> smallest model we've measured is *fourteen times* the size of the network it feeds. dwn2rtl
> refuses that file and tells you what's missing, rather than emitting a design that synthesizes
> cleanly and runs at chance.

### Afterwards — in a terminal

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
lut_node.v  popcount.v  argmax.v  pipe_reg.v      hand-written building blocks, copied in
dwn_core_params.vh  dwn_top_params.vh             widths and pipeline depth
vec_params.vh  top_params.vh                      how many test vectors there are
x_binarized.hex  expected.hex                     test vectors for the network alone
x_quant.hex  expected_top.hex                     test vectors for the whole design
input_scaling.json                                only if you trained on scaled features
tb/dwn_core_tb.v  tb/dwn_top_tb.v                 the self-checking testbenches
```

Self-contained — hand the directory to any simulator or synthesis tool.

### Then it's yours

Instantiate `dwn_top` in whatever harness your application needs:

```verilog
dwn_top u_dwn (.clk(clk), .x_flat(features), .class_idx(prediction));
```

Latency in cycles is in `dwn_top_params.vh`, and throughput is one classification per clock. The
board, the clock, the I/O and the synthesis strategy are yours — deliberately, and `tool-roadmap.md`
Q1 records why.

---

## 5. Where the line is

| it does | it never |
|---|---|
| read a trained model | train, tune, or choose an architecture |
| work out integer width exactly from your thresholds | guess fractional bits, or make you supply them |
| emit the encoder, the network and a top level | emit a board harness, a UART, or a vector store |
| generate test vectors from the *model* | need your dataset |
| report encoder and network cost separately | ship an area model, or need a vendor tool |

The only thing this project generates structurally is a **test fixture** — a synthetic checkpoint
of a chosen shape, used in CI to exercise the emitter at shapes we have no real model for. It's
test-only and never on a user's path, and it earns its keep: it caught emitter bugs at MNIST's
shape before an MNIST model existed.

---

## 6. How it was built

Ordered so that a simulator confirming the output comes as early as possible, because until that
happens nothing is actually known.

**Each phase leaves two documents behind, and they do different jobs:**

| | `docs/phaseN-ledger.md` | `docs/phaseN-report.md` |
|---|---|---|
| written | *during*, as things happen | *at the end*, once |
| shape | dated entries: built / hit / decided | a retrospective |
| for | whoever's working, and future-you asking "why is it like this?" | someone who wasn't there |
| wrong turns | kept, struck through, with the reason | only where they changed a later decision |

The ledger is the evidence and the report is the argument. Write only the report and you lose the
reasoning; write only the ledger and anyone catching up has to read a diary.

| phase | | status |
|---|---|---|
| **0** | **make it runnable** — `pyproject.toml`, package data, the CLI entry point, a venv, a simulator | ✅ closed |
| **1** | **close the loop** — `checkpoint.py`, synthetic fixtures, `build()`, `dwn_top_tb.v`, `verify.py`, and both levels passing in a simulator | ✅ closed |
| **2** | **make it a tool** — `conftest.py`, CI on Linux and Windows, every commit | ✅ closed |
| **3** | **make it usable by someone else** — README, a worked example, the LICENCE, the upstream pin, and getting outside citations out of the shipped `rtl/*.v` | ✅ closed |
| **4** | **measure, then optimize** — `estimate` via yosys, then changes justified by measurement | ✅ closed |
| **5** | **publish it** — a GitHub org, packaging metadata, a Trusted Publishing workflow, a TestPyPI rehearsal, and `0.1.0` on PyPI | ✅ closed |
| **6** | **make it legible, and get a second opinion** — a user guide, comments cut back to the traps, and Verilator in CI as a linter and a second simulator | ✅ closed |
| **7** | **harden it, then ship the fixes** — adversarial audits, coverage and mutation testing, macOS and dependency floors in CI, and `0.2.0` on PyPI | ✅ closed |
| **8** | **audit the seams the last audit left** — re-attack on the axes phase 7 §3.1 names, and correct two earlier headlines that claimed more ground than they measured | ✅ closed |

**Phase 1 is the milestone.** Everything before it is scaffolding and everything after is
packaging: a trained DWN goes in, and a simulator confirms the Verilog gives identical answers to
the software model.

### Phase 4, and the rule it ran on

**An optimization doesn't land because someone argued for it. It lands on a measurement, or it
doesn't land.** That's this project's own scar rather than a borrowed principle: an earlier area
model existed to filter out configurations too large to synthesize, and across two complete
studies it filtered out **none** (`tool-roadmap.md` Q7). So `estimate` had to come *before* any
change to the RTL — changing the emitter before you can tell whether it helped is the same mistake
wearing a different hat.

⚠️ **The thing that could have made the phase actively harmful:** yosys isn't what users
synthesize with. Tuning the emitter until *yosys* numbers improved would have been optimizing
against the wrong target. So there were two guardrails — compare against real Vivado figures
before changing anything, and justify every change **structurally** first, using measurement only
to confirm it cost nothing.

**Both candidate optimizations were then cancelled by measurement.** Narrowing the encoder
register would have saved nothing, because synthesis already trims below the driven bit count. An
explicit popcount tree would have saved nothing either, because the depth already tracks log2 of
the width. Two measurements instead of two rewrites of shipped RTL.

The phase also made precision **inferred from the thresholds' grid**, so `dwn2rtl build model.pt`
needs no `--input-bits` for quantised inputs and derives a 9-bit word for MNIST where it used to
fall back to something wider.

⚠️ The calibration was the most useful thing to come out of it: yosys matches Vivado **exactly**
for the LUT network and comes out **2.1× low** for the encoder, so `estimate` prints that caveat
with every report instead of letting one number borrow the other's credibility.

The details, including what was deferred, are in `phase4-ledger.md`.

**Permanently out of scope:** the area model, the board harness, the design-space sweep, and
Learnable Reduction. Each has a reason on the record in `tool-roadmap.md` §7 and Q1/Q7.
