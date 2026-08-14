# dwn2rtl

**A trained Differentiable Weightless Neural Network goes in. Synthesizable Verilog comes out —
along with the means to prove, in your own simulator, that the Verilog computes the same function
as the model.**

```bash
pip install git+https://github.com/Krithik4/dwn2rtl.git
```

```bash
dwn2rtl build model.pt --out rtl/ --input-bits 8
dwn2rtl verify rtl/
```

```
features 784, classes 10, layers [300], n=6, z=3   from checkpoint
integer bits 0                                     derived, exact
frac bits 8 -> Q0.8 signed (9-bit)                 from --input-bits, provably lossless

core      300 nodes, 3 cycles
encoder   720 comparators of 2352 thermometer bits
top       4 cycles latency, II=1

vectors   core 504, top 1227, 7/10 classes hit
note      1169 of 2352 thresholds quantise to a duplicate comparison (1183 distinct)
WARNING   this model was trained on SCALED features. Whatever drives x_flat must apply
          (x - mean) / scale first, using input_scaling.json -- raw features give a design
          that runs at chance and looks healthy doing it.
wrote     rtl/ (18 files)
```
```
iverilog 12.0
  dwn_core  504 vectors  PASS
  dwn_top   1227 vectors  PASS
RESULT   PASS
```

That is a real MNIST model, not an illustration.

> Not on PyPI yet, hence the git install above.

---

## What it is

A **generator**, and nothing else. Verilog-2001 with no vendor primitives, no board harness, no
sweep automation, and no dependency on Vivado, Quartus or any licence — including for
verification, which is the part most tools make you take on trust.

| in scope | out of scope |
|---|---|
| checkpoint → truth tables, wiring, thresholds | the board harness — UART, vector store, FSM |
| the thermometer encoder, the LUT core, a top wiring them | design-space sweeps |
| a numpy golden model | training, tuning, or architecture choice |
| **self-checking testbenches and golden vectors** | area models — see [why](docs/tool-roadmap.md) |

**The encoder always ships, and its cost is always reported separately from the network's.** It
is intrinsic to a DWN, not preprocessing you supply, and on the smallest model in the study below
it was **fourteen times** the network it feeds. Published DWN resource counts that omit it
understate designs by most of their cost.

## Getting started

```bash
python examples/quickstart.py
```

One file, no dataset, no training, a few seconds: it builds a DWN, saves it, emits Verilog, and
runs that Verilog through a simulator. Read it — it is written to be read, and it shows exactly
where your training script plugs in.

### Verification needs a simulator

`build` needs nothing but Python. `verify` needs any Verilog simulator, which you install
yourself:

| | |
|---|---|
| Windows | `winget install Icarus.Verilog` (it does **not** add itself to PATH; dwn2rtl looks in `C:\iverilog\bin` anyway) |
| Debian/Ubuntu | `apt install iverilog` |
| macOS | `brew install icarus-verilog` |

## Saving a model

**A DWN is two objects.** The thermometer is fitted *before* training and is not a parameter of
the model, so it is not in the `state_dict`. One line at the end of your training script, plain
PyTorch, no `dwn2rtl` import required:

```python
torch.save({'model': model, 'thermometer': thermometer}, 'model.pt')
```

If your training scaled its features, include the scaler too — the thresholds live in whatever
feature space training used:

```python
torch.save({'model': model, 'thermometer': thermometer, 'scaler': scaler}, 'model.pt')
```

> ⚠️ **`torch.save(model.state_dict())` silently loses the encoder.** It is what every PyTorch
> user reaches for and it drops the thermometer entirely, which on the smallest studied model is
> most of the design. dwn2rtl refuses such a file by name rather than emitting something that
> synthesizes cleanly and classifies at chance.

Upstream DWN saves nothing at all, so dwn2rtl defines the format and owns both ends of it. It
accepts the plain dict above, a live model via `dwn2rtl.from_model(model, thermometer)`, or
`dwn2rtl.save(model, thermometer, path)` — which validates while the objects are still in memory,
so a mistake surfaces then rather than on another machine weeks later.

## `--input-bits`, and the question the tool refuses to ask

Almost everything is derived from the checkpoint: features, classes, layers, `n`, `z`, the wiring,
the table contents, and the **integer** width — which follows exactly from the thresholds.

Exactly one thing cannot be: how many **fractional** bits are safe. That depends on whether
quantisation changes predictions, which depends on your data. So the tool never asks for
fractional bits. It asks `--input-bits`: *the precision of your input*, which you know.

When your input has a native quantum — 8-bit pixels, most ADCs, anything already digital —
`frac = input_bits` is **provably** lossless, not merely measured. Pixels are `k/255`, and
quantising at `frac=8` computes `floor(k·256/255)`, strictly increasing over `k = 0…255`. Order
is preserved exactly, and every encoder bit is an order comparison, so no bit can change.

Omit it and you get a documented default that is reported as **a default, not a measurement**.

## What `build` emits

```
dwn_core.v  thermometer_encoder.v  dwn_top.v     the design
lut_node.v  popcount.v  argmax.v  pipe_reg.v     hand-written primitives, copied in
dwn_core_params.vh  dwn_top_params.vh            widths and pipeline depth
vec_params.vh  top_params.vh                     vector counts for the testbenches
x_binarized.hex  expected.hex                    core-level golden vectors
x_quant.hex  expected_top.hex                    top-level golden vectors
input_scaling.json                               only if your model was trained on scaled features
tb/dwn_core_tb.v  tb/dwn_top_tb.v                self-checking testbenches
```

Self-contained — hand the directory to any simulator or synthesis tool. Then instantiate it:

```verilog
dwn_top u_dwn (.clk(clk), .x_flat(features), .class_idx(prediction));
```

Latency in cycles is in `dwn_top_params.vh`; throughput is one classification per clock (II=1).
The board, clock, I/O and synthesis strategy are yours.

## Why two testbenches

`dwn_core_tb` drives pre-binarized bits; `dwn_top_tb` drives quantized features through the
encoder as well. The split makes a failure **localize itself**:

| | |
|---|---|
| core PASS, top FAIL | the encoder — nothing else needs re-examining |
| core FAIL, top FAIL | the network; fix that and this follows |

`verify` says so out loud when it happens.

**And nothing that was not checked is reported as a pass.** A missing testbench, an empty one, a
compile error, a simulation that did not finish, or a directory with no runnable levels at all —
each is a failure, not a skip.

## Evidence

The generator in this repository comes from the study repository
**[dwn-fpga-study](https://github.com/Kanishk234/dwn-fpga-study)**, where it was used to build:

- **77 configurations** across two datasets, **every one bit-exact** against its software model
- **166,000 / 166,000** and **10,000 / 10,000** correct **on physical silicon**, one model per
  dataset
- the original authors' published area and accuracy, reproduced at matched convention

That study is the evidence this works; this repository is the tool. Numbers it recorded — both
datasets' fixed-point formats, both comparator counts, both quantisation-merge counts — are
reproduced exactly by this code and pinned by its test suite.

## Development

```bash
pip install -e ".[dev]"
pytest                  # everything
pytest -m sim           # the gate: emitted RTL through a simulator
```

**The gate is the rule**: emitted RTL is not correct until a simulator
says it matches the golden model on *every* vector. Not "looks right", and not "the emitter's
own read-back passed" — the study repo has a case where an emitter's read-back reported 20/20
correct while the design was wrong on 958 of 1,504 vectors. It runs in CI on Linux and Windows
on every commit.

| | |
|---|---|
| [`docs/overview.md`](docs/overview.md) | what the tool is, how it installs, the build plan |
| [`docs/checkpoint-format.md`](docs/checkpoint-format.md) | what the exporter reads, and why |
| [`docs/tool-roadmap.md`](docs/tool-roadmap.md) | the audited work list and the decisions behind it |
| `docs/phaseN-ledger.md`, `docs/phaseN-report.md` | how it was built, including the wrong turns |

## Licence

MIT — see [LICENSE](LICENSE).

Targets [Alan Bacellar's DWN implementation](https://github.com/alanbacellar/DWN).
