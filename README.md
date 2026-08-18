# dwn2rtl

[![PyPI](https://img.shields.io/pypi/v/dwn2rtl.svg)](https://pypi.org/project/dwn2rtl/)
[![Python](https://img.shields.io/pypi/pyversions/dwn2rtl.svg)](https://pypi.org/project/dwn2rtl/)
[![CI](https://github.com/dwn2rtl/dwn2rtl/actions/workflows/ci.yml/badge.svg)](https://github.com/dwn2rtl/dwn2rtl/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](https://github.com/dwn2rtl/dwn2rtl/blob/main/LICENSE)

**A trained Differentiable Weightless Neural Network goes in. Synthesizable Verilog comes out —
with the testbenches to prove, in your own simulator, that it computes the same function.**

```bash
pip install dwn2rtl
```

```bash
dwn2rtl build model.pt --out rtl/
dwn2rtl verify rtl/
```

```
features 784, classes 10, layers [300], n=6, z=3   from checkpoint
integer bits 0                                     derived, exact
frac bits 8 -> Q0.8 signed (9-bit)                 INFERRED from the thresholds' grid, provably lossless

core      300 nodes, 3 cycles
encoder   720 comparators of 2352 thermometer bits
top       4 cycles latency, II=1

vectors   core 504, top 1227, 7/10 classes hit
wrote     rtl/ (18 files)
```
```
iverilog 12.0
  dwn_core  504 vectors  PASS
  dwn_top   1227 vectors  PASS
RESULT   PASS
```

That is a real MNIST model, not an illustration. No flags, no config file, no vendor toolchain.

---

## Try it in ten seconds

```bash
python examples/quickstart.py
```

One file, no dataset, no training: it builds a DWN, saves it, emits Verilog, and runs that Verilog
through a simulator. It is written to be read, and it shows exactly where your training script
plugs in.

> `build` needs nothing but Python. **`verify` needs a Verilog simulator**, which you install
> yourself: `winget install Icarus.Verilog` (Windows), `apt install iverilog` (Debian/Ubuntu), or
> `brew install icarus-verilog` (macOS).

## Saving your model

**A DWN is two objects.** The thermometer is fitted *before* training and is not a model
parameter, so one line at the end of your training script:

```python
torch.save({'model': model, 'thermometer': thermometer}, 'model.pt')
```

Add `'scaler': scaler` if your training scaled its features — the thresholds live in whatever
feature space training used.

> ⚠️ **`torch.save(model.state_dict())` silently loses the encoder.** It is what every PyTorch
> user reaches for, and it drops the thermometer entirely — which on the smallest studied model is
> most of the design. dwn2rtl refuses such a file by name rather than emitting something that
> synthesizes cleanly and classifies at chance.

## Precision: usually you type nothing

Everything structural is derived from the checkpoint, including the **integer** width. The one
number that cannot be derived in general is how many **fractional** bits to use, because whether
quantisation changes a prediction depends on your data. So the tool never asks how much precision
you *need* — it works out how much your input already *has*, and always says which happened:

```
frac bits 8 -> Q0.8 signed (9-bit)      INFERRED from the thresholds' grid, provably lossless
frac bits 8 -> Q0.8 signed (9-bit)      from --input-bits, provably lossless
frac bits 12 -> Q3.12 signed (16-bit)   DEFAULT for a continuous input, NOT measured
```

Thermometer thresholds are quantiles of the training data, so quantised data leaves a grid to
find. On MNIST the tool finds `k/255` and derives a 9-bit word with nothing typed. Pass
`--input-bits N` to state or override it.

## What `build` emits

A self-contained directory. Hand it to any simulator or synthesis tool — dwn2rtl is not needed
again.

```
rtl/
├── dwn_top.v                 <- INSTANTIATE THIS. encoder + core, one class per clock
│   ├── thermometer_encoder.v    features -> thermometer bits. always ships; often the larger half
│   └── dwn_core.v               the network itself: one node = one LUT6
│
├── lut_node.v                hand-written primitives, copied in so the directory stands alone
├── popcount.v
├── argmax.v
├── pipe_reg.v
│
├── dwn_top_params.vh         `DWN_TOP_LATENCY -- cycles from x_flat to class_idx
├── dwn_core_params.vh        widths and pipeline depth
│
├── input_scaling.json        ⚠️ only if trained on scaled features -- see below
│
├── x_quant.hex               golden vectors: quantized features in...
├── expected_top.hex          ...and the class the software model gives for each
├── x_binarized.hex           the same, one level down: pre-encoded bits
├── expected.hex
├── top_params.vh             vector counts, so the testbenches size themselves
├── vec_params.vh
└── tb/
    ├── dwn_top_tb.v          self-checking; what `dwn2rtl verify` runs
    └── dwn_core_tb.v
```

**For synthesis** you need the three design files, the four primitives, and the two `.vh`
parameter files. The `.hex` files and `tb/` are for verification only.

```verilog
dwn_top u_dwn (.clk(clk), .x_flat(features), .class_idx(prediction));
```

`x_flat` packs feature `f` at `[f*W +: W]` as a **signed** fixed-point integer. Throughput is one
classification per clock (II=1) at the fixed latency in `dwn_top_params.vh` — no handshake, no
backpressure. The board, clock, I/O and synthesis strategy are yours.

> ⚠️ **If `input_scaling.json` is emitted, whatever drives `x_flat` must apply
> `(x - mean) / scale` first.** Raw features give a design that runs at chance and looks entirely
> healthy doing it — it will still pass `verify`, because the testbench feeds it correctly-scaled
> vectors. See [the user guide](https://github.com/dwn2rtl/dwn2rtl/blob/main/docs/user-guide.md).

**The encoder always ships, and its cost is always reported separately from the network's.** It is
intrinsic to a DWN, not preprocessing you supply, and on the smallest model in the study below it
was **fourteen times** the network it feeds. Published DWN resource counts that omit it understate
designs by most of their cost.

## Verification is the whole point

Two testbenches, so a failure localizes itself: `dwn_core_tb` drives pre-binarized bits,
`dwn_top_tb` drives quantized features through the encoder as well. Core passing while top fails
means the encoder, and nothing else needs re-examining.

**And nothing that was not checked is reported as a pass.** A missing testbench, an empty one, a
compile error, or a simulation that did not finish is a failure, not a skip.

## Will it fit?

`dwn2rtl estimate rtl/` synthesizes the emitted design with
[yosys](https://github.com/YosysHQ/yosys) and reports LUTs and flops **per module**, so the
encoder's cost never disappears into a total. It is the one optional command — `build` and
`verify` need no synthesis tool, and `estimate` skips cleanly when yosys is absent.

```
  thermometer_encoder       717 LUT        0 FF
  dwn_core                  110 LUT        0 FF
  dwn_top                   833 LUT        0 FF
```

Generic mapping is not your vendor toolchain, and the report says so every time it prints: against
Vivado on an `xc7a35t` the core agreed exactly and the encoder came out 2.1× low. Treat core
numbers as indicative, encoder numbers as a floor, and synthesize for figures you publish —
[the calibration is in the user
guide](https://github.com/dwn2rtl/dwn2rtl/blob/main/docs/user-guide.md).

## Evidence

The generator here comes from the study repository
**[dwn-fpga-study](https://github.com/Kanishk234/dwn-fpga-study)**, where it built:

- **77 configurations** across two datasets, **every one bit-exact** against its software model
- **166,000 / 166,000** and **10,000 / 10,000** correct **on physical silicon**, one model per
  dataset
- the original authors' published area and accuracy, reproduced at matched convention

That study is the evidence this works; this repository is the tool. Numbers it recorded — both
datasets' fixed-point formats, comparator counts, and quantisation-merge counts — are reproduced
exactly by this code and pinned by its test suite.

## Documentation

| | |
|---|---|
| [`docs/user-guide.md`](https://github.com/dwn2rtl/dwn2rtl/blob/main/docs/user-guide.md) | **start here** — using it on your own model: saving, integrating the RTL, troubleshooting |
| [`docs/checkpoint-format.md`](https://github.com/dwn2rtl/dwn2rtl/blob/main/docs/checkpoint-format.md) | what the exporter reads, and why |
| [`docs/overview.md`](https://github.com/dwn2rtl/dwn2rtl/blob/main/docs/overview.md) | what the tool is, how it installs, the build plan |
| [`docs/tool-roadmap.md`](https://github.com/dwn2rtl/dwn2rtl/blob/main/docs/tool-roadmap.md) | the audited work list and the decisions behind it |

## Development

```bash
pip install -e ".[dev]"
pytest                  # everything
pytest -m sim           # the gate: emitted RTL through a simulator
```

**The gate is the rule**: emitted RTL is not correct until a simulator says it matches the golden
model on *every* vector. Not "looks right", and not "the emitter's own read-back passed" — the
study repo has a case where a read-back reported 20/20 correct while the design was wrong on 958
of 1,504 vectors. It runs in CI on Linux and Windows on every commit.

## Licence

MIT — see [LICENSE](https://github.com/dwn2rtl/dwn2rtl/blob/main/LICENSE).

Targets [Alan Bacellar's DWN implementation](https://github.com/alanbacellar/DWN).
