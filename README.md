# dwn2rtl

[![PyPI](https://img.shields.io/pypi/v/dwn2rtl.svg)](https://pypi.org/project/dwn2rtl/)
[![Python](https://img.shields.io/pypi/pyversions/dwn2rtl.svg)](https://pypi.org/project/dwn2rtl/)
[![CI](https://github.com/dwn2rtl/dwn2rtl/actions/workflows/ci.yml/badge.svg)](https://github.com/dwn2rtl/dwn2rtl/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](https://github.com/dwn2rtl/dwn2rtl/blob/main/LICENSE)

**Turn a trained Differentiable Weightless Neural Network into synthesizable Verilog for FPGAs —
and prove, in your own simulator, that the Verilog computes exactly what your model does.**

A DWN is a neural network whose learned parameters *are* lookup tables: one neuron becomes one
LUT6, so the network is logic rather than arithmetic. That makes it unusually well suited to an
FPGA, and `dwn2rtl` is the generator that gets it there — plus the testbenches that prove the
translation was faithful.

## Installation

```bash
pip install dwn2rtl
```

Python 3.10+, on Linux, macOS and Windows. Pulls in numpy and torch.

## Requirements

| command | needs |
|---|---|
| `dwn2rtl build` | nothing but Python |
| `dwn2rtl verify` | a Verilog simulator — `winget install Icarus.Verilog` (Windows), `apt install iverilog` (Debian/Ubuntu), `brew install icarus-verilog` (macOS). [Verilator](https://verilator.org) also works on Linux and macOS: `dwn2rtl verify rtl/ --simulator verilator` |
| `dwn2rtl estimate` | [yosys](https://github.com/YosysHQ/yosys), and it is optional |

No vendor toolchain appears in that list. Nothing here needs Vivado, Quartus or a licence —
including the verification, which is the part most tools ask you to take on trust.

## Quickstart

```bash
dwn2rtl build model.pt --out rtl/     # checkpoint -> Verilog + golden vectors
dwn2rtl verify rtl/                   # compile and run it; must print PASS
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

That is a real MNIST model, not an illustration — no flags, no config file, no vendor toolchain.
Before it was packaged as a tool, this generator built **77 configurations across two datasets,
every one bit-exact** against its software model, two of them verified on physical silicon.

**No model yet?**
[`examples/quickstart.py`](https://github.com/dwn2rtl/dwn2rtl/blob/main/examples/quickstart.py)
needs no dataset and no training: it builds a DWN, saves it, emits Verilog, and simulates it. It
is written to be read, and it shows where your own training script plugs in. It ships in the
repository rather than the package:

```bash
curl -O https://raw.githubusercontent.com/dwn2rtl/dwn2rtl/main/examples/quickstart.py
python quickstart.py
```

## From Python

The CLI and the library are the same installed code, so anything `build` does you can do inline —
useful while the model is still in memory:

```python
import dwn2rtl

dwn2rtl.save(model, thermometer, 'model.pt', scaler=scaler)   # validates while you can still fix it
report = dwn2rtl.build('model.pt', 'rtl/')
result = dwn2rtl.verify('rtl/')
```

`import dwn2rtl` does not import torch — every emitter and the golden model are pure numpy.

## Using it on your own model

**A DWN is two objects.** The thermometer is fitted *before* training and is not a model
parameter, so one line at the end of your training script:

```python
torch.save({'model': model, 'thermometer': thermometer, 'scaler': scaler}, 'model.pt')
```

> ⚠️ **`torch.save(model.state_dict())` silently loses the encoder.** It is what every PyTorch
> user reaches for, and it drops the thermometer entirely — which on the smallest model we have
> measured is most of the design. dwn2rtl refuses such a file by name rather than emitting
> something that synthesizes cleanly and classifies at chance.

**You almost never specify precision.** Everything structural is derived from the checkpoint, and
rather than asking how much precision you *need* — unanswerable without your data — the tool works
out how much your input already *has*, then reports whether it inferred, was told, or fell back to
a default. `--input-bits N` overrides it.

The [user guide](https://github.com/dwn2rtl/dwn2rtl/blob/main/docs/user-guide.md) covers all of
this properly: every checkpoint shape accepted, the three precision cases, and what to do when
there is no grid to find.

## What you get

A self-contained directory. Hand it to any simulator or synthesis tool — dwn2rtl is not needed
again. Every file is a sibling; the indentation below shows what instantiates what.

```
rtl/
  dwn_top.v              <- INSTANTIATE THIS; encoder + core, one classification per clock
      thermometer_encoder.v   features -> thermometer bits. Always ships, often the larger half
      dwn_core.v              the network itself: one node = one LUT6

  lut_node.v  popcount.v  argmax.v  pipe_reg.v  primitives, copied in so the directory stands alone
  dwn_top_params.vh  dwn_core_params.vh         latency, widths, pipeline depth

  input_scaling.json    emitted only if the model was trained on scaled features -- see below
  x_quant.hex  expected_top.hex    golden vectors: quantized features in, expected class out
  x_binarized.hex  expected.hex    the same one level down, on pre-encoded bits
  top_params.vh  vec_params.vh     vector counts, so the testbenches size themselves
  tb/                              self-checking testbenches; what `dwn2rtl verify` runs
```

**For synthesis** you need the three design files, the four primitives, and the two `.vh`
parameter files. The `.hex` files and `tb/` are for verification only.

```verilog
dwn_top u_dwn (.clk(clk), .x_flat(features), .class_idx(prediction));
```

One classification per clock (II=1), at the fixed latency in `dwn_top_params.vh` — no handshake,
no backpressure. The board, clock, I/O and synthesis strategy are yours.

> ⚠️ **If `input_scaling.json` is emitted, whatever drives `x_flat` must apply
> `(x - mean) / scale` first.** Raw features give a design that runs at chance and looks entirely
> healthy doing it — and it will still pass `verify`, because the testbench feeds it
> correctly-scaled vectors. This is the likeliest way to build a working-looking, useless design.

How to pack `x_flat`, how wide `class_idx` is, and the rest of the integration contract are in the
[user guide](https://github.com/dwn2rtl/dwn2rtl/blob/main/docs/user-guide.md).

## Verifying and sizing

**Emitted RTL is not correct until a simulator says it matches the golden model on every vector.**
`build` writes two testbenches so a failure localizes itself — core passing while top fails means
the encoder, and nothing else. Nothing unchecked is ever reported as a pass: a missing testbench,
a compile error or a simulation that did not finish is a failure, not a skip.

`dwn2rtl estimate rtl/` synthesizes with yosys and reports **per module**, so the encoder's cost
never disappears into a total:

```
  thermometer_encoder       717 LUT        0 FF
  dwn_core                  110 LUT        0 FF
  dwn_top                   833 LUT        0 FF
```

Generic mapping is not your vendor toolchain, and the report says so every time it prints:
calibrated against Vivado, the core agreed exactly while the encoder came out 2.1× low. Treat core
numbers as indicative, encoder numbers as a floor, and synthesize for figures you publish — the
calibration table is in the
[user guide](https://github.com/dwn2rtl/dwn2rtl/blob/main/docs/user-guide.md).

**The encoder always ships, and its cost is always reported separately.** It is intrinsic to a
DWN, not preprocessing you supply, and on the smallest model we have measured it was **fourteen
times** the network it feeds. Published DWN resource counts that omit it understate designs by
most of their cost.

## Documentation

| | |
|---|---|
| [`docs/user-guide.md`](https://github.com/dwn2rtl/dwn2rtl/blob/main/docs/user-guide.md) | **start here** — using it on your own model: saving, precision, integrating the RTL, troubleshooting |
| [`docs/checkpoint-format.md`](https://github.com/dwn2rtl/dwn2rtl/blob/main/docs/checkpoint-format.md) | what the exporter reads, and why |
| [`docs/overview.md`](https://github.com/dwn2rtl/dwn2rtl/blob/main/docs/overview.md) | what the tool is, how it installs, the build plan |

## Evidence

Before it was packaged as a tool, this generator ran a full FPGA study:

- **77 configurations** across two datasets, **every one bit-exact** against its software model
- **166,000 / 166,000** and **10,000 / 10,000** correct **on physical silicon**, one per dataset
- the original authors' published area and accuracy, reproduced at matched convention

Every number that study recorded — both fixed-point formats, comparator counts, quantisation-merge
counts — is reproduced by the code here and pinned by its test suite, so the claim is checkable
from this repository rather than taken on trust.

## Citation

If `dwn2rtl` is useful in published work, please cite the tool and the paper it targets.

**The tool:**

```bibtex
@software{dwn2rtl,
  title  = {dwn2rtl: synthesizable Verilog from Differentiable Weightless Neural Networks},
  author = {Sama, Krithik and Sama, Kanishk},
  year   = {2026},
  url    = {https://github.com/dwn2rtl/dwn2rtl}
}
```

**The DWN paper**, whose [reference implementation](https://github.com/alanbacellar/DWN) this tool
reads checkpoints from:

```bibtex
@InProceedings{pmlr-v235-bacellar24a,
  title     = {Differentiable Weightless Neural Networks},
  author    = {Bacellar, Alan Tendler Leibel and Susskind, Zachary and
               Breternitz Jr, Mauricio and John, Eugene and John, Lizy Kurian and
               Lima, Priscila Machado Vieira and Fran{\c{c}}a, Felipe M.G.},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  pages     = {2277--2295},
  year      = {2024},
  volume    = {235},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR}
}
```

**The measurements** cited under Evidence — the 77 configurations, the silicon runs and the Vivado
figures — were produced in
[dwn-fpga-study](https://github.com/Kanishk234/dwn-fpga-study), which is where to look for how
they were obtained.

## Development

```bash
pip install -e ".[dev]"
pytest                  # everything
pytest -m sim           # the gate: emitted RTL through a simulator
```

**The gate is the rule**, and it is not "looks right" or "the emitter's own read-back passed" — we
have a recorded case where a read-back reported 20/20 correct while the design was wrong on 958 of
1,504 vectors. It runs in CI on Linux and Windows, under two independent simulators, every commit.

`docs/phaseN-ledger.md` and `phaseN-report.md` record how the tool was built, including the wrong
turns; [`docs/tool-roadmap.md`](https://github.com/dwn2rtl/dwn2rtl/blob/main/docs/tool-roadmap.md)
is the archived work list that shaped it. Both are history, not instructions.

## Licence

MIT — see [LICENSE](https://github.com/dwn2rtl/dwn2rtl/blob/main/LICENSE).
