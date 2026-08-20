# dwn2rtl

[![PyPI](https://img.shields.io/pypi/v/dwn2rtl.svg)](https://pypi.org/project/dwn2rtl/)
[![Python](https://img.shields.io/pypi/pyversions/dwn2rtl.svg)](https://pypi.org/project/dwn2rtl/)
[![CI](https://github.com/dwn2rtl/dwn2rtl/actions/workflows/ci.yml/badge.svg)](https://github.com/dwn2rtl/dwn2rtl/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](https://github.com/dwn2rtl/dwn2rtl/blob/main/LICENSE)

**Turn a trained Differentiable Weightless Neural Network into Verilog you can put on an FPGA —
and check, in your own simulator, that the Verilog gives the same answers your model does.**

A DWN is a neural network whose learned parameters *are* lookup tables. One neuron becomes one
LUT6, so the network is logic rather than arithmetic — which makes it a very good fit for an FPGA.
`dwn2rtl` does the translation, and gives you the testbenches to confirm it came out right.

## Installation

```bash
pip install dwn2rtl
```

Python 3.10+, on Linux, macOS and Windows. Pulls in numpy and torch.

## Requirements

| command | what it needs |
|---|---|
| `dwn2rtl build` | nothing but Python |
| `dwn2rtl verify` | a Verilog simulator — `winget install Icarus.Verilog` (Windows), `apt install iverilog` (Debian/Ubuntu), `brew install icarus-verilog` (macOS). [Verilator](https://verilator.org) works too on Linux and macOS: `dwn2rtl verify rtl/ --simulator verilator` |
| `dwn2rtl estimate` | [yosys](https://github.com/YosysHQ/yosys), and it's optional |

You don't need Vivado, Quartus, or a licence for any of this — including the checking step, which
is the part most tools ask you to take their word for.

## Quickstart

```bash
dwn2rtl build model.pt --out rtl/     # your checkpoint -> Verilog + test vectors
dwn2rtl verify rtl/                   # compile and run it
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

That's a real MNIST model, with no flags, no config file and no vendor toolchain. Before this was
packaged as a tool it built 77 different configurations across two datasets, and every one of them
gave identical answers to its software model. Two were confirmed on real hardware.

**Don't have a model handy?**
[`examples/quickstart.py`](https://github.com/dwn2rtl/dwn2rtl/blob/main/examples/quickstart.py)
needs no dataset and no training. It builds a small DWN, saves it, emits the Verilog and simulates
it — and it's written to be read, so you can see where your own training script would slot in.

```bash
curl -O https://raw.githubusercontent.com/dwn2rtl/dwn2rtl/main/examples/quickstart.py
python quickstart.py
```

## From Python

The CLI and the library are the same code, so anything `build` does you can do inline. Handy when
the model is still in memory:

```python
import dwn2rtl

dwn2rtl.save(model, thermometer, 'model.pt', scaler=scaler)   # checks it while you can still fix it
report = dwn2rtl.build('model.pt', 'rtl/')
result = dwn2rtl.verify('rtl/')
```

`import dwn2rtl` doesn't import torch — the emitters and the software model are pure numpy.

## Using it on your own model

**A DWN is two objects.** The thermometer encoder is fitted *before* training and isn't part of
the model, so you have to save both. One line at the end of your training script:

```python
torch.save({'model': model, 'thermometer': thermometer, 'scaler': scaler}, 'model.pt')
```

> ⚠️ **`torch.save(model.state_dict())` throws away the encoder.** It's what everyone reaches for,
> and on the smallest model we've measured the encoder is most of the design. dwn2rtl refuses that
> file and tells you what's missing, rather than building something that synthesizes fine and
> classifies at random.

**You almost never have to specify precision.** Everything structural comes out of the checkpoint.
Rather than asking how much precision you *need* — which nobody can answer without your data — the
tool works out how much your input already *has*, and tells you whether it figured that out, was
told, or fell back to a default. Pass `--input-bits N` to override it.

The [user guide](https://github.com/dwn2rtl/dwn2rtl/blob/main/docs/user-guide.md) goes through all
of this properly: every checkpoint shape it accepts, how precision is decided, and what to do when
there's nothing to infer from.

## What you get

A directory that stands on its own. Hand it to any simulator or synthesis tool — you don't need
dwn2rtl again. The files are all siblings; the indentation shows what instantiates what.

```
rtl/
  dwn_top.v              <- THIS IS THE ONE YOU INSTANTIATE; encoder + network
      thermometer_encoder.v   turns features into thermometer bits
      dwn_core.v              the network itself: one node = one LUT6

  lut_node.v  popcount.v  argmax.v  pipe_reg.v  building blocks, copied in
  dwn_top_params.vh  dwn_core_params.vh         latency, widths, pipeline depth

  input_scaling.json    only if your model was trained on scaled features -- see below
  x_quant.hex  expected_top.hex    test inputs, and the answers your model gives for them
  x_binarized.hex  expected.hex    the same thing one level down, on pre-encoded bits
  top_params.vh  vec_params.vh     how many test vectors there are
  tb/                              the testbenches `dwn2rtl verify` runs
```

**To synthesize**, you need the three design files, the four building blocks, and the two `.vh`
files. The `.hex` files and `tb/` are only for checking.

```verilog
dwn_top u_dwn (.clk(clk), .x_flat(features), .class_idx(prediction));
```

One classification per clock, at the fixed latency written into `dwn_top_params.vh`. No handshake,
no backpressure. The board, clock, I/O and synthesis strategy are yours.

> ⚠️ **If you got an `input_scaling.json`, whatever drives `x_flat` has to apply
> `(x - mean) / scale` first.** Feed it raw features and you get a design that runs at chance and
> looks perfectly healthy doing it. It'll even pass `dwn2rtl verify`, because the testbench feeds
> it correctly scaled inputs. This is the easiest way to end up with something that looks like it
> works and doesn't.

How to pack `x_flat`, how wide `class_idx` is, and the rest of the wiring details are in the
[user guide](https://github.com/dwn2rtl/dwn2rtl/blob/main/docs/user-guide.md).

## Checking it, and sizing it

`dwn2rtl verify rtl/` compiles the design and runs it against every test vector. You get two
results — one for the network on its own, one for the whole thing — so if something's wrong you
know which half to look at. A missing testbench, a compile error or a simulation that didn't
finish all count as failures, not as "skipped".

`dwn2rtl estimate rtl/` runs yosys and reports **per module**, so the encoder never disappears
into a total:

```
  thermometer_encoder       717 LUT        0 FF
  dwn_core                  110 LUT        0 FF
  dwn_top                   833 LUT        0 FF
```

yosys isn't your vendor's toolchain, and the report says so every time it prints. Checked against
real Vivado numbers, the network matched exactly and the encoder came out 2.1× low — so treat the
network figure as indicative, the encoder figure as a floor, and synthesize properly for anything
you publish. The comparison table is in the
[user guide](https://github.com/dwn2rtl/dwn2rtl/blob/main/docs/user-guide.md).

**The encoder is always built, and always reported on its own line.** It's part of the network
rather than preprocessing you supply, and it's often the bigger half — on the smallest model we've
measured, fourteen times the size of the network it feeds.

## Documentation

| | |
|---|---|
| [`docs/user-guide.md`](https://github.com/dwn2rtl/dwn2rtl/blob/main/docs/user-guide.md) | **start here** — saving your model, precision, wiring the RTL into a design, troubleshooting |
| [`docs/checkpoint-format.md`](https://github.com/dwn2rtl/dwn2rtl/blob/main/docs/checkpoint-format.md) | exactly what the tool reads out of a checkpoint |
| [`CHANGELOG.md`](https://github.com/dwn2rtl/dwn2rtl/blob/main/CHANGELOG.md) | what changed in each release, and whether you need to upgrade |

## Evidence

Before this was packaged as a tool, it ran a full FPGA study:

- **77 configurations** across two datasets, every one giving identical answers to its software model
- **166,000 / 166,000** and **10,000 / 10,000** correct **on real hardware**, one model per dataset
- the original authors' published area and accuracy figures, reproduced at matched convention

Every number that study recorded is reproduced by the code here and pinned by its test suite, so
you can check the claims from this repository rather than taking them on trust.

## Citation

If dwn2rtl is useful in published work, please cite the tool and the paper it targets.

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

**The measurements** under Evidence — the 77 configurations, the hardware runs and the Vivado
figures — were produced in
[dwn-fpga-study](https://github.com/Kanishk234/dwn-fpga-study), which is where to look for how
they were obtained.

## Development

```bash
pip install -e ".[dev]"
pytest                  # everything
pytest -m sim           # just the simulator checks
```

The rule the project runs on: emitted Verilog isn't considered correct until a simulator confirms
it matches the software model on every test vector. Not "it looks right", and not "the emitter
checked its own work" — there's a case on record where an emitter's self-check reported 20/20
correct while the design was wrong on 958 of 1,504 vectors. CI runs this on Linux and Windows,
under two independent simulators, on every commit.

`docs/phaseN-ledger.md` and `phaseN-report.md` record how the tool was built, wrong turns
included, and
[`docs/overview.md`](https://github.com/dwn2rtl/dwn2rtl/blob/main/docs/overview.md) and
[`docs/tool-roadmap.md`](https://github.com/dwn2rtl/dwn2rtl/blob/main/docs/tool-roadmap.md)
are the plans that shaped it. All of that is history, not instructions.

## Licence

MIT — see [LICENSE](https://github.com/dwn2rtl/dwn2rtl/blob/main/LICENSE).
