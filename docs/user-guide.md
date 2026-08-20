# dwn2rtl user guide

You've trained a DWN. This is how you get Verilog out of it and into a design.

The [README](https://github.com/dwn2rtl/dwn2rtl/blob/main/README.md) covers what the tool is and
why you'd want it. This covers actually using it — including the parts that only bite once you try
to wire the output into something real.

Everything below was run against a `pip install dwn2rtl` from PyPI and a real trained MNIST model
(784 features, 10 classes, one layer of 300 nodes, n=6, z=3). The terminal output is copied from
those runs, not made up.

---

## 1. Installing it

```bash
pip install dwn2rtl
```

That's everything you need for `dwn2rtl build`. It pulls in numpy and torch.

`dwn2rtl verify` also needs a Verilog simulator. That's a separate program, not a Python package,
so you install it yourself:

| | |
|---|---|
| Windows | `winget install Icarus.Verilog` — it won't add itself to your PATH, but dwn2rtl knows to look in `C:\iverilog\bin` |
| Debian/Ubuntu | `apt install iverilog` |
| macOS | `brew install icarus-verilog` |

`dwn2rtl estimate` needs `yosys`, and it's entirely optional.

**If you already have Verilator**, it works too, on Linux and macOS:

```bash
dwn2rtl verify rtl/ --simulator verilator
```

Both simulators run the same testbenches and have to agree — CI checks that on every commit.
iverilog is the default simply because it's much faster for this: **0.38 s against 14.7 s** on the
same design, since Verilator translates everything to C++ and compiles it before it runs a single
cycle. Verilator wins on long simulations, and these aren't long.

Verilator isn't listed for Windows on purpose — it's a Unix-oriented tool that needs a C++
compiler, so the route there is WSL. Nothing in dwn2rtl needs it either way.

> **On Windows, install into a short path.** A fresh venv about 120 characters deep failed to
> install *torch*, with `OSError: [Errno 2] No such file or directory:
> ...mem_eff_attention/epilogue_thread_apply_logsumexp.h`. That's Windows' 260-character path
> limit, hit by torch's own deeply-nested headers — nothing dwn2rtl does. Either turn on Long Path
> support or put the venv somewhere shallow like `C:\work\`.

---

## 2. The three commands

```bash
dwn2rtl build my_model.pt --out rtl/     # your checkpoint -> Verilog + test vectors
dwn2rtl verify rtl/                      # compile it and run it
dwn2rtl estimate rtl/                    # optional: how big is it
```

No config file, no project setup, and usually no flags at all. Here's a real run:

```
$ dwn2rtl build my_model.pt --out rtl/
features 784, classes 10, layers [300], n=6, z=3   from checkpoint
integer bits 0                                     derived, exact
frac bits 8 -> Q0.8 signed (9-bit)                 INFERRED from the thresholds' grid, provably lossless

core      300 nodes, 3 cycles
encoder   720 comparators of 2352 thermometer bits
top       4 cycles latency, II=1

vectors   core 504, top 1227, 7/10 classes hit
note      1169 of 2352 thresholds quantise to a duplicate comparison (1183 distinct)
WARNING   this model was trained on SCALED features. [...]
wrote     rtl/ (18 files)

$ dwn2rtl verify rtl/
iverilog 12.0 (devel) (C:\iverilog\bin\iverilog.exe)
  dwn_core  504 vectors  PASS
  dwn_top   1227 vectors  PASS
RESULT   PASS
```

**Run `verify` every time.** It isn't a formality. Until a simulator has actually run the design
and compared it against what your model says, nobody knows whether the translation came out right
— and the emitter checking its own output isn't the same thing. There's a case on record where an
emitter's self-check reported everything correct on a design that was wrong on most of its inputs.

### Why there are two of them

Two testbenches, so that when something fails you know which half to look at. `dwn_core_tb` drives
the network with already-binarized bits. `dwn_top_tb` drives real features through the encoder as
well. Read the two results together:

| | |
|---|---|
| core PASS, top FAIL | it's the encoder, and nothing else needs looking at |
| core FAIL, top FAIL | it's the network — fix that first, and this one usually follows |
| core FAIL, top PASS | neither. The design is fine and the *core-level test vectors* aren't. The top level runs the same network through the encoder, so if it's right there it's right everywhere. Look at `x_binarized.hex` / `expected.hex`, or at whatever edited them |

**If something didn't get checked, you won't be told it passed.** A missing testbench, an empty
one, a compile error, a simulation that stopped early, a directory with nothing runnable in it —
every one of those is a failure, not a skip.

---

## 3. Saving your model

**A DWN is two objects.** The thermometer encoder gets fitted *before* you train, and it isn't a
model parameter, so it's nowhere in the `state_dict`. You have to save both. One line at the end
of your training script:

```python
torch.save({'model': model, 'thermometer': thermometer}, 'my_model.pt')
```

If you scaled your features before training, save the scaler too — the thresholds live in whatever
feature space you trained in:

```python
torch.save({'model': model, 'thermometer': thermometer, 'scaler': scaler}, 'my_model.pt')
```

And if the model is still in memory, you can have dwn2rtl check it now, while you're still in a
position to fix anything:

```python
import dwn2rtl
dwn2rtl.save(model, thermometer, 'my_model.pt', scaler=scaler)
```

### The mistake almost everyone makes

`torch.save(model.state_dict())` is the reflex, and it **quietly throws the encoder away** — which
can be many times the size of the network it feeds. So dwn2rtl refuses that file and tells you
what's missing, rather than handing you a design that synthesizes beautifully and classifies at
random:

```
$ dwn2rtl build bare_state_dict.pt --out bad_rtl/
dwn2rtl build: this looks like a bare state_dict -- it has the model weights but NO THERMOMETER.
  keys: '0._LUTLayer__dummy_mapping', '0.luts', '0.mapping.weights'

  A DWN is two objects. The thermometer is fitted before training and is not a
  parameter of the model, so torch.save(model.state_dict()) drops the encoder
  entirely -- and the encoder can be many times the size of the network it feeds.

  Re-save with both:
      torch.save({'model': model, 'thermometer': thermometer}, 'model.pt')
```

Exit code 1, and nothing gets written.

### You don't need `torch_dwn` installed

dwn2rtl reads the checkpoint by shape rather than by importing anything, so the machine that emits
your Verilog never has to build the upstream CUDA/C++ extension. Handy, because `pip install
torch_dwn` fails to build a wheel on a plain Windows box — which doesn't matter here.

It accepts `{'model': ..., 'thermometer': ...}`, a live model through
`dwn2rtl.from_model(model, thermometer)`, and the `{'state_dict', 'thermometer', 'config'}` form
some upstream training scripts write. The details are in
[checkpoint-format.md](https://github.com/dwn2rtl/dwn2rtl/blob/main/docs/checkpoint-format.md).

---

## 4. Precision — you usually don't type anything

There's exactly one number the tool can't work out from your checkpoint: how many **fractional**
bits to use. Whether rounding changes a prediction depends on your data, and the tool doesn't have
your data. So instead of asking you a question you can't answer, it works out how much precision
your input *already has*, and then tells you which of three things happened:

```
frac bits 8 -> Q0.8 signed (9-bit)      INFERRED from the thresholds' grid, provably lossless
frac bits 8 -> Q0.8 signed (9-bit)      from --input-bits, provably lossless
frac bits 12 -> Q3.12 signed (16-bit)   DEFAULT for a continuous input, NOT measured
```

**INFERRED** is the usual one. Thermometer thresholds are quantiles of your training data, so if
that data was quantised the thresholds sit on the same grid. With MNIST the tool spots `k/255` and
picks a 9-bit word without you typing anything.

**`--input-bits N`** is you saying it outright. Use it when deployment differs from training — a
10-bit ADC feeding a model trained on 8-bit images — or when there's no grid to find.

**DEFAULT** means genuinely continuous input, like standard-scaled tabular features. It's reported
as a default rather than a measurement, and it's the one case worth a second thought.

So: read the right-hand column of the build report. If it says DEFAULT and you happen to know your
input's real precision, pass `--input-bits`.

---

## 5. What you get, and how to use it

```
the design      dwn_core.v  thermometer_encoder.v  dwn_top.v
building blocks lut_node.v  popcount.v  argmax.v  pipe_reg.v
parameters      dwn_core_params.vh  dwn_top_params.vh
test vectors    x_quant.hex  expected_top.hex  x_binarized.hex  expected.hex
                top_params.vh  vec_params.vh
scaling         input_scaling.json      (only if you trained on scaled features)
testbenches     tb/dwn_core_tb.v  tb/dwn_top_tb.v
```

The building blocks get copied in on purpose — the emitted design instantiates them, so the
directory stands on its own and you can hand it to any tool without also shipping dwn2rtl. It's
plain Verilog-2001 with **no vendor primitives**, so nothing in this flow needs Vivado, Quartus or
a licence.

**For synthesis**, take the three design files, the four building blocks and the two `.vh`
parameter files. The `.hex` files and `tb/` are only there for checking — leave them out.

### The interface

```verilog
module dwn_top #(
    parameter integer PIPE_ENC = 1,   // after the comparators
    parameter integer PIPE_LUT = 1,
    parameter integer PIPE_POP = 1,
    parameter integer PIPE_OUT = 1
)(
    input  wire clk,
    input  wire [7055:0] x_flat,      // 784 features x 9 bits, feature f at [f*9 +: 9]
    output wire [3:0] class_idx
);
```

Three things to get right.

**Packing.** `x_flat` is your features laid end to end, lowest first: feature `f` sits at
`x_flat[f*W +: W]`, where `W` is the word width from the build report. Each field is a **signed
two's-complement** fixed-point integer — the encoder compares them with `$signed(...)`.

**Timing.** The `PIPE_*` parameters are all 1 by default, which gives you the latency written into
`dwn_top_params.vh` (`DWN_TOP_LATENCY 4`) and one result per clock. Feed it a new vector every
cycle and you get an answer every cycle, `LATENCY` cycles behind. There's no handshake, no valid
signal and no backpressure — it's a fixed-latency pipeline, and it's your job to know when the
answer is ready. If you care more about latency than clock rate, set a `PIPE_*` to 0 to drop that
stage, then re-run `verify`.

**Index width.** `class_idx` is `ceil(log2(num_classes))` bits wide — 4 bits for 10 classes. Don't
hardcode it; read `IDX_W` out of `top_params.vh`. An early testbench of ours hardcoded 3, which
meant a 10-class design was only ever checked on 3 of its 4 index bits. It passed, of course.

### ⚠️ The scaling contract

If you trained on scaled features, the build report says so in capitals and you get an
`input_scaling.json`:

```json
{
  "note": "apply as (x - mean) / scale BEFORE quantizing to the word format below",
  "format": "Q0.8 signed (9-bit)", "frac_bits": 8, "word_bits": 9,
  "mean": [...], "scale": [...]
}
```

Whatever drives `x_flat` — an ADC front end, a DMA, a host CPU — has to apply that same fitted
`(x - mean) / scale` and then quantize, because the thresholds baked into the comparators live in
your training feature space.

**Feed it raw features and you get a design that runs at chance and looks completely healthy doing
it.** It will still pass `dwn2rtl verify`, because the testbench feeds it correctly scaled inputs.
This is far and away the easiest way to end up with something that looks like it works and
doesn't.

---

## 6. Checking it against your own data

The vectors the tool generates are random, plus a set of edge cases. That's deliberate — the job
is to prove the Verilog matches the model, and random inputs plus boundary cases do that better
than clustered real data. But you may also want the design exercised on your actual inputs, so the
numbers you quote come from your test set.

The easy version is just more vectors:

```bash
python -c "import dwn2rtl; dwn2rtl.build(dwn2rtl.load('my_model.pt'), 'rtl/', n_random=5000)"
```

The other option is to re-label the emitted design with your own samples. The script below runs
the same software model the testbench compares against, so if `verify` passes afterwards, the
hardware gave exactly the same answers your model did on *your* data:

```python
"""Re-label an emitted design with YOUR data instead of the random vectors."""
import json, os, numpy as np, dwn2rtl
from dwn2rtl.extract import (extract_tables, extract_wiring, layer_indices,
                             encode, forward, quantize, quantize_thresholds)
from dwn2rtl.vectors import words_to_hex, write_lines

OUT = 'rtl'
ck = dwn2rtl.load('my_model.pt')
cfg = ck['config']
n, num_classes = cfg['n'], cfg['num_classes']
sd = ck['state_dict']
layers = [(extract_tables(sd, i), *extract_wiring(sd, i, n)) for i in layer_indices(sd)]

meta = json.load(open(os.path.join(OUT, 'input_scaling.json')))
frac_bits, word_bits = meta['frac_bits'], meta['word_bits']
mean = np.asarray(meta['mean'], dtype=np.float64)
scale = np.asarray(meta['scale'], dtype=np.float64)

# ---- YOUR data goes here: (num_samples, num_features), raw, unscaled ----
x_raw = ...

x  = (x_raw - mean) / scale                      # the scaling contract
xq = quantize(x, frac_bits, word_bits)           # -> integers in the RTL's word format
thr_q = quantize_thresholds(ck['thermometer']['thresholds'].numpy(), frac_bits)
y = forward(encode(xq, thr_q), layers, num_classes)[0]

write_lines(os.path.join(OUT, 'x_quant.hex'), [words_to_hex(r, word_bits) for r in xq])
write_lines(os.path.join(OUT, 'expected_top.hex'), [f'{int(v):X}' for v in y])
write_lines(os.path.join(OUT, 'top_params.vh'), [
    '// regenerated from my own data',
    f'`define N_TOP {xq.shape[0]}',
    f'`define X_W {xq.shape[1] * word_bits}',
    f'`define IDX_W {max(1, int(np.ceil(np.log2(num_classes))))}',
])
```

Then run `dwn2rtl verify rtl/`. On 200 real samples, that gave:

```
  dwn_core  504 vectors  PASS
  dwn_top   200 vectors  PASS
RESULT   PASS
```

Three things to watch:

- ⚠️ **Point it at the same checkpoint you built from.** This script re-reads `my_model.pt`. Aim it
  at a different file than `build` used and you get a testbench that passes against the wrong RTL,
  which is worse than having no testbench at all.
- `dwn2rtl.extract` and `dwn2rtl.vectors` are **internal modules**, not the public API. The recipe
  above is covered by a test and works today, but treat it as pinned to the version you have
  installed.
- If you trained without a scaler there's no `input_scaling.json` — drop the mean and scale lines
  and quantize `x_raw` directly.

---

## 7. How big is it

```
$ dwn2rtl estimate rtl/
yosys 0.68+64

  thermometer_encoder      1040 LUT        0 FF
  dwn_core                  755 LUT      351 FF
  dwn_top                  1826 LUT      904 FF

  the encoder is 1.4x the core, as generic mapping sees it
```

**These come from generic mapping, not from your vendor's toolchain**, and the tool says so every
time it prints. That's not a hedge — the gap was actually measured. The same design has real
Vivado numbers, so the estimator was compared against them before anyone leaned on it:

| module | yosys | Vivado, `xc7a35t` | |
|---|---|---|---|
| `dwn_core` | 110 | **110** | exact — a LUT network is LUT6s, and both tools find the same ones |
| `thermometer_encoder` | 717 | **1519** | **0.47×** — generic mapping packs comparators together; Vivado puts them on carry chains |
| `dwn_top` | 833 | **1621** | 0.51× |

So: treat the network figure as indicative, treat the encoder figure as a floor, and run real
synthesis for anything you intend to publish.

⚠️ **The two modules are trustworthy to different degrees**, which is why they're never added into
one number. It changes the headline figure too: the encoder is **13.8× the network by Vivado and
6.5× by yosys**, so quoting the yosys ratio on its own would undersell the whole point.

**A design that didn't map is an error, not a small number.** If generic gates survive LUT mapping,
the count you get is a fragment of the design rather than its size — yosys 0.33 once reported *one
LUT for a 21-node network*. So `estimate` prints the yosys version and refuses to report, rather
than handing you a number that looks perfectly plausible.

> You'll need yosys, which you install yourself: `apt install yosys`, or the
> [OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build/releases) on Windows.
> Don't put the suite on your PATH — it ships its own `iverilog`, which would shadow the simulator
> you already installed. dwn2rtl finds it either way.

**The encoder always gets built, and always gets its own line.** It's part of the network rather
than preprocessing you supply, and it's often the bigger half — on the smallest model we've
measured, 14× the size of the network it feeds. A figure that counts only the network is missing
most of the design.

---

## 8. When something goes wrong

| What you see | What it means |
|---|---|
| `this looks like a bare state_dict` | You saved with `model.state_dict()`. See §3. |
| `frac bits ... DEFAULT ... NOT measured` | No grid was detectable in your thresholds. Fine for continuous inputs; pass `--input-bits N` if you know better. See §4. |
| `WARNING ... trained on SCALED features` | Not an error. Whatever drives `x_flat` has to apply `input_scaling.json` first. See §5. |
| `the RTL and the vectors ... came from DIFFERENT builds` | A previous `build` was interrupted partway, so the directory has new Verilog beside old test vectors. Re-run `build` over the whole directory. |
| `N of M features drive no comparator` | Some of your features aren't read by any node. Legitimate for a lopsided model — but it's also what a thermometer from a *different training run* looks like, so double-check you saved the matching pair. |
| `provably lossless -- EXCEPT at the word rail` | One threshold landed on the largest value the word can hold, so a feature past it saturates onto the threshold instead of over it. Widen the word by one integer bit if that matters to you. |
| `N of M thresholds quantise to a duplicate comparison` | Informational. Distinct thresholds collapsed onto the same integer at this precision. Harmless — the comparators just merge — but a large fraction means more fractional bits would buy you nothing. |
| `feature values are NaN, so they cannot be quantized` | Your input has NaN in it. A missing feature isn't a small one — it's on no side of any threshold — so the tool stops rather than guessing. |
| `verify` finds no simulator | It won't report success without actually running something. Install iverilog (§1) or pass `--simulator <path>`. |
| `x/y classes hit` is low | Random vectors won't reach every class on a model with rare ones. That doesn't affect how well the RTL is covered; use §6 if you want your real class distribution. |
| torch fails to install with a long-filename error | Windows path limit. Use a shallow directory. See §1. |
| you keep your own Verilog in the emitted directory | That's fine. `verify` only compiles the files it emitted, so your own harness can sit alongside them without getting dragged in. |

---

## 9. What this tool doesn't do

Listed so you know what's left for you to build:

- **No board harness.** No AXI wrapper, no DMA, no constraints, no bitstream. You get RTL with a
  clean fixed-latency interface, and you integrate it.
- **No vendor toolchain, ever.** Nothing here calls Vivado or Quartus, and nothing it emits needs a
  licence to simulate or synthesize.
- **No area model.** We built one once. It was calibrated on a single dataset and never ruled out a
  single configuration across two whole studies, so it's gone. `estimate` reports what a real tool
  measured, or it reports nothing.
- **No sweep automation, and no training.** It translates a model you already trained.

---

## Where to go next

- [README](https://github.com/dwn2rtl/dwn2rtl/blob/main/README.md) — what it is, and the evidence
  behind it
- [examples/quickstart.py](https://github.com/dwn2rtl/dwn2rtl/blob/main/examples/quickstart.py) —
  one runnable file, no dataset needed, showing where your own training script would slot in
- [checkpoint-format.md](https://github.com/dwn2rtl/dwn2rtl/blob/main/docs/checkpoint-format.md) —
  the exact file format, if you're writing your own exporter
- [CHANGELOG.md](https://github.com/dwn2rtl/dwn2rtl/blob/main/CHANGELOG.md) — what changed in each
  release
