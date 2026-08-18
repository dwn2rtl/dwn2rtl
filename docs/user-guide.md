# dwn2rtl user guide

How to take a DWN you trained and get Verilog you can put in a design.

The [README](https://github.com/dwn2rtl/dwn2rtl/blob/main/README.md) says what the tool is and
why. This says how to use it on your own model — including the parts that only matter once you
try to wire the output into something real.

Everything below was run end to end against a `pip install dwn2rtl` from PyPI and a real trained
MNIST checkpoint (784 features, 10 classes, one layer of 300 nodes, n=6, z=3). The transcripts are
the actual output.

---

## 1. Install

```bash
pip install dwn2rtl
```

That is the whole install for `build`. It pulls numpy and torch.

`verify` additionally needs a Verilog simulator, which is not a Python package and you install
separately:

| | |
|---|---|
| Windows | `winget install Icarus.Verilog` — it does **not** add itself to PATH, but dwn2rtl looks in `C:\iverilog\bin` anyway |
| Debian/Ubuntu | `apt install iverilog` |
| macOS | `brew install icarus-verilog` |

`estimate` needs `yosys` and is entirely optional.

> **Windows: install into a short path.** A fresh venv created ~120 characters deep failed to
> install *torch* — `OSError: [Errno 2] No such file or directory: ...mem_eff_attention/
> epilogue_thread_apply_logsumexp.h`. That is the 260-character `MAX_PATH` limit, hit by torch's
> own deep headers, not by anything dwn2rtl does. Either enable Long Path support or put the venv
> somewhere shallow like `C:\work\`.

---

## 2. The whole tool in three commands

```bash
dwn2rtl build my_model.pt --out rtl/     # checkpoint  -> Verilog + golden vectors
dwn2rtl verify rtl/                      # compile and run it; must print PASS
dwn2rtl estimate rtl/                    # optional: how big is it
```

There is no configuration file, no project setup, and — usually — no flags. Real output:

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

**`verify` is not a formality.** Until a simulator has printed PASS, the emitted design is
unproven — an emitter's own read-back is not evidence. Run it every time.

### Why two levels

`dwn_core_tb` drives pre-binarized bits; `dwn_top_tb` drives quantized features through the
encoder as well. The split makes a failure **localize itself**, and `verify` says so out loud when
it happens:

| | |
|---|---|
| core PASS, top FAIL | the encoder — nothing else needs re-examining |
| core FAIL, top FAIL | the network; fix that and this follows |

**Nothing that was not checked is reported as a pass.** A missing testbench, an empty one, a
compile error, a simulation that did not finish, or a directory with no runnable levels at all —
each is a failure, not a skip.

---

## 3. Saving your model so the tool can read it

**A DWN is two objects.** The thermometer is fitted *before* training and is not a model
parameter, so it is not in the `state_dict`. One line at the end of your training script:

```python
torch.save({'model': model, 'thermometer': thermometer}, 'my_model.pt')
```

If your training scaled its features, save the scaler too — the thresholds live in whatever
feature space training used:

```python
torch.save({'model': model, 'thermometer': thermometer, 'scaler': scaler}, 'my_model.pt')
```

If the model is still in memory you can let dwn2rtl validate it while you can still fix it:

```python
import dwn2rtl
dwn2rtl.save(model, thermometer, 'my_model.pt', scaler=scaler)
```

### The mistake everyone makes

`torch.save(model.state_dict())` is what every PyTorch user reaches for, and it **silently drops
the encoder** — which can be many times the size of the network it feeds. dwn2rtl refuses such a
file by name rather than emitting a design that synthesizes cleanly and classifies at chance:

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

Exit code 1. Nothing is written.

### You do not need `torch_dwn` installed

dwn2rtl duck-types the checkpoint, so the machine that emits Verilog never has to build the
upstream CUDA/C++ extension. (Worth knowing: `pip install torch_dwn` fails to build a wheel on a
plain Windows box. It does not matter here.)

Accepted shapes: `{'model': ..., 'thermometer': ...}`, a live model via
`dwn2rtl.from_model(model, thermometer)`, and the `{'state_dict', 'thermometer', 'config'}` form
an upstream training script may write. See
[checkpoint-format.md](https://github.com/dwn2rtl/dwn2rtl/blob/main/docs/checkpoint-format.md).

---

## 4. Precision: usually you type nothing

The only number the tool cannot derive is how many **fractional** bits to use, because whether
quantisation changes a prediction depends on your data. So it never asks. It works out how much
precision your input *already has*, and always tells you which of three things happened:

```
frac bits 8 -> Q0.8 signed (9-bit)      INFERRED from the thresholds' grid, provably lossless
frac bits 8 -> Q0.8 signed (9-bit)      from --input-bits, provably lossless
frac bits 12 -> Q3.12 signed (16-bit)   DEFAULT for a continuous input, NOT measured
```

- **INFERRED** — thermometer thresholds are quantiles of the training data, so if that data was
  quantised they sit on the same grid. On MNIST the tool finds `k/255` and derives a 9-bit word
  with nothing typed.
- **`--input-bits N`** — state it yourself. Use this when deployment differs from training (a
  10-bit ADC feeding a model trained on 8-bit data), or when there is no grid to find.
- **DEFAULT** — genuinely continuous inputs, e.g. standard-scaled tabular features. Reported as a
  default, *not* a measurement. This is the one case worth a second thought.

Read the third column of the build report. If it says DEFAULT and you know your input's real
precision, pass `--input-bits`.

---

## 5. What you get, and what to do with it

```
the design      dwn_core.v  thermometer_encoder.v  dwn_top.v
primitives      lut_node.v  popcount.v  argmax.v  pipe_reg.v
parameters      dwn_core_params.vh  dwn_top_params.vh
golden vectors  x_quant.hex  expected_top.hex  x_binarized.hex  expected.hex
                top_params.vh  vec_params.vh
scaling         input_scaling.json      (only if the model was trained on scaled features)
testbenches     tb/dwn_core_tb.v  tb/dwn_top_tb.v
```

The primitives are copied in deliberately: the emitted core *instantiates* them, so the directory
is self-contained and you can hand it to any simulator or synthesis tool without also shipping
dwn2rtl. It is Verilog-2001 with **no vendor primitives** — no Vivado, Quartus or licence is
needed anywhere in this flow.

**Add to your project:** the three design files, the four primitives, and the two `.vh` parameter
files. The `.hex` files and `tb/` are for verification only; leave them out of synthesis.

### The top-level interface

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

Three things to get right:

1. **Packing.** `x_flat` is your features concatenated little-end-first: feature `f` occupies
   `x_flat[f*W +: W]`, where `W` is the word width from the build report. Each field is a
   **signed two's-complement** fixed-point integer — the encoder compares with `$signed(...)`.
2. **Latency and throughput.** The `PIPE_*` parameters are all 1 by default, giving the latency in
   `dwn_top_params.vh` (`DWN_TOP_LATENCY 4`) at **II=1** — feed a new vector every clock, get a
   result every clock, `LATENCY` cycles behind. There is no handshake, no valid, no backpressure;
   it is a fixed-latency pipeline. Set a `PIPE_*` to 0 to remove that stage if you are chasing
   latency rather than clock rate, then re-run `verify`.
3. **`class_idx`** is `ceil(log2(num_classes))` bits — 4 bits for 10 classes. Do not hardcode a
   width; read `IDX_W` from `top_params.vh`. (An earlier testbench of ours hardcoded 3, which left a
   10-class design checked on 3 of its 4 index bits — and passing.)

### ⚠️ The scaling contract

If your training scaled its features, the build report says so in capitals and
`input_scaling.json` is emitted:

```json
{
  "note": "apply as (x - mean) / scale BEFORE quantizing to the word format below",
  "format": "Q0.8 signed (9-bit)", "frac_bits": 8, "word_bits": 9,
  "mean": [...], "scale": [...]
}
```

Whatever drives `x_flat` — an ADC front end, a DMA, a host CPU — must apply the *same fitted*
`(x - mean) / scale` and then quantize, because the thresholds baked into the comparators live in
the training feature space. **Raw features produce a design that runs at chance and looks
completely healthy doing it**: it will still pass `verify`, because the testbench feeds it
correctly-scaled vectors. This is the single most likely way to end up with a working-looking,
useless design.

---

## 6. Verifying against your own data

The generated vectors are random by construction, plus edge cases. That proves the RTL matches the
model, which is the point. But you may also want the design exercised on your real inputs — a
held-out test set — so the numbers you quote come from your data.

Rebuild with more vectors:

```bash
python -c "import dwn2rtl; dwn2rtl.build(dwn2rtl.load('my_model.pt'), 'rtl/', n_random=5000)"
```

Or re-label the emitted design with your own samples. This script runs the same golden model the
testbench trusts, so a PASS afterwards means *your data* was reproduced bit-exactly:

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

Then `dwn2rtl verify rtl/`. Run against 200 samples, this produced:

```
  dwn_core  504 vectors  PASS
  dwn_top   200 vectors  PASS
RESULT   PASS
```

Three cautions:

- ⚠️ **Vectors and RTL must come from the same checkpoint.** This script re-reads `my_model.pt`;
  point it at a different file than `build` used and you get a testbench that passes against wrong
  RTL — worse than shipping none.
- `dwn2rtl.extract` and `dwn2rtl.vectors` are **internal modules**, not the public API. The recipe
  above is tested and works today; treat it as pinned to your installed version.
- If the model was trained *without* a scaler there is no `input_scaling.json` — drop the mean and
  scale lines and quantize `x_raw` directly.

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

**This is an estimate from generic mapping, not your vendor toolchain**, and the tool says so
every time it prints. That caveat is not a disclaimer — it is a measurement. The same design has
real Vivado figures for this design, so the estimator was calibrated against them before it was
trusted with anything:

| module | yosys | Vivado, `xc7a35t` | |
|---|---|---|---|
| `dwn_core` | 110 | **110** | exact — a LUT core is LUT6s, and both tools find the same ones |
| `thermometer_encoder` | 717 | **1519** | **0.47×** — generic mapping packs comparators; Vivado puts them on carry chains |
| `dwn_top` | 833 | **1621** | 0.51× |

Treat core numbers as indicative, encoder numbers as a floor, and synthesize for anything you
intend to publish.

⚠️ **Two different levels of trust in one design**, which is why the numbers are never summed into
a single figure. It also cuts against this project's own headline: the encoder-to-core ratio is
**13.8× by Vivado and 6.5× by yosys**, so a tool that printed the yosys ratio bare would
understate the very thing it exists to show.

**An unmapped design is an error, not a small number.** If generic gate primitives survive LUT
mapping, the count is a fragment of the design rather than its size — yosys 0.33 once reported
*one LUT for a 21-node core* — so `estimate` names the yosys version and refuses. Same rule as
`verify`, one level up: nothing unmeasured is reported as a measurement.

> Requires yosys, which you install yourself — `apt install yosys`, or the
> [OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build/releases) on Windows.
> Don't put the suite on PATH: it ships its own `iverilog` and would shadow the simulator your
> gate runs against. dwn2rtl finds it either way.

**The encoder is always reported separately, and it always ships.** It is intrinsic to a DWN, not
preprocessing. On the smallest studied model it is 14× the network it feeds — work that quotes
only the network understates the design by most of its cost.

---

## 8. Troubleshooting

| What you see | What it means |
|---|---|
| `this looks like a bare state_dict` | You saved with `model.state_dict()`. See §3. |
| `frac bits ... DEFAULT ... NOT measured` | No grid was detectable in the thresholds. Fine for continuous inputs; pass `--input-bits N` if you know better. |
| `WARNING ... trained on SCALED features` | Not an error. Whatever drives `x_flat` must apply `input_scaling.json` first. See §5. |
| `N of M thresholds quantise to a duplicate comparison` | Informational. Distinct thresholds collapsed to the same integer at this precision. Harmless — the comparators merge — but a large fraction suggests more fractional bits would buy nothing. |
| `verify` finds no simulator | `verify` is the gate and refuses to report success without running. Install iverilog (§1) or pass `--simulator <path>`. |
| `x/y classes hit` is low | Random vectors will not reach every class on a model with rare classes. Coverage of the RTL is unaffected; use §6 if you want your real class distribution. |
| torch fails to install with a long filename error | Windows `MAX_PATH`. Use a shallow directory. See §1. |

---

## 9. What this tool does not do

By design, so you know what to build yourself:

- **No board harness.** No AXI wrapper, no DMA, no constraints, no bitstream. You get RTL with a
  clean fixed-latency interface and you integrate it.
- **No vendor toolchain, ever.** Nothing here calls Vivado or Quartus, and nothing emitted needs a
  licence to simulate or synthesize.
- **No area model.** One existed, was calibrated on one dataset, and filtered zero configurations
  across two complete studies. `estimate` reports what a real tool measured, or reports nothing.
- **No sweep automation, and no training.** It translates the model you already trained.

---

## Where to go next

- [README](https://github.com/dwn2rtl/dwn2rtl/blob/main/README.md) — what it is, and the evidence
- [examples/quickstart.py](https://github.com/dwn2rtl/dwn2rtl/blob/main/examples/quickstart.py) —
  one runnable file, no dataset, showing where your training script plugs in
- [checkpoint-format.md](https://github.com/dwn2rtl/dwn2rtl/blob/main/docs/checkpoint-format.md) —
  the exact file format
- [overview.md](https://github.com/dwn2rtl/dwn2rtl/blob/main/docs/overview.md) — how the tool is
  built, phase by phase
- [tool-roadmap.md](https://github.com/dwn2rtl/dwn2rtl/blob/main/docs/tool-roadmap.md) — open
  questions, and what is deliberately out of scope
