# DWN checkpoint format — what the exporter reads

**Source of truth.** Everything here was read directly from `third_party/DWN` at the pinned commit
`9f887a0`, not inferred from the paper, from tensor shapes, or from PyTorch convention. That is the
rule in CLAUDE.md and it exists because getting this wrong produces an exporter that looks correct
and silently emits wrong wiring (project-brief.md §12, risk #5).

Files read: `src/torch_dwn/lut_layer.py`, `src/torch_dwn/mapping.py`, `src/torch_dwn/utils.py`,
`src/torch_dwn/custom_operators/cuda/efd_cuda_kernel.cu` (forward kernel), `examples/mnist.py`.

If the pin ever moves, re-read all of it. Nothing below is safe to carry forward on faith.

---

## 1. A LUT node's output bit

`LUTLayer.forward` runs EFD and then, with `ste=True` (the default), `STEFunction`:

```python
# utils.py
class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return (x > 0).float()
```

and EFD's forward is a plain table lookup (`efd_cuda_kernel.cu`):

```cuda
output[i][j] = luts[j][addr];
```

So the hardware bit for node `j` at address `addr` is:

```
bit = (luts[j][addr] > 0)
```

**Strictly `> 0`, not `>= 0`.** An entry of exactly `0.0` emits 0. Unlikely in a trained model but
it is a real edge case, and Gate 1 covers edge cases, so the golden model must use `>`.

`luts` has shape `(output_size, 2**n)` and dtype float32.

### The `[-1, 1]` range is deliberate

```python
# lut_layer.py, forward()
if self.training and self.clamp_luts:
    with torch.no_grad():
        self.luts.clamp_(-1, 1)
```

Tables are initialised uniform in `[-1, 1]` (`torch.rand(...)*2 - 1`) and clamped back into that
range on every training forward pass. **Seeing `range [-1.000, 1.000]` in a checkpoint is normal and
expected, not a sign of a saturated or broken model.** Only the sign is ever used at inference.

---

## 2. Address construction — LSB first

From the forward kernel:

```cuda
uint addr = input[i][mapping[j][0]] > 0;
for(int l = 1; l < mapping.size(1); ++l)
    addr |= (uint)(input[i][mapping[j][l]] > 0) << l;
```

**Mapping slot `l` contributes bit `l` of the address. Slot 0 is the LSB.**

This is the single easiest thing in the whole project to get backwards, and reversing it produces a
model that is wrong on most inputs while still looking structurally plausible. The RTL's address
concatenation must match this, and the golden model must assert it.

Input bits are thresholded `> 0` here too, so the layer treats any positive value as 1.

---

## 3. Wiring — two completely different representations

A `LUTLayer` stores its mapping one of two ways depending on how it was constructed. **The exporter
needs both paths**; they share no code.

### 3a. `mapping='learnable'`

```python
# lut_layer.py
self.mapping = LearnableMapping(input_size, output_size*n, tau=lm_tau)
self.__dummy_mapping = torch.nn.Parameter(
    torch.arange(output_size*n).reshape(output_size, n).int(), requires_grad=False)
```

```python
# mapping.py
mapping = weights.argmax(dim=0)
output = x[:, mapping]
```

- `weights` has shape `(input_size, output_size * n)` — e.g. `(128, 1800)` for 128 input bits and
  300 nodes at n=6.
- **`argmax(dim=0)`** over the input-bit axis yields `output_size * n` indices, one per node slot.
- Node `j`, slot `k` reads input bit `weights.argmax(dim=0)[j*n + k]`.

The argmax is input-independent, so it is resolved once at export and costs nothing in hardware —
this is the paper's "Learnable Mapping is free at inference" claim (brief §4), confirmed in code.

`tau` (default `lm_tau=0.001`) appears only in the *backward* pass. It has no effect on export.

### 3b. `mapping='random'`, `'arange'`, or an explicit tensor

```python
self.mapping = torch.nn.Parameter(layer_mapping(...), requires_grad=False)
```

A plain `(output_size, n)` int32 tensor, already the final wiring. Node `j` slot `k` reads input bit
`mapping[j][k]`. No argmax, no transformation.

### ⚠️ `_LUTLayer__dummy_mapping` is a decoy

It appears in `state_dict` for learnable layers as a name-mangled `(output_size, n)` int32 tensor —
the *same shape and dtype* as a real 3b mapping. It is only `arange(output_size*n)` reshaped,
because after `x[:, mapping]` gathers the inputs the kernel reads consecutive slots.

**Exporting it as if it were the wiring yields a structurally valid, completely wrong model.** Key
off whether the layer has a `LearnableMapping` (i.e. whether `<i>.mapping.weights` exists in the
`state_dict`), never off tensor shape.

---

## 4. Output stage — `GroupSum`

```python
# utils.py
x = pad_if_needed(x, self.k)
x = x.view(*x.shape[:-1], self.k, int(x.shape[-1]/self.k))
return x.sum(dim=-1) / self.tau
```

- **Groups are contiguous slices, in order.** Class `c`'s score is the popcount of final-layer
  outputs `[c * (W/k) : (c+1) * (W/k)]`, for final layer width `W`. No interleaving.
- `randperm` defaults to `False`, so no permutation is applied. If a future config sets it, the
  permutation is generated at construction and **is not saved in `state_dict`** — such a checkpoint
  would not be exportable. Don't enable it.
- Division by `tau` is monotonic and identical across classes, so **argmax is unaffected**. The
  hardware needs the popcount and the argmax only; `tau` can be ignored entirely at inference.
- `pad_if_needed` zero-pads when `W % k != 0`. This is silent. **Keep final layer width divisible by
  `num_classes`** or the DSE will produce configs whose hardware and software disagree about group
  boundaries. Worth an assert in the exporter.

---

## 5. Not needed for export

- `alpha` (default `0.5 * 0.75**(n-1)`, so `0.11865` at n=6) and `beta` (`0.25/0.75`) — EFD backward
  only.
- `lm_tau` — LearnableMapping backward only.
- Everything in the backward kernel.

---

## 6. Upstream's reference recipe (`examples/mnist.py`)

The only worked example shipped at this pin. **There is no JSC example** — the paper's JSC configs
are not in this repo.

```python
thermometer = dwn.DistributiveThermometer(3).fit(x_train)
model = nn.Sequential(
    dwn.LUTLayer(x_train.size(1), 2000, n=6, mapping='learnable'),
    dwn.LUTLayer(2000, 1000, n=6),            # mapping defaults to 'random'
    dwn.GroupSum(k=10, tau=1/0.3)
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=14)
train_and_evaluate(..., epochs=30, batch_size=32)
```

Compared against `training/dwn_jsc_kaggle.ipynb` as of the t=8 run:

| Knob | Upstream | Ours | Match? |
|---|---|---|---|
| optimizer / lr | Adam, 1e-2 | Adam, 1e-2 | same |
| scheduler | StepLR, γ=0.1, step 14 | StepLR, γ=0.1, step 14 | same |
| epochs | 30 | 30 | same |
| `tau` | `1/0.3` | `1/0.3` | same |
| mapping pattern | learnable, then random | learnable, then random | same |
| thermometer | distributive, **3 bits** | distributive, 4→8 bits | ours is richer |
| **batch_size** | **32** | **256** | **differs, 8×** |

**This kills the learning-rate hypothesis.** The frozen training loss in the t=4 and t=8 runs cannot
be blamed on `lr=1e-2` or on the schedule — both are exactly what the authors use to reach the
paper's MNIST numbers. It also weakens the layer-1-mapping hypothesis, since upstream also leaves
the second layer random.

`batch_size` is the one place our recipe deviates from the authors'. That makes it the
evidence-backed next experiment.

Note also that upstream reaches strong MNIST accuracy on a **3-bit** thermometer, which is
independent support for the t=4 → t=8 finding that encoder resolution is a flat axis.