# What dwn2rtl reads out of a checkpoint

If you're writing your own exporter, or you want to know exactly what the tool is looking at, this
is the reference. Everything here was read straight out of the upstream DWN source — not guessed
from the paper, not inferred from tensor shapes, and not assumed from PyTorch convention. Get any
of it wrong and you get an exporter that looks fine and silently emits the wrong wiring.

## Which version this describes

```
upstream   https://github.com/alanbacellar/DWN
commit     9f887a0b4bd84dabf6d8c9ae35368ab2a7e0e3c0   ("9f887a0")
verified   2026-08-13
```

**dwn2rtl doesn't depend on that package.** Nothing imports it, at build time or ever. The commit
is here so you can check these claims yourself, and so a maintainer knows what to re-read when
upstream moves.

The files it was read from: `src/torch_dwn/lut_layer.py`, `src/torch_dwn/mapping.py`,
`src/torch_dwn/utils.py`, `src/torch_dwn/custom_operators/cuda/efd_cuda_kernel.cu` (the forward
kernel), and `examples/mnist.py`.

This document is older than the tool, so rather than trusting what it already said, every claim
that matters was checked against the source a second time:

| § | the claim | where it's confirmed |
|---|---|---|
| 1 | a table bit is `luts[j][addr] > 0`, **strictly** greater | `STEFunction.forward` is `(x > 0).float()` |
| 2 | mapping slot `l` is address bit `l` — **LSB first** | `addr \|= (input[mapping[j][l]] > 0) << l` |
| 3a | learnable wiring is `weights.argmax(dim=0)` | `mapping.py:17` |
| 3 | `__dummy_mapping` is a decoy — just `arange` reshaped | `lut_layer.py:60` |
| 4 | `GroupSum` holds `k` and `tau` and has **no parameters** | `utils.py:11-16` |

That last row is the reason `num_classes` can't be recovered from a `state_dict` at all, and has
to come off the live module instead.

If that commit ever moves, re-read all of it. None of this is safe to carry forward on faith.

> ⚠️ **Upstream's `LUTLayer` forward needs CUDA** — the CPU path raises. That's not a constraint on
> dwn2rtl, which never runs a model, but it's why the worked example builds its own stand-in
> classes instead of importing `torch_dwn`.

---

## 1. What a node outputs

`LUTLayer.forward` runs EFD, then passes the result through `STEFunction` (which is on by
default):

```python
# utils.py
class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return (x > 0).float()
```

and EFD's forward is a plain table lookup:

```cuda
// efd_cuda_kernel.cu
output[i][j] = luts[j][addr];
```

So the hardware bit for node `j` at address `addr` is:

```
bit = (luts[j][addr] > 0)
```

**Strictly greater than zero, not greater-or-equal.** An entry of exactly `0.0` gives you a 0.
That's unlikely in a trained model, but it's a real case and the test vectors include edge cases,
so the software model has to use `>` as well.

`luts` is `(output_size, 2**n)`, float32.

### Values pegged at ±1 are normal

```python
# lut_layer.py, forward()
if self.training and self.clamp_luts:
    with torch.no_grad():
        self.luts.clamp_(-1, 1)
```

Tables start out uniform in `[-1, 1]` and get clamped back into that range on every training
step. So if you open a checkpoint and see `range [-1.000, 1.000]`, that's expected — it isn't a
saturated or broken model. Only the sign is ever used at inference anyway.

---

## 2. How the address is built — LSB first

Straight from the forward kernel:

```cuda
uint addr = input[i][mapping[j][0]] > 0;
for(int l = 1; l < mapping.size(1); ++l)
    addr |= (uint)(input[i][mapping[j][l]] > 0) << l;
```

**Slot `l` of the mapping becomes bit `l` of the address. Slot 0 is the least significant bit.**

This is the easiest thing in the whole project to get backwards, and if you reverse it you get a
model that's wrong on most inputs while looking entirely plausible. Your RTL's address
concatenation has to match this, and your software model should check that it does.

Input bits are thresholded `> 0` here too, so anything positive counts as a 1.

---

## 3. Wiring — two representations that share no code

A `LUTLayer` stores its mapping one of two ways, depending on how it was built. **Your exporter
needs both.**

### 3a. Learnable mapping

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

`weights` is `(input_size, output_size * n)` — so `(128, 1800)` for 128 input bits and 300 nodes
at n=6. Taking `argmax(dim=0)` down the input-bit axis gives you one index per node slot, and node
`j` slot `k` reads input bit `weights.argmax(dim=0)[j*n + k]`.

That argmax doesn't depend on the input, so you resolve it once when exporting and it costs
nothing in hardware. This is the paper's claim that learnable mapping is free at inference, and it
holds up in the code.

`tau` (`lm_tau`, default `0.001`) only shows up in the backward pass. It has no effect on export.

### 3b. Fixed mapping

```python
self.mapping = torch.nn.Parameter(layer_mapping(...), requires_grad=False)
```

Used for `mapping='random'`, `'arange'`, or an explicit tensor. It's a plain `(output_size, n)`
int32 tensor that's already the final wiring — node `j` slot `k` reads input bit `mapping[j][k]`.
No argmax, no transformation.

### ⚠️ `_LUTLayer__dummy_mapping` is a trap

Learnable layers put this in their `state_dict` under a name-mangled key, and it's an
`(output_size, n)` int32 tensor — *the same shape and dtype as a real fixed mapping*. It's only
`arange(output_size*n)` reshaped, because once `x[:, mapping]` has gathered the inputs, the kernel
just reads consecutive slots.

**Export it as if it were the wiring and you get a structurally valid, completely wrong model.**
Decide which representation you're looking at by checking whether `<i>.mapping.weights` exists —
never by tensor shape.

---

## 4. The output stage — `GroupSum`

```python
# utils.py
x = pad_if_needed(x, self.k)
x = x.view(*x.shape[:-1], self.k, int(x.shape[-1]/self.k))
return x.sum(dim=-1) / self.tau
```

Four things follow from that.

**Groups are contiguous and in order.** For a final layer of width `W`, class `c`'s score is the
popcount of outputs `[c * (W/k) : (c+1) * (W/k)]`. Nothing is interleaved.

**`randperm` defaults to `False`**, so no permutation is applied. If some future config turns it
on, the permutation is generated at construction and **isn't saved in the `state_dict`** — which
means such a checkpoint can't be exported at all. Leave it off.

**`tau` doesn't matter to you.** Dividing by it is monotonic and identical across every class, so
it can't change which class wins. The hardware only needs the popcount and the comparison.

**Keep your final layer width divisible by `num_classes`.** If it isn't, `pad_if_needed` silently
zero-pads, and your hardware and your model end up disagreeing about where each class's group
starts. Worth refusing outright in an exporter — dwn2rtl does.

---

## 5. What you can ignore

None of this is needed for export:

- `alpha` (defaults to `0.5 * 0.75**(n-1)`, so `0.11865` at n=6) and `beta` (`0.25/0.75`) — EFD
  backward only
- `lm_tau` — LearnableMapping backward only
- everything in the backward kernel

---

## 6. Upstream's own example

`examples/mnist.py` is the only worked example shipped at that commit, and it's a useful sanity
check on what a real DWN looks like:

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

Two details worth noticing if you're building your own model:

- **The first layer is learnable and the second is random.** That's the authors' own pattern, not
  a simplification. dwn2rtl handles both in one checkpoint, which is why §3 covers both paths.
- **Three thermometer bits is enough for strong MNIST accuracy.** Encoder resolution is a flatter
  axis than people expect, and since the encoder is often the bigger half of the design (see the
  [user guide](https://github.com/dwn2rtl/dwn2rtl/blob/main/docs/user-guide.md) §7), spending bits
  there is expensive.

There's no JSC example at this commit — the paper's JSC configurations aren't in that repository.
