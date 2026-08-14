# Phase 1 ledger — close the loop

**Goal.** Get from a checkpoint to Verilog that a simulator says is bit-exact against the golden
model. This is the milestone; phase 0 was scaffolding and phases 2–3 are packaging around it.

**Status: OPEN.**

> Conventions in `overview.md` §6. Entries oldest first, recording *built* / *hit* / *decided*.
> Reversed decisions stay, struck through, with the reason.

**The gate, restated, because everything here serves it (CLAUDE.md):**

> Emitted RTL is not correct until a simulator says it matches the golden model on **every**
> vector. Not "looks right", not "the emitter's self-check passed" — the study repo has a case
> where an emitter's own read-back reported 20/20 correct while the design was wrong on 958 of
> 1,504 vectors.

## Plan

| # | unit | why it is in this order |
|---|---|---|
| 1 | **a synthetic checkpoint fixture** | nothing downstream can be tested without one, and the repo contains no checkpoint at all. Pure data, so it depends on nothing |
| 2 | **`checkpoint.py`** | defines what a checkpoint *is*: sniff the three accepted shapes, normalize, and fail by name on a bare `state_dict` |
| 3 | **`build()` in `__init__.py`** | wires `build_core` -> `build_encoder` -> `generate`, in that order. The encoder reads the core's real pipeline depth from `dwn_core_params.vh`, so it must run second |
| 4 | **`rtl/tb/dwn_top_tb.v`** | currently a 0-byte file. Half the gate does not exist |
| 5 | **the gate: `iverilog` green end to end** | the only unit that proves any of the others |

---

## 1. Built — the synthetic checkpoint fixture

**Why this is test-only, and stays test-only.** The tool translates; it does not generate DWNs.
This fixture exists because roadmap **P8** makes it mandatory rather than convenient: a learnable
mapping stores a `(features x z) x (width x n)` float32 matrix, so real checkpoints run **17 MB
for MNIST `1x300` and 471 MB for `1x1000 z=25`**. Anything over 100 MB cannot go in git, so CI
fixtures must be *synthesized*, not committed. It lives in `tests/`, ships in no wheel, and is on
no user's path.

It also earns its place on evidence rather than principle: in the study repo the equivalent script
passed Gate 1 at MNIST's shape — 784 features, 10 classes, two layers — **before any MNIST
training existed**, catching emitter bugs weeks before a real checkpoint could have.

`tests/fixtures.py` builds checkpoints that are **structurally real**: they obey every rule in
`docs/checkpoint-format.md`, including both mapping representations and the `__dummy_mapping`
decoy. They are not trained, so their accuracy is meaningless — which is fine, because the gate
compares RTL against the golden model, never against a dataset.

Five named shapes, each present for a stated reason:

| shape | why it exists |
|---|---|
| `tiny` | the fast path, and **n=2 gives 4-entry tables** — where the study repo's `np.packbits` bug lived. `bitorder='big'` pads a partial byte on the *low* side, so `[1,1,1,0]` emitted as `0x70` instead of `0x07`. n>=3 is a whole number of bytes and never showed it |
| `single` | one layer, so emit_core's layer-chaining loop never executes |
| `n6` | the architectural premise — one node is exactly one LUT6, tables are 64 bits |
| `ten_class` | `idx_w` is 4 bits. The study repo's testbench hardcoded 3 and silently checked three of four index bits on a 10-class design, which passed |
| `all_fixed` | the learnable path absent rather than merely unused |

The first layer is `learnable` and the rest `fixed`, which is upstream's own recipe and forces
**both** mapping representations through every test. They share no code, and one has a decoy
beside it.

### ⚠️ Hit: the first fixtures were degenerate, and would have made the gate meaningless

Measured immediately after writing them — every shape but one predicted **a single class for
every input**:

```
tiny        K= 2  group=2   classes_hit=1/2
n6          K= 3  group=2   classes_hit=1/3
ten_class   K=10  group=2   classes_hit=3/10
```

**Cause.** The popcount group — final layer width / `num_classes` — was 2. With untrained tables
a node is *constant* whenever its `2**n` table happens to be all-positive (probability 1/16 at
n=2), and a group of two constant nodes scores a permanent 2 that nothing can beat. Diagnosed by
printing the score distribution: `unique scores: [[0,2],[1,2]]`, tie fraction 0.0, class counts
`[0, 500]`. Class 1 did not win on merit; it was never in a contest.

**Why it mattered more than it looks.** A testbench whose expected output is a constant passes
against a design whose argmax, popcount and grouping are *all* wrong. It is precisely the shape
of the failure the gate exists to catch — the study repo's read-back reporting 20/20 while the
design was wrong on 958 of 1,504 vectors — reproduced in the fixture rather than the emitter.

**Fix, in two parts.** Groups raised to >= 3 in every shape, and `make()` now searches seeds
deterministically until the model discriminates at least `min_classes` classes. A *pinned* good
seed was rejected: it silently stops being good the moment any shape parameter changes, and it
fails as a collapsed fixture rather than as an obviously wrong seed. After:

```
tiny        K= 2  group=4   classes_hit=2/2
single      K= 2  group=4   classes_hit=2/2
n6          K= 3  group=3   classes_hit=3/3
ten_class   K=10  group=4   classes_hit=10/10
all_fixed   K= 2  group=4   classes_hit=2/2
```

`classes_hit()` is now asserted per shape, so a fixture cannot silently collapse again. It is the
same condition `vectors.py` already reports as `degenerate` on real builds — this is the
fixture-side check that stops one being built at all.

### Built — `tests/test_fixtures.py`

19 tests that the fixture is structurally real rather than merely plausible: table shape and the
`[-1, 1]` range, wiring indices in range of the *previous* layer's width, thresholds sorted within
each feature, final layer divisible by `num_classes`, determinism, and both mapping
representations present.

**The most valuable one is the decoy test.** `_LUTLayer__dummy_mapping` has the same
`(output_size, n)` shape and int dtype as a genuine fixed mapping, so nothing about its type gives
it away. The test asserts the fixture *ships* it, that it really is `arange()` reshaped, and that
`extract_wiring` returns the argmax-derived wiring **and that the two differ** — so the test
cannot pass by coincidence.

**Suite: 82 passed, 1 xfailed** (the empty `dwn_top_tb.v`, still tracked).

---

## 2. Built — `checkpoint.py`

Roadmap **Q8 closed**. The tool now defines its input format and owns both ends of it.

### The three accepted shapes converge

| shape | how a user gets it |
|---|---|
| `{'model': ..., 'thermometer': ...}` | plain `torch.save` — no `dwn2rtl` import in their training script |
| `{config, state_dict, thermometer, results}` | the study repo's existing checkpoints, unchanged |
| live objects | `from_model(model, thermometer)`, the notebook path |

Asserted equal across all five fixture shapes: same `config`, same `summary()`. A checkpoint's
meaning must not depend on how it happened to be saved.

**Nothing in `checkpoint.py` imports the upstream DWN package.** Every fact is read from tensors
or from duck-typed attributes, so a user does not need `torch_dwn` installed to convert a
checkpoint they already have, and an upstream version bump cannot break loading.

**Decided: `num_classes` comes off `GroupSum`, matched by class name.** It is the one fact no
tensor knows — `GroupSum` has no parameters, so it contributes nothing to a `state_dict`. Matched
on `type(m).__name__ == 'GroupSum'` plus a `k` attribute rather than an `isinstance` check,
because the alternative is importing upstream.

**Decided: the API is lazy at the package root.** `dwn2rtl.load` / `.save` / `.from_model` work
as if imported in `__init__.py`, but via PEP 562 `__getattr__`. `checkpoint.py` imports torch at
module level, so a plain top-level import would have quietly cost *every* invocation a
multi-second torch import — including `dwn2rtl verify`, which never reads a checkpoint. Both
properties are now asserted in one test: `False True` for torch-in-`sys.modules` before and after
touching `dwn2rtl.load`.

### The errors are part of the contract, and are tested as such

The failure this module exists to prevent produces a design that synthesizes cleanly, reports a
plausible area, and classifies at chance. A user told *"invalid checkpoint"* re-saves it the same
way; a user told *the thermometer is a separate object* fixes it in one line. So the text is
asserted, not just the exception type — it must name the missing object, say **why**, and show
the fix:

```
this looks like a bare state_dict -- it has the model weights but NO THERMOMETER.
  keys: '0.luts', '0._LUTLayer__dummy_mapping', '0.mapping.weights', '1.luts', ...

  A DWN is two objects. The thermometer is fitted before training and is not a
  parameter of the model, so torch.save(model.state_dict()) drops the encoder
  entirely -- and the encoder can be many times the size of the network it feeds.

  Re-save with both:
      torch.save({'model': model, 'thermometer': thermometer}, 'model.pt')
```

Four rejected shapes, each with its own message: a bare `state_dict`, a model alone, a dict with
a model but no thermometer, and `{'model': <a state_dict>, 'thermometer': ...}` — which *has* the
encoder but has lost `GroupSum` and with it the only record of the class count.

### Validation catches corruptions that would otherwise emit a plausible wrong design

Every path in runs the same checks. The one worth naming:

⚠️ **Transposed thresholds.** A `(z, features)` matrix has the right rank, the right dtype and
entirely plausible values, and would emit an encoder with features and thresholds swapped —
silently. It is detectable only because `config` states `z` independently, so the two can be
cross-checked:

```
thermometer thresholds are (2, 4) but config says thermometer_bits=2.
This looks TRANSPOSED -- the expected shape is (features, bits_per_feature), i.e. (4, 2).
```

Also caught: `n` disagreeing with table width, `layers` disagreeing with the tables, a class
count that does not divide the final layer (GroupSum would zero-pad silently), wiring that reads
past the input width, a non-power-of-two table, non-contiguous layer indices, and 1-D thresholds.

### ⚠️ Hit: the fixture was inventing an accuracy, and `emit_core` demanded one

The first fixture wrote `results: {'final_acc': 0.0, 'best_acc': 0.0}` into every checkpoint. It
was untrained, so that number was fabricated to fill a field — and `final_acc: 0.0` reads as
*"this model scores zero"*, not *"nobody measured"*. Removed; the fixture now records no results
at all, which is both honest and the case a real user hits constantly.

That immediately exposed a defect on the other side: **`emit_core.py` read
`ck['results']['final_acc']` unconditionally**, for a header comment. Accuracy is metadata — no
part of the emitted hardware depends on it — yet a user who saved a model without recording a
training statistic would have got a `KeyError` out of a code generator, over a comment. Now:

```
// accuracy   : not recorded in the checkpoint
```

**Defaulting to `0.0` was rejected for the same reason the fixture's was.** A comment that
confidently states a wrong number is worse than no comment. Output is byte-identical to before
for any checkpoint that *does* carry results, so nothing the study repo verified has moved.

**Suite: 117 passed, 1 xfailed.**

