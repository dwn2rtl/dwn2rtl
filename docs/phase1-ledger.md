# Phase 1 ledger — close the loop

**Goal.** Get from a checkpoint to Verilog that a simulator says is bit-exact against the golden
model. This is the milestone; phase 0 was scaffolding and phases 2–3 are packaging around it.

**Status: CLOSED.** Retrospective in `phase1-report.md`.

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
| 6 | **`verify.py`** | ⬅ pulled forward from phase 2: the tests had grown their own copy of simulator discovery, and two implementations would drift |

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

---

## 3. Built — `build.py`, and **the gate went green**

### 🎯 The milestone

**Emitted RTL passed a simulator, bit-exact against the golden model, on every fixture shape.**
504 vectors each, under **iverilog 12.0** — a non-Xilinx simulator, on generated files, which
this project had never done:

```
tiny         504 vectors  PASS (bit-exact on every vector)
single       504 vectors  PASS
n6           504 vectors  PASS      n=6, 64-bit tables
ten_class    504 vectors  PASS      K=10, 4-bit class index
all_fixed    504 vectors  PASS
```

Roadmap **Q6 is now measured for emitted files**, not only for the hand-written primitives.
`ten_class` matters specifically: the study repo's testbench hardcoded a 3-bit index and
silently checked three of four bits on a 10-class design, which passed. `IDX_W` is derived here
and the gate covers it.

⚠️ **This is the CORE gate only.** `dwn_top_tb.v` is still empty, so the encoder is emitted,
copied and unchecked. That is unit 4.

### What `build()` does

`build_core` -> `build_encoder` -> `generate` -> copy the hand-written modules. The order is
load-bearing rather than conventional: `build_encoder` **reads `dwn_core_params.vh`** to learn
the core's real pipeline depth instead of being told it twice, because `dwn_top` instantiates
`dwn_core` and its parameters *override* the core's — told independently, the two could disagree
and produce a top whose latency constant did not match its own pipeline. Asserted by a test that
calls the encoder first and requires `FileNotFoundError`.

**Decided: the output directory is self-contained.** The four hand-written primitives are copied
in beside the emitted files, so a user can hand the folder to any tool without knowing where pip
put the package. Testbenches go in `tb/` so that `*.v` in the root is exactly the design — two
testbenches in one compile would give two top modules. Copies are **byte-for-byte**: text mode on
Windows would rewrite `\n` to `\r\n`, and there is no reason for a copy to alter a file.

**Decided: `build_core` now returns its extracted layers, and `generate` is handed them.** The
invariant in CLAUDE.md is that vectors and RTL derive from the same checkpoint; passing the same
in-memory arrays makes that structural rather than merely likely. Re-extracting would be
deterministic and would almost always agree — "almost" is the problem.

**Decided: every derived number states its provenance.** `integer bits 0` and `frac bits 8` look
equally authoritative and are not: one is exact, the other is a proof or a guess depending on
`--input-bits`. The report's right-hand column is the point of it.

```
features 8, classes 3, layers [12, 9], n=6, z=3   from checkpoint
integer bits 0                                    derived, exact
frac bits 8 -> Q0.8 signed (9-bit)                from --input-bits, provably lossless
```

An unproved width is a **warning**, not a footnote.

### ⚠️ Hit: an empty testbench would have shipped silently

`_copy_package_rtl` copies `tb/*.v` — and `dwn_top_tb.v` is a 0-byte file. Copying it produces an
output directory that *looks* complete while that level's gate does nothing, which is the exact
"green light nobody has reason to distrust" failure this whole phase is organised against. It is
now skipped and warned about at build time:

```
WARNING   dwn_top_tb.v is EMPTY in the installed package and was not copied
          -- that level has no testbench, so nothing checks it.
```

### ⚠️ Hit: the test that proves the gate can FAIL did not, and the reason is the interesting part

A gate that cannot fail is worse than no gate, so there is a test that corrupts an emitted truth
table and requires a `FAIL`. The first corruption was `64'h` -> `64'hF`.

**It passed.** That produces a **seventeen**-digit literal in a 64-bit parameter, and Verilog
**silently truncates the excess high digit**, restoring the original sixteen. The design was
bit-identical and the gate was right to pass it.

That is precisely the silent truncation `emit_core.py`'s `MAX_N` assertion exists to prevent at
n>6 — *"a 2\*\*n-bit table into a 64-bit parameter and Verilog would TRUNCATE it silently"* —
reproduced by accident, in a test, on this machine. **A too-wide constant does not error; it
quietly becomes a different, valid one.** The assertion is not defensive programming; it is
guarding a real behaviour of the language. Corruption is now a bitwise inversion holding the
literal at exactly 16 digits, and the gate fails as required.

### ⚠️ Hit: a skipped test is not a passing test

The degeneracy test hunted for a seed that produced a collapsed model and **skipped** when it did
not find one — so it had been quietly not running. Degeneracy is now *constructed*: setting every
final-layer table all-positive makes every node output 1, every class tie, and argmax return
class 0 forever. Deterministic, and it actually executes.

### ⚠️ Hit, tooling: MSYS paths and Windows Python

Passing `/c/Users/...` from the Bash tool into Windows Python created a directory literally at
`\c\Users\...` on the current drive. The build reported success and wrote to a path that did not
exist from the shell's point of view. Native `C:\...` paths for anything crossing that boundary.
Same family as the `\v` path-escape bug CLAUDE.md records from the study repo.

### CLI

`dwn2rtl build model.dwn --out rtl/ --input-bits 8` works end to end, from
`dwn2rtl.save(model, thermometer, path)` through to a printed report. Exit codes are meaningful
and distinct: **0** built, **1** the checkpoint was bad, **2** not implemented yet. A bad
checkpoint prints the `CheckpointError` message rather than a traceback — those messages are the
contract, and a traceback buries the part the user needs.

**Suite: 146 passed, 1 xfailed. `pytest -m sim`: 6 passed.**

---

## 4. Built — `dwn_top_tb.v`, and the gate is complete

The 0-byte file from commit `646aebe` is written. **Both levels now pass on every shape:**

```
             core                     top
tiny         504 vectors  PASS        511 vectors  PASS
single       504 vectors  PASS        511 vectors  PASS
n6           504 vectors  PASS        519 vectors  PASS
ten_class    504 vectors  PASS        519 vectors  PASS
all_fixed    504 vectors  PASS        511 vectors  PASS
```

**The encoder is now verified**, and that is the point of this unit rather than a detail. It is
the piece published DWN resource counts leave out; on the smallest studied model it is fourteen
times the network it feeds. An unverified encoder is most of an unverified design.

### Why two testbenches instead of one

The split is what makes a failure **localize itself**:

| | meaning |
|---|---|
| core PASS, top FAIL | the encoder. Nothing else needs re-examining |
| core FAIL, top FAIL | the network; fix that first and this follows |

A single top-level testbench would say only that something, somewhere, was wrong — and the
encoder and the core are emitted by different files, from different parts of the checkpoint.

**That claim is now proved rather than asserted in a comment.** A test corrupts one comparator
constant in `thermometer_encoder.v` and requires the core gate to still **PASS** while the top
gate **FAILS** — including the emitted hint *"If dwn_core_tb PASSED, the fault is in the
thermometer encoder."* If both failed, or neither, the split would be buying nothing.

### Derived, never retyped

`LATENCY` comes from `dwn_top_params.vh` and `IDX_W` from `top_params.vh`, both written by the
same build that emitted the pipeline. The study repo's testbench **hardcoded `IDX_W` to 3**,
which left the upper bits undriven below five classes and truncated the comparison above eight —
a 10-class design was checked on three of its four index bits and passed. `ten_class` is in the
gate specifically to keep that closed.

Comparison is `!==`, not `!=`, so an `x` or `z` counts as a failure rather than propagating into
a comparison that returns `x`. That matters more at this level than at the core: undriven encoder
bits are a real possibility here.

### ✅ The strict xfail did its job

`test_top_testbench_has_content` was an expected failure with `strict=True`. Writing the
testbench turned it into:

```
[XPASS(strict)] dwn_top_tb.v is a 0-byte file -- commit 646aebe landed it empty.
```

— a hard failure that **forced the marker's deletion** rather than leaving a stale TODO nobody
greps for. Replaced by `test_both_testbenches_have_content`, which asserts both files contain a
`module` and a `$readmemh`, so a truncated testbench cannot ship. The build-time
empty-testbench guard stays too, as the second layer.

**The suite now has no xfails at all: 154 passed. `pytest -m sim`: 13 passed.**

A clean end-to-end CLI run, no warnings:

```
$ dwn2rtl build model.dwn --out rtl/ --input-bits 8
features 8, classes 3, layers [12, 9], n=6, z=3   from checkpoint
integer bits 0                                    derived, exact
frac bits 8 -> Q0.8 signed (9-bit)                from --input-bits, provably lossless

core      21 nodes, 4 cycles
encoder   23 comparators of 24 thermometer bits
top       5 cycles latency, II=1

vectors   core 504, top 519, 3/3 classes hit
wrote     rtl/ (17 files)
```

---

## 5. Built — `verify.py`

Pulled forward from phase 2, because `test_build.py` had grown its own copy of simulator
discovery and two implementations of *"where is iverilog"* would drift. That copy is deleted; the
tests now import from `verify.py`, which is what users actually hit.

The whole user journey now runs from the terminal:

```
$ dwn2rtl verify rtl/
iverilog 12.0 (devel) (s20150603-1539-g2693dd32b) (C:\iverilog\bin\iverilog.exe)
  dwn_core  504 vectors  PASS
  dwn_top   519 vectors  PASS
RESULT   PASS
```

**Roadmap V3 closed — verification no longer needs a Vivado licence.** The roadmap called this
*"the largest single barrier to anyone using the tool"*: every other vendor dependency was the
user's own synthesis, which they were doing anyway, but this one made *verification* require a
licence. It does not any more.

### The organising principle: not checking is not passing

Every design decision in this module is a form of that.

| situation | reported as |
|---|---|
| testbench absent | `MISSING`, and the run fails |
| testbench present but 0 bytes | `MISSING`, and the run fails |
| vectors absent | `MISSING`, named, *before* the simulator runs |
| compile error | `ERROR`, never a skip |
| no `RESULT` line printed | `ERROR` — the simulation did not finish |
| testbench printed PASS **and** a nonzero mismatch count | `ERROR` — its summary contradicts its own count |
| no levels ran at all | **FAIL** |

That last one is not hypothetical: `all([])` is `True`, so without an explicit emptiness check a
directory with nothing runnable in it would print a green `RESULT` having checked nothing. It has
its own test, and that test needs no simulator because it is a property of the report.

The missing-vectors check earns its place too: without it `$readmemh` merely warns, the testbench
compares against `x`, and the failure surfaces two layers from its cause.

### Localization is surfaced, not left to be inferred

The two-testbench split only pays off if the tool says which half broke, at the moment it
matters. Verified against a genuinely corrupted comparator:

```
  dwn_core  504 vectors  PASS
  dwn_top   519 vectors  FAIL (49 mismatches)
  the core is bit-exact and the top is not, so the fault is in the thermometer encoder
RESULT   FAIL
```

### Simulator discovery, and why it is not a one-liner

PATH first, then the directories Windows installers actually use. This is the phase 0 finding
made permanent: `winget install Icarus.Verilog` succeeds, installs to `C:\iverilog\bin`, and adds
**nothing** to PATH — so a PATH-only search would tell a user who plainly has a simulator that
they have none. `--simulator` takes an explicit path for anyone with several or one somewhere
unusual, and a not-found error lists what was searched plus the install command for three
platforms.

⚠️ **`iverilog -V` exits 255.** Checking its return code would report a working simulator as
broken. Deliberately ignored, with a test.

**Suite: 172 passed. `pytest -m sim`: 23 passed.**

---

## 6. Validated against REAL checkpoints — and it found a defect

Phase 1 was marked closed before this; it was reopened because the phase 1 report listed *"never
verified against a real checkpoint"* as the top thing phase 2 inherits, and the study repo turned
out to be sitting on the same machine at `Coding-Projects/dwn-fpga`. No upload, no export.

### Every study checkpoint builds and passes

**9 of 9, both levels, unmodified.** The study format loads with no conversion step, which is the
whole reason `checkpoint.py` sniffs rather than insisting on one format.

```
checkpoint                                shape                  nodes   cmp   gate
dwn_jsc_t200_distributive_50_l_b100       16f/5c/[50] z=200         50   202   core:PASS top:PASS
dwn_jsc_t4_distributive_300-100_lr        16f/5c/[300,100] z=4     400    64   core:PASS top:PASS
dwn_jsc_t8_distributive_300-100_lr_b32    16f/5c/[300,100] z=8     400   125   core:PASS top:PASS
dwn_jsc_t8_distributive_300-100_lr        16f/5c/[300,100] z=8     400   124   core:PASS top:PASS
mnist_n6_z3_distributive_w300             784f/10c/[300] z=3       300   720   core:PASS top:PASS
n6_z100_distributive_w2400                16f/5c/[2400] z=100     2400  1348   core:PASS top:PASS
n6_z200_distributive_w1600                16f/5c/[1600] z=200     1600  1986   core:PASS top:PASS
n6_z200_distributive_w2000                16f/5c/[2000] z=200     2000  2145   core:PASS top:PASS
n6_z50_distributive_w3000                 16f/5c/[3000] z=50      3000   763   core:PASS top:PASS
```

MNIST verifies in 3.4 s and JSC in 1.4 s, so real designs are cheap enough for CI.

### ✅ Nine independently-recorded numbers reproduced exactly

The two checkpoints in `COPY-ME.md`'s verification table let the tool be checked against values
measured by a *separate implementation*, months earlier. Every one matches:

| | study repo recorded | dwn2rtl derived |
|---|---|---|
| JSC precision, no `--input-bits` | Q3.12 signed, 16-bit | **Q3.12 signed (16-bit)** |
| JSC comparators / nodes | 202 / 50 | **202 / 50** |
| JSC merge at Q3.12 | 139 of 3,200 | **139 of 3,200** |
| JSC top vectors | 533 | **533** |
| MNIST precision at `--input-bits 8` | Q0.8 signed, 9-bit | **Q0.8 signed (9-bit)** |
| MNIST comparators | 720 of 2,352 | **720 of 2,352** |
| MNIST merge at Q0.8 | 1,169 of 2,352 | **1,169 of 2,352** |
| MNIST top vectors | 1,227 | **1,227** |
| MNIST `1x300` file size (roadmap P8) | 17 MB | **17 MB** |

Nothing was tuned to match. The precision policy in particular derives its answer from the
thresholds alone and independently lands on both datasets' known formats.

### ⚠️ Hit: the scaler was silently dropped, and the emitted RTL said otherwise

**The defect only a real checkpoint could have exposed.** JSC's checkpoint carries
`scaler: {mean, scale}` — a fitted StandardScaler — and `normalize()` discarded it. The synthetic
fixtures have no scaler, so nothing had ever noticed.

Why it matters is in the emitted encoder's own header: thresholds live in whatever feature space
training used, so driving `x_flat` with raw features when the model saw scaled ones *"produces a
design that runs at chance and looks entirely healthy doing it."* Every JSC design this tool
emitted was in exactly that trap.

Worse, that same header ended: **"The checkpoint carries the scaler for this reason."** True of
the checkpoint, false of what dwn2rtl emitted. A user following the instruction would have found
nothing to follow it with.

**Fixed in three places, because a warning without the numbers is a riddle:**

1. `checkpoint.py` preserves the scaler, accepts a live sklearn-style object or a `{mean, scale}`
   dict, and **refuses a per-feature vector whose length does not match the feature count** — an
   off-by-one there would be applied across the whole input, silently.
2. `build()` writes **`input_scaling.json`** with the mean, the scale, *and the word format they
   must be quantized into*, then warns.
3. The emitted header now points at that file, or states positively that the checkpoint records
   no scaler. Absence is a fact, not a silence to be interpreted.

An unrecognized scaler is recorded and warned about rather than ignored or refused — refusing
would reject a checkpoint over something the hardware does not depend on.

**Suite: 179 passed. `pytest -m sim`: 23 passed. Real checkpoints: 9/9 PASS.**

