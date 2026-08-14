# Phase 4 ledger — measure, then optimize

**Goal.** Ship `dwn2rtl estimate`, and use it to make the emitted RTL measurably better —
in that order, because the reverse is how the study repo's area model came to filter zero
configurations across two complete studies.

**Status: IN PROGRESS.** Unit 8 tier 1 is built. Units 1-7 not started.

> Conventions in `overview.md` §6. Entries oldest first, recording *built* / *hit* / *decided*.
> Reversed decisions stay, struck through, with the reason.

---

## The rule this phase is organised around

**No optimization lands on an argument. It lands on a measurement, or it does not land.**

That is not a general principle borrowed from elsewhere; it is this project's own scar. Roadmap
Q7: the study repo built an area model to filter configurations too large to synthesize, and
across two completed studies it filtered **zero**, because both were bounded by training time
rather than by area. A plausible model that nobody checked cost real work and returned nothing.

The corollary, which shapes the whole plan: **`estimate` comes first.** Changing RTL before there
is any way to tell whether the change helped is the same mistake in a different costume.

## What "optimized" can and cannot mean here

Worth stating plainly, because it bounds the phase.

**Most of the design is not ours to optimize.** The LUT tables *are* the model — one node is one
LUT6 by construction, and there is no freedom to make it cheaper. The generator's degrees of
freedom are narrow, and the two largest are already taken:

- only **used** thermometer bits get a comparator (720 of 2,352 on MNIST, not 2,352)
- `argmax` is a balanced tree, not a chain — worth **87.5 -> 108 MHz** when it was fixed

What is left is small, structural, and identified. This phase does those, and stops.

## Plan

| # | unit | why |
|---|---|---|
| 1 | **install yosys** | 703 MB, OSS CAD Suite, Windows. The prerequisite for everything else |
| 2 | **`estimate.py` + `dwn2rtl estimate`** | the last stubbed subcommand. Encoder and core reported **separately**, always |
| 3 | **calibrate against known Vivado numbers** | the study recorded **110 core / 1,519 encoder / 1,621 top LUTs** for JSC `1x50`. Estimating the same design says how much yosys can be trusted |
| 4 | **baseline the real checkpoints** | numbers before any change, recorded here, so later claims are diffs and not impressions |
| 5 | **balanced `popcount` tree** | the one clearly-identified structural defect. See below |
| 6 | **narrow the encoder pipeline register** | `pipe_reg #(.WIDTH(2352))` when 720 bits are driven |
| 7 | **mark `tool-handoff.md` and `tool-roadmap.md` historical** | they describe starting a project that is now built; a new reader meets three plans and cannot tell which is current |
| 8 | **`--data`** | the last designed-but-unbuilt piece of the command surface. See below |

**Not in this phase:** PyPI publishing, Verilator as a second backend. Both are real, neither is
optimization, and mixing them in would blur what this phase is for.

### PyPI — deferred, but the prerequisites are known

Recorded here so they are not rediscovered. Publishing itself is free, takes minutes, and needs
no review; the packaging work was done in phase 0 and a wheel has been verified to install and
run with no source tree present. **`dwn2rtl` was still an unclaimed name as of 2026-08-14.**

Four things to settle first, all small:

1. ⚠️ **The README's relative links break on PyPI.** It is rendered as the project page, so
   `](docs/overview.md)`, `](docs/checkpoint-format.md)`, `](docs/tool-roadmap.md)` and
   `](LICENSE)` resolve against `pypi.org` and 404 — five dead links on the page every new user
   sees. They need absolute GitHub URLs. Same class of defect as the two
   `test_readme_links_resolve` already catches, and that test should be extended to cover it.
2. **`[project.urls]` is missing** from `pyproject.toml` — that is the PyPI sidebar with
   Homepage / Repository / Issues. Without it the page has no route back to the source.
3. **Version is `0.1.0.dev0`.** `pip install dwn2rtl` skips dev releases by default, so the first
   real upload wants `0.1.0`. ⚠️ **Versions on PyPI are immutable** — a version can be *yanked*
   but never replaced, so the first upload is permanent.
4. **The repository is private.** Publishing makes the *package* public while the source stays
   closed, which is allowed but means every GitHub link on the PyPI page 404s for everyone, and
   the README's evidence section points at a repo nobody can open. Decide the two together.

**Prefer Trusted Publishing over an API token**: PyPI is told which GitHub repo and workflow may
publish, and no secret is ever stored. CI already exists to carry it. **TestPyPI** is a
full-fidelity practice instance and is worth one rehearsal, on the same reasoning as every other
rehearsal in this project.

## ⚠️ Finding, 2026-08-13 — the input precision IS usually derivable, and the roadmap said it was not

**Measured before building anything, and it changes unit 8.**

Roadmap Q9 states that fractional width is *"NOT derivable from the checkpoint"*. That is true of
the question it was answering — *is this width safe for my data* — and it was carried forward
into `precision.py`, the CLI and the README as though it settled the whole matter. It does not.

**The thermometer's thresholds are quantiles of the training data, so they inherit its grid.** If
the data had a native quantum, the thresholds sit on that quantum. Probed against real
checkpoints:

```
mnist_n6_z3_distributive_w300      2352 thresholds   on the k/(2^8-1) grid
dwn_jsc_t200_distributive_50       3200 thresholds   no dyadic grid -- genuinely continuous
dwn_jsc_t8_distributive_300-100     128 thresholds   no dyadic grid -- genuinely continuous
```

⚠️ ~~Max error exactly zero across 2,352 values.~~ **Withdrawn — that was float32 arithmetic
flattering itself.** Re-measured in float64:

| | residual against the k/(2^n - 1) grid |
|---|---|
| MNIST, n=8 | **7.57e-06** — on grid |
| MNIST, n=12 | 4.71e-01 — off |
| JSC, every n from 8 to 16 | **~5.0e-01** — off at every width |

**What makes the inference safe is not exactness, it is the margin**: on-grid 7.6e-06 against
off-grid 5e-01, five orders of magnitude apart. Any tolerance between roughly 1e-4 and 1e-2
separates them, so the threshold is not a delicate tuning choice — which is the property worth
having, and it is measured rather than assumed.

JSC does not false-positive: standard-scaled features land on no dyadic grid at any width, which
is the correct answer.

⚠️ **The smallest n is load-bearing, and this is a real trap.** MNIST also passes at n=16
(residual 1.9e-03), because `k/255 = 257k/65535` exactly — **every coarse grid is a subset of
every finer one**. The inference must therefore take the *smallest* n that fits, which is the
coarsest grid the data actually lives on and so its true quantum. Any other match gives a
needlessly wide word, which costs area in exactly the way this phase is otherwise trying to
reduce.

So `dwn2rtl build model.pt` with **no flags at all** can derive `Q0.8, 9-bit` for MNIST rather
than falling back to a 16-bit default — and can still say honestly that JSC's width is a default.

**Why this is sound, not a heuristic.** Finding the *smallest* n whose grid fits gives the coarsest
grid the data lies on, which is its native quantum. The §6 proof then applies unchanged: values
are `k/(2^n - 1)`, quantising at `frac = n` gives a step of `2^n/(2^n - 1) > 1`, so the mapping is
strictly increasing and no order comparison can change.

**What it does NOT establish**, and the report must not claim: that *inference-time* data will
share the grid of *training* data. Nothing can establish that from a checkpoint — but it is
exactly the same assumption a user makes when they type `--input-bits 8`, so inference is no
weaker than the flag it replaces. It should be labelled *inferred* rather than *given*, and
`--input-bits` stays as an override for anyone who knows better.

## Unit 8 in detail — zero-friction precision, in three tiers

The goal is `dwn2rtl build model.pt` and nothing else. Three tiers, in order:

| tier | when | result |
|---|---|---|
| **1. infer from thresholds** | the thresholds lie on a dyadic grid | automatic, **provably lossless**, no flag |
| **2. `--data sample.npz`** | they do not, but the user has data | infer from the sample, else measure bit-error at candidate widths |
| **3. default** | neither | the documented default, reported as a DEFAULT |

`--input-bits` remains an override at every tier.

Tier 1 is new, cheap, and removes the flag for images and any digital sensor — which is most
users. Tier 3 already works. Tier 2 is the original roadmap item, below.

### `--data`, for the user who has data but no grid

Roadmap §5.2 specified **three** ways to reach a fixed-point format. Two are built:

```
dwn2rtl build model.pt                    derives everything derivable; the width is a DEFAULT
dwn2rtl build model.pt --input-bits 8     you know the input's precision; provably lossless
dwn2rtl build model.pt --data test.npz    NOT BUILT -- measure what needs data
```

The gap matters for a specific person: someone who has a trained model and a data sample but
cannot say what precision their features carry. Today they must omit `--input-bits` and accept a
16-bit word that is probably wider than they need, flagged honestly as unproved. `--data` closes
that, in two stages that must not be confused with each other:

**1. Infer, when the input has a native quantum.** If every value in the sample is `k/(2^n - 1)`
for integer `k`, then `n` is the answer and quantising there is **provably** lossless — the same
argument that makes 8-bit pixels safe. This is a proof about the input domain, not a measurement
of a sample, so it does not depend on the sample being representative.

**2. Measure, when it does not.** For a genuinely continuous input there is no proof to be had.
Report the encoder bit-error at candidate widths and label the result **a stress test, not a
guarantee.**

⚠️ **Stage 2 is where this feature could do harm, and the scar is specific.** `docs/jsc/report.md`
§5.6 *(study repo)* records an encoder narrowing that was fitted **and validated on the same
1,000 samples**, and 8 of 15 features came out too narrow when checked against held-out data. A
width chosen from a sample and then blessed by that same sample is not evidence. So: `--data`
must never report a measured width as *proved*, the `Precision.proved` flag stays reserved for
the quantum argument, and the output must say which of the two it did.

**Not an optimization, and slightly outside this phase's theme** — included because it is the
last piece of the designed command surface, and because a user who over-widens their word by
four bits pays for it in exactly the area this phase is otherwise trying to measure.

## Unit 5 in detail — the one that is clearly worth doing

`rtl/popcount.v` is still a linear loop:

```verilog
for (i = 0; i < WIDTH; i = i + 1)
    count = count + bits[i];
```

Its comment says this is fine because addition is associative and the tool rebalances it.
**Nobody has ever checked that.** And roadmap §5.1 wrote the rule after `argmax` cost 20 MHz for
exactly this shape:

> ⚠️ **Emit balanced reductions explicitly, do not rely on synthesis to fix a loop.** Whether a
> generated reduction is fast should not depend on a property of the operator that the emitter
> never checks.

**The structural claim needs no simulator and no synthesis tool:** a balanced adder tree is
`ceil(log2(W))` deep against `W-1` for a chain. At MNIST's 30-wide group that is 5 against 29.
What measurement is for is confirming it does not *cost* anything in area — not proving it is
shallower, which is arithmetic.

## The risk that could make this phase actively harmful

⚠️ **yosys is not the tool users synthesize with.** They use Vivado, Quartus, or something else,
and yosys's generic LUT mapping is not those tools' mapping. Tuning the emitter until *yosys*
numbers improve would be optimizing against the wrong target, and would be a new form of the
mistake in Q7 — a model that looks authoritative and answers a question nobody asked.

Two guards:

1. **Unit 3 calibrates before anything is changed.** The study repo has real Vivado numbers for a
   design we can estimate. If yosys disagrees wildly, that is a finding, and the honest response
   is to report estimates with a stated confidence rather than to trust them.
2. **Every change must be justified structurally first** — shallower tree, fewer flops — with
   measurement used to confirm it *did not cost* something, never as the reason for the change.
   A change whose only argument is "yosys says the number went down" does not land.

**And `estimate` output must say what it is.** Not a vendor number, not a guarantee: an estimate
from a generic mapping, with the user told to synthesize for real figures. The tool already
distinguishes *derived*, *proved* and *default* in its precision reporting; this is the same
discipline applied to area.

## The gate applies to every one of these

Any change to `popcount.v` or the emitters is a change to the design, so
`pytest -m sim` must pass on all five synthetic shapes **and** the real checkpoints, before and
after. A faster design that computes the wrong answer is worth nothing, and this is precisely the
kind of change — a reduction rewritten for depth — where an off-by-one is easy and invisible.

## Expected shape of `dwn2rtl estimate`

```
$ dwn2rtl estimate rtl/
yosys 0.xx

  encoder   1519 LUT6      thermometer_encoder
  core       110 LUT6      dwn_core
  top       1621 LUT6      dwn_top

  the encoder is 13.8x the core

ESTIMATE -- yosys generic mapping, not your vendor's toolchain.
Synthesize for real numbers; treat these as relative, not absolute.
```

Reporting the **encoder-to-core ratio** directly is deliberate. It is the project's headline
finding — published DWN resource counts omit the encoder, and on the smallest studied model it is
fourteen times the network it feeds — and a tool that emits both and reports only their sum would
reproduce the reporting defect it exists to correct.

---

## 1. Built — unit 8 tier 1: precision inferred from the thresholds

**`dwn2rtl build model.pt` with no flags now derives the right word width for quantised inputs.**
Measured on the real checkpoints:

| | before | now |
|---|---|---|
| MNIST `1x300` | Q0.12, **13-bit**, unproved | **Q0.8, 9-bit, inferred, provably lossless** |
| JSC `1x50 z=200` | Q3.12, 16-bit, default | Q3.12, 16-bit, default — correct, no grid exists |
| JSC `300-100 z=8` | Q1.12, 14-bit, default | Q1.12, 14-bit, default |

Four bits narrower on every image model, with no user input, and it is the *provable* kind of
lossless rather than the hoped-for kind.

**`precision_for()` now runs three tiers:** `given` (the flag) -> `inferred` (the grid) ->
`default`. `Precision.proved` became a **property derived from `source`** rather than a field
stored beside it, so the two cannot drift.

**Decided: `inferred` is labelled distinctly from `given`, though both are equally proved.** Both
rest on the same assumption — that inference-time data shares training-time precision — so
inference is no weaker than the flag it replaces. But a user should be able to see that the tool
made the choice, and override it if their deployment differs.

### ⚠️ Hit: the first implementation would have emitted a ONE-BIT word, labelled provable

Caught by `test_a_scaled_integer_grid_is_recognised[16]`, whose `k/65535` inputs are all below
0.001. It inferred **n=1**.

**Cause.** Checking those values against the n=1 grid multiplies by 1, everything rounds to 0,
and the residual is ~9.6e-04 — under the 1e-3 absolute tolerance. So a grid that resolves nothing
at all looked like a perfect match. On any model with small-magnitude features this would have
emitted a **one-bit fractional word described as "provably lossless"**: silently catastrophic,
and exactly the false positive this function must never produce.

**Fix.** Closeness is not the criterion; **separation** is. A grid only qualifies if it maps the
distinct thresholds to *distinct* grid points:

```python
if np.unique(rounded).size == distinct.size:
```

That is the honest statement of what "these values lie on this grid" means — if two distinct
thresholds collapse onto one grid point, the grid is too coarse to be the quantum they came from.
It subsumes the tolerance rather than replacing it, and needs no tuning.

### ⚠️ Hit, the other way round: the test data was wrong and the code was right

`test_a_raw_fixed_point_grid_is_recognised[12]` failed, reporting 9 instead of 12. The
implementation was correct. The test sampled `arange(0, 4096, 40)`, so every `k` was a multiple
of 40 — and values that are all multiples of 40 over 4096 **genuinely do lie on the coarser
m/512 grid.**

Worth keeping, because it is true of real data too: **a subsampled grid is a different, coarser
grid.** A sensor read every fourth count is effectively lower precision, and the inference will
say so. Tests now use consecutive `k` including `k=1`, which pins the grid exactly because
`1/(2**n - 1)` lies on no coarser one.

**Two defects in one unit, one in each direction** — the implementation wrong where the test was
right, and the test wrong where the implementation was right. Neither was findable by reading.

**Suite: 229 passed, 1 skipped.**

## 2. ⚠️ Hit — a test that passed on Windows and was vacuous on Linux

`2656da3` added tests for the simulator fallback, which had been exercised constantly and checked
nowhere. Good commit: the fallback runs on every machine where iverilog is not on PATH — the
default after `winget install Icarus.Verilog` — so breaking it would have turned the gate red
immediately, and that *looked* like coverage while depending on one developer's PATH staying
broken.

**CI then failed on both Ubuntu jobs**, and only there:

```
test_path_wins_over_the_fallback
  assert '.../stale' == '.../chosen'
```

**Cause: the stub install was Windows-shaped.** `_fake_install` wrote `iverilog.exe` with no
execute bit, and `shutil.which` does not mean the same thing on the two platforms. Probed
directly rather than reasoned about:

| stub | Linux | Windows |
|---|---|---|
| `iverilog.exe`, no `+x` | **None** | found |
| `iverilog`, `+x` | found | **None** |
| `iverilog`, no `+x` | None | None |

So on Linux **nothing was on PATH at all**. The fallback won by default, and the assertion caught
a real difference between the platforms rather than a bug in `verify.py` — which is correct as
written: PATH does win, when there is anything on PATH to win.

⚠️ **The test was worse than failing: on Windows it was passing vacuously in the other
direction.** It only ever proved the fallback loses to a `.exe` that Windows happens to resolve.
The precedence it claimed to test was never exercised on either platform.

**Fix:** `_fake_install(..., on_path=True)` writes the platform's own name — suffix conditional on
`os.name` — and sets the execute bit. The branch is needed in **both** directions, per the table:
a bare name is invisible to Windows, a `.exe` is invisible to POSIX. One fixed name cannot work.

**The lesson is about the matrix, not the test.** This is the second defect CI's Linux job has
caught that no amount of local Windows running could (after `requires-python = ">=3.9"`, which
was unrunnable). Both were *claims that happened to hold on one machine*. That is precisely what
phase 2's report predicted a matrix is for, and it is now evidenced twice.

---

## 3. Built — yosys installed, and unit 3's calibration done BEFORE anything was changed

**yosys 0.68**, from the OSS CAD Suite Windows build (564 MB compressed, 2.1 GB extracted, at
`C:\oss-cad-suite`). Removable by deleting that one directory — no installer, no registry, and
**deliberately not on PATH**: the suite bundles its own `iverilog.exe` and `vvp.exe` among 136
binaries, and putting it ahead of `C:\iverilog\bin` would silently change which simulator the
entire gate runs against. `estimate` will find it the way `verify` finds iverilog — PATH, then
known locations — which is a design this project already has.

⚠️ **yosys needs its own `bin` and `lib` on PATH to run at all.** Invoked by absolute path with a
clean environment it prints nothing and exits 0 — a silent no-op, which is the worst possible
failure. `estimate` must pass a prepared environment to the subprocess. (The suite's *iverilog*
does not need this, so `verify.py` is unaffected — checked, not assumed.)

### 🎯 The calibration, and it changes what `estimate` is allowed to claim

JSC `1x50 z=200`, built **unpipelined** so it matches the study's out-of-context figures exactly
(FF = 0 in both, confirmed on both sides — the first attempt compared a 4-stage build against
Vivado's unpipelined numbers and was not apples-to-apples):

| module | yosys `$lut` | Vivado (study, `xc7a35t`) | ratio |
|---|---|---|---|
| `dwn_core` | **106** | **110** | **0.96x** |
| `thermometer_encoder` | **717** | **1519** | **0.47x** |
| `dwn_top` | **838** | **1621** | **0.52x** |

⚠️ ~~dwn_core 120 (1.09x), dwn_top 854 (0.53x)~~ — **withdrawn, and the reason is a trap worth
keeping.** The first run used `synth -top M` **without `-flatten`**, so yosys left every
`lut_node` as its own module in gate primitives (`$_AND_`, `$_NAND_`, ...) and `abc -lut 6`
mapped **only the top level**. The `$lut` figure scraped from that output was one stray module's
count, not a total, and it happened to look plausible. Fixed with `synth -top M -flatten`, after
which there is exactly **one** `$lut` line in the output — which is now the check that flattening
actually happened, rather than something assumed.

**Two completely different levels of agreement, in one design.**

- **The LUT core: yosys is within 4%.** That is close enough to act on. A 50-node core is 50
  LUT6 by construction plus the popcount and argmax trees, and both tools find essentially the
  same thing.
- **The encoder: yosys reports less than half.** 717 against 1519 for the same 202 comparators —
  **3.5 LUTs each against Vivado's 7.5**. Not a yosys bug: a 16-bit signed comparison against a
  constant genuinely fits in ~3 LUT6s, and Vivado on 7-series maps comparators onto carry chains
  instead, which costs more LUTs as carry inputs. Generic mapping is simply *optimistic* here.

⚠️ **This is the risk the plan named, now measured.** Tuning the encoder until yosys's number
falls would be optimizing against a tool that disagrees with the user's by 2.1x.

⚠️ **And it distorts this project's headline claim.** The encoder-to-core ratio is **13.8x by
Vivado and 6.8x by yosys**. A tool that printed the yosys ratio unqualified would *understate the
very thing it exists to highlight* — that published DWN resource counts omit an encoder which can
dominate the design.

**Consequences, decided here rather than discovered later:**

1. `estimate` reports LUT counts **with the module they belong to**, and states plainly that
   generic mapping under-reports comparator-heavy logic. It is an estimate, and the *encoder*
   half of it is the unreliable half.
2. **Unit 5 (balanced popcount) is measurable** — it is core-side, where yosys and Vivado agree
   to 4%.
3. **Encoder-side changes are not reliably measurable with this tool.** Unit 6 stands anyway,
   because "do not emit flops nothing drives" is structural and needs no synthesis tool to
   justify.

## 4. Built — `estimate.py` and `dwn2rtl estimate`

The last stubbed subcommand. Roadmap **P2's command surface is now complete**: `build | verify |
estimate` all work.

```
$ dwn2rtl estimate rtl/
yosys 0.68+64 (C:\oss-cad-suite\bin\yosys.exe)

  thermometer_encoder       717 LUT        0 FF
  dwn_core                  106 LUT        0 FF
  dwn_top                   838 LUT        0 FF

  the encoder is 6.8x the core, as generic mapping sees it

ESTIMATE -- yosys generic mapping, not your vendor toolchain.
Calibrated once against Vivado on xc7a35t: the CORE agreed within 4%, the ENCODER came
out 2.1x LOW because generic mapping packs comparators more tightly than a carry-chain
architecture does. Treat core numbers as indicative, encoder numbers as a floor, and
synthesize for figures you can publish.
```

**Decided: the calibration is printed with every report, not buried in a doc.** The two halves of
that output have genuinely different authority, and a reader given both without the caveat would
grant the encoder figure the core figure's credibility. Same discipline as `derived` / `inferred`
/ `DEFAULT` in the precision line.

**Decided: `find_yosys` rejects a candidate that cannot report a version.** The silent-no-op case
is not hypothetical — it is what this yosys does when invoked by absolute path with a clean
environment. Accepting it would produce empty measurements that look like a very small design.

⚠️ **`-flatten` is load-bearing and is now checked, not assumed.** Without it, `abc -lut 6` maps
only the top level and the scraped `$lut` figure is one stray submodule's count. After flattening
there is exactly **one** `$lut` line, and `estimate` treats more than one as an ERROR rather than
summing them.

## 5. ⚠️ Unit 6 measured — the dead flops are real, and yosys does NOT trim them away

The premise was `pipe_reg #(.WIDTH(2352))` when only 720 bits are driven, with the emitter's own
comment hoping synthesis would remove the rest. Measured on MNIST `1x300` by toggling that one
stage:

```
PIPE_ENC=1   dwn_top  1821 LUT   1808 FF
PIPE_ENC=0   dwn_top  1818 LUT    704 FF
```

**The encoder register costs 1,104 flops. Only 720 bits are driven.** So ~384 are dead — 53%
overhead on that register — and the hope in the comment is only half right: yosys trims some of
the 2,352 but not down to the useful set.

**Unit 6 is therefore justified by measurement as well as by structure**, and it is FF-side, so
the 2.1x encoder-LUT calibration gap does not apply — this is a flop count, which both tools
agree on.

**Suite: `pytest tests/test_estimate.py` 17 passed.**

## 6. Built — yosys in CI, on Linux only

`apt install yosys` added to the Ubuntu jobs, plus a step that **fails the run if yosys was
installed and is not usable**. Same reasoning as "The gate actually ran", one notch weaker:
`estimate` is optional, so its tests skipping without yosys is correct — but on a runner where we
just installed it, a skip means the estimate tests silently did not run and the install achieved
nothing.

**Linux only, deliberately.** ~9 MB from apt against a 564 MB OSS CAD Suite download on Windows,
which would also ship its own iverilog into the runner and could shadow the simulator the gate
uses. Covering `estimate` on one platform is enough; the gate itself still runs on both.

⚠️ **Unverified: Ubuntu ships yosys 0.33 and this was developed against 0.68.** Thirty-five
releases apart. The parser depends on `synth -top M -flatten`, `abc -lut 6`, and `stat` printing
`NNN $lut` — all long-stable, none of them confirmed on 0.33 from here, because installing it
locally needs an interactive sudo password. **CI is the test.** If it fails, the likely culprits
in order are the `stat` cell-name format, then `-flatten`.

Recording this as unverified rather than asserting it works is the point: the two defects CI has
already caught were both claims that happened to hold on one machine.
