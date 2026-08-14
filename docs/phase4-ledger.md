# Phase 4 ledger — measure, then optimize

**Goal.** Ship `dwn2rtl estimate`, and use it to make the emitted RTL measurably better —
in that order, because the reverse is how the study repo's area model came to filter zero
configurations across two complete studies.

**Status: PLANNED, not started.**

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
mnist_n6_z3_distributive_w300      2352 thresholds   on the k/(2^8-1) grid, max err 0.00e+00
dwn_jsc_t200_distributive_50       3200 thresholds   no dyadic grid -- genuinely continuous
dwn_jsc_t8_distributive_300-100     128 thresholds   no dyadic grid -- genuinely continuous
```

**Max error exactly zero across 2,352 values.** That is not a near miss to be thresholded, it is
the same grid. And JSC does not false-positive: standard-scaled features land on no dyadic grid,
which is the correct answer.

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
