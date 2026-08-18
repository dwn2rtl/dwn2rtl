# What the generator needs to become a tool — an audited work list

> ## ⚠️ HISTORICAL — superseded by `overview.md` §6 and the phase ledgers
>
> **This was the work list for building the tool, and it is done.** Phases 0–5 are closed and
> `dwn2rtl 0.1.0` is on PyPI. There is no live phase ledger; `overview.md` §6 has the map and
> `phase5-report.md` §6 lists what remains, all of it optional. Nothing in this file should be
> picked up as a task without checking there first.
>
> **P1–P8** (fork, package, licence, CI, publish) and **Q4** (the name, and where it lives —
> `dwn2rtl`, at `github.com/dwn2rtl/dwn2rtl`) are resolved, though the §8 status block below
> still shows them pending: it is a snapshot of what was known then, and is left as one.
>
> It stays because it is the **reasoning archive**: the `V*`/`P*`/`Q*` items record why the
> project does *not* do certain things, and those arguments are still load-bearing. Q7 (no area
> model) and §5.1 (emit balanced reductions explicitly) are cited by phase 4 as its organising
> rules.
>
> ⚠️ **One entry is now known to be wrong.** **Q9** states that input fractional width is *not*
> derivable from the checkpoint. Measured on 2026-08-13 against real checkpoints, it usually
> **is** — MNIST's 2,352 thresholds lie exactly on the k/(2⁸−1) grid, max error 0.00e+00. The
> withdrawal and what replaces it are in `phase4-ledger.md`. Read Q9 as the question it actually
> answered — *is this width safe for my data* — and not as settling the whole matter.
>
> Timings below (**BEFORE / DURING / AFTER** the MNIST port) are relative to a milestone that has
> already passed and no longer schedule anything.

**What this is.** Every change required to turn the DWN→RTL generator in this repo into something
another person could use on their own model, with each item marked by *when* it should happen
relative to the MNIST port.

**How it differs from the two docs beside it.** `docs/reference/reusable-generator.md` *(not in this repo)* argues *whether* to
build the tool and scopes it in weeks. `docs/mnist/plan.md` *(not in this repo)* covers what MNIST specifically needs.
This is the union, audited against the code as it stands, with file and line references so nothing
here is a guess. Where the three disagree, this one was checked most recently.

**Identifiers.** `B*`, `F*`, `R*` and `T*` are `docs/mnist/plan.md` *(not in this repo)* §2's; `M1a`–`M1g` are its §3
steps. Items that already have an ID there keep it — this document does not invent a second name
for the same thing. `B4`, and the `V*`/`P*`/`Q*` groups below, are new here because no existing ID
covers them.

**Timing legend.**

| | meaning |
|---|---|
| **BEFORE** | do it before MNIST. Either it is wrong today, or MNIST cannot start without it. |
| **DURING** | MNIST is what surfaces or validates it. Doing it earlier means guessing. |
| **AFTER** | packaging and polish. Real work, but it should follow a generator that has stopped moving. |
| **ANYTIME** | independent of MNIST; costs little and unblocks thinking. |

**The organising principle**, from `docs/mnist/plan.md` *(not in this repo)* §1.3, and worth repeating because it is the
test that decides most of these calls:

> *Would this still be right for a dataset we have not thought of?* Not: *does this work for MNIST?*

---

## 1. Defects in shared code — wrong today, not merely JSC-shaped

These are not generalisations. They are bugs that JSC's particular shape hides, and each fails
*silently* rather than erroring.

| # | Defect | Evidence | Why it is dangerous | When | Effort |
|---|---|---|---|---|---|
| ~~**B1**~~ ✅ **done 2026-08-11** (`fbc3be8`) | Feature count hardcoded in the core emitter: `input_bits = 16 * cfg['thermometer_bits']` | `rtlgen/emit_core.py:97` | Emits a core with the wrong input width for any non-16-feature model. No error — Gate 1 fails confusingly, one level away from the cause. **`rtlgen/emit_encoder.py:41` and `tb/gen_vectors.py:74` already derive it correctly** (`thresholds.shape[0]`), so this is one file disagreeing with the two beside it | **BEFORE** | 30 min |
| **R1** | Area model is JSC-calibrated and self-tests at five classes | `dse/area_model.py:39,48,87,168,183` | `predict()` now *takes* `features` and `num_classes`, so the shape is parameterised — what remains is the `JSC_FEATURES` default and the **calibration constants**, which are the part that is actually wrong for another dataset. Nothing in the Gate 1 path reads it, so it blocks *predicting* MNIST area, not measuring it | **DURING** — and see R3, which decides whether a corrected R1 can be right at all | 1 h |
| **R2** | Grid is JSC-shaped throughout — size ladder, `tau` anchors, slugs, `NUM_CLASSES = 5` | `dse/grid.py:41,59,95` | ~~Not a defect in the emitter~~ — but as of 2026-08-12 it is **blocking MNIST Phase 2** (`docs/mnist/phase2-ledger.md` *(not in this repo)* M2c). `dse/run.py` resolves a checkpoint *by grid slug*, so no MNIST slug exists and `--all` finds nothing to run | ⬅ **NOW**, promoted from AFTER | 2–3 h |
| **R3** 🆕 | **There is no live encoder cost model** | `docs/mnist/phase1-ledger.md` *(not in this repo)*, the 2026-08-11 retraction | Two hypotheses have now died: `features × z` comparators (the paper's MNIST config was projected at ~67,000 encoder LUTs and is really far less, because the **learned mapping only builds comparators for bits it picks** — 720 of 2,352 on the board model), and the amortisation explanation for LUTs-per-comparator. **R1 cannot be made correct until one replaces them**, and the tool's area reporting depends on it | **DURING** | 3 synthesis runs |

> **Status note, 2026-08-12.** **Every BEFORE item is done.** B1, B4 and F1 all landed, and the
> descriptor work went considerably further than B4 described (below). JSC reproduces exactly:
> 12/12, areas **110 / 1,519 / 1,621** — ⚠️ *not* the 108 / 1,519 / 1,619 this document was written
> with. The argmax tree costs +2 in the core and +2 in `dwn_top`; the pre-change values are pinned
> at the `jsc-complete` tag. Plus the two-layer `300-100` checkpoint bit-exact, and 166,000/166,000
> on silicon.

> ⚠️ **B4 was narrower than the defect it named, and this is the useful lesson.** It said "add a
> dataset descriptor." The descriptor was added — and **nothing imported it** for a month, while
> every consumer kept a private copy of JSC's constants. The refactor of 2026-08-12 found **seven**
> sites of the same shape, *a dataset constant sitting where a derived value belongs*, and the fix
> that makes it stop is `datasets.identify(ck)` plus removing module-level constants entirely, so
> a missing width is a `TypeError` at the call site rather than a plausible wrong number reaching
> the FPGA. **Adding the right data structure is not the same as making anything read it**, and
> only the second half stops the bug recurring.

**R1 and R2 sit in `dse/`**, which no part of emit → Gate 1 → synthesise touches, so neither can
corrupt a bring-up. That is still true — but R2 now blocks the MNIST *sweep*, which is a different
claim from blocking the port, and this document previously conflated them.

---

## 2. Generality gaps — where the flow assumes a dataset

Real work, no current failure. Ordered by whether MNIST is blocked without them.

| # | Gap | Evidence | When | Effort |
|---|---|---|---|---|
| ~~**F1**~~ ✅ **done 2026-08-11** | **Configurable precision.** `Q3.12` was a module-level constant | ~~`exporter/extract.py:118-119`~~ — **`extract.py` now has no module constants at all**; `quantize`, `quantize_thresholds`, `saturation_is_lossless` and `fits_in_word` take widths as **required** arguments. `HardwareConfig.from_dataset()` (`rtlgen/config.py:128`) resolves them from the descriptor. The surviving assert (`:210`) checks the *default instance* against `datasets.JSC` — a regression guard, not a constraint | **BEFORE** | 1 day incl. verification |
| ~~**B4**~~ ✅ **done 2026-08-11** (`9815062`) | **A dataset descriptor** so dimensions are data, not code | ~~none exists~~ `datasets/__init__.py`, frozen `Dataset` per dataset with `check_checkpoint()`. Its first act was to catch a real bug: `record_bytes()` used `word_bits // 8`, which gives one byte for an 11-bit word | **BEFORE** | half day |
| ~~**B2**~~ ✅ **done 2026-08-11** | **Board record format fixed at 33 bytes** | Now derived. `record_bytes()` rounds up rather than flooring, and `uart_loader.v` shifts bytes into the record instead of indexing a 6-bit counter that silently capped at 63. **Verified on silicon 2026-08-12** at 1,569 bytes/record | **DURING** — and per Q1, **not tool work**; this was the demo | 1 day |
| ~~**B3**~~ ✅ **done 2026-08-11** | **Vector store sized for JSC** — `DATA_W=256`, `DEPTH=1024` | Dimensions come from the model; `--depth` is the one argument a checkpoint genuinely cannot supply, being a property of the bitstream | **DURING** — same, not tool work | half day |
| **P7** | **Default checkpoint paths name a JSC file** in four scripts | `run_gate1.py:27`, `run_tb.py:25`, `host.py:50`, `verify_phase1.py:145` — ⚠️ **also `experiments/make_test_checkpoint.py:51`**, which is five, not four | **AFTER** — they are defaults with overrides, harmless until packaging | 1 h |

**F1 was the one that decided whether MNIST fits at all**, and it delivered: MNIST runs at
**Q0.8, a 9-bit word**, and the format turned out to be *lossless on the full test set* —
0 of 10,000 predictions differ from float32 despite 56,835 values saturating, because 8-bit pixels
have only 256 distinct values. `docs/jsc/report.md` *(not in this repo)* §7's JSC finding (11-bit gives a 5.80× smaller encoder
for 0.142 pp) generalised, and then some.

⚠️ **And it moved the headline constraint.** Phase 1 projected the paper's `2x[1000,500]` at
**102.5% of the device at 16-bit and 38.0% at 11-bit** — so **word width, not `z`, is what decides
whether a model fits**. That is the opposite of what this document assumed when it called
`784 × z` "the entire area problem" (§3), and the correction is R3's.

**On `LABEL_W`:** the harness already parameterises it (`benchmark_fsm.v:35`,
`dwn_basys3_top.v:47`, default 3). MNIST's ten classes need 4 bits, which is a parameter change
rather than a rewrite. Good news, and worth recording so nobody re-audits it.

---

## 3. What only MNIST can settle — **DURING**

These cannot be decided by inspection. Attempting them earlier means guessing.

**Five of six are now settled**, all by MNIST Phase 1 (2026-08-11/12).

| Question | Answer |
|---|---|
| ~~Does the generator emit correct RTL at 784 features and 10 classes?~~ | ✅ **Yes.** Gate 1 bit-exact, then **Gate 1b 10,000/10,000 on silicon**. This is the claim the whole exercise existed to test |
| ~~How many thermometer thresholds per pixel?~~ | ✅ **z=3**, which is what upstream uses. And the premise was wrong: `784 × z` is *not* the area problem, because the learned mapping builds comparators only for bits it selects (720 of 2,352). **Word width is** — see F1 |
| ~~Are MNIST pixels standard-scaled or min-max?~~ | ✅ **min-max, [0,1]**, via `transforms.ToTensor()`. No scaler in the pipeline |
| ~~Does the upstream MNIST recipe binarise differently?~~ | ✅ **No.** Same `DistributiveThermometer`, `feature_wise=True` |
| ~~Does the paper's `1000, 500` fit at any precision?~~ | ✅ **BUILT 2026-08-12, and it fits at 9-bit.** `2x[1000,500]`: 97.76%, **3,464 LUTs (16.65%)**, 103.8 MHz — the best MNIST design that meets the board clock. The 11-bit projection was unnecessarily pessimistic |
| ~~Why is LUTs-per-comparator lower on real data than synthetic?~~ | ✅ **Settled 2026-08-13: TWO mechanisms, previously conflated.** *Within* a dataset it is logic sharing — cost per comparator falls monotonically as comparators-per-feature rises (0.218 → 0.086 LUT/bit over three measured points). *Between* datasets that fails to explain it, since JSC has the most comparators per feature and is still dearest, leaving threshold **values** as the cross-dataset mechanism. See `docs/mnist/phase2-ledger.md` *(not in this repo)* |

**The most valuable output of MNIST is a list of bugs**, not an accuracy number — and it delivered.
Beyond the four JSC-shaped assumptions Phase 2 and 3 found, MNIST produced **seven more** in one
refactor (§1), three silent-cap bugs (`uart_loader`'s 6-bit counter, `host.py`'s byte alignment, a
testbench checking three of four class-index bits), a linear `argmax` chain that cost 20 MHz, and
two dead cost models. **Not one of them was findable by inspection.** That is the argument for P5
(CI on a second dataset) stated as evidence rather than as principle.

---

## 4. The verification story — what makes the tool worth using

This is the strongest differentiator over rolling your own, and most of it already exists.

| # | Item | State | When |
|---|---|---|---|
| **V1** | **Ship Gate 1 with the tool** | `tb/dwn_core_tb.v` already self-checks and prints `PASS (bit-exact on every vector)` | **ANYTIME** (decision), **AFTER** (packaging) |
| **V2** | **Vectors without a dataset** — generate random inputs, run both golden model and RTL | **half built.** `experiments/make_test_checkpoint.py` synthesizes a checkpoint *and* its `_testvectors.npz` from a `datasets/` descriptor, and it proved itself: **Gate 1 passed at MNIST's shape — 784 features, 10 classes, 2 layers — before any MNIST training existed.** What is missing is the other direction: `tb/gen_vectors.py:97` still requires a saved npz beside a *real* checkpoint, so a user with their own model and no test set cannot verify it | **AFTER** |
| **V3** | **Simulator independence** — Verilator or Icarus alongside `xsim` | Unchanged. `run_gate1.py` hardcodes Vivado's `xsim` (`:74,115,132`), but behind `find_vivado_bin()`/`run_xsim()`, so it is a backend swap. ⚠️ **Now the largest single barrier to anyone using the tool** — every other Vivado dependency is the user's own synthesis, which they were doing anyway; this one makes *verification* need a Vivado licence | **AFTER** |
| ~~**V4**~~ ✅ **exercised 2026-08-12** | **A precision-choice procedure**, not a constant | Worked, and the contract below held exactly. MNIST derived **Q0.8**; `saturation_is_lossless()` said the format was safe and the **full 10,000-sample test set confirmed it — 0 divergences**. Note which half did the work: the *derivation* gave the floor, the *data* upgraded it to measured | **DURING** — done |

**On V1 — ship it.** A generator whose output nobody can check is worth much less, and this
project has the evidence: the emitter's own read-back check reported 20/20 correct while the design
was wrong on 958 of 1,504 vectors. Only an independent golden model caught it.

**On V2 — random vectors are *better* than dataset vectors** for this purpose. You are verifying
the emitter, not the model, and random inputs hit address patterns a trained model's data may never
produce. `experiments/make_test_checkpoint.py` already uses exactly this argument to run Gate 1 at
n=2 and n=4 without a trained model. It also makes `verify()` fully self-contained.

**On V4 — this is the design question the tool turns on.** Precision splits cleanly:

- **integer bits** — derivable exactly from the checkpoint's thresholds, and the *renormalisation*
  trick in `docs/jsc/report.md` *(not in this repo)* §7 removes the question entirely: map each feature affinely into [−1, 1) and
  the integer width is 1 for every dataset, forever, with no retraining because a comparison is
  unchanged by a monotonic rescale applied to both sides.
- **fractional bits** — **not** derivable from the checkpoint. Whether quantisation changes
  predictions depends on the data. `docs/jsc/report.md` *(not in this repo)* §5.6's scar applies directly: the encoder-narrowing
  result was fitted and validated on the same 1,000 samples, and 8 of 15 features were narrowed too
  far. A tool that picks fractional width from a checkpoint-only heuristic reproduces that bug for
  every user.

So the honest contract: **derive a floor from the checkpoint and say it is a floor; upgrade to
"measured" only when given data.** Never silently claim a width is safe.

---

## 5. Packaging — **AFTER**

Only start once the generator has stopped changing. `docs/reference/reusable-generator.md` *(not in this repo)* §5 gives the
reason: maintaining a fork while the emitters move means merging every change twice.

| # | Item | Notes | Effort |
|---|---|---|---|
| **P1** | Fork `main`, then prune | Fork rather than fresh repo — the history is where the reasoning lives (address bit order, the `__dummy_mapping` trap, the packbits shift) | 1 day |
| **P2** | `pyproject.toml`, one CLI entry point | none exists today. **The command surface is designed in §5.2 — it is a decision, not a formatting job** | 1 day |
| **P3** | **A licence** | **none exists.** Blocks anyone using it, and blocks citing it | 1 h |
| **P4** | README, worked examples | `rtl/example-model-1x50/` is already a good artifact to build on | 1–2 days |
| **P5** | CI on a second dataset | The claim is generality; it needs a second dataset to be a claim at all | 2–3 days |
| **P6** | Decide the upstream-version policy | Currently one pinned commit (`9f887a0`). Drift fails silently — `__dummy_mapping` has the same shape and dtype as a real mapping | design call |
| **P8** 🆕 | **Checkpoints are enormous, and it is almost all dead weight** | A learnable mapping stores a `(features × z) × (width × n)` float32 matrix — **17 MB** for MNIST `1x300`, **113 MB** for `1x2000`, **471 MB** for `1x1000 z=25`. `extract.py:60` does `weights.argmax(axis=0)`, so the exporter uses **~7 KB** of that 471 MB. Consequences: `dwn2rtl build model.pt` must stream rather than `torch.load` whole where it can, anything over 100 MB cannot go in git (so P5's CI fixtures must be synthesized, not committed), and "email me your checkpoint" is not a support channel. **`make_test_checkpoint.py` already generates fixtures of any shape**, which is the answer for CI | half day |

### 5.1 Emit balanced reductions, always

**Decided 2026-08-11 by measurement.** `rtl/argmax.v` was a sequential `for` loop, which
synthesizes to a chain of `K-1` dependent compare-selects. At JSC's five classes that is four
deep and invisible; at MNIST's ten it is nine deep, 17 logic levels, and it held `dwn_top` to
87.5 MHz against a 100 MHz board. A balanced tree closed it at 108.0 MHz.

**It is unconditional here too, as of 2026-08-11.** It was briefly conditional (`K > 5`) to keep
JSC's published 108 and 1,619 from moving, and that was withdrawn: a branch may encode a
discontinuity in the *target* (`MAX_N = 6` is one — at n=7 a node stops being one LUT6) but not a
fact about the project's own history. The published figures are pinned by the `jsc-complete` tag
instead, and `verify_phase1.py` re-measured to 110 / 1,621.

**The general rule that came out of it, and it belongs in the tool:** if a conditional's condition
can only be explained by referring to the past rather than to the hardware, it does not belong in
a generator. Those branches do not compose — each new case adds another, and every one needs its
own verification.

⚠️ **The wider rule: emit balanced reductions explicitly, do not rely on synthesis to fix a
loop.** `popcount.v` is written as the same shape of loop and is fine, because addition is
associative and the tool rebalances it. `argmax` is a data-dependent select and the tool leaves it
linear. Whether a generated reduction is fast should not depend on a property of the operator that
the emitter never checks.

### 5.2 The command surface — derive, do not ask

**The target is `dwn2rtl build model.pt` producing Verilog.** Not twenty flags. A tool that makes
the user restate what is already in the checkpoint is not a tool, it is a wrapper.

**Almost everything is derivable, and this repo already derives it.** Read straight from a
checkpoint today, with no input from anyone:

| | from | example |
|---|---|---|
| features, classes | `config`, threshold shape | 16 / 5 |
| layers, `n`, `z` | `config`, `state_dict` | `[300, 100]`, 6, 8 |
| wiring and table contents | `state_dict` | 300 tables, 1,800 indices |
| **integer bits** | thresholds — **exact** | 3 for JSC's `1x50`, 1 for the `t8` models |
| **threshold-separation floor** | wired thresholds per feature | 11 bits for `1x50`, 9 for `t8 300-100` |
| pipeline depth | a safe default, JSC-proven at four stages | |

**Exactly one thing is not derivable: how many FRACTIONAL bits are safe.** That depends on whether
quantisation changes predictions, which depends on the data. §4's V4 argument, and docs/jsc/report.md §5.6
is the scar — a narrowing fitted and validated on the same 1,000 samples put 8 of 15 features too
narrow.

So the surface is **two commands and an escape hatch**, not a flag per parameter:

```
dwn2rtl build model.pt                    # derives everything derivable; prints every choice
dwn2rtl build model.pt --data test.npz    # measures what needs data; narrower, and it says so
dwn2rtl build model.pt --width 11         # you already know; the tool verifies and obeys
```

**Without data it must not guess.** It uses the derived separation floor, which is conservative,
and says which parts were derived and which were assumed:

```
features 16, classes 5, layers [50], n=6, z=200          from checkpoint
integer bits 3                                            derived, exact
fractional bits 8  -> Q3.8, 12-bit word                   FLOOR, not measured
  pass --data to measure the safe width; it is usually narrower
```

**The generator-only decision shrinks this further.** No part, no clock target, no synthesis
strategy — the user's toolchain, not ours. Those are three flags that never have to exist.

**This repo's scripts are not the tool's interface.** `run_gate1.py --word-bits/--frac-bits` is
development plumbing for sweeping and experimenting, and it is fine for it to be explicit. The
product is allowed a smaller surface than the workshop that built it.

---

## 6. Decisions to settle early — **ANYTIME**, and they change the scope

Free to answer, and each one narrows what the work above actually is.

| # | Decision | Why it matters now |
|---|---|---|
| ~~**Q1**~~ | ✅ **RESOLVED 2026-08-11: the TOOL is generator-only.** It emits synthesizable Verilog for the network and nothing around it; users bring their own harness, as hls4ml does. **B2 and B3 are therefore not tool work.** ⚠️ This does **not** decide whether MNIST runs on our own board in this repo — that is a separate question, still open, tracked in `docs/mnist/phase1-ledger.md` *(not in this repo)*. If the answer is yes, B2 and B3 happen for the demo without becoming part of the tool |
| ~~**Q2**~~ | ✅ **RESOLVED 2026-08-11: yes, Gate 1 ships.** A generator whose output nobody can check is worth much less, and this repo has the evidence — the emitter's own read-back check reported 20/20 correct while the design was wrong on 958 of 1,504 vectors |
| **Q3** | **Which upstream DWN versions?** (P6) | One pinned commit is honest and cheap; a range needs a compatibility layer plus silent-drift detection |
| **Q4** | **Name, and where it lives** | `mnist/plan.md` §1.6 forbids branch- or person-named files; the same discipline should apply to the fork |
| ~~**Q5**~~ | ✅ **RESOLVED 2026-08-13: new repository, not a fork.** The tool is ~2,300 of this repo's ~19,600 lines — **12%**. A fork's first commit would delete 88%, and its history would be 90+ commits of study retractions irrelevant to using it. The knowledge lives in the code comments, which travel with a copy; the ledgers should not travel at all. **This repo stays the tool's evidence** and the tool's README links to it |
| ~~**Q6**~~ | ✅ **RESOLVED 2026-08-13: vendor-neutral, with an optional estimate tier.** No Vivado dependency. Verified by inspection: the emitted RTL instantiates **no vendor primitives** and is Verilog-2001 (`timescale`, `default_nettype`, `generate`, `$readmemh`) — the only `logic` in the tree is the English word, in a comment. Estimates, if wanted, shell out to **yosys** when present. ⚠️ Not yet *tested* under a non-Xilinx simulator; see the new blocker below |
| ~~**Q7**~~ | ✅ **RESOLVED 2026-08-13: do not ship the area model.** It was built to filter configs too big to build and across both completed studies filtered **zero**, because both were bounded by training rather than by area. hls4ml does not ship one either — it reports what a vendor tool says. See `dse/area_model.py`'s header |

### ⚠️ Q8 — the one genuine blocker, discovered 2026-08-13

**Upstream DWN saves no checkpoint at all.** `examples/mnist.py` trains, prints accuracy, and
exits — there is no `torch.save` anywhere in it. The `{config, state_dict, thermometer, scaler,
results}` dict this project reads is **our own invention**, written by our Kaggle notebooks.

It cannot simply be `state_dict`, either: a DWN is **two objects**, and the thermometer's
thresholds live outside the model. Measured on a real checkpoint — `state_dict` holds
`['0._LUTLayer__dummy_mapping', '0.luts', '0.mapping.weights']` and **no thresholds**. So the
obvious `torch.save(model.state_dict())` silently loses the encoder.

**Consequences for the tool:**

1. **It must define the format**, and own both ends: `dwn2rtl.save(model, thermometer, path)` and
   `dwn2rtl build <path>`. The CLI is the interface; the save step exists because there is nothing
   else to build from. `docs/checkpoint-format.md` already specifies the schema against
   the pinned commit and is most of the work.
2. **It should also accept live objects** — `from_model(model, thermometer)` — because that is what
   a user has the moment training finishes, and it is what hls4ml does.
3. **It must fail loudly on a bare `state_dict`**, naming the missing thermometer rather than
   emitting a broken encoder.

### ✅ Q9 — RESOLVED 2026-08-13: ask for the INPUT precision, never for fractional bits

The original framing was wrong. Three separate quantities were being conflated, and only one of
them is genuinely undecidable.

**1. Integer width — derivable exactly, always.** `required_int_bits()`, per config. MNIST needs
Q0.8 at z≤8 and **Q1.8 at z=25**, where one quantile threshold lands on exactly 1.0.
`widen_for_checkpoint()` in `dse/run.py` already does this. No user input.

**2. The comparator-merge floor — derivable exactly, from the thresholds.** Below
`ceil(-log2(min gap between distinct thresholds))`, two distinct thresholds quantise to the same
value and **two comparators collapse into one**. Measured:

| checkpoint | min gap | implied floor | we used |
|---|---|---|---|
| MNIST `1x300` | 3.922e-03 | **8** | **8** |
| MNIST `1x1000 z=25` | 3.922e-03 | **8** | 8 |
| JSC `1x50 z=200` | 3.576e-07 | 22 | 12 |

⚠️ **This is a warning, not the answer.** JSC shows why: the rule demands 22 bits and 12 is fine,
because thresholds 3.6e-7 apart have essentially no data between them, so merging them costs
nothing measurable. Report it; do not enforce it.

**3. Whether quantisation flips an encoder bit — NOT derivable from a checkpoint.** A bit flips
when a *data point* lands in the sub-LSB interval immediately above a threshold. That is a property
of the data.

#### The reframe: `--input-bits`, not `--frac-bits`

"How many fractional bits?" is a question no user can answer. **"What precision is your input?"**
is one they almost always can — and it determines the answer exactly when the input has a native
quantum.

**MNIST is the proof, and it is a proof rather than an observation.** Pixels are `k/255`, and
thresholds are quantiles of those same values, so both sit on the same grid. With frac = 8,
`floor(k · 256/255)` is strictly increasing over k = 0…255, so ordering is preserved exactly and
the quantised encoding is **identical** to the float one for every possible input. That is why we
measured 0 divergence across all 10,000 test samples — not luck, and not something that needed
measuring at all once the argument is made.

**So the policy is:**

| input | what the tool does |
|---|---|
| native n-bit integers, scaled (images, most sensors) | `frac = n` → **provably lossless**, say so |
| user supplies a sample | infer: if every value is `k/(2ⁿ−1)`, take n. Otherwise measure bit-error at candidate widths |
| genuinely continuous, no sample | default, and report the measured bit-error on self-generated random vectors — labelled a **stress test, not a proof** |

**The tool never asks for fractional bits, and never silently picks one.** ⚠️ And a measured
bit-error on the tool's own random vectors must not be reported as a guarantee:
`docs/jsc/report.md` *(not in this repo)* §5.6 records this project fitting *and* validating an encoder narrowing on the
same 1,000 samples, and 8 of 15 features coming out too narrow as a result.

**Q1 was the highest-leverage question in this document, and it is now answered: generator-only.**
That removes B2, B3 and the whole MNIST harness rework in one stroke.

**Two consequences worth stating, because they are not obvious.**

**The interface becomes the product surface.** Ship only Verilog and what a user actually consumes
is the port list, the latency in cycles, and the initiation interval — not the internals. That is
already close to right: `dwn_top` takes `clk` and a flat feature word and returns a class index,
with `dwn_top_params.vh` carrying the pipeline depth so a testbench and a harness cannot disagree
about it. It needs documenting rather than building.

**The encoder always ships, and its area is always reported separately.** Thermometer encoding is
not preprocessing a user can supply — it is intrinsic to a DWN, and on the smallest JSC model it is
fourteen times the network it feeds. `docs/jsc/report.md` *(not in this repo)* §5.2 criticises published work for reporting
core-only LUT counts; a tool of ours that emitted the network and left the encoder to the user
would commit exactly that error, and hand users a number that understates their design by most of
its cost.

**Gate 1b stays in this repo, and stays a claim.** The board flow is not part of the tool, but
"166,000 of 166,000 exact on real silicon" is the strongest evidence the generator's output is
correct. It belongs in the tool's README as evidence, not as a feature.

---

## 7. Explicitly out of scope

Recorded so they are not silently reconsidered:

- **Adopting the encoder narrowing for JSC.** `docs/jsc/report.md` *(not in this repo)* §7 argues against it — the binding
  constraint on this device is timing, and the encoder is not on the critical path. F1 makes it
  *possible*; it should not become the default.
- **Learnable Reduction.** ~~Never explored enough to belong in a tool.~~ **Now explored, and the
  answer is still no** — `docs/mnist/reduction-ledger.md` *(not in this repo)*, 35 trained configurations. A learned
  taper genuinely works (**+3.69 pp** over a plain narrow layer at the same group size) and is
  **dominated anyway**: `1x500` reaches 97.70% with 500 nodes and the identical 500-bit popcount
  that the best taper spends 2,800 nodes to reach 97.43% with. ⚠️ One exception, worth a line in
  the tool's docs rather than a feature: a **mild** 2:1 taper (`2x[2000,1000]`) is the best model
  in earlier work with half the adder tree. So the tool should keep emitting `GroupSum` and say
  nothing about pyramids.
- **Reorganising the JSC artifacts.** `mnist/plan.md` §1.4 — `docs/jsc/report.md` *(not in this repo)* and the `jsc-complete`
  tag reference current paths.
- 🆕 **Running the emitted RTL back through yosys and shipping *that* Verilog.** Measured
  2026-08-14 and recorded here because it is an obvious-sounding idea that will be suggested
  again. yosys can indeed `synth; write_verilog`, but on `dwn_core`:

  | | |
  |---|---|
  | size | 156 lines -> **523 lines**, 3.6x larger, every wire renamed `_000_` |
  | LUT nodes come out as | `assign _000_ = 64'h...0007150 >> {_098_, _106_, ...};` |
  | vendor primitives | **none** — so Vivado re-synthesizes it from scratch anyway |
  | room to improve | **none**: `dwn_core` measures **110 LUTs against Vivado's 110** |

  ⚠️ **That output line is what `lut_node.v` already is** — a 64-bit constant indexed by a 6-bit
  address. The tool already emits the optimal form; yosys only removes the names. The core is
  nothing but trained tables and two reductions, the tables cannot be smaller, and both
  reductions are already logarithmic (phase 4 ledger §9).

  Three costs against zero benefit: it makes yosys a **hard dependency of `build`** (today it is
  optional and only for `estimate`, and on Windows it is a 564 MB download); it destroys
  reviewability, including the emitters' own read-back checks, which parse the emitted text; and
  it hands the vendor tool pre-flattened anonymous logic instead of structure. On that last
  point — Vivado maps comparators to carry chains at 7.5 LUTs each where yosys uses 3.5 generic
  LUTs. That is **two architectures, not one tool being smarter**, and we found timing
  rather than area was binding.

  **When it would be legitimate:** targeting a flow that cannot synthesize behavioral Verilog, or
  ASIC tech-mapping to a specific cell library. Neither is what a vendor-neutral generator ships;
  see Q6.

---

## 8. Suggested order

**Rewritten 2026-08-12. Everything BEFORE and DURING is done; the remaining work is all AFTER,
plus one item that jumped the queue.**

```
DONE      B1  hardcoded feature count            ✅ 2026-08-11 (fbc3be8)
          B4  dataset descriptors                ✅ 2026-08-11 (9815062)
              ...and actually IMPORTED           ✅ 2026-08-12 -- the half that mattered
          F1  configurable precision             ✅ 2026-08-11 -- MNIST runs at Q0.8
          B2  board record format                ✅ 2026-08-11, on silicon 2026-08-12
          B3  vector store dimensions            ✅ 2026-08-11
          Q1  generator-only                     ✅ RESOLVED -- removed B2/B3 from TOOL scope
          Q2  Gate 1 ships                       ✅ RESOLVED
          V4  precision procedure                ✅ exercised, and the contract held
          5.1 balanced argmax tree               ✅ 2026-08-11 -- 87.5 -> 108.0 MHz
                    ↓  JSC reproduced exactly after every one (12/12, 110/1,519/1,621)

NOW       R2  make dse/ dataset-aware            ⬅ promoted from AFTER. MNIST Phase 2 (M2c)
                                                    cannot run without it, and it is the only
                                                    remaining item on BOTH the tool's path and
                                                    earlier
          R3  find an encoder cost model         ⬅ 3 synthesis runs. R1 cannot be made correct
                                                    before it, and the tool reports area
          R1  fix area_model's calibration       ⬅ after R3, not before

AFTER     V2, V3   self-contained verification   ← V3 is the real barrier: verification
                                                    currently needs a Vivado licence
          P1-P8    fork, package, licence, CI, checkpoint size
          P7       stop defaulting to a JSC path (five sites, not four)

OPEN      Q3  upstream-version policy
          Q4  name, and where it lives
```

**Rough total:** ~half a day for R1–R3, one to two weeks for AFTER — consistent with
`docs/reference/reusable-generator.md` *(not in this repo)* §4, and still "almost none of it RTL". The estimate did not move;
what moved is that the *before* column is empty.

---

## 9. Pointers

- `docs/mnist/plan.md` *(not in this repo)* — ground rules for the branch, and the JSC-must-not-break gate
- `docs/mnist/phase1-ledger.md` *(not in this repo)* — the dated log for the port, and where every ✅ above is evidenced
- `docs/mnist/phase2-ledger.md` *(not in this repo)* — the sweep, and M2c, which is R2 under another name
- `docs/mnist/reduction-ledger.md` *(not in this repo)* — why Learnable Reduction stays out of scope (§7)
- `docs/reference/reusable-generator.md` *(not in this repo)* — whether to build the tool at all, and how to split it off
- `docs/jsc/report.md` *(not in this repo)* §7 — the precision measurement F1 exists to expose
- `docs/checkpoint-format.md` — what the exporter reads, verified against JSC only (M4)