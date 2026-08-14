# Phase 4 report — measure, then optimize

**Status: closed, 2026-08-14.**

The day-by-day record is `phase4-ledger.md`. This is the retrospective.

---

## 1. The result, stated honestly

The phase set out to ship `dwn2rtl estimate` and then use it to make the emitted RTL measurably
better. It shipped `estimate`. **Both candidate optimizations were then cancelled — by
measurement.**

That is the phase succeeding, and it is worth being blunt about why. Roadmap **Q7** records the
study repo building an area model on plausible reasoning that filtered **zero** configurations
across two complete studies. Here, two changes that looked equally sound were tested before being
made, and both dissolved. The cost was two measurements rather than two rewrites of shipped RTL
plus the verification they would have required.

**What actually shipped:** an `estimate` command, precision that needs no flag, and — most
valuably — **a calibration that says what its numbers are worth.**

## 2. What was delivered

| | |
|---|---|
| `estimate.py` + `dwn2rtl estimate` | the last stubbed subcommand; **roadmap P2's command surface is complete** |
| the yosys/Vivado calibration | `dwn_core` **110 vs 110**, `thermometer_encoder` **717 vs 1519** |
| precision inference | `dwn2rtl build model.pt` derives a 9-bit word for MNIST with no flags |
| yosys in CI | Ubuntu jobs, plus a step that fails if it installed and is unusable |
| `popcount.v` | comment upgraded from an assumption to a measurement |
| the stub machinery | deleted — no subcommand parses and refuses any more |

## 3. The calibration, which is the phase's most useful output

JSC `1x50 z=200`, built unpipelined to match the study's out-of-context Vivado figures exactly
(FF = 0 on both sides):

| module | yosys 0.68 | Vivado, `xc7a35t` | ratio |
|---|---|---|---|
| `dwn_core` | **110** | **110** | **1.00x** |
| `thermometer_encoder` | **717** | **1519** | **0.47x** |
| `dwn_top` | **833** | **1621** | **0.51x** |

**Two completely different levels of trust, in one design.** The LUT core lands on Vivado exactly.
The encoder is out by 2.1x — and not because yosys is wrong: a 16-bit signed comparison against a
constant genuinely fits in ~3 LUT6s, which is what generic mapping finds, while Vivado on
7-series maps comparators onto carry chains that cost more. Generic mapping is simply optimistic
for comparator-heavy logic.

⚠️ **It also distorts this project's headline claim.** The encoder-to-core ratio is **13.8x by
Vivado and 6.8x by yosys**. A tool printing the yosys ratio unqualified would understate the very
thing dwn2rtl exists to highlight — that published DWN resource counts omit an encoder which can
dominate the design. So `estimate` prints the calibration with every report, not in a footnote.

## 4. Findings that outlive this phase

### 4.1 A measurement tool's first duty is to know when it has not measured

Ubuntu ships yosys **0.33**; development was on **0.68**. On 0.33 the tool reported:

```
dwn_core: 1 LUT, 72 flops, status OK
```

**One LUT for a 21-node core, presented as a successful measurement.** The root cause was that
scraping `stat` was never sound: it prints one section per surviving module, including leftovers
still holding pre-mapping gate cells, so reading a number out means picking the right section and
picking wrong is silent.

Now: `select -count` instead of `stat`, explicit `hierarchy; flatten` instead of
`synth -flatten` — and the check that matters, **a mapped design must have zero generic gate
primitives left**. Any survivor means `abc -lut 6` did not cover the design, so the count is a
fragment and it is an ERROR naming the yosys version, not a number.

`verify` already refuses to call an unchecked thing a pass. `estimate` now refuses to call an
unmapped design small. **Same rule, one level up.**

### 4.2 Both optimizations were wrong, and the arguments for them were good

| | the argument | the measurement |
|---|---|---|
| **narrow the encoder register** | 2,352 bits nominal, 720 driven; the emitter's own comment invited it | costs **552** flops — synthesis already trims *below* the driven count, because duplicate comparators share |
| **balanced popcount tree** | a loop is a chain of `W-1` dependent adds, and argmax proved that costs 20 MHz | depth tracks **log2(W)**: 6 levels at width 30, 12 at width 600 |

**The popcount result came with a control**, which is what makes it a conclusion rather than an
artifact. Small depths everywhere could just mean the metric cannot see depth — so the same
measurement was run on a chain of data-dependent selects, the shape `argmax` had before it became
a tree. At width 100: popcount **8** levels, the control **39**. The metric sees depth; popcount
does not have any.

**And the distinction is principled, not tool-specific.** Addition is associative, so any
competent tool *may* rebalance it. Selection is not, so none *can*. That is exactly why `argmax`
needed an explicit tree and `popcount` does not — and it is now recorded in `popcount.v` as a
table of measurements rather than the assumption that was there before.

### 4.3 Precision is derivable after all, and the roadmap said it was not

Roadmap Q9 states fractional width is *"NOT derivable from the checkpoint"*. True of the question
it was answering — *is this width safe for my data* — but it had been carried into the code, the
CLI and the README as though it settled everything.

**A thermometer's thresholds are quantiles of the training data, so they inherit its grid.**
Measured: MNIST's 2,352 thresholds sit 7.6e-06 from the `k/(2^8-1)` grid while JSC's sit ~5e-01
from every grid at every width. Five orders of magnitude of margin, so the tolerance is not a
delicate choice.

`dwn2rtl build model.pt` now derives **Q0.8, 9-bit** for MNIST with no flags, where it previously
fell back to a wider default. ⚠️ Two traps found on the way, both by tests:

- **The smallest grid is the only correct one.** Every coarse grid is a subset of every finer one
  (`k/255 == 257k/65535`), so any but the smallest gives a needlessly wide word.
- **Closeness is not enough — the grid must SEPARATE the thresholds.** Values of ~0.001 all round
  to 0 on a coarse grid, so a naive tolerance check inferred **n=1**: a one-bit fractional word,
  labelled provably lossless. Requiring injectivity is the honest statement of what "these values
  lie on this grid" means.

### 4.4 Do not put a toolchain bundle on PATH

The OSS CAD Suite ships its own `iverilog` and `vvp` among 136 binaries. Adding
`C:\oss-cad-suite\bin` to PATH would have silently changed which simulator the entire gate runs
against. It is found the way `verify` finds iverilog — PATH, then known locations — and yosys is
handed its own `bin`/`lib` per subprocess, because without them it prints nothing and exits 0.

## 5. Roadmap movement

- **P2 — pyproject and one CLI entry point** ✅ complete. `build | verify | estimate` all work,
  and the "not implemented yet" machinery is deleted.
- **Q7 — no area model** upheld, and now *demonstrated*: `estimate` shells out to a real
  synthesis tool and reports what it says, with its error bars.
- **§5.1 — emit balanced reductions, do not rely on synthesis** refined rather than followed.
  The rule's own justification is that a reduction should not be fast by luck; measuring the
  operator satisfies that, and the associative/non-associative split says which reductions need
  an explicit tree.
- **Q9 — fractional bits are not derivable** partially withdrawn. The *safety* question is not
  derivable; the input's native *quantum* usually is.

## 6. What comes next

Nothing in the tool is unfinished. What remains is optional:

- **`--data`** — deferred with a reason (ledger §10): tier 1 absorbed most of its value, its
  measuring half carries the JSC scar, and **it cannot be tested** without a checkpoint whose
  thresholds are off-grid. Build it when a real user needs it and there is a case to test against.
- **PyPI** — four small prerequisites recorded in the ledger.
- **Verilator** as a second simulator backend, behind the existing interface.

## 7. By the numbers

| | |
|---|---|
| tests | **252 passing**, 1 skipped; 30 the gate, 18 estimate |
| calibration | core **1.00x** Vivado, encoder **0.47x** |
| optimizations shipped | **0 of 2** — both cancelled on measurement |
| optimizations that would have shipped without measuring | **2** |
| defects found by CI's second yosys version | 1, and it improved the tool |
