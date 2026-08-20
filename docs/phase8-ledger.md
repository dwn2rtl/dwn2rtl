# Phase 8 ledger — audit the seams the last audit left

**Goal.** Phase 7 closed with a claim about where the remaining risk is. Test that claim, fix
what it turns up, and correct the two earlier sections that this round proves overstated.

**Status: CLOSED.** Retrospective in `phase8-report.md`.

> Conventions in `overview.md` §6. Entries oldest first, recording *built* / *hit* / *decided*.
> Reversed decisions stay, struck through, with the reason.

---

## The rule this phase is organised around

**Phase 7 §3.1 is a method, not just a retrospective observation.** Every serious defect that
phase found lived on an axis with exactly one tested value, and the fix was never a cleverer
assertion — it was a second value. That is a procedure anyone can run: list the parameters, ask
which have only ever had one value, and go there first.

⚠️ **So the target this round is not the tool, it is the audit.** Phase 6 §10–29 ran seven
adversarial rounds and closed 29 defects. Every finding below sits in a *seam* of that work —
next to something it checked, inside a blind spot one of its own headlines created, or in the
uncovered half of a fix it made. None is in ground it skipped.

Phase 7 §5 said the remaining risk "is not in code this project can reach: it is in a real user's
model." **That was half right.** The untried values that matter are indeed outside this
repository — but seven of them were reachable from inside it, and this ledger is what happens
when you go and try them.

---

## 1. 🎯 Found — a model with more than 256 classes fails its own gate, and the design is correct

Both testbenches declare the golden answers eight bits wide and then slice them to the index
width:

```verilog
reg [7:0] expected [0:N_VEC-1];
...
if (class_idx !== expected[j][`IDX_W-1:0])
```

At `num_classes >= 257`, `IDX_W` is 9, so that part-select reads past the end of an 8-bit reg and
yields `x`. `!==` is strict about `x` — deliberately, so an undriven bit cannot pass — so **every
vector mismatches.**

The boundary is exact:

| `num_classes` | `IDX_W` | gate |
|---|---|---|
| 255 | 8 | PASS |
| 256 | 8 | PASS |
| **257** | **9** | **FAIL — 12/12 core, 27/27 top** |

⚠️ **The hardware is bit-exact.** On a 300-class build, widening that one declaration to
`reg [31:0]` and changing nothing else:

```
  vectors tested : 24
  mismatches     : 0
  RESULT         : PASS (bit-exact on every vector)
```

This is the shape phase 7 §3.4 named the most dangerous — *a correct design reported as broken*.
A user meeting it has no way to tell it from a genuine emitter bug, and the natural response is
to distrust the tool and stop.

⚠️ **The most useful part is that this line was already inspected.** `phase6-ledger.md` §26
quotes it exactly, and its analysis is correct — for the *other* direction:

> A too-narrow `IDX_W` does not break the comparison, it **narrows** it.

The fix §26 chose was a test pinning `IDX_W == ceil(log2(K))` per shape. That test passes at
K=257, because the derivation is right; it is the *storage* that is too narrow. One line, read
carefully, in one direction. Tested class counts across the whole suite are `{0, 2, 3, 10}`.

## 2. 🎯 Found — the thermometer check covers the learnable half only

`phase6-ledger.md` §22 found that a thermometer from a different training run loaded happily
against a model, and stated the consequence precisely: the design builds, **the gate passes**,
because the vectors are generated from the same wrong assumption, and only real features reveal
it by landing in the wrong bit positions.

It was fixed. The fix covers models whose first layer has a *learnable* mapping, and §22's own
closing sentence says why:

> Detectable exactly: a **learnable** mapping's `weights` is `(input_size, output_size * n)`, so
> the first layer states how many bits it expects.

A fixed mapping states nothing, so nothing checks it. The same attack, both kinds — a 12-feature
thermometer bolted onto a 6-feature model:

| first layer | result |
|---|---|
| `learnable` | `CheckpointError: the thermometer does not match this model...` |
| **`fixed`** | **builds clean, `dwn_core PASS`, `dwn_top PASS`** |

⚠️ **A green gate on hardware that is wrong.** This is the most severe finding of the round, and
the uncovered half was never recorded as remaining — the ledger reads as though the case is
closed.

`SHAPES['all_fixed']` exists, so the fixed-mapping path is exercised. Nothing bolts a *wrong
thermometer* onto it: the mapping kind is an axis with two values, and the mismatch attack had
only ever been run against one of them.

## 3. 🎯 Found — `quantize()` corrupts saturation at wide words, and at 64 bits it flips the sign

`np.clip(q, lo, hi)` runs in float64 before `.astype(np.int64)`. `2**63 - 1` is not representable
in float64, so it rounds up to `2**63` and the cast overflows:

| `word_bits` | input that should saturate to the top rail | got |
|---|---|---|
| 54 | 9007199254740991 | correct |
| 60 | 576460752303423487 | 576460752303423488 (off by one) |
| 63 | 4611686018427387903 | 4611686018427387904 (off by one) |
| **64** | **9223372036854775807** | **−9223372036854775808** |

At 64 bits a saturating input comes out as the most negative representable value — a sign flip
that inverts every comparator it feeds. NumPy emits `RuntimeWarning: invalid value encountered in
cast`, which nothing checks and a script never shows.

⚠️ **The gate cannot see this**, and the reason is structural: `build` never calls `quantize()`.
Top-level vectors are generated as integers already, so the emitted testbench exercises the
comparators and not the front end. But `input_scaling.json` and the user guide both instruct the
*user* to call `quantize()` on their own data — so the one function a user is told to run is the
one the gate does not cover. That is the fourth instance of the shape phase 7 §3.2 identified: a
specification choice the gate is blind to by construction, after the forgotten scaler, the
mismatched thermometer and `quantize`'s floor-vs-round.

⚠️ **And it sits inside a blind spot an earlier headline created.** `phase6-ledger.md` §20 is
titled "the `input_bits` axis is clean," and it went hunting for exactly this class of hazard:

> Quantising a threshold computes `t * 2**frac`, and with a wide word that product can pass
> float64's 53-bit mantissa — the obvious place for silent precision loss. Checked against exact
> rational arithmetic at ten combinations up to a 63-bit word: **error of zero, every time.**

That is right, and it is about the *multiply*, in `quantize_thresholds`. The bug is in the
*clip*, in `quantize`. Sweeping `--input-bits` 0–63 end to end passes the gate at every width,
because the gate never reaches the broken line. A correct measurement, a correct conclusion, and
a headline broader than either.

## 4. Found — `python -O` deletes every emitter self-check

`assert` is the mechanism for both emitters' read-back, for the `n <= MAX_N` guard, and for the
golden model's group-divisibility check. `python -O` removes all of them.

Planting this project's own named worst-case bug — a reversed address concatenation, the thing
`verify_emitted` exists to catch:

```
normal      read-back caught it -> layer 0 node 0: wiring [9, 13] != expected [13, 9]
python -O   build SUCCEEDED with a reversed address concatenation on disk
```

`build_core`'s docstring promises the opposite:

> Raises AssertionError if the read-back disagrees with the checkpoint. That is deliberate: an
> emitter that half-succeeded should not leave a file on disk anyone might synthesize.

Under `-O` it leaves exactly that. `n=7` also builds, emitting tables silently truncated to
lut_node's 64-bit `TABLE` parameter.

⚠️ **The gate still fails in both cases**, which is what keeps this below the first two in
severity — the design is caught, just not by the check that was supposed to catch it, and not
with a message that names the cause. `PYTHONOPTIMIZE` and `-O` appear nowhere in the docs, the
tests or CI.

## 5. Found — nothing binds the vectors to the RTL they were generated from

`build()` writes RTL in steps 1–2 and vectors in step 3. Interrupt between them and the directory
holds new RTL beside old vectors, and `verify` processes it without a word. A directory with an
11-bit encoder (`Q0.10`) and `X_W 54` — the 9-bit vectors from the previous build — reported:

```
  dwn_core  44 vectors  PASS
  dwn_top   55 vectors  PASS
RESULT   PASS
```

1 of 5 seeds gives that spurious PASS; the other 4 FAIL. **So it is not systematic — but a
provably inconsistent directory can and does report PASS**, which is the invariant in `CLAUDE.md`
with nothing enforcing it at verify time:

> ⚠️ Vectors and RTL must derive from the same checkpoint. Otherwise you ship a testbench that
> passes against wrong RTL — worse than shipping none.

⚠️ **The adjacent case is recorded as safe, and is.** `phase6-ledger.md` §29 says "Interrupted
builds recover. A build killed after 0, 4 or 9 files, **then re-run**, produces a design that
passes the gate every time." Kill-then-*re-run* was tested. Kill-then-*verify* was not, and it is
the likelier user action: the build printed an error, so you re-run the thing that checks.

The version stamp from phase 7 §1 is the right mechanism aimed at a different question — it
records which *tool* wrote a file, not which *checkpoint*.

## 6. Found — `saturation_is_lossless()` is dead code, and the tool checks the weaker condition

`extract.py` defines it, and its own docstring names it as the thing to use:

> Saturates to the word range, which is lossless as long as every threshold sits strictly inside
> it -- check with saturation_is_lossless() rather than assuming.

Nothing calls it. `emit_encoder` calls `fits_in_word()` instead, which is inclusive where the
specification is strict. A threshold quantising to exactly the top rail:

```
thr_q max      : 511   word max : 511
fits_in_word           (what the tool checks) : True
saturation_is_lossless (what the docs specify): False
precision reported as  : Q1.8 signed, source=given, proved=True

a real feature x = 2.49609, threshold t = 1.99609
  float model says     x > t : True
  hardware computes q(x) > T : False
```

The build reports `proved=True` — "provably lossless" — on a format that is measurably lossy at
its own boundary. The gate agrees, because the golden model saturates identically.

⚠️ **This function did real work before it was packaged.** `tool-roadmap.md` V4 records it
deciding MNIST's Q0.8 was safe, confirmed against the full 10,000-sample test set with 0
divergences. It became uncalled during the port and nothing noticed, because nothing it would
have rejected was ever built.

## 7. Found — a scalar `results` field gives the user a traceback

```
TypeError: 'int' object is not iterable
  checkpoint.py:405   'results': dict(obj.get('results') or {}),
```

`cmd_build` catches `CheckpointError, ValueError, OSError` — not `TypeError`. (`results='great'`
raises `ValueError` and *is* reported cleanly; only the scalar case escapes.)

This contradicts the rule `comment_safe` was written to enforce — "**Metadata must never be able
to break a design**" — and sits one step upstream of where that rule was applied.
`phase6-ledger.md` §14 tested `results={'final_acc': 'ninety'}`, a bad value *inside* a dict, and
fixed it in `accuracy_line`. A `results` that is not a mapping at all dies earlier, in
`normalize`, before `accuracy_line` is ever reached. §29's "17 hostile invocations, zero
tracebacks" was about the CLI's *arguments*, not the checkpoint's fields.

## 8. Measured — what held up, and why that matters here

Aimed at deliberately and unbroken, every one bit-exact through the gate:

- `n` = 1, 3, 5, 6 — every legal table width, not just the two in the fixtures
- `z` = 1; a single feature with a single threshold; 17 of 18 thermometer bits dead
- `num_classes` = 1; a popcount group of exactly 1
- six chained layers
- `Pipeline(3, 2, 0, 5)` and `Pipeline(0, 0, 0, 0)`
- `--input-bits` 0 through 63
- `n_random=0`; all-identical thresholds; thresholds scaled by 1e9

⚠️ **The phase 6 and 7 hardening is holding**, and that is the context the seven findings belong
in. The odd-width packing fix, the pipeline-depth fix and the zero-latency fix all survive
re-attack from a different direction. What did not hold were the seams: one line read in one
direction, one fix scoped to one of two representations, one headline broader than its
measurement.

## 9. Decided — the corrections owed to phase 6

Two of that ledger's headlines are now overstated by measurement, and `CLAUDE.md` requires the
correction to land on the original claim rather than as a fresh section somewhere else:

| section | claim | what this round measured |
|---|---|---|
| §20 | "the `input_bits` axis is clean" | true of `quantize_thresholds`' multiply, which is what it checked; false of `quantize`'s clip (§3) |
| §22 | the thermometer is "now checked" against its model | true for a learnable first layer, false for a fixed one (§2) |

⚠️ **Neither is a wrong measurement.** §20's rational-arithmetic check is exact and its result
stands; §22's detection mechanism is sound and its fix works. Both headlines simply claim more
ground than the work under them covers, which is the failure mode a ledger is most prone to —
the summary line outliving the scope of the thing it summarises.

---

## 10. Built — §2: the fixed-mapping half of the thermometer check, as far as it can go

⚠️ **This one cannot be made exact, and saying so is part of the fix.** A learnable mapping's
`weights` is `(input_size, ...)` and states the expected bit count outright. A fixed mapping is a
list of indices and states nothing, so there is no fact in the checkpoint to check a thermometer
against. No amount of care recovers information that is not there.

What *is* observable is the shape of what got left out: a thermometer larger than its model
leaves whole **trailing** features driving no comparator at all. So `build_encoder` computes the
dead-feature set and `build` reports it, with the trailing case named as the mismatch signature:

```
WARNING   the LAST 6 of 12 features drive no comparator (features 6, 7, 8, 9, 10, 11), and this
          model has a FIXED first-layer mapping, which states no input width for the loader to
          check. That is the signature of a thermometer from a different training run -- it
          would build, pass the gate, and put real features in the wrong bit positions.
          Confirm the thermometer was fitted for THIS model.
```

⚠️ **Reported, never raised, and that is a decision rather than caution.** A genuinely lopsided
model has dead features too — one of §8's own passing shapes drives 17 of 18 thermometer bits
into nothing. Refusing on a heuristic would reject correct models to catch an incorrect one,
which is the trade `tool-roadmap.md` Q7 already rejected once for the area model. A signature
that names itself is worth more than a filter that is wrong both ways.

Three tests: the learnable case still refuses exactly, the fixed case warns with the signature,
and **every shipped shape must raise no such warning** — the half that stops it becoming noise.

## 11. Built — §1: the testbenches store the golden answer wider than the index

`reg [7:0] expected` became `reg [EXP_W-1:0]` with `EXP_W = 32`, in both testbenches.

⚠️ **The width is not `IDX_W`, and both reasons matter.** Too narrow was the bug. But equal to
`IDX_W` would have re-created the one `phase6-ledger.md` §26 identified and could not close:
slicing *both* sides to `IDX_W` means a wrong index width narrows the comparison rather than
breaking it, so a truncating bug agrees with itself. So the golden answer is held wider and
`class_idx` is zero-extended up to meet it:

```verilog
if ({{(EXP_W-IDX_W){1'b0}}, class_idx} !== expected[j])
```

Now a golden answer that does not fit the index width mismatches instead of being truncated into
agreement — the direction §26 wanted and had to settle for a derivation test on.

Pinned at 256 and 257, both levels, plus a test that corrupts one golden answer to a value
outside the index width and requires a FAIL. The shape is deliberately **not** in `SHAPES`:
every shape there must discriminate all its classes and keep a popcount group of at least 3,
which 257 classes cannot do at a simulable size. Weakening either invariant to admit one shape
would cost more than the shape is worth.

## 12. Built — §5: every generated `.vh` carries a digest of the build that wrote it

`build_id()` hashes the config, thresholds, every state_dict tensor, the precision and the
pipeline, and the first 16 hex digits go into all four parameter files. `verify` reads them
before it compiles anything and refuses a directory holding more than one:

```
dwn_core  ERROR: the RTL and the vectors in this directory came from DIFFERENT builds, so a
          PASS here would mean nothing [...] Found 2 build stamps: 8f5b953b... from
          dwn_core_params.vh (RTL), dwn_top_params.vh (RTL); e4c020e3... from
          top_params.vh (vectors), vec_params.vh (vectors).
```

All five seeds that produced the §5 attack now ERROR, including the one that had reported PASS.

Three choices worth recording:

- ⚠️ **Content, not a uuid or a timestamp.** A rebuild of one checkpoint must stay byte-identical,
  which is what makes "it passed yesterday" a checkable claim. `test_a_rebuild_is_byte_identical`
  would have caught the lazy version immediately.
- ⚠️ **`n_random` and `seed` are excluded.** They change *which* vectors exist, not whether the
  vectors and the RTL agree about the model. Folding them in would flag a legitimate rebuild
  with a larger vector count, and a check that cries wolf gets switched off.
- ⚠️ **A missing stamp is not a mismatch.** Files written before this change carry none, so they
  are skipped rather than assumed stale. That is the compatible direction to be wrong in, and
  mixing tool versions is what the phase 7 version stamp already makes diagnosable.

## 13. Built — §3: the clip moved out of float, and NaN stopped being a very small number

`np.clip(q, lo, hi).astype(np.int64)` is gone. The bounds are now tested against the two values
that *are* exact in float64 — both powers of two — and only values known to fit are ever cast:

```python
low  = q < float(lo)            # lo     = -2**(word_bits-1), exact
high = q >= float(hi + 1)       # hi + 1 =  2**(word_bits-1), exact
```

Correct at every width from 2 to 64, both rails, verified against `fractions.Fraction` on the
in-range path.

⚠️ **NaN now raises instead of saturating.** It could not saturate: every comparison against it
is False, so it fell through both branches and landed on int64's most negative value — a feature
that is *missing* silently becoming a feature that is *extremely small*. Infinities are left to
saturate, and the distinction is exact rather than stylistic: an infinity is on a definite side
of every threshold, and a NaN is on no side of any.

The tests live in `test_golden_model.py` with a note saying why they are there — `build` never
calls `quantize()`, so this is the one piece of the numerics with no simulator behind it.

## 14. Built — §4: the checks that a runtime flag could delete are now exceptions

Ten `assert`s across four modules, every one load-bearing, replaced by real checks. Both
emitters' read-backs raise a new `EmitterMismatch(ValueError)`; the rest raise `ValueError`.

Verified the only way it can be: a subprocess run twice, once with `-O` and once without,
asserting *both* that the run really had its asserts stripped and that the guard fired anyway.
Checking only the second half would pass on a broken test.

⚠️ **`build_core`'s docstring was making a promise the flag could revoke** — "an emitter that
half-succeeded should not leave a file on disk anyone might synthesize" — and it now says why it
is an exception. The read-back also gained the regression test it never had: plant a reversed
address concatenation on disk, and the emitter must refuse to leave it there.

## 15. Built — §6: `saturation_is_lossless()` is called again, and the claim it qualifies is qualified

`build_encoder` now calls it beside `fits_in_word`, and a threshold on the word's rail is
reported — including in the line that makes the claim, not only in a warning below it:

```
frac bits 8 -> Q1.8 signed (10-bit)   from --input-bits, provably lossless -- EXCEPT at the
                                      word rail, see WARNING
WARNING   a threshold quantises to the maximum value Q1.8 signed (10-bit) can hold [...] so a
          feature past it saturates onto the threshold instead of over it and the comparison
          flips. [...] Widen the word by one integer bit to remove the exception.
```

⚠️ **The qualification belongs where the claim is made.** A warning three lines below "provably
lossless" is read by someone who already believes the headline. `precision.proved` is left
alone: it is about where `frac_bits` came from, which is a different question with a different
answer, and overloading it would make two facts share one bit.

## 16. Built — §7: metadata cannot break a design, one step further upstream

`_results_of()` replaces `dict(obj.get('results') or {})`, and a non-mapping becomes
`{'unrecognized': <type>}` — the convention `_scaler_of` already used. The emitted header then
says what is true:

```verilog
// accuracy   : the results field is a int, not a mapping; nothing to read
```

⚠️ **Unrecognized rather than dropped.** "Nobody recorded an accuracy" and "the accuracy is
unreadable" are different facts, and printing the first when the second is true is the small
version of the mistake this project keeps finding in large.

Also fixed while in the file: the winget line in `verify.py` was a non-raw string containing a
Windows path, so Python emitted an invalid-escape `SyntaxWarning` on every import — one day a
`SyntaxError`, and the same defect class as the `\v` that turned `scripts\verify` into a
vertical tab.

## 17. Measured — the suite after the phase

| | before | after |
|---|---|---|
| tests | 416 passing, 6 skipped | **463 passing, 6 skipped** |
| import warnings | 1 (`SyntaxWarning`, invalid escape) | **0** |
| the seven findings | all reproducible | **all closed, re-attacked** |
| §8's passing shapes | all bit-exact | **all still bit-exact** |

Every attack script from §1–§7 was re-run against the fixed tool, and §8's twelve shapes were
re-run to check the fixes cost nothing.
