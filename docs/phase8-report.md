# Phase 8 report — audit the seams the last audit left

**Status: closed, 2026-08-19.**

The day-by-day record is `phase8-ledger.md`. The audit it re-attacks is `phase6-ledger.md`
§10–29, and the method it applies is `phase7-report.md` §3.1.

---

## 1. The result

**Seven defects found and fixed, and every one of them sits in a seam of the previous audit.**
Two produced a wrong result on a green gate or a false alarm on correct hardware; the rest
weakened a check the project believed it had.

Phase 7 closed by locating the remaining risk outside this repository:

> The remaining risk is not in code this project can reach: it is in a real user's model.

**That was half right, and the half it got wrong is the finding of this phase.** The supply of
untried values outside the repo genuinely is larger. But seven of them were still reachable from
inside it — and every one was next to something phase 6 had already examined.

## 2. What was delivered

| | |
|---|---|
| the two silent-wrong fixes | the thermometer's fixed-mapping half, the vectors/RTL binding |
| the false-alarm fix | >256 classes no longer fails its own gate |
| numerics | `quantize()` correct at every word width, and NaN refused rather than saturated |
| robustness | ten load-bearing `assert`s that `python -O` deleted, now real checks |
| honesty | `saturation_is_lossless()` called again, and "provably lossless" qualified where it is claimed |
| metadata | a non-mapping `results` can no longer end the build in a traceback |
| corrections | phase 6 §20 and §22 struck through with what was withdrawn |
| **463 tests** | up from 416, and zero import warnings |

## 3. Findings that outlive this phase

### 3.1 A summary line outlives the scope of the thing it summarises

This is the sharpest pattern here, and it is not about code at all.

| section | the headline | what it actually measured |
|---|---|---|
| §20 | "the `input_bits` axis is clean" | `quantize_thresholds`' multiply — exactly, correctly, and only that |
| §22 | the thermometer is "now checked" | true for a learnable first layer; a fixed one states no width to check |
| §26 | the `IDX_W` comparison analysed and pinned | the *narrowing* direction; the storage was still 8 bits wide |

⚠️ **Not one of those is a wrong measurement.** §20's rational-arithmetic check is exact. §22's
detection mechanism is sound and its fix works. §26's reading of the comparison is correct. In
all three the *work* was right and the *headline* claimed more ground than the work covered —
and the headline is what the next reader trusts, because a ledger is read by someone catching up,
not someone re-deriving.

That suggests a cheap discipline: **when a section closes an axis, write what was actually
varied, not the axis's name.** "Checked `t * 2**frac` against exact rationals" would not have
protected `np.clip` for a year.

### 3.2 Phase 7 §3.1 works as a procedure, not just as hindsight

"List the parameters and ask which have only ever had one value" produced six of the seven
directly:

| finding | the axis with one value |
|---|---|
| >256 classes | `num_classes` was 2, 3 or 10 — always |
| the thermometer | the mismatch attack had only run against a learnable mapping |
| `quantize` saturation | nothing had ever asked for a word wider than 32 |
| `python -O` | asserts were only ever run with asserts enabled |
| stale directories | interruption was tested as kill-then-*re-run*, never kill-then-*verify* |
| `results` | it had only ever been a dict, or absent |

⚠️ **None needed a new technique.** Phase 7 reached for coverage and mutation testing when
hand-picked cases ran dry; this round needed neither. It needed the same procedure phase 7 wrote
down, pointed at the audit itself rather than at the tool.

### 3.3 Some checks cannot be exact, and the fix is to say so

The thermometer case is the honest one. A learnable mapping states its input width; a fixed
mapping is a list of indices and states nothing, so **there is no fact in the checkpoint to check
against.** No care recovers information that is not there.

What shipped is a signature — trailing features driving no comparator — reported and never
raised, because a genuinely lopsided model looks the same and refusing one would reject correct
models to catch an incorrect one. That is the trade `tool-roadmap.md` Q7 rejected for the area
model, and it is the same trade here.

⚠️ **The warning is paired with a test that it does NOT fire on any shipped shape.** A signature
that fires on ordinary models is noise, and noise gets switched off, which is worse than no
check at all.

### 3.4 The gate's blind spots are a growing list, and they have one shape

`quantize()` is the fourth entry:

| blind spot | why the gate cannot see it |
|---|---|
| a forgotten scaler | the testbench feeds correctly-scaled vectors |
| a mismatched thermometer | the vectors come from the same wrong assumption |
| `quantize`'s floor vs round | RTL and golden model share the function |
| **`quantize`'s saturation** | **`build` never calls it — the USER does** |

⚠️ **Every one is a specification choice rather than a translation error.** The gate proves the
Verilog matches `extract.forward()`, which is exactly what it claims and exactly its limit. What
it cannot prove is that `extract.forward()` describes the thing the user will actually build and
drive. The second implementation added in phase 6 §23 closes half of that; the other half is the
contract at the boundary — what the user is told to do with their own data — and it has no
simulator behind it by construction.

### 3.5 A promise in a docstring is only as strong as the runtime that keeps it

`build_core` said an emitter that half-succeeded "should not leave a file on disk anyone might
synthesize." One interpreter flag made that false, and no test would ever have noticed, because
tests do not run under `-O`.

The general form: **a guarantee implemented with `assert` is a guarantee with an opt-out.** Worth
a one-line grep on any project that states invariants in prose.

## 4. What is not done, and why

- **A thermometer check that is exact for fixed mappings** — impossible from the checkpoint
  alone, per §3.3. It would need the input width recorded at save time, which validates nothing,
  because `save()` derives it from the very thermometer in question.
- **`-O` in CI** — the guard is now flag-independent and directly tested in both modes, so a
  whole CI job would re-run 463 tests to check a property two tests already pin.
- **A build-id in the `.hex` files** — `$readmemh` parses values, not comments, so there is
  nowhere to put one. The four `.vh` files are the whole surface, and both halves are covered.
- **Refusing a threshold on the word's rail** — it is a legitimate format, just not one whose
  losslessness extends to the last value. Reported, not blocked.

## 5. Where the risk actually is now

⚠️ **Phase 7's answer was right about direction and wrong about exclusivity, and this phase's
answer should not repeat that.** The largest reservoir of untried values is still a real user's
model. What phase 8 shows is that "this project cannot reach it" was too strong — six of seven
came from re-reading the project's own claims and asking what was actually varied.

So the specific thing left is narrower than "audit again": **the tool's remaining blind spots are
now enumerated (§3.4), and three of the four are at the user's boundary rather than inside the
translation.** That boundary — the scaling contract, the quantization the user performs, the
thermometer they pair — is where a defect still turns into a working-looking, useless design, and
none of it is reachable by a simulator this project can run.

## 6. By the numbers

| | |
|---|---|
| defects found and fixed | **7** |
| of those, silently wrong on a green gate | **2** |
| of those, a false alarm on correct hardware | **1** |
| sitting in a seam of the previous audit | **7 of 7** |
| earlier headlines corrected | **2** (`phase6-ledger.md` §20, §22) |
| load-bearing `assert`s converted | **10**, across 4 modules |
| tests | **463 passing**, 6 skipped (from 416) |
| import warnings | **0** (from 1) |
| shapes re-attacked after the fixes | **12**, all still bit-exact |
