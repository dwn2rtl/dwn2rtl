# Phase 7 report — harden it, then ship the fixes

**Status: closed, 2026-08-19.**

The day-by-day record is `phase7-ledger.md`, and the round-by-round audit it grew out of is
`phase6-ledger.md` §10-29. This is the retrospective.

---

## 1. The result

**29 defects found and fixed, and `0.2.0` on PyPI carrying them.** Three produced a wrong result;
one hung the tool forever; the rest failed in a way that blamed the wrong thing.

The phase began as "try to break it" and ended somewhere more useful: the techniques that found
the most were the ones that **measure the tests rather than the tool.**

## 2. What was delivered

| | |
|---|---|
| **`0.2.0` on PyPI** | verified from the index, not just from a green publish job |
| the three headline fixes | vector packing at odd widths, pipeline depth, zero-latency designs |
| a second implementation | the golden model is finally cross-checked |
| macOS + dependency floors in CI | both were supported in name only; the floors found a defect |
| a version stamp on every generated file | so a mixed-version design says so |
| `CHANGELOG.md` | leading with the reason to upgrade |
| 406 tests | measured by mutation, not assumed |

## 3. Findings that outlive this phase

### 3.1 Every serious bug lived on an axis with exactly one tested value

This is the sharpest pattern in the whole audit, and it held every single time:

| the bug | the axis nobody varied |
|---|---|
| core vectors wrong at odd widths | every fixture and both studied models were byte-aligned |
| pipeline depth did nothing above 1 | only `Pipeline(1,1,1,1)` was ever gated |
| zero-latency designs mis-sampled | the same |
| `IDX_W` unverified | only one class count was checked against its width |
| torch/numpy incompatibility | only the newest of each was ever installed |
| macOS untested | listed as supported since day one |

⚠️ **None of these needed a cleverer test. They needed a second value.** The fix in each case was
a fixture, a parameter sweep, or a CI job -- not a smarter assertion. That is a cheap thing to
check for on any project: list the parameters, and ask which ones have only ever had one value.

### 3.2 Measuring the tests beats writing more of them

Seven rounds of adversarial input had decayed to finding message-quality issues. Then:

- **Coverage** found the suite never ran `verify` or `estimate` through `main()` -- the two
  commands users actually type had no test behind their success path.
- **Mutation testing** found two holes in an afternoon that seven rounds had missed, both being
  places where the spec said something no test checked: `luts > 0.0` could become `>= 0.0`, and
  `quantize`'s floor could become round.

⚠️ **And the second of those is invisible to the gate by construction** -- the RTL and the golden
model share that one function, so they agree whichever way it rounds. It matters because
`input_scaling.json` tells the USER to quantise their own inputs. That is the third instance of
the same shape: a specification choice the gate cannot see (the others being a forgotten scaler
and a mismatched thermometer).

### 3.3 A mutation score measures the tests you ran, not the tests that exist

Three separate sweeps produced false survivors from their own narrow selection -- 11, then 3,
then 3 -- and every time the fix was to widen the run rather than change the code. The `pipe_reg`
mutants "survived" only because the harness built with the default pipeline, where `STAGES=1`
makes all three no-ops.

⚠️ **A local score is also a lower bound**, because the strictest checker in the project is not
installed on the primary development platform. The GroupSum slice mutant simulates correctly --
Verilog truncates a wide expression from the MSB, keeping exactly the right bits -- and is
caught only by Verilator's `WIDTHTRUNC`. Behaviourally equivalent, structurally wrong, and dead
in CI while alive on a laptop.

### 3.4 The dangerous failures are the ones that look like success

Of the 29, the three that mattered most all shared a shape: **a correct design reported as
broken, or a broken input accepted quietly.**

- odd-width vectors -> `core FAIL` on hardware that was fine
- `Pipeline(lut=2)` -> mismatches on hardware that was fine
- a mismatched thermometer -> a design that builds, gates green, and puts real features in the
  wrong bit positions

A tool that crashes gets fixed. A tool that lies gets trusted.

## 4. What is not done, and why

- **`--data`** -- still deferred, still because it cannot be tested without a checkpoint whose
  thresholds are off-grid while its data is quantised.
- **Property-based testing** -- would explore axes now swept by hand; real, but the marginal
  yield is low against the cost.
- **A full mutation run over every module** -- mechanical, hours of runtime, and the sampled runs
  are already hitting equivalent mutants rather than holes.
- **`estimate` under real yosys** -- only reachable in CI, which covers it.

## 5. Where the risk actually is now

Every technique available here has been applied, and the last few rounds found progressively
narrower things. ⚠️ **The remaining risk is not in code this project can reach: it is in a real
user's model.** Every serious defect this phase found came from a value nobody had tried, and the
supply of untried values that matter is now much larger outside this repository than inside it.

## 6. By the numbers

| | |
|---|---|
| defects found and fixed | **29** |
| of those, producing a wrong result | **3** |
| of those, hanging forever | **1** |
| audit rounds before the technique changed | **7** |
| holes found by coverage and mutation after that | **6** |
| shipped Verilog mutation score | **11 / 11** non-equivalent mutants killed |
| tests | **406 passing**, 16 skipped |
| CI | 3 platforms, 2 simulators, a linter, a synthesis estimator, dependency floors |
| releases | `0.1.0`, then `0.2.0` fixing it |
