# Phase 6 report — make it legible, and get a second opinion

**Status: closed, 2026-08-18.**

The day-by-day record is `phase6-ledger.md`. This is the retrospective.

---

## 1. The result

Two goals that looked unrelated and were not. The docs and code were made readable by someone
who is not us, and a second simulator was pointed at the emitted RTL for the first time.

**The second one found a defect the gate cannot catch, and then changed the plan it was meant to
serve.** Verilator was probed in order to decide whether to build it in as a backend. The probe
found the *linting* was the valuable half -- the backend was built too, but second, and knowing
it was the smaller prize.

⚠️ **And then an adversarial audit found something bigger than either**: the emitted core-level
golden vectors were wrong for every model whose encoder width is not a multiple of 8. The suite
had always passed, because nothing in the repository -- or in two completed studies -- had such
a width.

Twenty-six commits. ⚠️ ~~Nothing here changed what `build` emits~~ -- **withdrawn, and the
exception is the point**: the vector-packing fix changes `x_binarized.hex` for any model whose
encoder width is off a byte boundary. The Verilog is byte-for-byte identical; the testbench's
input file was wrong. `verify` also gained a second simulator, and several bad-input paths
gained real errors.

## 2. What was delivered

| | |
|---|---|
| `docs/user-guide.md` | using the tool on your own model, with the recipes now under test |
| a linter in CI | catches malformed-for-another-tool, which the gate structurally cannot |
| a second simulator in CI | core 504 ✓, top 519 ✓ -- two independent implementations agreeing |
| docs that stand alone | no outside repository named; every claim checkable from here |
| the comment cull | longest block in `src/` 34 lines -> 13; the four primitives 120 -> 20 |
| `tests/test_user_guide.py` | the published recipe executed, so a rename breaks a test |
| `--simulator verilator` | optional second backend, Linux and macOS; iverilog stays the default |
| bad input fails with a reason | four raw tracebacks, found by an audit of 30 wrong-input cases |
| **correct vectors at any width** | a byte-packing bug that made 12-bit vectors 16x too large |
| an `odd_width` fixture | 18 bits, so the gate drives the off-byte path on every commit |
| metadata that cannot break a design | a newline in `run_name` emitted uncompilable Verilog |
| a pipeline depth that means what it says | `pipe_reg` inserted one register for any count |

## 3. Findings that outlive this phase

### 3.1 The gate has a blind spot, and it is not about arithmetic

CLAUDE.md's rule is that emitted RTL is not correct until a simulator says it matches the golden
model. That rule is sound and it is not enough, because it only ever asks one question.

Two prose lines in `argmax.v` began with the word "verilator". That is a **pragma**, so the
sentences after them were rejected as unknown ones and `--lint-only` exited with an ERROR --
while **iverilog printed PASS on the same files.** The gate was green. Nothing in the suite
noticed. A user linting an emitted design would have hit a hard error in a file we shipped.

⚠️ **The generalisation: the gate proves the RTL computes the right thing, and says nothing
about whether the file is well-formed for a tool the gate does not run.** Those are different
questions and they need different checks. A linter in CI is now the second one, and it exists
because it caught something on its first day.

The same lesson has a second edge, found in `tests/test_user_guide.py`: **omitting the input
scaling still PASSES the gate.** The golden model is computed from the same quantised input
written to the vector file, so both sides agree on whatever is supplied. The gate proves
RTL == golden model. It cannot prove the inputs were the ones you meant, and no simulator can.

### 3.2 A probe can move the value, not just the risk

The Verilator probe was framed as risk reduction: find out whether it can run these testbenches
before building anything. It answered yes, easily -- `--binary --timing` compiles them unmodified
and `verify.py`'s existing regexes parse the output with **zero changes**.

But the same probe measured two things that dissolved the case for the backend:

- **Speed: no gain.** 0.04-0.07 s under both simulators. Verilator's advantage is large designs.
- **Agreement: exact.** 504 and 519 vectors, 0 mismatches, matching iverilog.

So what the backend uniquely offers is *user-facing choice of simulator* -- and the verification
value it was wanted for could be had in CI without touching the product.

⚠️ It was declined on that basis and then **built the same day**, because the only load-bearing
argument turned out to be "nobody has asked" and the owner asked. The measurements survived the
reversal and now shape the design instead of blocking it: iverilog stays the default *because*
of the 39x, and the CLI help says so. Recording the decision with its evidence is what made
reversing it a two-minute conversation rather than a re-argument.

**This is phase 4's rule doing something new.** There, measurement cancelled two changes. Here
it *ordered* them: the CI check came first and found a bug, the backend followed and is
deliberately not the default. Neither would have been true from the argument alone.

### 3.3 A control turns a plausible story into a fact, twice

Verilator called `argmax.v` "circular combinational logic", fatally. The structural argument
against that is easy to make and easy to get wrong.

**So a control was built instead:** a trivially acyclic adder tree of the same shape, nothing but
`+`, which raises the identical warning on the identical line. That proves the warning follows
from the array-of-levels shape, not from anything in `argmax.v` -- and a packed vector raises it
too, so it is dependency granularity rather than declaration style.

The second control ran the other way. A test docstring claimed the gate would catch a forgotten
input scaling; checking it showed the opposite, and the docstring was corrected. **A control is
as useful for disproving your own claim as for proving it**, and both happened in one phase.

⚠️ Related, and the reason the waiver is a waiver: *"restructuring is worth it because UNOPTFLAT
costs simulation speed"* was withdrawn on measurement. It costs nothing measurable here.

### 3.4 A test suite can be exhaustive and still never touch the bug

The vector-packing defect is the sharpest thing this phase produced, and the reason is not that
it was subtle. `np.packbits` pads a partial byte on the low side; the project **already knew
that**, had been bitten by it once, and documents it in `emit_core.table_to_hex`. The fix simply
never travelled to the second place the same trick was used.

⚠️ **What kept it invisible was the data, not the code.** Every fixture was 8 or 24 bits wide.
Both studied models are too -- MNIST 784x3 = 2352, JSC 16x8 = 128. Seventy-seven configurations
across two datasets, every one bit-exact, and not one of them had a width off a byte boundary.
No amount of running that suite harder would have found it.

**So the fix that matters is the fixture, not the patch.** `odd_width` (6x3 = 18) now runs
through both gate levels on every commit. Before it existed, reverting the patch broke nothing;
now it breaks eight tests.

The generalisation is uncomfortable and worth keeping: **"the tests pass" and "the tests
exercise this" are different claims**, and a suite that grew alongside one shape of data will
agree with itself indefinitely. The only thing that broke the tie here was deliberately building
inputs nobody had built before.

### 3.5 Comments compete with each other

The cull was asked for as readability work. It turned up two references to things that do not
exist in this repository -- a `jsc-complete` tag and "the training notebook" -- plus a
pre-existing defect where a comment describing one constant had been stranded above another.

**All three were invisible because they were buried in paragraphs nobody read.** That is the
argument for brevity that survives taste: a warning sitting alone gets read, and the same
warning in the ninth line of a block competes with eight other sentences for attention.

The rule applied everywhere was **keep what stops a bug, cut what tells a story** -- decisions
belong in these ledgers, which is what they are for. Nothing was deleted that is not recorded
somewhere reachable.

### 3.6 Governing files go stale silently

`CLAUDE.md` still said *"Cite that; do not reproduce it here"*, naming the outside repository,
after the README had deliberately stopped naming it. Nothing enforces a file that instructs
future work, so the next session would have followed it and undone the change.

⚠️ And it is **gitignored**, so the correction lives on one machine. Worth knowing about a file
that governs everyone's work.

## 4. What is not built, deliberately

- ~~**`dwn2rtl verify --simulator verilator`.** Declined; see §3.2 and the ledger's three
  triggers.~~ ⚠️ **Built** (`phase6-ledger.md` §8). `Simulator` now owns its two commands, so
  `_run_level` never knows which tool it drives; discovery prefers iverilog on the 39x
  measurement; Linux and macOS only, and the docs say why not Windows.
- **A restructured `argmax`.** The UNOPTFLAT warning is a measured false positive and costs
  nothing measurable; the waiver carries its reasoning.
- **`--data`** (phase 4, unit 8 tier 2). Unchanged and still correct: it cannot be tested without
  a checkpoint whose thresholds are off-grid while its data is quantised.

## 5. Correction to phase 5

`phase5-report.md` §6 listed *"Verilator as a second simulator backend"* as the top remaining
item, on the reasoning that a second simulator is the only way to know the gate is not passing on
an iverilog quirk. **The reasoning was right and the conclusion was wrong**: that assurance did
not need a backend, and is now had in CI. §6 is struck and points here.

## 6. By the numbers

| | |
|---|---|
| commits | **33** |
| tests | **370 passing**, 16 skipped |
| wrong-input cases audited | **30**, of which **4** reached the user as a traceback |
| defects found by the adversarial audit | **21**, three of them producing a wrong result |
| hostile inputs that behaved correctly | **60+** across the CLI, the reader and damaged output |
| studied configurations that could have caught the vector bug | **0 of 77** |
| decisions reversed by their own evidence | **2** -- the UNOPTFLAT fix, and the backend itself |
| simulators agreeing on every vector | **2** |
| defects found by the linter on its first day | **1**, invisible to the gate |
| docstring claims disproved by their own control | **1** |
| stale references found while shortening comments | **2**, plus 1 pre-existing defect |
| longest comment block in `src/` | 34 lines -> **13** |
