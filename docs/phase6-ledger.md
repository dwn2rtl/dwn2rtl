# Phase 6 ledger — make it legible, and get a second opinion

**Goal.** Two things that turned out to belong together: make the code and docs readable by
someone who is not us, and find out whether a second simulator has anything to say about the
emitted RTL.

**Status: CLOSED.** Retrospective in `phase6-report.md`.

> Conventions in `overview.md` §6. Entries oldest first, recording *built* / *hit* / *decided*.
> Reversed decisions stay, struck through, with the reason.

---

## The rule this phase is organised around

**A comment earns its place by preventing a bug, not by explaining a decision.** Decisions
belong in these ledgers, which is what they are for. Code that carries its own history dilutes
the warnings sitting next to it -- and the warnings are the part that stops someone shipping
something broken.

That is a rule about *where* knowledge lives, not about having less of it. Nothing was deleted
here that is not written down somewhere a reader can reach.

---

## 1. Built — the docs were made to stand alone

`README.md` was cut to what a new user needs, `docs/user-guide.md` was added for using the tool
on a real model, and outside references were dropped from the docs, the comments, the package
metadata and the shipped RTL.

⚠️ **Reversal, recorded in `phase5-ledger.md` §2.** That ledger had decided the earlier research
repository could be cited, because the link resolved. It was removed anyway: standing alone
outranked saving the work. The README now states the evidence as this project's own history and
rests it on the numbers its test suite pins, which is checkable from here.

## 2. ⚠️ Hit — a find-and-replace left four names for one thing, and three broken sentences

Replacing the outside repository's name mechanically produced `the earlier implementation` (24
uses), `An earlier iteration` (4), `the earlier research repository` (3) -- and three sentences
that no longer parsed, of which the worst was circular:

```
The study repository it came from is the earlier research repository.
```

It also broke the wrap convention in 22 places, measured against a before/after control: **5
lines over 100 columns before the rewrite, 27 after.** Every one had the same cause -- the
replacement was twelve characters longer than what it replaced.

**Fixed by choosing a shorter word rather than rewrapping 22 lines.** `the study` is the
README's own vocabulary, drops no outside reference, and is shorter than the original, so the
overflow disappeared instead of being reflowed. One name, 26 uses, back to the pre-existing 5
long lines.

## 3. ⚠️ Hit — `CLAUDE.md` still instructed the opposite of what the docs now did

It read *"Cite that; do not reproduce it here"*, naming the outside repository, after the README
had deliberately stopped naming it. That is worse than a stale doc: it is the file that governs
future work, so the next session would have followed it and put the reference back.

Rewritten to state the evidence as this project's own history and to require that the docs stand
alone. ⚠️ **`CLAUDE.md` is gitignored**, so the fix lives on one machine and does not travel --
worth knowing about a file that governs everyone's work.

## 4. Built — the user guide's recipe is now actually tested

`docs/user-guide.md` tells a user to re-label an emitted design with their own data, using
`dwn2rtl.extract` and `dwn2rtl.vectors` -- internal modules with no API promise -- and claimed
the recipe *"is tested and works today"*. Every function was covered; **the recipe was not**.
Nothing ran it end to end.

`tests/test_user_guide.py` makes the claim true rather than softening it, transcribing the recipe
as literally as a test can. A rename in those internals now fails a test instead of silently
invalidating published instructions.

### ⚠️ Hit — the control disproved the test's own docstring

The gate test claimed a PASS proved the user's data was reproduced, listing "forget the scaling"
among the mistakes it would catch. **Checked, and it does not:** omitting the scaling still
PASSES, because the golden model is computed from the same `xq` written to the hex file, so both
sides agree on whatever inputs are supplied.

That is correct behaviour and exactly what the README already warned about. A defect that made
it into a docstring while the code was right. Corrected in place, and the limit is now recorded:
**the gate proves RTL == golden model; it cannot prove the inputs were the ones you meant, and no
simulator can.** A real defect was then found for it to catch -- quantising thresholds one bit
off gives **47 mismatches**.

## 5. Built — the Verilator probe, before building anything

Roadmap-style unit 1: measure first. Three questions, with a control already banked (WSL's own
iverilog runs the same directory to 519 vectors, 0 mismatches).

| | |
|---|---|
| lints clean? | **No** -- 2 warnings, both `UNOPTFLAT` in `argmax.v` |
| `--binary --timing` compiles the testbench? | **Yes**, exit 0 -- the `always #5` clock is fine on 5.020 |
| runs and prints the RESULT line? | **Yes** -- core 504, top 519, matching iverilog exactly |

🎯 **And `verify.py`'s existing regexes parse Verilator's output unchanged** -- `_RESULT` -> PASS,
`_VECTORS` -> 519, `_MISMATCHES` -> 0. The whole simulator-specific surface is two subprocess
command lines.

**No WIDTH warnings anywhere**, which was the class most expected on RTL that had never seen a
strict linter, and is a genuinely good signal about the emitters.

## 6. ⚠️ UNOPTFLAT — a false positive, proven with a control rather than argued

Verilator reported "circular combinational logic" on `argmax.v`'s two level arrays, fatally.
Structurally there is no loop: `lvl_*[l+1]` reads only `lvl_*[l]`.

**The control settles it.** A trivially acyclic adder tree of the same shape -- nothing but `+`
-- raises the identical warning on the identical declaration line. A packed vector instead of an
unpacked array raises it too, so the cause is per-signal dependency granularity, not the
declaration style.

⚠️ ~~"Restructuring is worth it because UNOPTFLAT costs simulation speed"~~ — **withdrawn on
measurement.** 0.04-0.07 s for 519 vectors, the same as iverilog on the same design. Rewriting
logic that is bit-exact under two simulators and maps to the vendor's exact LUT count, to buy
nothing measurable, is the trade phase 4 declined twice. Waived, with the reasoning in the file.

### ⚠️ Hit — the comment explaining the waiver broke linting

Two prose lines began with the word "verilator". **That is a pragma**, so the sentences after
them were rejected as unknown ones: `--lint-only` exited with an ERROR while **iverilog still
printed PASS**, so the gate stayed green and this would have shipped invisibly.

The project's own thesis, aimed back at us: the gate proves the RTL matches the golden model and
says nothing about whether the file is well-formed for a tool the gate does not run.
`tests/test_lint.py` now pins it, deliberately *unmarked* so it runs on Windows too -- where the
offending comment gets written and no linter is installed.

## 7. Built — Verilator in CI, as lint and then as a cross-check

Added the way `yosys` was: a marker, a conftest skip, an apt line on the Ubuntu jobs, and a guard
step that fails if the tool installed but is unusable. **Optional in exactly the same sense --
nothing in `src/` references Verilator, it is in no dependency list, and `build`/`verify` behave
identically without it.**

The cross-check compiles with `--binary --timing` and asserts `RESULT : PASS`, zero mismatches,
**and a non-zero vector count** -- a testbench that silently ran nothing would otherwise print
PASS. Green on real runners: **core 504, top 519, two independent simulators agreeing.**

⚠️ Platform support, stated at the level it was actually verified: **Linux tested**; macOS
standard but untested here; Windows is not a supported route for Verilator (it generates C++ and
needs a compiler; WSL is the way). Hence Linux-only in CI.

**Marker renamed `lint` -> `verilator`**, since it now means "needs this tool" rather than "does
linting", matching how `yosys` is named.

## 8. ⚠️ ~~Decided — the `--simulator` backend is NOT built~~ — REVERSED, and built

> **Reversed the same day.** The reasoning below was sound on its own terms and the conclusion
> did not follow: the only load-bearing argument was "nobody has asked", and the owner then
> asked. Kept in full because the *measurements* in it still hold and still shape the design --
> iverilog remains the default because of them.

Units 2-5 of the original plan let a user run `dwn2rtl verify --simulator verilator`.

**What the backend adds is user-facing choice.** The verification value -- two independent
simulators agreeing -- was already had in CI without touching `verify.py`.

**The argument for it:** a user with Verilator and no iverilog could not run `verify` at all,
since `find_simulator` looked only for iverilog and `_run_level` hardcoded its two-command
shape. Verilator is the common simulator in RISC-V and academic toolchains.

~~**Why it still loses:** the cost to that user is `apt install iverilog`, about 9 MB and one
command, against a permanent second discovery path and second command shape in the one module
where correctness matters most.~~ **Withdrawn.** That cost is real but small, and "optional extra
tool support, documented as Linux/macOS" is an ordinary thing for a tool to have -- the yosys
precedent is exactly it.

### What was built

- `Simulator` owns its two commands, so `_run_level` no longer knows which tool it drives and
  adding a simulator never touches the code that decides PASS or FAIL. Output parsing is
  unchanged, because the *testbench* prints the RESULT line.
- `--simulator verilator`, or an explicit path. `verify(simulator=...)` from Python.
- Discovery order: iverilog anywhere (PATH, then the installer directories) -> verilator ->
  fail. ⚠️ **A fallback-located iverilog outranks an on-PATH verilator, deliberately**, on the
  measurement below.
- Docs say Linux and macOS, and say why Windows is absent: Verilator generates C++ and needs a
  compiler, so WSL is the route there.

### The measurement that decides the default

| | compile + run, same design |
|---|---|
| iverilog | **0.38 s** |
| verilator | **14.74 s** |

**39x, and the opposite of the usual intuition.** Verilator's advantage is throughput on long
simulations; here the simulation is 0.04 s and the C++ compile dominates. So Verilator is never
selected automatically when iverilog exists, and the CLI help says why.

⚠️ **One test could not run locally.** `test_verify_drives_verilator_end_to_end` needs Verilator,
which does not install natively on Windows, so the assembled backend path first executed on CI's
Linux jobs. The command shape itself was proven during the probe (504 and 519 vectors, PASS).

## 9. Built — the comment cull, across every code file

Applied one rule everywhere: **keep what stops a bug, cut what tells a story.**

| | before | after |
|---|---|---|
| the four shipped primitives | ~120 comment lines | **20** |
| longest block in `src/` | 34 lines | **13** |
| `src/` average non-code | 30% | **24%** |
| `tests/` docstrings over 4 lines | 44 | **17** |
| longest docstring in `tests/` | 32 lines | **13** |

What went: which commit found a bug, how many crashes it caused, what a previous version of a
test did. What stayed: the address bit order, the `n <= 6` silent truncation, `all([])` is True,
the POSIX/Windows `shutil.which` split, yosys 0.33's one-LUT core, the pragma word.

⚠️ **Two stale references surfaced while cutting**, both in shipped RTL: a `jsc-complete` tag
that does not exist in this repository, and "the training notebook", which this tool does not
have. Both were invisible while buried in paragraphs nobody read.

⚠️ **And a pre-existing defect**: in `checkpoint.py` a comment describing `REQUIRED_CONFIG` had
been stranded above `UPSTREAM_URL`, two unrelated comments run together with the constant they
described several lines away. Confirmed by git to predate this phase. Reattached.

---

## 10. ⚠️ Built — an error-path audit, and four tracebacks it found

The happy paths were exhaustively tested; the wrong-input paths were not. A harness walked **30
ways to hold the tool wrong** -- bad checkpoint shapes, bad config values, bad build arguments,
and `verify` against five kinds of damaged output -- asking of each: does it fail with a NAMED
reason, and does it ever silently succeed?

**26 gave a clean error. Four reached the user as a raw traceback**, each fixed at the layer
that owns it:

| | was | now |
|---|---|---|
| a file that is not a checkpoint | `UnpicklingError` from inside torch | says what it expected, shows the `torch.save` line |
| `num_classes = 0` | `ZeroDivisionError` from inside the golden model | a positive-integer guard, before anything divides |
| `--input-bits 999` | `OverflowError: int too big to convert` | bounded -- values are int64, so 64 bits is a real ceiling |
| `--out` pointing at a file | raw `FileExistsError [WinError 183]` | "is a file, not a directory" |

⚠️ The CLI's `build` handler also caught only `CheckpointError` and `FileNotFoundError`, so two
of the new errors would have tracebacked anyway. Widened to the bad-input family.

**What held up:** `verify` against damage. Emptying `x_quant.hex`, `expected_top.hex`,
`dwn_core.v` or `dwn_top_params.vh`, or deleting `tb/` entirely -- **none reported PASS.** Each
gave FAIL, ERROR or MISSING with real detail. The "nothing unchecked reads as a pass" claim
survives contact with a broken directory, which was the likeliest place to find a hole.

**Left as a finding, not a change:** `--input-bits 0` is legitimate (integer-valued inputs) but
yields `Q0.0 signed (1-bit)`, and the report says *"provably lossless"* two lines above *"10 of
24 thresholds quantise to a duplicate comparison"*. Both are true about different things --
the quantiser is order-preserving, the comparator collapse is a separate loss -- but printed
together they read as "nothing was lost".

**And the scaling question that prompted deferring `--data` was measured here**: integer bits
are derived exactly from the thresholds, so the word auto-widens (Q2.12 for standard-scaled,
Q10.12 for ±500) and saturation stays lossless in every case tried. Resolution is not
guaranteed, but insufficiency is *reported* rather than silent. A visible, correctable failure
mode is what makes `--data` optional rather than necessary.

## 11. 🎯 Found — the emitted core vectors were wrong at any width off a byte boundary

The audit went adversarial: build deliberately awkward models and try to break the tool. One
shape failed the gate, and the failure was a combination the docs do not list.

```
n_features=6, z=3  ->  core=FAIL (330 mismatches), top=PASS
```

⚠️ **core FAIL while top PASSES is not "the network".** The top level drives the same core
through the encoder and is bit-exact, so the RTL is right and the *core-level vectors* are
wrong. Bisecting the shape:

| core input width (`n_features * z`) | gate |
|---|---|
| 8, 24 | PASS |
| 12, 18 | **FAIL** |

Failing widths are exactly those not divisible by 8, which is byte packing. `bits_to_hex` used
`np.packbits`, and **packbits pads a partial byte on the LOW side**, so the value came out
shifted left by the padding. Measured directly:

| width | emitted |
|---|---|
| 8, 16, 24 | correct |
| 12 | **x16** |
| 18 | **x64** |

**This is the same defect, from the same cause, that `emit_core.table_to_hex` documents in its
own comment.** It was found there, fixed there, and the identical bug sat in `vectors.py`
untouched -- the fix never travelled to the second call site.

### ⚠️ Why the gate ran green over it for the project's whole life

Every fixture was 8 or 24 bits wide. **Both studied models are too** -- MNIST 784x3 = 2352, JSC
16x8 = 128, both divisible by 8. The broken path was never driven, by anything, ever. 77
configurations across two datasets did not touch it.

That is the sharpest example yet of the difference between "the tests pass" and "the tests
exercise the thing": no amount of running the existing suite harder would have found this.

**Fixed** by padding at the high end, where the extra bits become leading zeros. Verified across
widths 1-39. And the structural fix is the fixture: **`odd_width` (6x3 = 18)** now runs through
both gate levels on every commit, so the case cannot go dark again. Reverting the fix fails 8
tests; before the fixture existed it failed none.

⚠️ **What this did and did not affect.** The emitted *hardware* was always correct -- this was
the testbench's input file. A user with an odd width saw `core FAIL` on a good design and had no
way to know it was a false alarm. `0.1.0` on PyPI has it, which makes 0.2.0 a fix rather than a
feature.

## 12. ⚠️ Found — a non-finite threshold hung the tool forever

`required_int_bits` searches upward with `while (1 << bits) <= span`. With `span = inf` that
condition is always true: **no error, no traceback, no timeout -- it simply never returns.**
NaN was quieter and equally wrong, reporting 0 integer bits for a model that cannot be
represented at all.

Worse than a crash, because nothing tells the user anything. Now refused at load, naming NaN in
the training data as the likely cause, and the function itself can no longer spin whoever calls
it.

## 13. Found — two ways to be quietly given the wrong thing

**A stale `input_scaling.json`.** Every other emitted file is overwritten on rebuild; this one
is conditional, so building an unscaled model over a scaled one in the same directory left the
PREVIOUS model's mean and scale behind. A user following it applies another model's
transformation -- the design then runs at chance and looks healthy, which is the exact failure
the file exists to prevent. Now removed, with a warning.

**A user's own Verilog compiled into the gate.** `verify` globbed `*.v` from the directory, and
the README tells users to instantiate `dwn_top` in their own harness. A harness dropped in there
was compiled into the test, so a syntax error in a file *not under test* broke verification
entirely. `verify` now compiles the emitted design by name, and reports a missing one by name
rather than as "unknown module type".

## 14. ⚠️ Found — metadata that only reaches a comment could stop a design compiling

`run_name` and `results` are written into the emitted Verilog header and nowhere else. Nothing
in the hardware depends on them, and all four ways they could go wrong were worse than a missing
comment:

| the checkpoint held | what happened |
|---|---|
| `run_name` with a newline | escaped the `//` and emitted **Verilog that does not compile** |
| `run_name` with non-ASCII | **the build crashed** -- `UnicodeEncodeError` writing the file |
| `run_name` of 5,000 chars | an unbounded header line |
| `results={'final_acc': 'ninety'}` | raw `TypeError` from a format string, over a comment |

⚠️ **The unicode one is this project's own documented hazard in a place nobody checked.**
CLAUDE.md requires stdout to be ASCII because Windows consoles are cp1252 -- and the same
default applies to `open(path, 'w')`. The rule was known; the second place it applied was not.

Fixed with `comment_safe()` and `accuracy_line()` in `emit_core`: flatten whitespace,
transliterate to ASCII, truncate, and format a number only when it is one. **Metadata must never
be able to break a design.**

## 15. Found — two rough edges on the public surface

- **`from_model('a/path.pt', ...)`** raised `'str' object has no attribute 'state_dict'`, naming
  an implementation detail rather than the argument. Passing a path is the likely mistake, so
  the message now says so and points at `load()`.
- **An unrunnable `--simulator`** surfaced a bare `[WinError 2] The system cannot find the file
  specified`. Discovery deliberately accepts a simulator that merely exists (a version probe is
  cosmetic and must not cost a working install, `test_verify.py`), so the fix belongs where it
  is actually run: the failure now names the tool inside the normal report.

## 16. Found — four public arguments that named a library internal

Round three pointed at the API's own arguments rather than at checkpoints or files. Everything
structural held; what did not was the wording when a caller passes the wrong thing.

| call | said | says now |
|---|---|---|
| `build(..., pipeline={'enc': 1})` | `'dict' object has no attribute 'lut'` | "must be a dwn2rtl.Pipeline", with an example |
| `build(..., n_random='500')` | numpy's `UFuncTypeError` | "n_random must be an int" |
| `verify(..., levels=('core','topp'))` | bare `KeyError: 'topp'` | "unknown level(s); expected core, top" |
| `verify(..., levels='core')` | `KeyError: 'c'` | "must be a sequence, not a string" |

⚠️ **The string one is the classic Python trap**: a bare string is a sequence, so it iterated
into characters and failed on the first one. It is exactly the kind of argument a user passes
from memory.

**What held in the same round**, and is the larger half again: `pathlib.Path` works for both the
checkpoint and the output directory; `n_random=0` and `=1` build; `seed=0` builds; `timeout=0`
fails rather than passing; `levels=()` returns **not ok**, because an empty run is not a pass;
and a partially-copied directory reports MISSING or refuses outright at every prefix length
tried, never PASS.

## 17. Found — `estimate` argued with the wrong thing, and the CLI missed a whole error family

**`estimate`'s arguments were validated after the tool search**, so the error depended on what
was installed: a caller without yosys got *"no working yosys found"* for what is really a typo,
and a caller with yosys got a bare `KeyError`. Now checked first, which also makes them testable
on a machine that has no yosys at all.

| call | said | says now |
|---|---|---|
| `modules='dwn_core'` | iterated the characters -> `KeyError: 'd'` | "must be a sequence, not a string" |
| `modules=['dwn_coree']` | bare `KeyError` | "unknown module(s); expected ..." |
| `modules=[]` | ⚠️ silently measured **ALL** modules | measures nothing, needs no yosys, and is **not ok** |

⚠️ The empty case is the interesting one. `list(modules or MODULES)` treats an empty list as
"unset", so asking for nothing got you everything -- the opposite of what the caller wrote, and
the opposite of `verify(levels=())`, which honours empty and reports not-ok. **Two commands
disagreeing about what "nothing" means is worse than either answer.**

### ⚠️ And the CLI was catching subclasses instead of the family

`except (CheckpointError, FileNotFoundError, NotADirectoryError, IsADirectoryError, ValueError)`
names two `OSError` subclasses. **`PermissionError` is a third**, and it is what you get from a
file an editor left open or a copy left read-only -- so a read-only `dwn_core.v` in the output
directory reached the user as a traceback. All three subcommands now catch `OSError` itself.

The pattern is worth naming: **enumerating subclasses is a bet that you thought of all of them.**

## 18. 🎯 Found — the pipeline depth parameter did not do what it says, and only the default was ever gated

Round five pointed at `Pipeline`, a documented public argument. **Every non-default depth failed
the gate.** Two independent defects, both hidden by the same gap: the gate had only ever run
against `Pipeline(1, 1, 1, 1)`.

### `pipe_reg` treated a COUNT as a FLAG

`config.Pipeline` says *"Each is a stage count"* and `latency()` sums them --
`enc + lut * n_layers + pop + out`. The hardware said otherwise:

```verilog
if (ENABLE != 0) begin : g_reg   // ONE register, whatever the number
```

So `Pipeline(lut=2)` on a two-layer model claimed latency 6, the hardware delivered 4, and the
testbench sampled two cycles late: **201 mismatches on a design that was fine.** Fixed by making
`pipe_reg` a chain of `STAGES` registers -- and renamed from `ENABLE`, because a parameter named
for a flag invites exactly this.

### The testbench compared before it drove

```verilog
@(negedge clk);
if (i >= LATENCY) compare against expected[i - LATENCY];
x = vectors[i];                  // the input for i, applied AFTER the comparison
```

Correct for `LATENCY >= 1`, wrong at zero: a combinational design's answer for vector `i`
depends on an input that had not been applied yet, so every comparison was one step early.
⚠️ **`Pipeline(enc=1, lut=0, pop=0, out=0)` is a plausible build** -- register the encoder, leave
a small core combinational -- and it produced `core FAIL` on a correct design. Same false-alarm
class as the vector-packing bug, from the opposite direction.

Now drives, settles (`#1`, well inside the half period), then compares -- correct at every
latency including zero.

### ⚠️ Changing a testbench needs its own proof

These are the files everything else is checked against, so "the suite still passes" is not
enough: a testbench that passes *unconditionally* would also pass. Verified separately that the
comparison is still latency-sensitive by lying to it about the latency:

| claimed | real | result |
|---|---|---|
| 4 | 4 | PASS |
| 3 | 4 | **FAIL**, 192 mismatches |
| 5 | 4 | **FAIL**, 192 mismatches |
| 6 | 4 | **FAIL**, 201 mismatches |

Both fixes are now gated at six depths including `(0,0,0,0)` and `(1,2,1,1)`, and the
latency-sensitivity check is a test of its own.

## 19. Found — a scaler nobody could apply, and a file that was not valid JSON

`input_scaling.json` is not metadata: it is an **instruction the user must follow** before
driving `x_flat`. So a scaler that cannot be applied is worse than no scaler at all.

| the checkpoint held | what shipped |
|---|---|
| `scale = 0` | "apply (x - mean) / 0" -- an instruction that divides by zero |
| `scale = NaN` or `inf` | an instruction that is meaningless |
| `mean = NaN` | the same |

⚠️ **And the file stopped being JSON.** Python's `json.dump` writes bare `NaN` and `Infinity`,
which the JSON spec does not have -- so `input_scaling.json` was **rejected by strict parsers**
(JavaScript's `JSON.parse`, Go, Rust serde all refuse it). A user's toolchain would fail on a
file this tool told them to read, for a reason nothing in the file explains.

Both symptoms have one cause, so there is one fix: `_scaler_of` now refuses a non-finite or
zero scale, naming the likely reason (a constant or empty feature at fitting time, where
scikit-learn substitutes 1.0 rather than 0). A negative scale is still accepted -- it flips a
sign, which is unusual but perfectly followable. And the emitted file is now checked to parse
under a strict reader, not merely under Python's.

## 20. Measured — ~~the `input_bits` axis is clean~~, and the reason is worth keeping

> ⚠️ **Headline narrowed on 2026-08-19.** What was measured below is `quantize_thresholds`'
> multiply, and that result stands exactly as written. It is not the whole axis: `quantize()`'s
> `np.clip` was never checked, and it corrupts saturation from about a 55-bit word upward — at
> 64 bits a saturating input comes out with its sign flipped. `phase8-ledger.md` §3. The section
> below is a correct measurement under a headline that claimed more ground than it covered.

Tests only ever used `input_bits=8`, so the axis got the same treatment as the byte-boundary
one. **Every width from 1 to 63 builds and passes the gate**, and 64 is refused at the boundary
because the word would exceed int64.

⚠️ The interesting part is a bug that is NOT there. Quantising a threshold computes
`t * 2**frac`, and with a wide word that product can pass float64's 53-bit mantissa -- the
obvious place for silent precision loss. Checked against exact rational arithmetic at ten
combinations up to a 63-bit word: **error of zero, every time.** Multiplying a float64 by a
power of two only changes the exponent, so it is exact, and `word_bits <= 64` bounds the product
below int64's maximum by construction.

Recorded because a negative result that was actually measured is worth more than a hazard nobody
checked -- and because the next person to look at this will have the same suspicion.

## 21. Measured — `words_to_hex` and the top level are clean, at every width

`bits_to_hex` had a packing bug at any width off a byte boundary, so its counterpart was the
obvious next suspect. It is not affected, and the reason is structural: `words_to_hex` builds
the value with **Python integers** rather than `np.packbits`, so there is no byte to pad.

Checked against an exact reference at word widths 1-33, feature counts 1-8, and the two's
complement extremes of each width: **0 wrong values, 0 truncations.** The field width is
`len(words) * word_bits // 4`, which truncates -- but Python's format width is a MINIMUM, so a
value needing another nibble simply prints one. Under-specified padding, never a wrong value.

**End to end, 32 combinations** of feature count (1-7), thermometer bits and `--input-bits`,
covering every residue of `X_W` mod 8: **all pass.** Both helpers are now pinned by tests across
widths, so neither can regress into the other's bug.

## 22. Found — the thermometer was never checked against the model it came with

A DWN is two objects, saved separately, so pairing the wrong ones is a mistake the README
already warns about in the other direction. **A thermometer with 99 features loaded happily
against a model whose first layer expects 8.**

⚠️ **The existing range check cannot catch this**, and that is the point: it rejects a wiring
index that is out of range, so a thermometer that is too SMALL fails -- while one that is too
LARGE leaves every index comfortably in range. The design would build, and **the gate would
pass**, because the test vectors are generated from the same wrong assumption. Only real
features would reveal it, by landing in the wrong bit positions.

Detectable exactly: a learnable mapping's `weights` is `(input_size, output_size * n)`, so the
first layer states how many bits it expects, and that must equal `features x z`. Now checked,
naming the likely cause -- a model and thermometer from different training runs.

> ⚠️ **~~Now checked~~ — half of it was, corrected 2026-08-19.** The mechanism above only exists
> for a *learnable* first layer; a fixed mapping states nothing about its input width, so the
> same wrong thermometer still builds clean and still gates PASS on both levels. The diagnosis
> here is right and the fix works — it was scoped to the representation that made detection easy,
> and the other half was not recorded as remaining. `phase8-ledger.md` §2.

## 23. 🎯 Built — the golden model now has a second implementation behind it

**The one failure the gate cannot detect, closed.** `verify` proves the RTL matches
`extract.forward()`. If that function were itself wrong, the RTL would match it perfectly and
both would be wrong together -- and `verify.py`'s own docstring said the quiet part out loud:
*"there is no second independent implementation to cross-check against."*

Now there is. `tests/naive_reference.py` implements the forward pass and the encoder **from
docs/checkpoint-format.md**, structured to share as little as possible with the original:
explicit per-node loops and bit shifts where `extract.py` gathers and vectorises. Vectorised
code and loop code fail differently, which is the whole point.

Result across all six shapes, 200 samples each: **group sums identical, predicted classes
identical, encoder bits identical.**

⚠️ **Two things that make it evidence rather than decoration.**

The tie-break is the subtlest rule in the model -- numpy, torch and the RTL all keep the LOWEST
tied index -- so a comparison that never hits a tie says nothing about it. Measured rather than
hoped:

| shape | classes | vectors tying for top |
|---|---|---|
| tiny | 2 | 123 / 500 |
| n6 | 3 | 210 / 500 |
| odd_width | 3 | 243 / 500 |
| ten_class | 10 | 271 / 500 |

And the reference must be able to DISAGREE, or agreement proves nothing. A deliberately
corrupted final-layer table makes the two diverge, and that is a test of its own.

⚠️ **What it cannot do**, stated so nobody over-reads it: both implementations were written here,
so a shared misreading of upstream's semantics would survive. That risk is managed separately by
the pinned upstream commit and `docs/checkpoint-format.md`. What this catches is transcription
error -- a reversed address, a wrong group boundary, an off-by-one table index -- which is the
class the gate is structurally blind to.

## 24. Built — coverage and mutation testing, which measure the TESTS rather than the tool

Seven rounds of adversarial input had reached the point of finding only message quality, so the
next step was to stop adding cases and start measuring whether the existing ones can fail.

**Coverage** (branch-aware, 89%) found one real gap: the suite **never ran `dwn2rtl verify` or
`estimate` through `main()`**. Only a CI step did, so the two commands users actually type had no
test behind their success path. Everything else uncovered is either a defensive branch or the
yosys path, which needs the tool installed.

**Mutation testing found two holes that seven rounds of auditing had missed.** Ten deliberate
breakages, each applied alone, with the suite run against it:

| mutation | caught? |
|---|---|
| binarized hex: drop the byte padding | yes |
| LUT address: reverse the bit order | yes |
| `luts > 0.0` becomes `>= 0.0` | ⚠️ **NO** |
| argmax: ties keep the highest index | yes |
| `quantize`: floor becomes round | ⚠️ **NO** |
| thresholds: floor becomes ceil | yes |
| RTL argmax: `>` becomes `>=` | yes |
| RTL lut_node: invert the address | yes |
| RTL popcount: an equivalent rewrite | survived, correctly |
| `required_int_bits`: off by one | yes |

**Both survivors were places where the spec says something no test checked.**

`extract.py`'s own comment claimed *"an exact 0.0 emits 0 -- unlikely in a trained model, but
Gate 1 covers edge cases."* Gate 1 did not: fixtures draw uniform floats, which never land on
exactly 0.0, so the strictness of `> 0` was never exercised.

⚠️ And the quantiser's rounding mode is invisible to the gate **by construction** -- the RTL and
the golden model both use that one function, so they agree whichever way it rounds. It still
matters, because `input_scaling.json` tells the USER to quantise their own inputs, and a
different mode puts boundary features on the wrong side of a comparator. Same shape as the
missing-scaling finding: a specification choice the gate is structurally blind to.

Both closed, and the mutants now die: **9 killed, 0 uncaught, 1 equivalent mutant correctly
surviving.**

## 25. 🎯 Found — the declared dependencies permit a combination that cannot work

macOS was listed as supported from the first release and **had never once run**, and the floors
`numpy>=1.22` / `torch>=2.0` had never been installed -- every CI run resolved to the newest
release. Both are now jobs, on the ledger's own principle that a supported thing nothing
executes on is a claim rather than a fact.

Building the floor environment found a real defect immediately:

```
numpy 2.4.6 | torch 2.0.1  ->  209 failed
RuntimeError: Numpy is not available
```

⚠️ **torch builds before ~2.3 were compiled against NumPy 1.x and cannot hand a tensor to NumPy
2 at all.** Every emitter reads tensors through `.numpy()`, so the tool does not work -- and our
metadata permits it: the two constraints are each satisfiable while being mutually incompatible,
and pip resolves numpy to the newest release, so **anyone holding an older torch gets the broken
pair by default.**

**Measured before choosing a fix:** torch 2.0.1 with numpy 1.26 passes all 394 tests. So
raising the torch floor would forbid a combination that works, to prevent one that pip produces
by accident. Instead the broken pairing is detected once, at the entry points, and explained
with both ways out:

```
this torch (2.0.1+cpu) cannot exchange arrays with this numpy (2.4.6): Numpy is not available
  torch builds before ~2.3 were compiled against NumPy 1.x and cannot work with NumPy 2.
  Fix either side:  pip install "numpy<2"   or   pip install -U torch
```

Verified end to end in a genuinely broken environment, on the path a real user takes -- saving a
checkpoint from a healthy install and loading it from the broken one.

## 26. 🎯 Found — a systematic mutation sweep reopened the bug this project already paid for

Hand-picked mutants had stopped finding anything, so the next sweep was **generated rather than
chosen**: flip every comparison, min/max, floor/ceil and off-by-one in the three
correctness-critical modules, one at a time, and run the suite against each.

⚠️ **The most important survivor is a bug the ledger already records once.** `phase1` documents a
testbench that hardcoded `IDX_W=3`, checked a 10-class design on three of its four index bits,
and **passed**. The emitter derives `IDX_W` now -- but nothing ever checked the derivation, and
it cannot fail loudly:

```verilog
if (class_idx !== expected[j][`IDX_W-1:0])   // BOTH sides truncated to the same wrong width
```

A too-narrow `IDX_W` does not break the comparison, it **narrows** it. Mutating `ceil` to
`floor` (4 bits to 3 on a ten-class model) or `max` to `min` (down to 1 bit) left the entire
suite green. The fix is a test that pins `IDX_W == ceil(log2(K))` per shape, because the gate
structurally cannot notice.

**Three more survivors, all real:**

| mutant | why it survived | now |
|---|---|---|
| the word range narrowed by one | vectors simply never reached the word's extremes, so the saturation boundary went unexercised | a test asserts both extremes appear in `x_quant.hex` |
| `degenerate: classes < 2` -> `<= 2` | the positive case was tested and the negative one was not | a TWO-class model must not be flagged -- ten classes would not have killed it |
| an error message's wording | genuinely equivalent | left alone |

⚠️ **And the first re-check taught its own lesson.** Fifteen mutants "survived" the first pass
because the fast test subset did not cover `vectors.py` at all. Re-running the candidates against
the FULL suite separated four real holes from eleven artifacts of my own test selection. **A
mutation score is only as meaningful as the tests you run against it** -- which is the same trap
as a coverage number that counts a subprocess as uncovered.

**Result: 4 real holes closed, and every one of them was in code that 25 sections of auditing had
already walked past.**

## 27. Measured — the second sweep, and why a local mutation score under-reports

The first sweep covered three modules; this one took the other four (`emit_core`,
`emit_encoder`, `build`, `verify`) against the **full** suite. 19 mutants, 7 survivors -- and
triaging them was more instructive than the count.

| survivor | verdict |
|---|---|
| 3 in `emit_core` / `emit_encoder` | ⚠️ **not code** -- the sweep mutated prose inside docstrings and emitted comment strings |
| `len(text) > MAX_COMMENT` -> `>=` | equivalent: truncates at 119 instead of 120 |
| `width = max(len(left) ...)` -> `min` | report column alignment only |
| GroupSum slice `(c + 1) * group - 1` -> `(c + 2)` | ⚠️ **killed by Verilator, not by pytest** -- see below |
| `if n_random < 0` -> `<= 0` | ⚠️ **real**: `n_random=0` was never pinned as legitimate |

### ⚠️ The slice mutant is the finding, and it is about the harness rather than the code

`(c + 2) * group - 1` emits a popcount whose input slice is twice the port width. Verilog
truncates a wide expression to the port from the MSB side, keeping the LSBs -- which are exactly
the right bits -- so the design still **simulates correctly** and the gate passes. It is
behaviourally equivalent and structurally wrong.

Verilator says so immediately:

```
%Warning-WIDTHTRUNC: 'bits' expects 3 bits on the pin connection,
                     but pin connection's SEL generates 6 bits.        exit 141
```

**So the mutant dies in CI and survives here**, because the `verilator`-marked tests skip on a
machine without it. Two things follow:

1. **A local mutation score is a lower bound.** The strictest checker in this project is not
   installed on the primary development platform, so any sweep run here under-reports.
2. **The linter earned its place again.** It is the only thing in the suite that catches a
   design which is correct by simulation and wrong by construction -- the same category as the
   pragma bug it found on its first day.

## 28. Measured — the shipped Verilog, mutated

The emitters and the golden model had been swept; the four hand-written primitives had only ever
had four operators flipped by hand. Twelve mutants, each a real build + iverilog gate + a
Verilator lint:

| | |
|---|---|
| killed by the gate | 8 |
| killed by gate and lint together | 1 |
| killed by the pipeline-depth tests | 3 |
| survived, correctly | 1 -- summing bits in reverse gives the same count |

**11 of 11 non-equivalent mutants died**, including a wrong tree root, a carried-forward wrong
entry, an off-by-one LUT address, a dropped popcount bit and a swallowed pipeline input.

⚠️ **And the harness lied to me a third time.** Three `pipe_reg` mutants -- one stage short,
tapping the wrong stage, and `STAGES=0` no longer bypassing -- reported as surviving. All three
are invisible at `STAGES=1`, and the harness built with the DEFAULT pipeline only. Re-run against
the depth tests from §18, every one dies.

That is the same lesson as §26 and §27, and it is now the most repeated finding of the whole
audit: **a mutation score measures the tests you happened to run, not the tests that exist.**
Every sweep in this phase produced false survivors from its own narrow selection -- 11 the first
time, 3 the second, 3 here -- and each time the fix was to re-run against the full suite rather
than to change the code.

It also, incidentally, confirms the pipeline-depth tests are doing real work: they are the only
thing standing between a wrong `pipe_reg` and a green run.

## 29. What survived the audit

Worth recording, because it is the larger half:

- **`verify` never called anything wrong a pass** -- truncated hex, corrupt hex, a single flipped
  expected answer, an emptied `.v`, a deleted `tb/`: all FAIL, ERROR or MISSING with detail.
- **The emitter is robust across shapes** -- K=1, n=1, a single feature, z=1, five layers, 240
  nodes, every threshold identical, thresholds at 1e9: all bit-exact.
- **Interrupted builds recover.** A build killed after 0, 4 or 9 files, then re-run, produces a
  design that passes the gate every time -- the second build does not inherit the first's mess.
- **A file held open by a reader** does not stop a rebuild on Windows.
- **`pathlib.Path`** works for the checkpoint and the output directory.
- **Builds are deterministic** -- the same checkpoint and seed produce byte-identical output,
  and so do two runs with no seed at all.
- **Tiny and large models** -- a one-node one-class model, and 5,000 vectors, both pass.
- **Degenerate model shapes** -- a group of exactly one node per class, every node reading the
  same input bit, identical thresholds, descending thresholds, all-negative thresholds: all
  bit-exact. And a final layer that does not divide num_classes is refused by name.
- **The report objects** stay ASCII, refuse to divide by a zero-LUT core, and an empty estimate
  is not ok -- `all([])` is True, and that trap is guarded in both commands now.
- **Paths** -- spaces, unicode, relative, trailing separators all work; an over-long Windows path
  is refused rather than half-written.
- **The CLI** -- 17 hostile invocations (bad flags, missing arguments, wrong file types, an
  empty file, a directory where a checkpoint belongs): **zero tracebacks**.
- **The checkpoint reader** -- 23 hostile files: out-of-range and negative wiring, non-power-of-
  two tables, transposed or empty thresholds, non-contiguous layer indices, `layers` as a
  string, `n` as a bool. Every one a named error.
- **Odd dtypes are accepted AND correct** -- int8 tables, float64 tables, a float mapping all
  pass the gate. Duck-typing working as designed, not a silent hazard: checked rather than
  assumed, because "accepted" and "correct" are different claims.

## Phase 6 outcome

**Both halves landed, and the second one reordered the first.** The probe was meant to decide
whether to build a Verilator backend. It found the *linting* was the valuable half -- a real
defect the gate structurally cannot catch -- and that the cross-check gives the independent
verification the backend was wanted for. The backend was built anyway, second, and deliberately
not as the default.

**Then the audit found the phase's most important defect**, and it was not in any of that: the
core-level golden vectors were wrong for every model whose encoder width is not a multiple of 8.
The tests all passed the whole time, because nothing in the repository -- or in two completed
studies -- had such a width.

**Suite: 283 passing, 16 skipped. Two simulators, one linter, green on both platforms.**
