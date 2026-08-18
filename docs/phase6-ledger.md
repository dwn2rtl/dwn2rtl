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

## 8. ⚠️ Decided — the `--simulator` backend is NOT built

Units 2-5 of the original plan would let a user run `dwn2rtl verify --simulator verilator`. They
are not built, and the reason is that the probe moved the value somewhere else.

**What the backend would add is only user-facing choice.** The verification value -- two
independent simulators agreeing -- is already had, in CI, on every commit, without touching
`verify.py`.

**The honest argument for it**, recorded so it is not rediscovered: a user with Verilator and no
iverilog cannot run `verify` at all today, since `find_simulator` looks only for iverilog and
`_run_level` hardcodes its two-command shape. That is a real gap, and Verilator is the common
simulator in RISC-V and academic toolchains.

**Why it still loses:** the cost to that user is `apt install iverilog`, about 9 MB and one
command, against a permanent second discovery path and second command shape in the one module
where correctness matters most.

**Build it when any of these happens** -- the probe has de-risked every unknown, so it is roughly
half a day:

1. a real user reports being blocked with only Verilator installed
2. designs get big enough for simulation time to matter (Verilator's actual advantage, which
   current sizes do not expose -- 0.04 s against 0.04 s)
3. the CI cross-check goes red, i.e. the two simulators genuinely disagree, which would make
   multi-simulator support a correctness feature rather than a convenience

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

## Phase 6 outcome

**Both halves landed, and the second one changed what the first was for.** The probe was meant
to decide whether to build a Verilator backend. It found the backend was the least valuable part
of the idea: linting found a real defect the gate structurally cannot catch, the cross-check
gives the independent verification a backend was wanted for, and neither required touching the
product.

**Suite: 246 passing, 15 skipped. Two simulators, one linter, green on both platforms.**
