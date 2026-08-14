# Phase 2 ledger — make it a tool

**Goal.** The generator works and is verified (phase 1). This phase makes it *stay* working
without anyone remembering to check, and proves it works somewhere other than one Windows laptop.

**Status: OPEN.**

> Conventions in `overview.md` §6. Entries oldest first, recording *built* / *hit* / *decided*.
> Reversed decisions stay, struck through, with the reason.

**What CLAUDE.md requires of this phase, quoted, because it is the whole brief:**

> ```
> iverilog -o sim <emitted>/*.v src/dwn2rtl/rtl/*.v && vvp sim    # must print PASS
> ```
> This runs in CI on every commit. A change that cannot pass it is not finished.

## Plan

| # | unit | why |
|---|---|---|
| 1 | **`conftest.py`** | `import fixtures` currently works by accident of pytest inserting the test directory on `sys.path`. Implicit, and it will surprise someone |
| 2 | **CI on Linux and Windows** | the gate on every commit. Windows is the primary target *and* the one with the odd failures; Linux is where everyone else will run it |
| 3 | **Linux, at all** | every measurement in this project so far is from one Windows machine. iverilog's behaviour, path handling and CRLF are all untested elsewhere |

**Explicitly not in this phase:** `estimate`/yosys, the README, and the worked example. Those are
phase 3, and they should follow a generator that has stopped moving.

---

## 1. Built — `tests/conftest.py`

Two things that were happening by accident or by repetition.

**`import fixtures` is now deliberate.** It worked because pytest inserts a test file's directory
into `sys.path` when that directory has no `__init__.py` — a real rule, but an implicit one
depending on a layout nobody wrote down. Adding an `__init__.py`, or running the suite through a
different runner, would have produced `ImportError` with no obvious cause.

**The gate's skip is decided in one place.** Three test files had each grown their own copy of
*"is there a simulator?"*, all delegating to `verify.py` but each with its own `skipif`. A
`pytest_collection_modifyitems` hook now skips anything marked `sim`, so a test declares that it
needs a simulator without also saying how to look for one.

⚠️ **That hook is the one piece of machinery that could hide the project's only correctness
signal**, so it is loud about it. The skip reason is `no Verilog simulator -- THE GATE DID NOT
RUN`, and a `pytest_report_header` states the situation *before* the results rather than after:

```
dwn2rtl: gate simulator iverilog 12.0 (C:\iverilog\bin\iverilog.exe)
```
```
dwn2rtl: NO SIMULATOR FOUND -- gate tests will be SKIPPED, not passed
```

A header, not a footer: someone scrolling past a green summary should not have to infer from a
skip count whether the thing that matters executed.

**Hit: stripping PATH does not simulate a machine without a simulator.** `verify.py` falls back
to `C:\iverilog\bin` and finds it anyway — which is the phase 0 finding working exactly as
designed. The guard is therefore tested by patching `conftest.SIMULATOR` and calling the hook
directly, asserting both that gate tests get skipped and that non-gate tests do not.

**Hit: removing the old `skipif` decorators left two tests that would FAIL rather than skip.**
`test_find_simulator_returns_something_runnable` called `find_simulator()` directly; without one
it would have raised `SimulatorNotFound` and read as a real failure. Both now take the
`simulator` fixture.

## 2. Built — Linux, and it works

**Every measurement in this project had come from one Windows laptop.** WSL Ubuntu 24.04 was
available with `iverilog` already installed, so this is measured rather than assumed.

```
platform linux -- Python 3.12.3, pytest-9.1.1
dwn2rtl: gate simulator iverilog 12.0 (stable) (/usr/bin/iverilog)

181 passed, 8 skipped in 5.36s
```

The 8 skips are the real-checkpoint tests, correctly absent — the study repo is not in WSL. The
gate ran: `pytest -m sim` gave **23 passed**. And the full user journey works end to end:

```
$ dwn2rtl build /tmp/model.dwn --out /tmp/rtl --input-bits 8
wrote     /tmp/rtl (17 files)
$ dwn2rtl verify /tmp/rtl
  dwn_core  504 vectors  PASS
  dwn_top   519 vectors  PASS
RESULT   PASS
```

**Linux is ~5x faster** — 5.4 s against 25 s for the same suite. Nothing needed changing: no path
handling, no CRLF, no simulator differences. The emitted Verilog and both testbenches are
portable in practice, not only by inspection.

**Fixed a cosmetic difference it exposed.** Linux packages report `iverilog -V` as
`12.0 (stable) ()` — an empty build-id parenthesis that rendered as a stray `()` in every report
line. Stripped.

The WSL copy was deleted afterwards (4.7 GB, nearly all of it torch). It is not needed again:
CI runs Linux on GitHub's machines, which is the point of the next unit.

## 3. Built — `.github/workflows/ci.yml`

Two jobs.

**`test`** — three combinations: `ubuntu-latest` and `windows-latest` on 3.12, plus
`ubuntu-latest` on the declared floor. `fail-fast: false`, so a Windows-only failure cannot hide
a Linux one. It installs iverilog (apt on Linux, choco on Windows — winget is not on the
runners), runs the suite, runs `pytest -m sim` on its own, and finally does a full
save -> build -> verify exactly as a user would.

⚠️ **The step that matters most is "The gate actually ran".** `conftest.py` skips gate tests when
no simulator is found, which is right on a contributor's laptop and *wrong* in CI: a run whose
gate silently did not execute is indistinguishable from a green one. That step calls
`find_simulator()` and **exits 1** if it raises, before any test runs.

**`package`** — builds a wheel, installs it into an environment with no source tree, and runs the
whole flow from there. This exists because a development install *cannot* catch a packaging
failure: `pip install -e` leaves the source in place, so `files('dwn2rtl')/'rtl'` resolves back
into the repo and a missing package-data block looks perfectly healthy right up until a user's
wheel references four Verilog modules that do not exist.

### Rehearsed locally, because a workflow that has never run is a guess

The YAML was parsed and its matrix expanded, the "gate actually ran" step was executed, and the
whole `package` job was performed by hand: build a wheel, create a clean venv, install only that
wheel, then `dwn2rtl build` and `dwn2rtl verify` from a directory that is not the repo.

```
ok - 6 Verilog files; installed at ...\d2r\env\Lib\site-packages\dwn2rtl\__init__.py
wrote     rtl (17 files)
  dwn_core  504 vectors  PASS
  dwn_top   519 vectors  PASS
RESULT   PASS
```

**Hit: the first attempt failed on a Windows long-path limit**, unpacking a torch header into a
deeply nested scratchpad. Not a wheel defect — the same install at a 37-character path
succeeded — but a reminder that `pip install torch` on Windows is sensitive to how deep the
target is.

**Hit, again: MSYS paths.** `pip install /c/Users/.../x.whl` fails silently-ish on Windows
Python. Third occurrence of this family in the project (after the build output directory and the
`\v` escape from the study repo).

### ⚠️ Hit: `requires-python` was a claim, not a fact

`pyproject.toml` declared `>=3.9`. Nothing had ever run there, and two things say it was wrong:

- **torch 2.13 itself declares `Requires-Python >=3.10`**, so the floor could not install.
- **`tomllib` is stdlib only from 3.11**, and `test_version_matches_pyproject` imports it — that
  test would have errored on any 3.9 or 3.10 job.

Corrected to **`>=3.10`**, the CI floor job set to 3.10, and the `tomllib` import turned into a
`pytest.importorskip` — right here, because that test guards two files agreeing with each other
and the answer does not vary by interpreter.

**A supported version that nothing ever runs on is a claim.** The CI matrix is what converts it
into a fact, which is the general reason this job exists.

**Suite: 198 passed, 1 skipped (Windows). 181 passed, 8 skipped (Linux).**
