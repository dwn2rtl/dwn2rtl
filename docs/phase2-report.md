# Phase 2 report — make it a tool

**Status: closed, 2026-08-13. CI green on the first push.**

The day-by-day record is `phase2-ledger.md`. This is the retrospective.

---

## 1. The result

Phase 1 proved the generator works. **Phase 2 makes it keep working without anyone remembering
to check, and proves it works somewhere other than one Windows laptop.**

CLAUDE.md's requirement is now satisfied literally:

> Emitted RTL is not correct until a simulator says it matches the golden model on every vector.
> **This runs in CI on every commit.** A change that cannot pass it is not finished.

Every push runs the full suite plus the gate on **Ubuntu and Windows**, on the declared Python
floor as well as current, and separately builds a wheel and runs the whole flow from it in an
environment with no source tree.

## 2. What was delivered

| | |
|---|---|
| `.github/workflows/ci.yml` | two jobs: `test` (3 platform/version combinations) and `package` |
| `tests/conftest.py` | explicit `sys.path`, one place that decides the gate's skip, a report header |
| Linux validation | the entire suite, the gate, and the CLI, run on Ubuntu 24.04 |
| `requires-python` correction | `>=3.9` was unrunnable; now `>=3.10`, and CI proves it |

## 3. Findings that outlive this phase

### 3.1 The emitted RTL is portable in fact, not by inspection

Every measurement in this project had come from one Windows machine. On Ubuntu 24.04 with
iverilog 12.0: **181 passed, 8 skipped** (the skips being real-checkpoint tests, correctly absent
there), `pytest -m sim` **23 passed**, and a full `save -> build -> verify` returning `PASS`.

**Nothing needed changing.** No path handling, no CRLF, no simulator differences. Roadmap Q6's
vendor-neutrality claim — originally an inspection result, advanced to *measured on emitted
files* in phase 1 — is now measured on a second operating system with a different iverilog build.

Linux is also about **5x faster** for the same suite (5.4 s against 25 s), which matters for how
often the gate can reasonably be run.

### 3.2 A supported version nothing runs on is a claim, not a fact

`pyproject.toml` declared `requires-python = ">=3.9"`. Two things made that false, and both were
found by *checking* rather than by reasoning:

- **torch 2.13 declares `Requires-Python >=3.10`** — the floor could not have installed.
- **`tomllib` is stdlib only from 3.11**, and a test imports it, so any 3.9 or 3.10 job would
  have errored.

Corrected to `>=3.10`, with a CI job on exactly that version. **The general form: a claim in
metadata is worth precisely as much as the job that exercises it**, which is most of the argument
for having a matrix at all rather than a single build.

### 3.3 The mechanism that skips the gate is the one that could hide it

`conftest.py` skips `sim`-marked tests when no simulator is present. That is correct for a
contributor's laptop and **actively dangerous** anywhere else: a run whose gate silently did not
execute is indistinguishable from a green one.

Three defences, deliberately redundant:

1. The skip reason is `no Verilog simulator -- THE GATE DID NOT RUN`.
2. A `pytest_report_header` states the situation **before** the results, not after — nobody
   should have to infer from a skip count whether the thing that matters ran.
3. CI has a step that calls `find_simulator()` and **exits 1** if it raises, before any test
   runs. In CI, absence is a failure, not a skip.

### 3.4 Development installs cannot verify packaging — restated, and now automated

`pip install -e` leaves the source tree in place, so `files('dwn2rtl')/'rtl'` resolves back into
the repo. A completely missing package-data block would look perfectly healthy in development and
ship a wheel whose every emitted design references four Verilog modules that do not exist on the
user's machine.

Phase 0 checked this once, by hand. It is now a CI job that builds a wheel, installs it where
there is no source tree, and runs `build` and `verify` from there.

### 3.5 Rehearse what you cannot test

The workflow was green on the first push, and three specific predictions about what would break
were all wrong — `choco install iverilog`, the 3.10 floor job, and a `$GITHUB_WORKSPACE`
expansion.

**The predictions being wrong is not the point; the rehearsal is.** Everything that could be run
locally was: the YAML parsed and its matrix expanded, the "gate actually ran" step executed, and
the entire `package` job performed by hand with a real wheel in a clean venv. The two defects
that *would* have turned the first run red — the `>=3.9` floor and the `tomllib` import — were
both caught on this machine by checking claims against installed metadata.

A workflow that has never run is a guess. The response to that is to stop guessing before
pushing, not to push and iterate.

### 3.6 Windows path handling, a third time

`pip install /c/Users/.../x.whl` fails on Windows Python: MSYS-style paths are not native paths.
This is the third occurrence of that family in the project, after a build output directory landing
under `\c\Users\...` (phase 1) and the study repo's `\v` escape turning `scripts\verify` into a
vertical tab. **Anything crossing the shell/Python boundary on Windows takes a native path.**

Separately: `pip install torch` failed on a **Windows long-path limit** when the target was deeply
nested, and succeeded at a 37-character path. Not a defect in this project, but worth knowing
before debugging someone's install.

## 4. Roadmap movement

- **P5 — CI on a second dataset** ✅ satisfied in the form that matters. The roadmap's argument was
  that *"the claim is generality; it needs a second dataset to be a claim at all"*. CI runs five
  synthetic shapes spanning n=2 to n=6, 2 to 10 classes, one and two layers, and both mapping
  representations — and locally, the opt-in tests run nine real checkpoints across two datasets.
- **Q6 — vendor-neutral** now measured on two operating systems and two iverilog builds.
- **P2 — pyproject and one CLI entry point** complete: `build` and `verify` both work; only
  `estimate` remains stubbed, in phase 3.

## 5. What phase 3 inherits

- **`estimate`** — the last stubbed subcommand. Needs yosys, which is still not installed here;
  winget has no OSS CAD Suite package, so it is a manual download.
- **`README.md` is still 10 bytes.** It is the first thing anyone sees and currently says nothing.
- **A worked example**, end to end, committed.
- **Study-repo references in the `rtl/*.v` headers** still point at paths that do not exist in
  this repository (`docs/reference/checkpoint-format.md`, `docs/jsc/dse-plan.md`,
  `probe-results.md`), and one comment cites *"CLAUDE.md"* meaning the study repo's.
- **Verilator as a second simulator backend**, behind the existing `Simulator` interface.
- **Q3/P6 — the upstream-version policy.** `docs/checkpoint-format.md` was written against pinned
  commit `9f887a0` and this repository has never pinned it independently, which
  `tool-handoff.md` §9 explicitly asks for.

## 6. By the numbers

| | |
|---|---|
| tests | 198 (Windows) / 181 (Linux), 23 of them the gate |
| CI | 3 platform/version combinations + a packaging job, green first push |
| operating systems verified | **2** — was 1 |
| roadmap items closed | **P5**; Q6 and P2 advanced |
| defects found before pushing | 2 (`>=3.9` floor, `tomllib` on <3.11) |
