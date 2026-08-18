# Phase 3 ledger — make it usable by someone else

**Goal.** The tool works (phase 1) and stays working (phase 2). Nobody else can use it: the
README is 10 bytes, there is no example, and the emitted RTL still cites documentation that does
not exist in this repository.

**Status: CLOSED.** Retrospective in `phase3-report.md`.

> Conventions in `overview.md` §6. Entries oldest first, recording *built* / *hit* / *decided*.
> Reversed decisions stay, struck through, with the reason.

## Plan

| # | unit | why |
|---|---|---|
| 1 | **a worked example, end to end** | the README should quote real output, so this comes first |
| 2 | **`README.md`** | the first thing anyone sees, and it currently says nothing. Also where the study repo is cited as evidence |
| ~~3~~ | ~~**`estimate` via yosys**~~ | ⬅ **deferred to phase 4** — see §5. No light yosys exists on Windows, and writing it untested would put the first unverified claim in the repo |
| 4 | **clean the study-repo references out of `rtl/*.v`** | the shipped Verilog cites `docs/reference/checkpoint-format.md`, `docs/jsc/dse-plan.md` and `probe-results.md`, none of which exist here, and one comment means the *study repo's* CLAUDE.md |
| 5 | **pin the upstream DWN commit** | `tool-handoff.md` §9 asks for this explicitly: do not inherit the study repo's pin, re-read `checkpoint-format.md` against whatever this project pins |

---

## 1. Built — `examples/quickstart.py`

One file, no dataset, no training, a few seconds. It builds a DWN, saves it the way a user
would, emits Verilog, and runs that Verilog through a simulator:

```
1. build a DWN and save it
   model.dwn  (19 KB)
2. emit Verilog
   features 16, classes 5, layers [60], n=6, z=8   from checkpoint
   core      60 nodes, 3 cycles
   encoder   122 comparators of 128 thermometer bits
3. prove the Verilog matches the model
     dwn_core  504 vectors  PASS
     dwn_top   535 vectors  PASS
   RESULT   PASS
```

**Decided: the example does not train, and says why.** dwn2rtl translates a model's *structure*;
whether the model is any good is a training question the tool has no opinion about, and the thing
being demonstrated is that the Verilog matches the model, not that the model matches a dataset.
An untrained model exercises the emitter identically. Stating that outright is better than an
example that quietly implies training is part of the flow.

**Decided: the example does not import `torch_dwn`.** Upstream builds a CUDA/C++ extension, and
requiring a user to compile one before they can see the tool work would be backwards — especially
since dwn2rtl genuinely does not need it, because `checkpoint.py` duck-types. The example defines
three small stand-in classes and `for_a_real_model()` shows the actual upstream recipe in a
docstring. A test asserts the import stays absent.

**Decided: no simulator is exit code 2, not a cheerful finish.** The design would be emitted and
*nothing* would have checked it — a green-looking example that verified nothing is exactly the
failure this project is organised against.

**Built `tests/test_examples.py`**, because an example is documentation that claims to be
executable and therefore rots in a way prose does not: stale prose is merely wrong, a stale
example is wrong *and* was promised to run. It is the first thing a new user tries.

## 2. Built — `README.md`

Was **10 bytes**. Now opens with a real MNIST build and gate result — real output from a real
checkpoint, quoted verbatim, re-run and diffed against the file before committing.

Covers: what the tool is and is not, the quickstart, installing a simulator per platform, the
one-line save and the `state_dict` trap, `--input-bits` and the losslessness proof, what `build`
emits, the port list, why there are two testbenches, and the study repo as evidence.

### ⚠️ Hit: the README's first code block contained two false claims

Both found by checking rather than reading, and both in the part every visitor sees first.

1. **`pip install dwn2rtl`** — `https://pypi.org/pypi/dwn2rtl/json` returns **404**. The package
   is not published. Replaced with the git install, plus a line saying so.
2. **A link to `CLAUDE.md`** — which is in `.gitignore`, so it is not in the repository and the
   link 404s on GitHub.

**Also: the quoted build output was missing a line.** The real run of that MNIST checkpoint also
prints the scaled-features warning, and leaving it out made the block not-quite-real. Added — it
is honest, and it demonstrates the tool catching something worth catching.

**`tests/test_examples.py` now checks the README**: every repo-relative link must point at a
*git-tracked* file, the PyPI instruction must not reappear before it is true, and the example it
points at must exist. All three defects above were of exactly this shape — mechanically
checkable, and checked by nothing. A README is the one document every user reads and the one
nothing else verifies.

**Suite: 206 passed, 1 skipped.**

## 3. Built — the shipped Verilog no longer cites documents that do not exist

⚠️ **These comments are read by users, not by us.** The four primitives are *copied into every
emitted directory*, so their headers are documentation shipped with the product — and they came
across from the study repo verbatim.

They cited `docs/reference/checkpoint-format.md`, `docs/jsc/dse-plan.md` and `probe-results.md`,
none of which exist here, and referred to *"Phase 1"*, *"Phase 2"*, *"brief §9/§10"* and
*"CLAUDE.md"* meaning the **study repo's** phases and rules. **A reference that resolves nowhere
is worse than none**: it tells the reader there is an authority to consult, then wastes their
time looking for it.

Fixed by making each comment self-contained or citing something real:

| was | now |
|---|---|
| `n=6 is fixed for Phase 1 bring-up (CLAUDE.md)` | the actual constraint: 2\*\*n entries must fit a 64-bit parameter, above which Verilog **truncates silently**, and one node stops being one LUT6 |
| `confirmed by the Phase 1a probe, docs/reference/probe-results.md` | "measured on Vivado in the research repository" (named, at the time) — a citation a reader could follow |
| `a Phase 2 sweep axis (brief §10)` | "a build parameter, not a property of the model" |
| `docs/jsc/dse-plan.md, Group A` | the actual finding: 35 trained configurations showed a learned taper is dominated by a plain layer |
| `docs/reference/checkpoint-format.md §2/§4` | `dwn2rtl docs/checkpoint-format.md §2/§4` — which exists |

> ⚠️ **Row 2 was superseded on 2026-08-18.** Naming the study repository was the right call *here*
> — it replaced a dead path with a citation a reader could follow. It was later withdrawn on a
> different ground: the shipped `rtl/*.v` are copied into every user's output directory, so the
> citation travelled into projects that have no connection to it. It now reads "confirmed on
> Vivado". The Phase 3 reasoning was not wrong; a later requirement outranked it.

The technical content was left alone; only provenance changed. **Gate re-run: 30 passed.**

`test_shipped_verilog_cites_nothing_a_user_cannot_open` now parametrizes over all six shipped
files, rejects the known-dangling prefixes, and — the part that keeps working — resolves every
`docs/*.md` a comment cites against the actual filesystem.

## 4. Built — the upstream pin, verified rather than inherited

```
upstream   https://github.com/alanbacellar/DWN
commit     9f887a0b4bd84dabf6d8c9ae35368ab2a7e0e3c0
verified   2026-08-13, independently
```

`tool-handoff.md` §9 asked for exactly this: *"Pin it here independently — do not assume the
study repo's pin, and re-read `docs/checkpoint-format.md` against whatever commit this project
pins."* So each load-bearing claim was re-checked against the source rather than carried forward:

| § | claim | confirmed at |
|---|---|---|
| 1 | table bit is `luts[j][addr] > 0`, **strictly** greater | `STEFunction.forward` is `(x > 0).float()` |
| 2 | slot `l` is address bit `l`, **LSB first** | `addr \|= (input[mapping[j][l]] > 0) << l` |
| 3a | learnable wiring is `weights.argmax(dim=0)` | `mapping.py:17` |
| 3c | `__dummy_mapping` is only `arange` reshaped | `lut_layer.py:60` |
| 4 | `GroupSum` has **no parameters** | `utils.py:11-16` |

That last one is the reason `num_classes` cannot come from a `state_dict` — the design of
`checkpoint.py` rests on it, and it is now confirmed rather than assumed.

**Decided: the pin is provenance, not a version gate.** `UPSTREAM_COMMIT` lives in
`checkpoint.py` and a test keeps it identical to the document's. But **loading never checks it**,
and a test asserts that too: duck-typing exists precisely so a user whose model was trained
against a different commit is not refused. A rename upstream should surface as a clear failure
with a real message, never as *"your checkpoint is the wrong version"*.

**Found while verifying: upstream's `LUTLayer` forward requires CUDA** — the CPU path raises.
Not a constraint on dwn2rtl, which never runs a model, but it independently confirms the
example's decision not to import `torch_dwn`.

## 5. Decided — `estimate` defers to phase 4

**yosys is not a light dependency on Windows**, and this is measured rather than assumed:

| | |
|---|---|
| winget | no package |
| chocolatey | not installed, and no known package |
| YosysHQ `yosys` releases | **source tarballs only** — no prebuilt Windows binary |
| OSS CAD Suite | **703 MB** |
| `apt install yosys` on Linux | ~9 MB of packages |

Windows is this project's primary target, so the ~9 MB figure is not available to us.

**Deferred rather than written blind.** `estimate` is the only feature the roadmap itself calls
*optional*, and this project's rule is that nothing counts until it runs. Implementing it now
would put the first unverified claim in the repository, to save a download that can happen later
just as easily. It stays a stub exiting 2 and naming its phase — which is the policy `cli.py`
was built around from phase 0.

**Suite: 214 passed, 1 skipped.**

## Phase 3 — closed

Units 1, 2, 4 and 5 done. Unit 3 (`estimate`) deferred to phase 4, with a reason.
