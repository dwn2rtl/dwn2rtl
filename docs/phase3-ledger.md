# Phase 3 ledger — make it usable by someone else

**Goal.** The tool works (phase 1) and stays working (phase 2). Nobody else can use it: the
README is 10 bytes, there is no example, and the emitted RTL still cites documentation that does
not exist in this repository.

**Status: OPEN.**

> Conventions in `overview.md` §6. Entries oldest first, recording *built* / *hit* / *decided*.
> Reversed decisions stay, struck through, with the reason.

## Plan

| # | unit | why |
|---|---|---|
| 1 | **a worked example, end to end** | the README should quote real output, so this comes first |
| 2 | **`README.md`** | the first thing anyone sees, and it currently says nothing. Also where the study repo is cited as evidence |
| 3 | **`estimate` via yosys** | the last stubbed subcommand |
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
