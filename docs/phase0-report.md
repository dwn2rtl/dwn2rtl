# Phase 0 report — make it runnable

**Status: closed, 2026-08-13.** Four commits, `809b456`..`9aa36ab`.

The day-by-day record is `phase0-ledger.md`. This is the retrospective: what phase 0 delivered,
what it found, and what the next phase inherits.

---

## 1. What phase 0 was for

The repository arrived as ~2,300 lines of good code that **could not be run at all**. There was no
`pyproject.toml`, so it was not installable; no `__init__.py`, so `from .precision import
Precision` could not resolve and not one module could be imported; no CLI; no virtual environment;
and no simulator on the machine, so the project's own non-negotiable gate could not have been run
even if the Python had executed.

Phase 0 is the scaffolding that fixes that. **It emits no Verilog.** Its whole job is to make
everything after it runnable and checkable.

## 2. What was delivered

| | |
|---|---|
| `pyproject.toml` | deps, `rtl/**.v` as package data, the `[project.scripts]` entry point |
| `src/dwn2rtl/__init__.py` | `__version__` plus the two subsystems that work without a checkpoint |
| `src/dwn2rtl/cli.py` | argparse skeleton — `build`, `verify`, `estimate` |
| `tests/test_packaging.py` | packaging and CLI invariants |
| `tests/test_precision.py` | the `--input-bits` precision policy |
| `LICENSE` | MIT — roadmap **P3 closed** |
| `.gitignore` | Python and emitted-design entries |
| `docs/phase0-ledger.md` | the running log |

Plus a project venv and **Icarus Verilog 12.0** installed and on PATH.

## 3. Evidence it works

```
$ dwn2rtl --version                        dwn2rtl 0.1.0.dev0
$ dwn2rtl build foo.pt --out rtl/          "not implemented yet (phase 1)", exit 2
$ pytest                                   39 passed, 1 xfailed
$ python -m build --wheel                  4 primitives + 2 testbenches inside the wheel

>>> dwn2rtl.precision_for(thr_mnist, input_bits=8)    Q0.8  signed (9-bit)
>>> dwn2rtl.precision_for(thr_jsc)                    Q3.12 signed (16-bit)
```

Those last two are the strongest signal in the phase and were free. `precision_for` derives a
fixed-point format from thresholds alone, and on MNIST-shaped and JSC-shaped inputs it returns
**exactly the formats those two completed studies actually used** — numbers nobody typed in. A
policy that independently reproduces two known-correct answers is evidence; one that reproduces
none is a guess.

## 4. Findings that outlive this phase

Four results here change what later phases have to do. They are the reason this report exists
rather than just a commit log.

### 4.1 Vendor-neutrality is now measured, for the primitives

Roadmap Q6 declared the emitted RTL vendor-neutral **by inspection** — no vendor primitives,
Verilog-2001 only — and explicitly flagged that it had never been tested under a non-Xilinx
simulator. `argmax.v` was the named worry: 2-D wire arrays declared inside a `generate`, the kind
of construct tools disagree about.

All four primitives now compile and run clean under iverilog 12.0, no warnings, no workarounds:

```
lut_node  addr=3, TABLE=DEADBEEFCAFEF00D  -> 1     correct
popcount  1011001101                       -> 6     correct
argmax    K=10, tie at score 9             -> 2     correct, lowest index wins
argmax    K=5,  tie at score 7             -> 2     correct, and exercises the odd-K carry
pipe_reg  ENABLE=1 -> a5, ENABLE=0 -> 5a           correct, both paths
```

Both argmax cases were built to contain a **tie**, because the tie-break — strict `>` at every
merge, so equal scores keep the lower index — is the rule that must match numpy's `argmax`, and a
design that gets it backwards is right on ~97% of vectors and wrong on the rest. That is precisely
the failure Gate 1 exists to catch and spot-checking never does.

⚠️ **This is measured for the four hand-written primitives, not for emitted files.** Nothing has
been emitted yet. Gate 1 is phase 1.

### 4.2 A simulator installed but invisible is the *default* on Windows

`winget install --id Icarus.Verilog` succeeded, put iverilog in `C:\iverilog\bin`, and **added it
to neither the machine nor the user PATH.** It was invisible to every shell until PATH was edited
by hand.

This is a requirement on `verify.py`, not an anecdote: searching PATH and giving up is wrong,
because the common case is a user who *has* a simulator and would be told they do not.
`verify.py` must fall back to well-known install locations and report what it found and where.

### 4.3 Development installs cannot verify packaging

`pip install -e` leaves the source tree in place, so `files('dwn2rtl')/'rtl'` resolves straight
back into the repo. A **completely missing** `[tool.setuptools.package-data]` block would
therefore look perfectly healthy in development, and ship a wheel whose every emitted design
references four Verilog modules that do not exist on the user's machine.

The block is correct — verified by building an actual wheel and listing its contents — but the
general point stands for every phase: packaging claims have to be checked against a built
artifact, never against the working tree.

### 4.4 "Git is tracking it" and "my editor is drowning in it" look identical

Source control appeared to be tracking the 23,002-file venv. It was not:
`git ls-files` returned zero matches and `git check-ignore -v` confirmed the rule matched. The
real causes were an editor watching and indexing those files, and a `.gitignore` fix that was
still uncommitted while the *committed* version predated it.

Diagnose with `git ls-files` and `git check-ignore -v` before editing `.gitignore`.

## 5. Decisions, and what they cost

| decision | reasoning | cost if reversed |
|---|---|---|
| **Input format is sniffed, not fixed** | Upstream DWN saves nothing, so any format is our invention and no user arrives holding it. Sniffing keeps the study repo's checkpoints readable — and they are the evidence the generator works | a few lines in `checkpoint.py` |
| **The CLI and the library are one package** | `[project.scripts]` over a thin argparse layer. One implementation of everything, so the two front doors cannot drift | low |
| **`import dwn2rtl` must not import torch** | Every emitter and the golden model are pure numpy; torch is only needed to *read* a checkpoint, and `verify` has none to read. Asserted in a subprocess test | rises as more is added to `__init__` |
| **Subcommands exist before they work** | Omitting them makes `--help` misrepresent the tool; stubbing them to exit 0 is indistinguishable from success. They exit 2 naming their phase | none |
| **Known gaps are `strict=True` xfails** | The empty `dwn_top_tb.v` is asserted as an expected failure. When phase 1 writes it, the XPASS becomes an error and forces the marker's removal | none |

## 6. Roadmap movement

- **P3 — a licence** ✅ closed. MIT. It was blocking both use and citation.
- **Q6 — vendor-neutral** advanced from *inspection* to *measured*, for the primitives only.
- **P2 — pyproject and one CLI entry point** substantially done; the command surface exists, the
  implementations are phases 1–3.

## 7. What phase 1 inherits

Two gaps found during the phase-0 audit, both blocking the gate:

- ⚠️ **`src/dwn2rtl/rtl/tb/dwn_top_tb.v` is a 0-byte file.** Commit `646aebe` "add testbenches"
  landed one empty, so half the gate — encoder plus core — does not exist. Currently held by a
  strict xfail that will force its own removal.
- ⚠️ **There is no checkpoint in the repo**, real or synthetic, so nothing can be built end to
  end. Roadmap P8 makes a synthetic fixture mandatory rather than convenient: real checkpoints run
  17–471 MB and cannot go in git.

Carried forward deliberately, not oversights:

- **`yosys` not installed** — only `dwn2rtl estimate` needs it (phase 3), and winget has no OSS
  CAD Suite package, so it will be a manual download.
- **`README.md` is 10 bytes** — phase 3; nothing is blocked.
- **No CI** — phase 2, and it needed tests to exist first.
- **Editor configuration is not version-controlled** — a `.vscode/settings.json` excluding the
  venv from watching and indexing was written and not kept. Each contributor handles it locally.
- **The `LICENSE` copyright line reads "The dwn2rtl authors"** — the project has more than one and
  no file should guess legal names. Replace if a named holder is wanted.

## 8. By the numbers

| | |
|---|---|
| starting code that could be imported | **0 modules** |
| lines added | ~800, none of them a generator |
| tests | 40 collected — **39 pass, 1 xfail** (a tracked gap) |
| commits | 4 |
| roadmap items closed | 1 (P3), 1 advanced (Q6) |
| Verilog emitted | **none** — that is phase 1 |
