# Phase 0 ledger — make it runnable

**Goal.** Turn a folder of loose files into something pip can install and Python can import.
Nothing in this phase emits Verilog; it is the scaffolding everything after it needs.

**Status: OPEN.** See §6 for what remains.

> **One ledger per phase**, named `phaseN-ledger.md`, matching the study repo's convention.
> Within a phase, entries run oldest first and record *what was built*, *what was hit* (a problem
> encountered, with its resolution), and *what was decided*. A wrong turn that got reversed stays
> in the file, struck through, with the reason — per CLAUDE.md, correct rather than append. A tidy
> ledger is a less useful ledger. Phases are defined in `overview.md` §6.

---

## 1. Starting state — 2026-08-13

Audited the whole tree before touching it. What existed: six Python modules (`extract`,
`precision`, `emit_core`, `emit_encoder`, `vectors`, `config`), four hand-written Verilog
primitives, one testbench, four docs. ~2,300 lines.

What did **not** exist, and therefore what phase 0 is:

- no `pyproject.toml` — not an installable package
- no `__init__.py` — `from .precision import Precision` cannot resolve, so **not a single module
  could be imported**
- no `cli.py` — no terminal command
- no venv
- no `iverilog` or `yosys` on PATH — the gate could not have been run even if the code ran

Two further gaps recorded now and fixed later, so they are not rediscovered:

- ⚠️ **`src/dwn2rtl/rtl/tb/dwn_top_tb.v` is a 0-byte file.** Commit `646aebe` "add testbenches"
  landed one empty. Half the gate — encoder plus core — does not exist. **Phase 1.**
- ⚠️ **There is no checkpoint anywhere in the repo**, real or synthetic, so nothing can be built
  or tested end to end. **Phase 1.**

## 2. Decided — the input format is sniffed, not fixed

`dwn2rtl build` accepts any of three shapes and normalizes them internally:

| shape | where it comes from |
|---|---|
| `{'model': ..., 'thermometer': ...}` | plain `torch.save` — the documented primary path |
| `{config, state_dict, thermometer, results}` | the study repo's existing checkpoints |
| a live model + thermometer | `dwn2rtl.save()` / `from_model()`, optional sugar |

**Why not one strict format.** Upstream DWN saves nothing at all (roadmap Q8), so any format we
pick is our invention and no user arrives already holding it. Forcing `dwn2rtl.save()` would put
an import into everyone's training script to solve a problem one line of plain `torch.save`
already solves. Sniffing costs a few lines in `checkpoint.py` and keeps the study repo's existing
checkpoints readable — which matters, because they are the evidence the generator works.

**What does not soften:** a bare `state_dict` is refused by name. It has no thermometer, and
emitting without one produces a design that synthesizes cleanly and runs at chance.

## 3. Decided — the CLI and the library are one package

Not two products. `[project.scripts]` makes pip write a `dwn2rtl` launcher into the same directory
as `pip` itself; the launcher imports `dwn2rtl.cli` and calls `main()`. The CLI is a thin argument
parser over the library, so there is exactly one implementation of everything. User-facing version
in `overview.md` §2.

## 4. Built — the package

| file | what it does |
|---|---|
| `pyproject.toml` | metadata, deps, `rtl/**.v` as package data, the `[project.scripts]` entry point |
| `src/dwn2rtl/__init__.py` | `__version__` and the two things that work standalone |
| `src/dwn2rtl/cli.py` | argparse skeleton: `build`, `verify`, `estimate` |
| `.gitignore` | Python and emitted-design entries |

```
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
```

torch 2.13.0, numpy 2.5.2, pytest 9.1.1. Both front doors verified:

```
$ dwn2rtl --version                  ->  dwn2rtl 0.1.0.dev0
$ dwn2rtl build foo.pt --out rtl/    ->  "not implemented yet (phase 1)", exit 2
>>> dwn2rtl.precision_for(thr, input_bits=8)  ->  Q3.8 signed (12-bit)
>>> files('dwn2rtl')/'rtl'                    ->  the 4 primitives + 2 testbenches resolve
```

**The continuous-input default reproduces JSC's known format.** `precision_for(thr)` with no
`--input-bits` returned **Q3.12 signed, 16-bit** — exactly what the study repo's JSC models used.
Not a test, but a free consistency check on the extraction, and it agrees.

**Decided: subcommands exist before they work.** `build`/`verify`/`estimate` all parse and all
exit 2 naming the phase they land in. Omitting them until implemented would make `--help`
misrepresent the tool's shape; stubbing them to print nothing and exit 0 would be
indistinguishable from success. `_not_yet()` in `cli.py` is that policy.

**Decided: `import dwn2rtl` must not import torch.** Confirmed by assertion —
`'torch' in sys.modules` is `False` after importing the package. Every emitter and the golden
model are pure numpy; torch is only ever needed to *read* a checkpoint, and `dwn2rtl verify` on an
already-emitted directory has no checkpoint at all. Keeping the boundary sharp is also what would
make a torch-free `.npz` path possible later without touching anything downstream. `__init__.py`
therefore exports only `config` and `precision`.

**Hit: `[project.scripts]` names `dwn2rtl.cli:main`, so `cli.py` had to exist in phase 0.** A
missing module there is not a soft failure — pip installs the launcher regardless and the command
dies with `ModuleNotFoundError` on first use. Hence the skeleton now rather than in phase 2.

**Verified: the wheel actually contains the RTL.** `python -m build --wheel`, then listed the
archive. All four primitives and both testbenches are inside under `dwn2rtl/rtl/`.

⚠️ **This check was worth running specifically because the editable install cannot fail it.**
`pip install -e` leaves the source tree in place and resolves `files('dwn2rtl')/'rtl'` straight
back to the repo, so a completely missing `[tool.setuptools.package-data]` block would look
perfectly healthy in development and produce a wheel whose every emitted design references four
modules that do not exist on the user's machine. Development installs cannot verify packaging.

## 5. Built — a simulator, and the first non-Xilinx measurement

`iverilog` and `yosys` were both absent. Installed Icarus Verilog **12.0** via
`winget install --id Icarus.Verilog`.

**Hit: the silent install does not add itself to PATH.** It lands in `C:\iverilog\bin`, and
neither the machine nor the user PATH mentions it, so `iverilog` is invisible to every shell.
Added `C:\iverilog\bin` to the **user** PATH.

⚠️ **This is a requirement on `verify.py`, not a one-off.** A simulator installed but not on PATH
is evidently the *default* outcome of a Windows install, not an unusual one. `verify.py` must
search well-known install locations after PATH fails, and report what it found and where, rather
than telling a user who plainly has a simulator that they have none.

**Then: the RTL ran under a non-Xilinx simulator for the first time.** Roadmap Q6 resolved
vendor-neutrality by *inspection* — no vendor primitives, Verilog-2001 only — and flagged that it
had never been *tested*. `argmax.v` was the specific worry: 2-D wire arrays declared inside a
`generate`, the kind of construct tools disagree about.

A scratch elaboration harness compiled all four primitives and checked five behaviours:

```
lut_node  addr=3, TABLE=DEADBEEFCAFEF00D  -> 1     correct
popcount  1011001101                       -> 6     correct
argmax    K=10, tie at score 9             -> 2     correct, lowest index wins
argmax    K=5,  tie at score 7             -> 2     correct, and exercises the odd-K carry
pipe_reg  ENABLE=1 -> a5, ENABLE=0 -> 5a           correct, both paths
```

**Clean compile, no warnings, no workarounds.** Both argmax shapes were chosen to contain a tie,
because the tie-break — strict `>` at every merge, so equal scores keep the lower index — is the
rule that must match numpy's argmax, and a design that gets it backwards is right on ~97% of
vectors and wrong on the rest.

Q6 is now **measured for the primitives**. It is **not** measured for the *emitted* files, which
is Gate 1, in phase 1.

## 6. Built — the editor config, and a false alarm worth recording

**Reported: source control appears to be tracking `.venv`.** It is not. Measured:

```
files tracked under .venv/   0
untracked shown by git       none
git check-ignore             .gitignore:5:.venv/   -> matched, correctly ignored
files inside .venv/          23,002
```

Two real causes behind the appearance, neither of them git tracking anything:

1. **`.gitignore` stops git, not the editor.** 23,002 files were being watched, indexed and
   searched. Added `.vscode/settings.json` with `files.watcherExclude`, `search.exclude` and a
   pinned interpreter path. Committed deliberately, so anyone who clones and makes a venv gets a
   responsive editor without rediscovering why. `.venv` is left *visible* in the explorer — only
   excluded from watching and searching.
2. **The `.gitignore` fix was uncommitted.** The committed version at `e394093` lists only
   `CLAUDE.md` and `COPY-ME.md`; `.venv/` existed solely in the working tree. Anything reading
   committed state, or a cache from before the edit, would show 23,002 untracked files.

⚠️ **The general lesson: "git is tracking it" and "my editor is drowning in it" look identical
from the GUI and have completely different fixes.** Measure with `git ls-files` and
`git check-ignore -v` before editing `.gitignore` again.

## 7. Built — the test suite

`tests/test_packaging.py` and `tests/test_precision.py`. **39 passed, 1 xfailed.**

`test_packaging.py` asserts phase 0's actual deliverable — the things that break silently between
"works on my machine" and "a user ran pip install": the package imports, the version in
`pyproject.toml` matches the importable one, the entry point resolves, the four primitives and
both testbenches ship *inside* the package, unimplemented subcommands exit 2 rather than 0, and
`--help` output is ASCII.

`test_precision.py` covers the Q9 policy. Two cases are deliberately anchored to reality rather
than invented: `precision_for(thr, input_bits=8)` on MNIST-shaped thresholds gives **Q0.8, 9-bit**
and `precision_for(thr)` on JSC-shaped ones gives **Q3.12, 16-bit** — both the formats those
studies actually used. A policy that agrees with two independently-derived formats is evidence; a
policy that agrees with none is a guess. The `required_int_bits` boundary is pinned at `1.0`
costing an integer bit, which is exactly the MNIST `z=25` case from roadmap Q9.

**Decided: known gaps get a `strict=True` xfail, not a TODO.** `dwn_top_tb.v` being empty is
asserted as an expected failure. When phase 1 writes it, the test starts passing, `strict` turns
that XPASS into a failure, and whoever wrote the testbench is forced to delete the marker. A gap
that announces its own resolution beats a comment nobody greps for.

**Hit: a test that passed only because of file ordering.** `test_version_flag_exits_zero` called
`dwn2rtl.cli.main(...)`, but `__init__.py` deliberately does not import its own `cli` submodule —
so the attribute existed only because an earlier test in the same file had imported it. It would
have broken under `pytest-randomly` or if run alone. Now imported locally. Verified by running
that test in isolation.

## 8. Built — the licence

MIT, `LICENSE` at the repo root, matching the `license` field already in `pyproject.toml`.
Roadmap **P3 closed** — it was blocking both use and citation.

⚠️ The copyright line reads **"The dwn2rtl authors"** rather than a personal name, because the
project has more than one and none of their legal names is something this file should guess.
Replace it if a named holder is wanted.

## 9. What remains before phase 0 closes

| # | item | why it belongs to phase 0 |
|---|---|---|
| ~~**1**~~ | ~~**`tests/` does not exist** — `pytest` exited 0 with zero tests collected, a green light that meant nothing~~ | ✅ **done**, §7 — 39 passed, 1 xfailed |
| ~~**2**~~ | ~~**`LICENSE` file** — `pyproject.toml` claimed MIT with no file backing it~~ | ✅ **done**, §8 — roadmap P3 closed |
| **3** | **Nothing is committed.** Ten changed/untracked paths | ⬅ the only item left |

**Explicitly deferred, with reasons, so they are not mistaken for oversights:**

- **`yosys` not installed.** Only `dwn2rtl estimate` needs it, which is phase 3. winget has no OSS
  CAD Suite package, so it will be a manual download when the time comes.
- **`README.md` is 10 bytes.** `pyproject.toml` points at it and the build succeeds, so nothing is
  blocked. Phase 3.
- **No CI.** Phase 2, and it needs tests to run first.
