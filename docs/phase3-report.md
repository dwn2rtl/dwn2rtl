# Phase 3 report — make it usable by someone else

**Status: closed, 2026-08-13.** One unit deferred to phase 4, with a reason.

The day-by-day record is `phase3-ledger.md`. This is the retrospective.

---

## 1. The result

The tool worked after phase 1 and stayed working after phase 2. **Nobody else could use it.**
The README was 10 bytes, there was no example, the shipped Verilog cited documentation that did
not exist, and the format's claims rested on a commit this repository had never checked.

All four are fixed. A new user can now `pip install` from git, run one file, and watch a DWN
become verified Verilog in a few seconds.

## 2. What was delivered

| | |
|---|---|
| `examples/quickstart.py` | end to end in one file — no dataset, no training, no upstream package |
| `README.md` | 10 bytes -> a real document, opening with real MNIST output |
| `tests/test_examples.py` | the example and the README are checked like anything else |
| `rtl/*.v` provenance | six shipped files no longer cite documents that do not exist |
| the upstream pin | `9f887a0b4bd…`, verified independently rather than inherited |

## 3. Findings that outlive this phase

### 3.1 The shipped Verilog is documentation, and nobody was reading it as such

The four hand-written primitives are **copied into every emitted directory**. Their comments are
therefore a user-facing artifact — and they had come across from the study repo verbatim, citing
`docs/reference/checkpoint-format.md`, `docs/jsc/dse-plan.md` and `probe-results.md`, none of
which exist here, plus *"Phase 1"*, *"brief §10"* and *"CLAUDE.md"* meaning the **study repo's**.

**A reference that resolves nowhere is worse than no reference**: it tells the reader there is an
authority to consult and then wastes their time. Each is now self-contained or points at
something real, and a test resolves every `docs/*.md` a comment cites against the filesystem.

The fix improved the content, not just the links. `lut_node.v` said *"n=6 is fixed for Phase 1
bring-up"* — which is a fact about someone else's schedule. It now states the actual constraint:
2\*\*n entries must fit a 64-bit parameter, above which Verilog truncates **silently**, and one
node stops being one LUT6. **A comment that explains the hardware survives; one that explains the
project's history does not travel.**

### 3.2 A README is the one document everything reads and nothing checks

Its opening code block contained two false claims, both mechanically checkable, and both found by
*checking* rather than re-reading:

- **`pip install dwn2rtl`** — PyPI returns 404. The package is not published.
- **a link to `CLAUDE.md`** — which is in `.gitignore`, so it 404s on GitHub.

A third was an omission: the quoted MNIST output was missing the scaled-features warning the real
run prints, which made a "real output" block not-quite-real.

There is now a test that every repo-relative link resolves to a **git-tracked** file, that the
PyPI instruction cannot reappear before it is true, and that the example it points at exists.
**Documentation that makes checkable claims should be checked**, and the cost of doing so is a
dozen lines.

### 3.3 An example is documentation that promised to run

Prose that goes stale is merely wrong. An example that goes stale is wrong *and* was promised to
work, and it is the first thing a new user tries. `examples/quickstart.py` runs in CI like
anything else.

Two decisions inside it are worth keeping:

- **It does not train, and says so.** dwn2rtl translates structure; what is being demonstrated is
  that the Verilog matches the model, not that the model matches a dataset. An untrained model
  exercises the emitter identically. Saying that plainly beats an example that quietly implies
  training is part of the flow.
- **It does not import `torch_dwn`.** Upstream builds a CUDA/C++ extension — and its `LUTLayer`
  forward *requires CUDA*, the CPU path raises. Making someone compile that before they can see
  the tool work would be backwards, and dwn2rtl genuinely does not need it.

### 3.4 A pin is provenance, not a version gate

`docs/checkpoint-format.md` was inherited from the study repo, and `tool-handoff.md` §9
explicitly asked that its pin **not** be taken on trust. So every load-bearing claim was
re-checked against the upstream source at `9f887a0`: the strict `> 0` table threshold, the
LSB-first address order, `argmax(dim=0)`, the `arange` decoy, and `GroupSum` having no
parameters. All five confirmed.

The last one matters beyond bookkeeping: **`GroupSum` having no parameters is why `num_classes`
cannot be recovered from a `state_dict`**, which is a load-bearing part of `checkpoint.py`'s
design. It is now confirmed rather than assumed.

`UPSTREAM_COMMIT` lives in code, a test keeps it identical to the document's, and **a second test
asserts loading never checks it**. Duck-typing exists so a user whose model was trained against a
different commit is not refused; an upstream rename should surface as a clear failure with a real
message, never as *"your checkpoint is the wrong version"*.

### 3.5 Deferring beats writing blind

`estimate` needs yosys, and on Windows — this project's primary target — there is no light
option. Measured, not assumed:

| | |
|---|---|
| winget / chocolatey | no package |
| YosysHQ `yosys` releases | source tarballs only, no prebuilt Windows binary |
| OSS CAD Suite | **703 MB** |
| `apt install yosys` | ~9 MB, but that is Linux |

`estimate` is the only feature the roadmap itself calls *optional*, and this project's rule is
that nothing counts until it runs. Writing it now would have put **the first unverified claim in
the repository** to save a download that can happen later just as easily. It stays a stub that
exits 2 and names its phase.

## 4. Roadmap movement

- **P4 — README and worked examples** ✅ closed.
- **P7 — stop defaulting to a JSC path** ✅ moot: no default checkpoint path exists anywhere in
  this repository. Verified rather than assumed.
- **Q3 / P6 — the upstream-version policy** ✅ decided: **one pinned commit**, recorded in code
  and in docs, verified independently, and **not enforced at load time**. The roadmap called a
  single pin "honest and cheap"; the addition here is that duck-typed loading makes it cheap
  *without* being brittle.
- **V3 — simulator independence** was closed in phase 1; the README now documents installing one
  on all three platforms, which is what makes it usable rather than merely possible.

## 5. What phase 4 inherits

- **`estimate` via yosys** — the last stubbed subcommand, and the only one left.
- **Publish to PyPI**, at which point the README's install line changes and one test is deleted.
- **Verilator as a second simulator backend**, behind the existing `Simulator` interface.
- **`docs/tool-handoff.md` and `docs/tool-roadmap.md` are now historical.** They were written to
  start this project and describe work that is done. They should either be marked as such or
  folded into `overview.md`, because a new reader currently meets three documents describing the
  plan and has no way to tell which is current.

## 6. By the numbers

| | |
|---|---|
| tests | **214 passing**, 1 skipped, 30 of them the gate |
| README | 10 bytes -> ~8,400 |
| shipped files citing documents that do not exist | 6 -> **0** |
| roadmap items closed | **P4, P7, Q3/P6** |
| false claims found in the new README, by testing it | **2** |
