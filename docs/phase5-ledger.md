# Phase 5 ledger — publish it

**Goal.** Put `dwn2rtl` on PyPI, so the install instruction is `pip install dwn2rtl` and the tool
is available to someone who has never heard of this repository.

**Status: CLOSED.** Retrospective in `phase5-report.md`.

> Conventions in `overview.md` §6. Entries oldest first, recording *built* / *hit* / *decided*.
> Reversed decisions stay, struck through, with the reason.

---

## The rule this phase is organised around

**Publishing is the first irreversible thing this project has done.** Every phase before it could
be undone by an edit. A version on PyPI can be *yanked* but never replaced, re-uploaded, or
deleted — so the ordering matters in a way it never has here, and anything that must be right
has to be right *before* the upload rather than after it.

The corollary, and it is the phase's shape: **rehearse on TestPyPI first**, on the same reasoning
as every other rehearsal in this project. TestPyPI is a full-fidelity instance, so a rehearsal
there is not a simulation of publishing — it *is* publishing, to a throwaway index.

---

## 1. Built — `estimate` documented, and a stale calibration found on the way

Phase 4's headline deliverable appeared **zero times in the README**. A reader of the front door
met a two-command tool. Added a "Will it fit?" section with the yosys/Vivado calibration table
and the unmapped-design refusal.

⚠️ **Hit: the estimator's own docstring was quoting a withdrawn measurement.** `estimate.py`
still carried `dwn_core 106 / 0.96x / ratio 6.8x` and printed *"the CORE agreed within 4%"* with
every report. Phase 4 ledger §7 had already superseded those when explicit `hierarchy; flatten`
replaced `synth -flatten` and brought the core to **110 against Vivado's 110, exactly**.

**The correction runs the unusual way: the tool agreed with the vendor BETTER than it claimed.**
That is still a defect — a number that does not match its evidence is wrong in either direction,
and this one was about to be copied into the README, which is the document that becomes the PyPI
project page.

## 2. ⚠️ Decided — a GitHub organisation, because a personal repo cannot share control

The repository was `Krithik4/dwn2rtl`, with 38 commits from one author and 27 from the other.

⚠️ ~~"Add the other as an admin collaborator"~~ — **withdrawn, it is not possible.** Repositories
owned by a personal account have exactly **two** permission levels, owner and collaborator; the
granular roles (Triage, Maintain, Admin) are organisation-only. So a collaborator can push, merge,
tag and cut a release, but **cannot** manage secrets or environments, change visibility, or
configure the publishing pipeline.

That asymmetry is precisely what Trusted Publishing needs configured, so it was load-bearing
rather than cosmetic. Transferred to a `dwn2rtl` organisation with both as owners.

**Timing was the whole argument.** A move after publishing means reconfiguring the Trusted
Publisher and leaving stale URLs frozen in already-released metadata. Before the first release it
costs nothing. `github.com/dwn2rtl/dwn2rtl`.

**Decided: the study repo stays where it is.** `Kanishk234/dwn-fpga-study` was already public —
checked rather than assumed, with a control — so the README's evidence link already resolved and
the planned conversion from link to citation was dropped as unnecessary work.

## 3. ⚠️ Hit — CI broke, and it took three wrong diagnoses to find out why

The Ubuntu jobs began stalling on `apt-get install iverilog yosys`. Recorded in full because the
*pattern* of the error is more useful than the fix.

| # | diagnosis | what disproved it |
|---|---|---|
| 1 | the dpkg lock, held by `unattended-upgrades` | it hit **both** matrix jobs; lock contention is random per-runner |
| 2 | `Acquire::Retries=3`, added in fix 1, multiplying the wait | a run **without** the flag showed the same four attempts — Ubuntu 24.04 defaults Retries to 3, so the flag was a no-op |
| 3 | apt's 120s default timeout; shorten it | with 10s timeouts it still ran 15 minutes and timed out |

**Root cause, from the log rather than from reasoning:** the runner's `/etc/apt/apt-mirrors.txt`
leads with `azure.archive.ubuntu.com`, which was unreachable. *Every index item* pays its own
retry cycle against it before the mirrorlist falls back to `archive.ubuntu.com`, which works —
visible as the `Hit:`/`Get:` lines at the bottom of every slow run.

**Fix: replace the mirrorlist.** Do not tune how long to wait for a host that will never answer.

⚠️ **The first two diagnoses were made without reading a log**, and both were labelled as
inference at the time, which is the only thing that makes them defensible. The third was made
*with* a log and was still wrong, because it read the symptom (slow) instead of the structure
(one dead host, many items, each paying separately).

**Also fixed: `timeout-minutes`.** GitHub's default is **360**, which is not a timeout so much as
an afternoon. A hung job looked green for six hours. The suite runs in ~2 minutes, so 15/30 are
generous. Same rule as everything else here: *an unmeasured thing must not look like a measured
one*, and a spinner is the purest form of that.

## 4. ⚠️ Hit — the Windows guard failed a working install, and the guard was mine

Windows went red at "The gate actually ran" with no simulator found. **That guard working as
designed**: `choco install iverilog` had exited 0 having installed nothing, and without the guard
the run would have gone green with the gate silently skipped.

A verification step was added to the install so the failure would land *at* the install. It then
failed a run where choco had genuinely succeeded:

```
Chocolatey installed 1/1 packages.   iverilog v11.0.0
FATAL: choco reported success but there is no iverilog.exe
```

**Cause: the check looked in the wrong places.** Chocolatey does not use the installer layout. It
unpacks to `C:\ProgramData\chocolatey\lib\iverilog\tools` and writes **shims** into
`C:\ProgramData\chocolatey\bin`, which is already on PATH — which is why Windows CI had been
green for dozens of commits. The check tested `C:\iverilog\bin` and the two `Program Files`
variants, the *winget* layout, and never looked at PATH at all.

⚠️ **The check duplicated logic `verify.py` already owns, and drifted from it immediately.**
Commit `b958f3f` deleted exactly this kind of second copy a phase earlier. Rewritten to ask the
same question in the same order — PATH first, then the installer directories.

### ⚠️ Hit inside the fix: a check that would have failed *because the tool works*

`iverilog -V` **exits 255 on success** — `verify.py:74` documents this and deliberately ignores
the return code. GitHub runs `pwsh` with `$ErrorActionPreference='stop'`, and PowerShell 7.3+ can
promote a native command's non-zero exit to a *terminating* error. A verification step ending in
`iverilog -V` would therefore have failed on a perfectly healthy simulator.

Caught by reading `verify.py`'s comment before shipping the step, not by CI. **The note that
saved it was written by whoever hit the same thing in `verify.py` originally** — the comment paid
for itself in a completely different file.

## 5. Built — packaging metadata, and the README link trap

Four prerequisites, all recorded in `phase4-ledger.md` and all still true when checked:

| | |
|---|---|
| `[project.urls]` | absent — that is the PyPI sidebar, and without it the page has no route to source |
| `authors` | **absent entirely** — the page would have credited nobody |
| version | `0.1.0.dev0`; pip skips dev releases by default |
| README links | five relative links |

⚠️ **PyPI renders the README as the project page, where a relative link resolves against
`pypi.org` and 404s.** Four docs links and `LICENSE` worked perfectly on GitHub and would have
been dead on the page every new user sees. Nothing about the file looks wrong — the same text is
correct in one renderer and broken in the other.

**Decided: `authors` carries names, no email addresses.** PyPI project pages are scraped
continuously, a published version's metadata cannot be edited, and the practical value is near
zero because bug reports go to Issues, which `[project.urls]` links. Adding an address later is
trivial; removing one is impossible, and that asymmetry decides it.

⚠️ **Making the links absolute silently removed the old test's teeth** — a URL cannot be checked
against the filesystem, so `test_readme_links_resolve` would have passed on a link to a file that
does not exist. Split in two: one rejects relative links, the other maps repo-internal URLs back
to paths and checks them against `git ls-files`. **Each was verified to fail on its own defect**
rather than assumed to work.

## 6. ⚠️ Decided — rehearse the real mechanism, not a convenient one

`twine upload` with an API token would have put an artifact on TestPyPI in five minutes. It was
rejected: the real path is Trusted Publishing from GitHub Actions, and **a rehearsal of a
different mechanism is not a rehearsal**.

`publish.yml`, four jobs, `workflow_dispatch` only:

```
build  ->  gate + verify  ->  publish
```

- **`build`** — sdist and wheel, then `twine check --strict`, so a README PyPI cannot render is an
  ERROR. The project page *is* the README; a rendering failure ships a blank page.
- **`gate`** — the full suite on the commit being released.
- **`verify`** — installs the **built wheel** with no source tree present and drives `build` +
  `verify` through a simulator. `pip install -e` cannot check this: it leaves the repo in place,
  so `files('dwn2rtl')/'rtl'` resolves back into it and a missing package-data block looks healthy
  (`phase0-ledger.md` §4).
- **`publish`** — needs all three. OIDC only; there is no token to leak or rotate.

⚠️ **Deliberately NOT triggered by tag push.** An immutable, irreversible act should not be a
side effect of a typo in a tag name. A person chooses it.

## 7. Built — the TestPyPI rehearsal, `0.1.0rc1`

Published through the real pipeline, then verified the way a stranger would:

```
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ dwn2rtl==0.1.0rc1
```

⚠️ **The `--extra-index-url` is not optional.** numpy and torch are not mirrored on TestPyPI, so
a plain install fails in a way that reads as *your package is broken*.

| checked | result |
|---|---|
| import origin | `…\rehearsal\Lib\site-packages\dwn2rtl` — not the repo |
| `quickstart.py` | `dwn_core` 504 ✓, `dwn_top` 535 ✓, **PASS** |
| CLI launcher | `dwn2rtl.exe --version` -> `0.1.0rc1` |
| CLI end to end | `build` + `verify` -> 504 ✓, 511 ✓, **PASS** |

**The CLI was tested separately on purpose.** `quickstart.py` exercises only the Python API,
while the README leads with `dwn2rtl build` / `dwn2rtl verify`. A broken console script would
have shipped invisibly.

**The rehearsal found nothing.** Recorded as a result rather than an anticlimax — see the report.

## 8. ⚠️ Decided — the README must be right BEFORE the build, not after the upload

**The README is baked into the uploaded artifact**, so PyPI renders whatever the file said when
the wheel was built. Publishing `0.1.0` while the README still said
`pip install git+https://github.com/Krithik4/dwn2rtl.git` would have frozen that instruction —
pointing at a *stale org path* — on the project page for that version.

So the install line changed first, and the test that forbade it was **inverted rather than
deleted**: it now asserts `pip install dwn2rtl` is present and no `pip install git+` survives, so
the stale-URL defect cannot return. Two more stale `Krithik4` URLs were found in `overview.md`
during the same sweep.

⚠️ **And the immutability is narrower than it first appears.** PyPI renders the *latest*
non-yanked release's description, so a README fix costs a version number (`0.1.0.post1` for a
docs-only change) rather than being permanent. What is genuinely frozen is the per-version page
and the fact that a version exists.

## 9. Built — `0.1.0` on PyPI, and verified there too

```
pip install dwn2rtl          # no flags, exactly what the README says
```

| | |
|---|---|
| index | wheel + sdist, both with **provenance attestations** — Trusted Publishing, not a token |
| import origin | `…\live\Lib\site-packages\dwn2rtl` |
| CLI | `dwn2rtl --version` -> `0.1.0` |
| the gate | `dwn_core` 504 ✓, `dwn_top` 535 ✓, **RESULT PASS** |

**The project's own rule held to the end**: a simulator said the emitted RTL matched the golden
model, from an artifact fetched off the public index rather than out of a working tree.

**Suite: 243 passed, 11 skipped.**
