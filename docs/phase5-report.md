# Phase 5 report — publish it

**Status: closed, 2026-08-18.**

The day-by-day record is `phase5-ledger.md`. This is the retrospective.

---

## 1. The result

`pip install dwn2rtl` works. **0.1.0** is on PyPI, published through Trusted Publishing with
provenance attestations, and verified after the fact the way a stranger would meet it: a clean
virtual environment, no source tree, the worked example and the CLI both driven through a
simulator to `PASS`.

Twelve commits. **Not one of them changed a line of `src/` logic** — the phase was metadata, CI,
documentation and process, which is what "the code was done" actually looks like when it is true.

## 2. What was delivered

| | |
|---|---|
| **`dwn2rtl` on PyPI** | `0.1.0`, name claimed, wheel + sdist, provenance attestations |
| `.github/workflows/publish.yml` | build -> gate + verify -> publish; OIDC, no stored secret |
| the `dwn2rtl` GitHub org | both authors as owners, which a personal repo cannot express |
| packaging metadata | `authors`, `[project.urls]`, absolute README links, `0.1.0` |
| the README's `estimate` section | phase 4's headline deliverable, previously absent from the front door |
| CI robustness | the dead apt mirror, real timeouts, and a Windows install that proves itself |

## 3. Findings that outlive this phase

### 3.1 A rehearsal that finds nothing is evidence, not waste

The TestPyPI rehearsal found **no defects**. Everything that would have been caught there had
already been caught by working through the prerequisites first: the relative README links, the
missing `authors`, the absent `[project.urls]`, the dev version pip would skip.

That is the correct outcome and worth stating, because the opposite reading is available and
wrong. The rehearsal's value is not the bugs it finds; it is that **after it, nothing about the
real upload was unknown.** The first time the pipeline ran against an immutable index, it was the
second time it had run.

This is the same shape as phase 4's cancelled optimizations. There, two measurements said "do not
change this" and the phase counted that a success. Here a rehearsal said "nothing is wrong" and
that is a success for the same reason: **the cost of finding out was paid before it was
expensive.**

### 3.2 Three wrong diagnoses in a row, and they share a shape

CI stalled, and the cause was named wrong three times (`phase5-ledger.md` §3): the dpkg lock,
then a retries flag that turned out to be a no-op against an apt default, then a timeout that was
shortened and did not help. The actual cause was an unreachable mirror the runner tries first.

**The shape they share: each fix tuned how long to wait for a broken thing instead of not using
it.** Two of the three were guesses made without reading a log — defensible only because they
were labelled as inference at the time, and settled by the log within one exchange. The third was
made *with* a log and was still wrong, because it read the symptom (slow) rather than the
structure (one dead host, dozens of items, each paying its own retry cycle).

⚠️ **The useful generalisation is about timeouts, not apt.** The reason a six-hour stall was
possible at all is that GitHub's default job timeout is **360 minutes**. A job that hangs looks
identical to a job that is working, and this repository has a whole philosophy about exactly that
— `verify` refuses to call an unchecked thing a pass, `estimate` refuses to call an unmapped
design small. **A spinner is the purest form of an unmeasured thing looking measured**, and
nothing in the CI was enforcing a bound on it.

### 3.3 A guard that duplicates the product's logic will drift from it

Windows CI went red twice, for opposite reasons, and both are instructive.

**The first was real** and the existing guard caught it: `choco install iverilog` exited 0 having
installed nothing. Without "The gate actually ran", that run would have been green with the gate
silently skipped — which is the precise failure phase 2 built that step to prevent.

**The second was self-inflicted.** A new verification step added to the install checked
`C:\iverilog\bin` and the two `Program Files` paths — the *winget* layout — while Chocolatey
installs shims to `C:\ProgramData\chocolatey\bin`, on PATH, which is why Windows had been green
for dozens of commits. The check failed a working install.

⚠️ **The lesson is not "be careful with paths."** It is that the check re-implemented discovery
that `verify.py` already owns, and was wrong on its first day. Commit `b958f3f` — *"test verify
and drop the duplicated simulator discovery"* — deleted this same class of second copy a phase
earlier. The rewrite asks the product's question in the product's order.

**And the near-miss is the better story.** The fixed step ends in `iverilog -V`, which **exits
255 when it succeeds**. Under `pwsh` with `$ErrorActionPreference='stop'`, that would have failed
the step *because the simulator works*. What prevented it was a comment in `verify.py:74` written
by whoever hit it there first. **A note explaining a non-obvious workaround paid for itself in a
different file, years-equivalent later** — which is the argument for this project's comment
density, made concrete.

### 3.4 The README is an artifact, not a file

**PyPI renders the README as the project page, from the copy baked into the uploaded wheel.** Two
consequences that are not obvious until they bite:

1. **Relative links 404 there.** Five of them — four docs and `LICENSE` — worked perfectly on
   GitHub and would have been dead on the page every new user sees. The same text is correct in
   one renderer and broken in the other, which is why it is now a test.
2. **The install instruction must be right *before* the build.** Publishing with
   `pip install git+…` still in the README would have frozen that line — pointing at an org path
   the repository had already moved away from — on the `0.1.0` page.

⚠️ Making the links absolute also **silently removed the old test's teeth**, since a URL cannot
be resolved against the filesystem. Split into two tests, each verified to fail on its own defect
rather than assumed to work — the same discipline phase 4 applied when a vacuous test passed on
one platform and proved nothing on either.

### 3.5 Personal repositories cannot express shared ownership

With 38 and 27 commits, the project is genuinely two people's. A repository owned by a personal
account has exactly **two** permission levels — owner and collaborator — and the granular roles
are organisation-only. A collaborator can push, merge and cut a release but cannot manage
environments or secrets, which is exactly what Trusted Publishing requires configured.

**The timing was the entire decision.** Moving after publishing means reconfiguring the Trusted
Publisher and leaving stale URLs frozen in released metadata; moving before the first release
costs nothing. It was the last cheap moment, and it was recognised as such.

## 4. What publishing changed about the rules

**Everything before this phase was reversible.** A wrong emitter could be fixed, a wrong document
rewritten, a wrong measurement withdrawn — and this project's ledgers are full of exactly that,
kept and struck through rather than tidied away.

A published version is not like that. It can be yanked but never replaced. So the phase adopted
one rule the earlier ones did not need: **anything that must be right has to be right before the
upload**, which is why the README ordering, the metadata, and the rehearsal all came first.

The compensating discovery is that the blast radius is smaller than it looks: PyPI renders the
*latest* release's description, so a documentation fix costs a version number
(`0.1.0.post1`) rather than being permanent. **What is truly frozen is narrow — the per-version
page, and the fact that a version exists.** Worth knowing before it induces more caution than the
situation deserves.

## 5. Roadmap movement

- **P1–P8 — fork, package, licence, CI, publish** ✅ complete. Every item the `AFTER` column of
  `tool-roadmap.md` §8 listed is done. That block still *shows* them pending and is deliberately
  left alone — it is a historical snapshot, so the correction went in the file's banner instead.
- **Q4 — the name, and where it lives** ✅ resolved: `dwn2rtl`, claimed on PyPI, at
  `github.com/dwn2rtl/dwn2rtl`.
- **Q3 — upstream-version policy** remains **OPEN**. Publishing did not settle it and did not
  need to; the pin is recorded and verified (`phase3-ledger.md`).

## 6. What comes next

Nothing is unfinished. What remains is optional and unchanged from phase 4's list:

- **`--data`** (unit 8 tier 2) — still deferred, still for the right reason: it cannot be tested
  without a checkpoint whose thresholds are off-grid while its data is quantised.
- **Verilator** as a second simulator backend, behind the existing interface. Now slightly more
  valuable than before: a second simulator is the only way to know the gate is not passing on an
  iverilog quirk.
- **Real users.** The tool is installable by strangers for the first time, which is the one thing
  no amount of internal work could produce.

## 7. By the numbers

| | |
|---|---|
| commits | **12**, none touching `src/` logic |
| tests | **243 passing**, 11 skipped |
| PyPI | `0.1.0`, wheel + sdist, provenance attestations |
| rehearsals | 1, finding **0** defects — see §3.1 |
| wrong diagnoses before CI was fixed | **3** |
| CI failures caused by a guard rather than a defect | **1** |
| stored publishing secrets | **0** |
