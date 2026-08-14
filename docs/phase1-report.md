# Phase 1 report — close the loop

**Status: closed, 2026-08-13.**

The day-by-day record is `phase1-ledger.md`. This is the retrospective: what phase 1 delivered,
what it found, and what the next phase inherits.

---

## 1. The result

**A trained DWN goes in, and Verilog that a simulator certifies bit-exact against the golden
model comes out.** That is the whole product claim, and as of this phase it is a measurement
rather than an intention.

```
$ dwn2rtl build model.dwn --out rtl/ --input-bits 8
features 8, classes 3, layers [12, 9], n=6, z=3   from checkpoint
integer bits 0                                    derived, exact
frac bits 8 -> Q0.8 signed (9-bit)                from --input-bits, provably lossless

core      21 nodes, 4 cycles
encoder   23 comparators of 24 thermometer bits
top       5 cycles latency, II=1

vectors   core 504, top 519, 3/3 classes hit
wrote     rtl/ (17 files)

$ dwn2rtl verify rtl/
iverilog 12.0 (C:\iverilog\bin\iverilog.exe)
  dwn_core  504 vectors  PASS
  dwn_top   519 vectors  PASS
RESULT   PASS
```

Both levels pass on all five fixture shapes:

```
             core                     top
tiny         504 vectors  PASS        511 vectors  PASS
single       504 vectors  PASS        511 vectors  PASS
n6           504 vectors  PASS        519 vectors  PASS      n=6, 64-bit tables
ten_class    504 vectors  PASS        519 vectors  PASS      K=10, 4-bit index
all_fixed    504 vectors  PASS        511 vectors  PASS
```

## 2. What was delivered

| | |
|---|---|
| `checkpoint.py` | the format: sniff three shapes, normalize, validate, refuse a bare `state_dict` by name |
| `build.py` | `build_core` -> `build_encoder` -> `generate` -> copy the primitives; the `BuildReport` |
| `verify.py` | find a simulator, compile, run, parse, and never call an unchecked thing a pass |
| `rtl/tb/dwn_top_tb.v` | the encoder-plus-core gate — a 0-byte file since commit `646aebe` |
| `tests/fixtures.py` | synthetic checkpoints, five shapes, test-only |
| `tests/test_*.py` | 172 tests, 23 of them the gate |

`emit_core.py` gained two small changes: it returns its extracted layers, and it reports an
unrecorded accuracy instead of demanding one.

## 3. Roadmap movement

- **Q8 — there is no upstream checkpoint format** ✅ closed. The tool defines it and owns both
  ends: `save`/`from_model` in, `build` out, and a named refusal for the bare `state_dict` that
  silently drops the encoder.
- **V3 — simulator independence** ✅ closed. The roadmap called Vivado-only verification *"the
  largest single barrier to anyone using the tool"*: every other vendor dependency was the
  user's own synthesis, which they were doing anyway, but that one made *verification* need a
  licence. It does not any more.
- **V1/V2 — ship the gate, vectors without a dataset** ✅ done. Random plus edge-case vectors
  derived from the model alone.
- **Q6 — vendor-neutral** advanced from *inspection* (phase 0: the primitives) to **measured on
  emitted files**, at five shapes, both levels.

## 4. Findings that outlive this phase

Five results here change what later phases must do, or correct something previously believed.

### 4.1 A too-wide Verilog constant does not error — it silently becomes a different valid one

The test that proves the gate can *fail* corrupts an emitted truth table and requires a `FAIL`.
The first corruption was `64'h` -> `64'hF`. **It passed, correctly**: that makes a *seventeen*-
digit literal in a 64-bit parameter, and Verilog silently truncates the excess high digit,
restoring the original sixteen. The design was bit-identical.

That is exactly the hazard `emit_core.py`'s `MAX_N` assertion exists to prevent at n>6 — *"a
2\*\*n-bit table into a 64-bit parameter and Verilog would TRUNCATE it silently"* — reproduced by
accident on this machine. **The assertion is not defensive programming; it guards a real
behaviour of the language.** Anyone tempted to relax it should read this first.

### 4.2 A fixture can be degenerate, and a degenerate fixture makes the gate meaningless

The first synthetic checkpoints predicted **one class for every input**. The cause was a popcount
group of 2: with untrained tables a node is *constant* whenever its `2**n` table happens to be
all-positive (1/16 at n=2), and two constant nodes score a permanent maximum nothing can beat.

**A testbench whose expected output is a constant passes against a design whose argmax, popcount
and grouping are all wrong.** It is the same failure the gate exists to catch — the study repo's
read-back reporting 20/20 on a design wrong 958 of 1,504 times — relocated from the emitter into
the fixture, where it would have been far harder to notice, because everything would have been
green.

Fixed by groups >= 3 and a deterministic seed search, with `classes_hit()` asserted per shape. A
*pinned* good seed was rejected: it stops being good the moment a shape parameter changes, and
fails as a collapsed fixture rather than as an obviously wrong seed.

**The general rule: a test fixture needs its own correctness criterion.** "It ran" is not one.

### 4.3 A skipped test is indistinguishable from a passing one

The degeneracy test originally *hunted* for a collapsed model and called `pytest.skip` when it
did not find one — so it had been quietly not running. Degeneracy is now constructed
deterministically. Worth generalising: a conditional skip inside a test body is a test that may
never execute, and the suite reports it in the same green summary line either way.

### 4.4 Metadata must never be mandatory, and absent must never print as zero

`emit_core.py` read `ck['results']['final_acc']` unconditionally — for a **header comment**. A
user who saved a model without recording a training statistic would have got a `KeyError` out of
a code generator, over something no part of the hardware depends on.

Defaulting to `0.0` was rejected for the same reason the fixture's invented `0.0` was: it reads
as *"this model scores zero"* rather than *"nobody measured"*. It now prints `not recorded in the
checkpoint`. **A comment that confidently states a wrong number is worse than no comment.**

### 4.5 Structural checks and behavioural checks are not interchangeable

Both emitters carry a read-back check that re-parses their own output and asserts every table,
wire index and comparator against the checkpoint. Those checks are good and they caught nothing
this phase — because they can only prove the file *says* what the checkpoint says. Everything
this phase actually found was found by running the thing: the truncation, the degeneracy, the
skipped test.

## 5. Decisions, and their reasoning

| decision | why |
|---|---|
| **Input format is sniffed** | Upstream saves nothing, so no user arrives holding our format. One line of plain `torch.save` should be enough, with no `dwn2rtl` import in a training script |
| **`num_classes` comes off `GroupSum` by class name** | It is the one fact no tensor knows — `GroupSum` has no parameters. Matched by name so nothing here imports upstream |
| **`build_core` hands its layers to `generate`** | The CLAUDE.md invariant is that vectors and RTL derive from the same checkpoint; passing the same in-memory arrays makes that structural, not merely likely |
| **Output directories are self-contained** | A user should not need to know where pip put the package. Testbenches in `tb/` so `*.v` in the root is exactly the design |
| **Two testbenches, always both** | A failure localizes itself: core PASS + top FAIL means the encoder and nothing else. Proved by a test that corrupts a comparator and requires exactly that split |
| **Not checking is not passing** | Missing, empty, error, no-verdict and no-levels all fail. `all([])` is `True`, and that would have printed green having checked nothing |
| **The checkpoint API is lazy at the package root** | `dwn2rtl.load` works, and `import dwn2rtl` still does not import torch — which matters most for `verify`, which never reads a checkpoint |

## 6. What phase 2 inherits

The remaining work is packaging, not generation.

- **CI**, on Linux and Windows both. The gate runs in ~15 s on the synthetic fixtures, so it can
  run on every commit — which is what CLAUDE.md requires. Fixtures must be synthesized, never
  committed (roadmap P8: real checkpoints are 17–471 MB).
- **A `conftest.py`.** `tests/fixtures.py` is imported by bare `import fixtures`, which works
  because pytest inserts the test directory on `sys.path`. That is implicit and will surprise
  someone.
- **`estimate`** — the only remaining stubbed subcommand, phase 3, needs yosys.
- **README and a worked example** — phase 3. The README is still 10 bytes.
- **`verify.py` supports only iverilog.** Verilator would be a second backend behind the same
  `Simulator` interface; nothing in the design assumes there is only one.
- **Never verified against a REAL checkpoint.** Everything here is synthetic by necessity. The
  study repo's own checkpoints are the obvious first real input, and `checkpoint.py` accepts
  their format specifically so that is a one-command test.

## 7. By the numbers

| | |
|---|---|
| tests | 172 passing, **23 of them the gate**, 0 xfail |
| gate coverage | 5 shapes x 2 levels, 504–519 vectors each |
| roadmap items closed | **Q8, V3, V1, V2**; Q6 advanced |
| lines added | ~1,900, about a third of it tests |
| Verilog emitted and verified | encoder, core, top — all three |
