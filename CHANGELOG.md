# Changelog

## 0.3.0

⚠️ **Upgrade if you're on 0.2.0 -- two of these are silent-wrong fixes.** A model whose first
layer has a fixed mapping could be built against a thermometer from a different training run:
clean build, `PASS` on both levels, and every real feature landing in the wrong bit position.
And a directory holding RTL from one build beside test vectors from another could report
`PASS`, which checks the wrong pairing rather than your design.

⚠️ **Two behaviours changed rather than being fixed**, so a script relying on the old ones will
notice. `verify` now refuses a directory whose RTL and vectors came from different builds, and
`quantize()` now raises on a NaN feature where it used to return int64's minimum. Both are
corrections, and both can turn a previously "working" run into a loud failure -- which is the
point.

Full record in `docs/phase8-ledger.md`; the retrospective is `docs/phase8-report.md`.

### Fixed

- **A design with more than 256 classes failed its own testbench while being bit-exact.** Both
  testbenches stored the correct answer in a `reg [7:0]` and then sliced it to the index width,
  so at `IDX_W = 9` the part-select read past the end of the register and returned `x` -- and
  `!==` treats `x` as a definite mismatch, so *every* vector failed. 256 classes passed; 257
  failed 100%. Every class count anywhere in the test suite was 2, 3 or 10.
- **A mismatched thermometer was only detected for a learnable first layer.** A learnable
  mapping's `weights` states the input width it expects; a fixed mapping is just a list of
  indices and states nothing, so the existing check couldn't run. Exact detection isn't possible
  from the checkpoint alone, so `build` now reports the signature -- trailing features driving
  no comparator -- and names the likely cause. A warning rather than a refusal, because a
  lopsided model looks the same.
- **`verify` accepted a directory whose RTL and vectors came from different builds.** An
  interrupted rebuild leaves exactly that, and one such directory -- an 11-bit encoder against
  9-bit vectors -- reported `RESULT PASS`. Every generated `.vh` now carries a digest of the
  checkpoint, precision and pipeline it came from, and `verify` refuses a directory holding more
  than one before it compiles anything.
- **`quantize()` saturated to the wrong value at wide words.** The clamp ran in float64, where
  `2**63 - 1` isn't representable, so at a 64-bit word an input that should have saturated to
  `+9223372036854775807` came out as `-9223372036854775808` -- a sign flip. Wrong from about a
  55-bit word up, reachable via `--input-bits 62`. `build` never calls this function; the user
  guide and `input_scaling.json` tell *you* to.
- **A NaN feature became an extremely negative one.** `quantize()` now refuses it. Every
  comparison against NaN is False, so it saturated in neither direction and fell through to
  int64's minimum. Infinities still saturate, which is correct -- an infinity is on a definite
  side of every threshold, a NaN is on no side of any.
- **`python -O` deleted every emitter self-check.** Both read-backs, the `n <= MAX_N` guard and
  the software model's group-divisibility check were `assert`s, so under `-O` a build with a
  reversed address concatenation on disk completed and reported nothing. They're real exceptions
  now; the read-backs raise `EmitterMismatch(ValueError)`.
- **A `results` field that wasn't a mapping ended the build in a traceback**
  (`TypeError: 'int' object is not iterable`), over a value that reaches one comment line.
- **`verify.py` emitted a `SyntaxWarning` on every import** -- a Windows path in a non-raw
  string.

### Changed

- **`build` now tells you when a threshold sits on the word's rail.** `saturation_is_lossless()`
  was defined, named by `quantize()`'s docstring as the check to use, and called by nothing --
  the tool used the inclusive `fits_in_word` instead. A format can now be reported as lossless
  *except at that rail*, qualified in the line that makes the claim rather than only in a
  warning below it.
- `BuildReport` gained `saturation_lossless`.

## 0.2.0

⚠️ **Upgrade if you are on 0.1.0.** It emits wrong core-level test vectors for any model whose
thermometer width is not a multiple of 8, which shows up as `core FAIL` on a design that is
actually correct.

### Fixed

- **Core-level vectors were wrong at widths off a byte boundary.** `bits_to_hex` packed with
  `np.packbits`, which pads a partial byte on the low side, so a 12-bit vector came out 16x too
  large and an 18-bit one 64x. The emitted *hardware* was always correct; the testbench's input
  file was not, so `verify` reported `core FAIL` on a good design. Both studied models and every
  fixture happened to be byte-aligned, which is why it survived so long.
- **Pipeline depths above 1 did nothing.** `pipe_reg` treated its parameter as an on/off flag
  while `Pipeline` documented a stage count, so `Pipeline(lut=2)` claimed a latency the hardware
  did not have and the testbench sampled the wrong cycle.
- **Zero-latency designs failed their own testbench.** A fully combinational build --
  `Pipeline(enc=1, lut=0, pop=0, out=0)` is a reasonable one -- was compared one step before its
  input was applied.
- **A NaN or infinite threshold hung the tool forever** rather than failing.
- **A stale `input_scaling.json` survived a rebuild**, so an unscaled model could inherit the
  previous model's mean and scale.
- **Your own Verilog in the output directory broke `verify`**, which globbed `*.v` and compiled
  whatever it found. It now compiles only the files it emitted.
- Many bad inputs produced a Python traceback instead of a message: a file that is not a
  checkpoint, `num_classes=0`, an absurd `--input-bits`, `--out` pointing at a file, a read-only
  output file, and several wrong-typed API arguments.

### Added

- **`dwn2rtl verify --simulator verilator`** (Linux and macOS). iverilog stays the default and is
  dramatically faster here -- 0.38 s against 14.7 s end to end, because Verilator compiles to C++
  before it runs.
- **The thermometer is checked against the model it came with.** A learnable first layer states
  how many input bits it expects, so a thermometer from a different training run is now refused
  instead of silently producing a design whose features land in the wrong bit positions.
- **An unusable scaler is refused** -- a zero scale divides by zero, and NaN or infinity also made
  `input_scaling.json` invalid JSON that strict parsers reject.
- **A torch that cannot talk to your numpy is explained.** Builds before ~2.3 were compiled
  against NumPy 1.x and cannot hand a tensor to NumPy 2, which previously surfaced as
  `RuntimeError: Numpy is not available`.

### Changed

- `pipe_reg`'s parameter is `STAGES`, not `ENABLE` -- it is a count, and the old name invited the
  bug above. Emitted designs carry the new primitive, so nothing to do unless you kept a copy.

### Internal

- macOS and the declared dependency floors (`numpy==1.22`, `torch==2.0`) now run in CI; both were
  supported in name only.
- Verilator lints every emitted design in CI, and runs the gate a second time as an independent
  simulator.
- The golden model is cross-checked against a second, independently written implementation.
- 405 tests, and the suite is measured by mutation rather than assumed: the shipped Verilog
  scores 11 of 11.

## 0.1.0

First public release. A trained DWN goes in, synthesizable Verilog-2001 comes out, with
self-checking testbenches and golden vectors that prove the RTL matches the model in your own
simulator.
