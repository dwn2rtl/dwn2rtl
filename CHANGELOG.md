# Changelog

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
