"""build() -- checkpoint in, a self-contained design directory out.

⚠️ The tests marked `sim` are THE GATE. Everything else checks the right files appeared with the
right numbers in them; only the simulator checks the Verilog COMPUTES the right thing, and an
emitter's own read-back once reported 20/20 while the design was wrong on 958 of 1,504 vectors.

    pytest -m sim        run only the gate
    pytest -m "not sim"  skip it (no simulator installed)
"""

import os
import subprocess

import pytest

import fixtures
from dwn2rtl.build import PRIMITIVES, build


ALL_SHAPES = sorted(fixtures.SHAPES)


# Simulator discovery lives in verify.py, and the skip is applied centrally by conftest.py to
# anything marked `sim`. This file briefly carried its own copy of both, written before
# verify.py existed so the gate did not have to wait for it; three files had grown one by the
# end of phase 1.
from conftest import SIMULATOR


# ---------------------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize('shape', ALL_SHAPES)
def test_build_emits_a_self_contained_directory(shape, tmp_path):
    """A user must be able to hand the folder to any tool without also knowing where pip put
    the package -- so the hand-written primitives are copied in, not referenced."""
    r = build(fixtures.make(shape), str(tmp_path / 'rtl'), input_bits=8)

    expected = {'dwn_core.v', 'thermometer_encoder.v', 'dwn_top.v',
                'dwn_core_params.vh', 'dwn_top_params.vh',
                'vec_params.vh', 'top_params.vh',
                'x_binarized.hex', 'expected.hex', 'x_quant.hex', 'expected_top.hex',
                *PRIMITIVES}
    assert expected <= set(os.listdir(r.outdir))


def test_testbenches_land_in_a_subdirectory(tmp_path):
    """tb/ rather than the root, so `*.v` in the output root is exactly the design. Two
    testbenches in one compile would give two top modules."""
    r = build(fixtures.make('tiny'), str(tmp_path / 'rtl'), input_bits=8)
    assert os.path.isdir(os.path.join(r.outdir, 'tb'))
    assert not any(f.endswith('_tb.v') for f in os.listdir(r.outdir))


def test_every_testbench_is_either_copied_or_warned_about(tmp_path):
    """A zero-byte file in tb/ would make the directory look complete while that level's gate does
    nothing. Both are real now, so this asserts both copied and no warning -- kept as the guard
    against a future truncated one.
    """
    r = build(fixtures.make('tiny'), str(tmp_path / 'rtl'), input_bits=8)
    empty = [w for w in r.warnings if 'EMPTY' in w]
    copied = sorted(os.listdir(os.path.join(r.outdir, 'tb')))

    assert len(empty) + len(copied) == 2, 'a testbench was neither copied nor warned about'
    assert copied == ['dwn_core_tb.v', 'dwn_top_tb.v'] and not empty


@pytest.mark.parametrize('shape', ALL_SHAPES)
def test_report_agrees_with_the_checkpoint(shape, tmp_path):
    ck = fixtures.make(shape)
    r = build(ck, str(tmp_path / 'rtl'), input_bits=8)

    assert r.layers == list(ck['config']['layers'])
    assert r.classes == ck['config']['num_classes']
    assert r.n == ck['config']['n']
    assert r.nodes == sum(ck['config']['layers'])
    assert r.features == ck['thermometer']['thresholds'].shape[0]
    assert r.thermometer_bits == ck['thermometer']['thresholds'].numel()


def test_encoder_and_core_are_reported_separately(tmp_path):
    """A design invariant, not a formatting choice. The encoder is intrinsic to a DWN, and on
    the smallest studied model it is fourteen times the network it feeds -- reporting a
    core-only count understates a design by most of its cost."""
    r = build(fixtures.make('n6'), str(tmp_path / 'rtl'), input_bits=8)
    text = '\n'.join(r.lines())
    assert 'core' in text and 'encoder' in text
    assert str(r.nodes) in text and str(r.comparators) in text


def test_only_used_thermometer_bits_get_a_comparator(tmp_path):
    """The encoder builds comparators only for bits some node actually reads. That is the
    finding that killed the `features x z` cost model in the study repo."""
    r = build(fixtures.make('n6'), str(tmp_path / 'rtl'), input_bits=8)
    assert r.comparators <= r.thermometer_bits


def test_latency_is_the_sum_of_the_enabled_stages(tmp_path):
    from dwn2rtl import Pipeline
    ck = fixtures.make('tiny')
    r = build(ck, str(tmp_path / 'a'), input_bits=8, pipeline=Pipeline())
    assert r.core_latency == 1 * len(ck['config']['layers']) + 1 + 1
    assert r.latency == r.core_latency + 1

    flat = build(ck, str(tmp_path / 'b'), input_bits=8,
                 pipeline=Pipeline(enc=0, lut=0, pop=0, out=0))
    assert flat.latency == 0


def test_pipeline_depth_reaches_both_the_params_file_and_the_top(tmp_path):
    """emit_encoder reads the core's depth back out of dwn_core_params.vh rather than being
    told it twice, so the two emitters cannot disagree about it."""
    from dwn2rtl import Pipeline
    r = build(fixtures.make('tiny'), str(tmp_path / 'rtl'), input_bits=8,
              pipeline=Pipeline(enc=1, lut=2, pop=1, out=1))
    core_params = open(os.path.join(r.outdir, 'dwn_core_params.vh')).read()
    top_params = open(os.path.join(r.outdir, 'dwn_top_params.vh')).read()

    assert '`define DWN_CORE_PIPE_LUT 2' in core_params
    assert f'`define DWN_CORE_LATENCY {r.core_latency}' in core_params
    assert f'`define DWN_TOP_LATENCY {r.latency}' in top_params


def test_input_bits_changes_the_word_and_says_which_it_is(tmp_path):
    proved = build(fixtures.make('tiny'), str(tmp_path / 'a'), input_bits=8)
    default = build(fixtures.make('tiny'), str(tmp_path / 'b'))

    assert proved.precision.frac_bits == 8 and proved.precision.proved
    assert not default.precision.proved
    assert 'provably lossless' in '\n'.join(proved.lines())
    assert any('DEFAULT' in w for w in default.warnings)


def test_build_accepts_a_path_as_well_as_a_dict(tmp_path):
    import dwn2rtl
    model, therm = fixtures.make_live('tiny')
    p = tmp_path / 'model.dwn'
    dwn2rtl.save(model, therm, p)

    from_path = build(str(p), str(tmp_path / 'a'), input_bits=8)
    from_dict = build(fixtures.make('tiny'), str(tmp_path / 'b'), input_bits=8)
    assert from_path.nodes == from_dict.nodes
    assert from_path.comparators == from_dict.comparators


def test_a_scaled_model_emits_its_scaling_parameters(tmp_path):
    """Telling a user "apply your scaler" while withholding its parameters is not a warning,
    it is a riddle. The numbers are written out, and the encoder header points at them."""
    import json
    ck = fixtures.make('tiny')
    n = ck['thermometer']['thresholds'].shape[0]
    ck['scaler'] = {'mean': [0.25] * n, 'scale': [4.0] * n}

    r = build(ck, str(tmp_path / 'rtl'), input_bits=8)

    path = os.path.join(r.outdir, 'input_scaling.json')
    assert os.path.exists(path)
    data = json.load(open(path))
    assert data['mean'] == [0.25] * n and data['scale'] == [4.0] * n
    assert data['frac_bits'] == 8, 'the scaling is only usable alongside the word format'

    assert any('SCALED features' in w for w in r.warnings)
    enc = open(os.path.join(r.outdir, 'thermometer_encoder.v')).read()
    assert 'input_scaling.json' in enc


def test_an_unscaled_model_says_so_positively(tmp_path):
    """Absence of a scaler is a fact worth stating, not a silence to be interpreted."""
    r = build(fixtures.make('tiny'), str(tmp_path / 'rtl'), input_bits=8)
    assert not os.path.exists(os.path.join(r.outdir, 'input_scaling.json'))
    assert not any('SCALED' in w for w in r.warnings)
    enc = open(os.path.join(r.outdir, 'thermometer_encoder.v')).read()
    assert 'records no scaler' in enc


def test_report_is_ascii(tmp_path):
    """CLAUDE.md: stdout must be ASCII. cp1252 consoles raise on anything else, turning a
    successful build into a traceback."""
    r = build(fixtures.make('n6'), str(tmp_path / 'rtl'))
    '\n'.join(r.lines()).encode('ascii')


def test_encoder_before_core_is_refused(tmp_path):
    """build_encoder reads dwn_core_params.vh for the core's real pipeline depth. Called first,
    it must fail loudly rather than assume a depth."""
    from dwn2rtl.emit_encoder import build_encoder
    from dwn2rtl.precision import precision_for

    ck = fixtures.make('tiny')
    out = tmp_path / 'rtl'
    out.mkdir()
    thr = ck['thermometer']['thresholds'].numpy()
    with pytest.raises(FileNotFoundError, match='build_core'):
        build_encoder(ck, str(out), precision_for(thr, input_bits=8))


def test_vectors_and_rtl_come_from_the_same_extraction(tmp_path):
    """THE invariant: build_core hands its layers to generate() rather than letting it re-read the
    checkpoint. Proved by moving together -- a different checkpoint must change both the emitted
    core and the expected outputs, never one.
    """
    a = build(fixtures.make('tiny', seed=11), str(tmp_path / 'a'), input_bits=8)
    b = build(fixtures.make('tiny', seed=23), str(tmp_path / 'b'), input_bits=8)

    core_a = open(os.path.join(a.outdir, 'dwn_core.v')).read()
    core_b = open(os.path.join(b.outdir, 'dwn_core.v')).read()
    exp_a = open(os.path.join(a.outdir, 'expected_top.hex')).read()
    exp_b = open(os.path.join(b.outdir, 'expected_top.hex')).read()

    assert core_a != core_b, 'different checkpoints emitted identical RTL'
    assert exp_a != exp_b, 'different checkpoints produced identical expected outputs'


def test_a_rebuild_is_byte_identical(tmp_path):
    """Same checkpoint, same output -- vectors included. A testbench whose contents move between
    runs cannot be diffed, and 'it passed yesterday' stops being checkable."""
    a = build(fixtures.make('tiny'), str(tmp_path / 'a'), input_bits=8)
    b = build(fixtures.make('tiny'), str(tmp_path / 'b'), input_bits=8)
    for name in ('dwn_core.v', 'thermometer_encoder.v', 'x_quant.hex', 'expected_top.hex'):
        assert open(os.path.join(a.outdir, name), 'rb').read() == \
               open(os.path.join(b.outdir, name), 'rb').read(), name


def test_a_degenerate_model_is_warned_about(tmp_path):
    """Every vector landing on one class would pass a testbench against a design whose argmax,
    popcount and grouping were all wrong. Reported, not raised -- a lopsided model is legitimate.
    Degeneracy is CONSTRUCTED, not fished for with seeds.
    """
    import torch
    ck = fixtures.make('tiny')
    last = max(int(k.split('.')[0]) for k in ck['state_dict'] if k.endswith('.luts'))
    ck['state_dict'][f'{last}.luts'] = torch.ones_like(ck['state_dict'][f'{last}.luts'])

    assert fixtures.classes_hit(ck) == 1, 'the construction failed to degenerate the model'

    r = build(ck, str(tmp_path / 'rtl'), input_bits=8)
    assert r.degenerate
    assert any('ONE class' in w for w in r.warnings)


# ---------------------------------------------------------------------------------------
# THE GATE
# ---------------------------------------------------------------------------------------

def run_gate(outdir, testbench):
    """Compile and run one level, from inside the output directory.

    cwd matters: the testbench does $readmemh("x_quant.hex") and `include "top_params.vh",
    both resolved against the simulator's working directory, not against the source file.
    """
    sim = SIMULATOR
    sources = [f for f in os.listdir(outdir) if f.endswith('.v')]
    out = f'{os.path.basename(testbench)}.vvp'

    compile_ = subprocess.run([sim.compiler, '-o', out, *sources, testbench],
                              cwd=outdir, capture_output=True, text=True)
    assert compile_.returncode == 0, f'iverilog failed:\n{compile_.stderr}'

    run = subprocess.run([sim.runner, out], cwd=outdir, capture_output=True, text=True)
    assert run.returncode == 0, f'vvp failed:\n{run.stderr}'
    return run.stdout


@pytest.mark.sim
@pytest.mark.parametrize('shape', ALL_SHAPES)
def test_gate_core_is_bit_exact(shape, tmp_path):
    """GATE 1. The emitted core against the numpy golden model, every vector, no exceptions.

    This is the only correctness signal in the project: there is no second independent
    implementation to cross-check against.
    """
    r = build(fixtures.make(shape), str(tmp_path / 'rtl'), input_bits=8)
    stdout = run_gate(r.outdir, os.path.join('tb', 'dwn_core_tb.v'))

    assert 'PASS (bit-exact on every vector)' in stdout, stdout
    assert 'mismatches     : 0' in stdout, stdout


@pytest.mark.sim
@pytest.mark.parametrize('shape', ALL_SHAPES)
def test_gate_top_is_bit_exact(shape, tmp_path):
    """GATE 1, top level: quantized features through the encoder and the core. This is the half
    that checks the encoder, which is most of an unverified design when it is unverified.
    """
    r = build(fixtures.make(shape), str(tmp_path / 'rtl'), input_bits=8)
    stdout = run_gate(r.outdir, os.path.join('tb', 'dwn_top_tb.v'))

    assert 'PASS (bit-exact on every vector)' in stdout, stdout
    assert 'mismatches     : 0' in stdout, stdout
    assert f'vectors tested : {r.vectors["top"]["count"]}' in stdout


@pytest.mark.sim
def test_the_two_levels_test_different_things(tmp_path):
    """The split is what makes a failure localize itself, so the two must not be the same run.

    Different vector counts prove it: the core level drives pre-binarized bits, the top level
    drives quantized features and gets per-feature threshold edge cases the core cannot have.
    """
    r = build(fixtures.make('n6'), str(tmp_path / 'rtl'), input_bits=8)
    core = run_gate(r.outdir, os.path.join('tb', 'dwn_core_tb.v'))
    top = run_gate(r.outdir, os.path.join('tb', 'dwn_top_tb.v'))

    assert 'PASS' in core and 'PASS' in top
    assert r.vectors['core']['count'] != r.vectors['top']['count']
    assert 'dwn_core vs golden' in core
    assert 'dwn_top (encoder + core)' in top


@pytest.mark.sim
def test_a_broken_encoder_fails_only_the_top_gate(tmp_path):
    """The localization claim, proved rather than asserted. Corrupt a comparator constant: the core
    must still PASS and the top must FAIL, or the two-testbench split buys nothing.
    """
    r = build(fixtures.make('n6'), str(tmp_path / 'rtl'), input_bits=8)
    enc = os.path.join(r.outdir, 'thermometer_encoder.v')

    import re
    src = open(enc).read()
    # Push one threshold far out of range so that comparator's output flips for every input.
    broken, count = re.subn(r"> \$signed\(\d+'sd\d+\)", "> $signed(9'sd255)", src, count=1)
    assert count == 1 and broken != src
    open(enc, 'w').write(broken)

    assert 'PASS' in run_gate(r.outdir, os.path.join('tb', 'dwn_core_tb.v')), \
        'a broken encoder must not affect the core gate'
    top = run_gate(r.outdir, os.path.join('tb', 'dwn_top_tb.v'))
    assert 'FAIL' in top, f'a broken encoder passed the top gate:\n{top}'
    assert 'fault is in the thermometer encoder' in top


@pytest.mark.sim
def test_gate_reports_a_real_failure(tmp_path):
    """The gate must be capable of FAILING, or a PASS means nothing. One truth table is corrupted
    after emission and the gate is required to notice.
    """
    r = build(fixtures.make('tiny'), str(tmp_path / 'rtl'), input_bits=8)
    core = os.path.join(r.outdir, 'dwn_core.v')

    src = open(core).read()

    # Invert every table, in place, keeping the literal EXACTLY 16 hex digits.
    #
    # ⚠️ The obvious corruption does not work, and finding out why was worth the detour. The
    # first attempt here was `64'h` -> `64'hF`, which produces a SEVENTEEN-digit literal in a
    # 64-bit parameter -- and Verilog silently truncates the excess high digit, restoring the
    # original 16 and leaving the design bit-identical. The gate passed, correctly.
    #
    # That is precisely the silent truncation emit_core.py's MAX_N assertion exists to prevent
    # at n>6, reproduced by accident in a test. A too-wide constant does not error; it quietly
    # becomes a different, valid one.
    def invert(m):
        return f"64'h{(~int(m.group(1), 16)) & 0xFFFFFFFFFFFFFFFF:016X}"

    import re
    broken, count = re.subn(r"64'h([0-9A-F]{16})", invert, src)
    assert count > 0 and broken != src, 'corruption did not change the file'
    open(core, 'w').write(broken)

    stdout = run_gate(r.outdir, os.path.join('tb', 'dwn_core_tb.v'))
    assert 'FAIL' in stdout, f'a corrupted design still passed:\n{stdout}'


def test_out_pointing_at_a_file_is_refused_not_an_oserror(tmp_path):
    """os.makedirs raised a raw FileExistsError/WinError, which names the syscall rather than
    the mistake."""
    target = tmp_path / 'notadir'
    target.write_text('x', encoding='utf-8')

    with pytest.raises(NotADirectoryError, match='is a file, not a directory'):
        build(fixtures.make('tiny'), str(target), input_bits=8)


@pytest.mark.parametrize('width', [1, 3, 7, 8, 12, 15, 16, 18, 24, 31, 33])
def test_binarized_vectors_survive_any_width(width):
    """value = sum(row[i] << i), at every width -- not just the byte-aligned ones.

    np.packbits pads a partial byte on the LOW side, which multiplied 12-bit vectors by 16 and
    18-bit ones by 64 while leaving 8, 16 and 24 correct.
    """
    import numpy as np
    from dwn2rtl.vectors import bits_to_hex

    row = np.random.default_rng(width).integers(0, 2, size=width).astype(bool)
    expected = sum(int(b) << i for i, b in enumerate(row))
    assert int(bits_to_hex(row, width), 16) == expected


def test_a_stale_input_scaling_is_removed_on_rebuild(tmp_path):
    """⚠️ Every other emitted file is overwritten; this one is conditional, so it survived.

    Building an unscaled model over a scaled one in the same directory left the PREVIOUS model's
    mean and scale behind, and a user following them applies another model's transformation --
    the design then runs at chance and looks healthy doing it.
    """
    import json

    out = str(tmp_path / 'rtl')
    scaled = fixtures.make('tiny')
    n = scaled['thermometer']['thresholds'].shape[0]
    scaled['scaler'] = {'mean': [7.0] * n, 'scale': [2.0] * n}
    build(scaled, out, input_bits=8)

    path = os.path.join(out, 'input_scaling.json')
    assert json.load(open(path))['mean'] == [7.0] * n

    r = build(fixtures.make('tiny'), out, input_bits=8)          # no scaler this time
    assert not os.path.exists(path), 'the previous model\'s scaling survived the rebuild'
    assert any('stale input_scaling.json' in w for w in r.warnings), \
        'removing it silently is not enough -- the user has to know it was there'


@pytest.mark.parametrize('run_name', [
    'line one\nline two',            # escaped the // comment -> uncompilable Verilog
    'caf\u00e9 \u65e5\u672c\u8a9e',  # crashed the BUILD with UnicodeEncodeError on cp1252
    'x' * 5000,
    '`define EVIL 1',
    '',
])
def test_a_hostile_run_name_cannot_break_the_design(run_name, tmp_path):
    """⚠️ run_name reaches a header comment and nothing else, so it must never be able to stop a
    design compiling. A newline escaped the comment; non-ASCII killed the build outright."""
    ck = fixtures.make('tiny')
    ck['run_name'] = run_name
    outdir = build(ck, str(tmp_path / 'rtl'), input_bits=8).outdir

    core = open(os.path.join(outdir, 'dwn_core.v'), encoding='utf-8').read()
    assert core.isascii(), 'emitted Verilog must stay ASCII'
    assert '\n// run' in core
    for line in core.splitlines():
        assert len(line) < 200, 'a header line grew without bound'


@pytest.mark.parametrize('results', [
    {'final_acc': 'ninety'},
    {'final_acc': None},
    {'final_acc': 0.9, 'best_acc': [1, 2]},
    {'final_acc': float('nan')},
])
def test_odd_results_metadata_cannot_break_the_build(results, tmp_path):
    """`final_acc` was formatted with :.4f, so a string or None raised TypeError from a code
    generator -- over a comment that no hardware depends on."""
    ck = fixtures.make('tiny')
    ck['results'] = results
    outdir = build(ck, str(tmp_path / 'rtl'), input_bits=8).outdir
    assert 'accuracy' in open(os.path.join(outdir, 'dwn_core.v'), encoding='utf-8').read()


def test_public_arguments_name_the_argument_not_an_internal(tmp_path):
    """Every one of these named a library internal before: a dict pipeline gave "'dict' object
    has no attribute 'lut'", and a string n_random reached numpy as a UFuncTypeError."""
    from dwn2rtl import Pipeline

    ck = fixtures.make('tiny')
    with pytest.raises(TypeError, match='must be a dwn2rtl.Pipeline'):
        build(ck, str(tmp_path / 'a'), input_bits=8, pipeline={'enc': 1})
    with pytest.raises(TypeError, match='n_random must be an int'):
        build(ck, str(tmp_path / 'b'), input_bits=8, n_random='500')
    with pytest.raises(ValueError, match='n_random must be >= 0'):
        build(ck, str(tmp_path / 'c'), input_bits=8, n_random=-5)

    # and the valid forms still work
    assert build(ck, str(tmp_path / 'd'), input_bits=8, pipeline=Pipeline(enc=0)).outdir


@pytest.mark.sim
@pytest.mark.parametrize('depths', [
    (0, 0, 0, 0),      # fully combinational -- LATENCY 0, which the testbench mis-sampled
    (1, 0, 0, 0),      # register the encoder only: a plausible low-latency build
    (0, 1, 0, 0),
    (1, 2, 1, 1),      # ⚠️ lut=2: pipe_reg inserted ONE register for any non-zero count
    (2, 3, 2, 2),
    (8, 8, 8, 8),
])
def test_the_gate_passes_at_every_pipeline_depth(depths, tmp_path):
    """⚠️ Only the default (1,1,1,1) was ever gated. `pipe_reg` treated its parameter as a FLAG
    while `Pipeline` documented a stage COUNT and latency() summed counts, so any depth above 1
    claimed a latency the hardware did not have. And at zero latency the testbench compared
    before driving, so a combinational design was checked one step early.
    """
    from dwn2rtl import Pipeline
    from dwn2rtl.verify import verify

    enc, lut, pop, out = depths
    r = build(fixtures.make('tiny'), str(tmp_path / 'rtl'), input_bits=8,
              pipeline=Pipeline(enc=enc, lut=lut, pop=pop, out=out))
    report = verify(r.outdir)
    assert report.ok, f'depths={depths}\n' + '\n'.join(report.lines())


@pytest.mark.sim
def test_the_testbench_is_still_sensitive_to_latency(tmp_path):
    """⚠️ The guard on the fix above. Driving before comparing must not make the comparison
    latency-blind -- a testbench that passes whatever the pipeline does is worse than none."""
    import re

    from dwn2rtl.verify import verify

    outdir = build(fixtures.make('tiny'), str(tmp_path / 'rtl'), input_bits=8).outdir
    path = os.path.join(outdir, 'dwn_core_params.vh')
    src = open(path).read()
    real = int(re.search(r'DWN_CORE_LATENCY (\d+)', src).group(1))

    for claimed in (real - 1, real + 1):
        open(path, 'w').write(re.sub(r'(DWN_CORE_LATENCY )\d+', rf'\g<1>{claimed}', src))
        assert not verify(outdir, levels=('core',)).ok, \
            f'a design claiming latency {claimed} instead of {real} was reported as correct'


def test_the_emitted_scaling_file_is_strict_json(tmp_path):
    """⚠️ Bare NaN and Infinity are Python's json.dump, not JSON. A user's toolchain reading
    input_scaling.json with a strict parser -- JavaScript, Go, Rust -- would fail on a file we
    told them to read."""
    import json

    ck = fixtures.make('tiny')
    n = ck['thermometer']['thresholds'].shape[0]
    ck['scaler'] = {'mean': [0.5] * n, 'scale': [2.0] * n}
    outdir = build(ck, str(tmp_path / 'rtl'), input_bits=8).outdir

    raw = open(os.path.join(outdir, 'input_scaling.json')).read()

    def reject(c):
        raise ValueError(f'bare {c} is not valid JSON')

    json.loads(raw, parse_constant=reject)
