"""The recipes docs/user-guide.md hands to users, executed.

The guide's re-labelling recipe uses `dwn2rtl.extract` and `dwn2rtl.vectors` -- internal modules
with no API promise -- and claims it "is tested". It was not, so it is now: a rename there breaks
this rather than silently invalidating published instructions.

Transcribed as literally as a test can manage; a tidied version would stop pinning the text.
"""

import json
import os

import numpy as np
import pytest

import fixtures
from dwn2rtl.build import build


def _n_features(ck):
    """(features, z) -- the feature count is never a literal anywhere in this project, because
    a literal is how one dataset's shape gets baked into every design (extract.py)."""
    return ck['thermometer']['thresholds'].shape[0]


def _checkpoint_with_a_scaler():
    """The recipe reads input_scaling.json, which is only emitted for a scaled model."""
    ck = fixtures.make('n6')
    n = _n_features(ck)
    ck['scaler'] = {'mean': [0.25] * n, 'scale': [4.0] * n}
    return ck


def _relabel(outdir, ck, x_raw):
    """docs/user-guide.md, "Or re-label the emitted design with your own samples", verbatim."""
    from dwn2rtl.extract import (extract_tables, extract_wiring, layer_indices,
                                 encode, forward, quantize, quantize_thresholds)
    from dwn2rtl.vectors import words_to_hex, write_lines

    OUT = outdir
    cfg = ck['config']
    n, num_classes = cfg['n'], cfg['num_classes']
    sd = ck['state_dict']
    layers = [(extract_tables(sd, i), *extract_wiring(sd, i, n)) for i in layer_indices(sd)]

    meta = json.load(open(os.path.join(OUT, 'input_scaling.json')))
    frac_bits, word_bits = meta['frac_bits'], meta['word_bits']
    mean = np.asarray(meta['mean'], dtype=np.float64)
    scale = np.asarray(meta['scale'], dtype=np.float64)

    x = (x_raw - mean) / scale
    xq = quantize(x, frac_bits, word_bits)
    thr_q = quantize_thresholds(ck['thermometer']['thresholds'].numpy(), frac_bits)
    y = forward(encode(xq, thr_q), layers, num_classes)[0]

    write_lines(os.path.join(OUT, 'x_quant.hex'), [words_to_hex(r, word_bits) for r in xq])
    write_lines(os.path.join(OUT, 'expected_top.hex'), [f'{int(v):X}' for v in y])
    write_lines(os.path.join(OUT, 'top_params.vh'), [
        '// regenerated from my own data',
        f'`define N_TOP {xq.shape[0]}',
        f'`define X_W {xq.shape[1] * word_bits}',
        f'`define IDX_W {max(1, int(np.ceil(np.log2(num_classes))))}',
    ])
    return xq


def test_the_relabelling_recipe_runs(tmp_path):
    """Every internal name the guide imports exists and still takes these arguments.

    Cheap, needs no simulator, and it is the half that rots: these are internal modules, so
    nothing else stops a rename from silently invalidating the published recipe.
    """
    ck = _checkpoint_with_a_scaler()
    outdir = build(ck, str(tmp_path / 'rtl'), input_bits=8).outdir
    assert os.path.exists(os.path.join(outdir, 'input_scaling.json'))

    rng = np.random.default_rng(0)
    n_features = _n_features(ck)
    xq = _relabel(outdir, ck, rng.uniform(-1.0, 1.5, size=(64, n_features)))

    assert xq.shape == (64, n_features)
    assert '`define N_TOP 64' in open(os.path.join(outdir, 'top_params.vh')).read()


@pytest.mark.sim
def test_the_relabelled_design_still_passes_the_gate(tmp_path):
    """The guide says a PASS here means YOUR data was reproduced bit-exactly. Measured: quantising
    thresholds one bit off gives 47 mismatches, so this has teeth.

    ⚠️ It cannot catch a missing scaling -- both sides are built from the same xq, so they agree
    on whatever inputs are supplied. No simulator can prove the inputs were the ones you meant.
    """
    from dwn2rtl.verify import verify

    ck = _checkpoint_with_a_scaler()
    outdir = build(ck, str(tmp_path / 'rtl'), input_bits=8).outdir

    rng = np.random.default_rng(1)
    n_features = _n_features(ck)
    _relabel(outdir, ck, rng.uniform(-1.0, 1.5, size=(200, n_features)))

    report = verify(outdir)
    assert report.ok, '\n'.join(report.lines())

    top = next(r for r in report.levels if r.level == 'top')
    assert top.vectors == 200, f'the testbench did not run the 200 supplied vectors: {top}'
    assert top.mismatches == 0
