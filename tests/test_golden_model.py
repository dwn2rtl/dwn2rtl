"""The golden model, cross-checked against a second implementation.

⚠️ THE ONE FAILURE THE GATE CANNOT DETECT. `verify` proves the RTL matches extract.forward().
If extract.forward() is itself wrong, the RTL matches it perfectly and both are wrong together
-- verify.py's own docstring says there is no second implementation to cross-check against.

`naive_reference.py` is that second implementation: written from docs/checkpoint-format.md and
structured to share as little as possible with the original -- explicit per-node loops and bit
shifts where extract.py gathers and vectorises. It cannot catch a shared misreading of upstream
(the pinned commit covers that), but it does catch transcription errors: a reversed address, a
wrong group boundary, an off-by-one in a table index.
"""

import numpy as np
import pytest

import fixtures
from dwn2rtl.extract import encode, forward, quantize, quantize_thresholds
from naive_reference import naive_encode, naive_forward

ALL_SHAPES = sorted(fixtures.SHAPES)
SAMPLES = 200


@pytest.mark.parametrize('shape', ALL_SHAPES)
def test_the_forward_pass_agrees_with_an_independent_implementation(shape):
    ck = fixtures.make(shape)
    layers = fixtures.extract_layers(ck)
    width_in = ck['thermometer']['thresholds'].numel()

    x_bits = np.random.default_rng(1).integers(
        0, 2, size=(SAMPLES, width_in)).astype(bool)

    mine, my_scores = forward(x_bits, layers, ck['config']['num_classes'])
    theirs, their_scores = naive_forward(x_bits, layers, ck['config']['num_classes'])

    assert np.array_equal(my_scores, their_scores), 'group sums disagree'
    assert np.array_equal(mine, theirs), 'predicted classes disagree'


@pytest.mark.parametrize('shape', ALL_SHAPES)
def test_ties_are_actually_exercised_by_that_comparison(shape):
    """⚠️ The tie-break is the subtlest rule in the model -- numpy, torch and the RTL all keep
    the LOWEST index -- so a comparison that never hits a tie proves little about it.

    Measured rather than hoped: every shape ties on a quarter to a half of random vectors.
    """
    ck = fixtures.make(shape)
    _, scores = forward(
        np.random.default_rng(1).integers(
            0, 2, size=(SAMPLES, ck['thermometer']['thresholds'].numel())).astype(bool),
        fixtures.extract_layers(ck), ck['config']['num_classes'])

    top = scores.max(axis=1, keepdims=True)
    tied = int(((scores == top).sum(axis=1) > 1).sum())
    assert tied > SAMPLES // 20, f'only {tied}/{SAMPLES} vectors tie; the rule is untested'


@pytest.mark.parametrize('shape', ALL_SHAPES)
def test_the_encoder_agrees_with_an_independent_implementation(shape):
    """Every encoder bit is a strict `x > t`, laid out feature-major."""
    ck = fixtures.make(shape)
    thr = ck['thermometer']['thresholds'].numpy()
    thr_q = quantize_thresholds(thr, 8)

    rng = np.random.default_rng(2)
    xq = quantize(rng.uniform(-1.0, 2.0, size=(SAMPLES, thr.shape[0])), 8, 16)

    assert np.array_equal(encode(xq, thr_q), naive_encode(xq, thr_q))


def test_the_reference_can_disagree():
    """⚠️ The guard on all of the above. A reference that agrees no matter what proves nothing,
    so it must notice a deliberately corrupted table."""
    ck = fixtures.make('tiny')
    layers = fixtures.extract_layers(ck)
    x_bits = np.random.default_rng(3).integers(
        0, 2, size=(SAMPLES, ck['thermometer']['thresholds'].numel())).astype(bool)

    corrupt = [(t.copy(), w, k) for t, w, k in layers]
    # The FINAL layer, so the damage reaches a group sum directly. A first-layer flip can wash
    # out before the argmax, which is a fact about this model rather than about the reference.
    corrupt[-1][0][0, :] = ~corrupt[-1][0][0, :]

    _, my_scores = forward(x_bits, corrupt, ck['config']['num_classes'])
    _, their_scores = naive_forward(x_bits, layers, ck['config']['num_classes'])
    assert not np.array_equal(my_scores, their_scores),         'the reference agrees with a corrupted model, so it proves nothing'


def test_a_table_entry_of_exactly_zero_emits_zero():
    """⚠️ Found by mutation testing: `luts > 0.0` could be relaxed to `>= 0.0` and the whole
    suite still passed.

    docs/checkpoint-format.md §1 says the hardware bit is STRICTLY greater, and extract.py's own
    comment says "an exact 0.0 emits 0 -- unlikely in a trained model, but Gate 1 covers edge
    cases". Gate 1 did not: fixtures draw uniform floats, which never land on exactly 0.0, so
    nothing ever exercised the boundary the rule is about.
    """
    import torch

    from dwn2rtl.extract import extract_tables

    sd = {'0.luts': torch.tensor([[0.0, -1.0, 1e-9, -1e-9]])}
    assert extract_tables(sd, 0)[0].tolist() == [False, False, True, False]


def test_quantisation_truncates_rather_than_rounds():
    """⚠️ Also found by mutation testing: floor could become round with nothing failing.

    The gate cannot see this by construction -- it compares RTL against the golden model, and
    both sides use this same function, so they agree whichever way it rounds. It matters anyway,
    because the emitted design's USER quantises their own inputs: `input_scaling.json` tells them
    to, and a different rounding mode puts boundary features on the wrong side of a comparator.
    """
    import numpy as np

    from dwn2rtl.extract import quantize

    x = np.array([[0.9, -0.5, 1.5, -1.5, 0.5]])
    assert quantize(x, 0, 8).tolist() == [[0, -1, 1, -2, 0]], 'must floor, including negatives'

    # and at a real fractional width, a value just under a step must not round up into it
    assert quantize(np.array([[(2 ** 8 - 1) / 2 ** 8]]), 8, 16).tolist() == [[255]]
