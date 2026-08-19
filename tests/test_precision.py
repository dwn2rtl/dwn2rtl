"""The precision policy: integer width DERIVED from the thresholds, fractional width ASKED for.

The two are never confused, and the distinction was paid for -- a narrowing fitted and validated
on the same 1,000 samples put 8 of 15 features too narrow.

Where a case reproduces a real model's known format it says so: a policy agreeing with two
independently-derived formats is evidence, one agreeing with none is a guess.
"""

import numpy as np
import pytest

from dwn2rtl import Pipeline, Precision, precision_for, required_int_bits
from dwn2rtl.config import BuildConfig
from dwn2rtl.precision import (DEFAULT_CONTINUOUS_FRAC_BITS, comparator_merge_floor,
                               infer_input_bits)


# --------------------------------------------------------------------------------------
# required_int_bits -- exact, not a heuristic
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize('span, expected', [
    (0.0,   0),
    (0.5,   0),
    (0.9,   0),      # MNIST at z<=8: pixels in [0,1], no threshold reaches 1.0 -> Q0.x
    (1.0,   1),      # MNIST at z=25: one quantile threshold lands on EXACTLY 1.0 -> Q1.x.
                     # The boundary is `<=`, so 1.0 costs an integer bit. Roadmap Q9.
    (1.5,   1),
    (3.99,  2),
    (4.0,   3),
    (7.9,   3),      # JSC: extreme thresholds -4.55 / +4.34 -> Q3.x
    (8.0,   4),
])
def test_required_int_bits_is_exact_at_the_boundary(span, expected):
    assert required_int_bits(np.array([[span]])) == expected


def test_required_int_bits_uses_the_largest_magnitude_either_way():
    """A negative threshold is just as far from zero as a positive one."""
    assert required_int_bits(np.array([[-4.55, 0.1, 4.34]])) == 3
    assert required_int_bits(np.array([[-7.99, 0.0]])) == 3
    assert required_int_bits(np.array([[-8.01, 0.0]])) == 4


# --------------------------------------------------------------------------------------
# precision_for -- the --input-bits policy
# --------------------------------------------------------------------------------------

def test_input_bits_gives_a_proved_format():
    """8-bit pixels -> frac=8, provably lossless. Values are k/255 and quantising at frac=8 is
    strictly increasing, so order is preserved and no order comparison can change.
    """
    thr = np.array([[0.1, 0.5, 0.9]])            # MNIST-shaped: everything under 1.0
    p = precision_for(thr, input_bits=8)
    assert (p.int_bits, p.frac_bits, p.word_bits) == (0, 8, 9)
    assert p.proved is True
    assert str(p) == 'Q0.8 signed (9-bit)'       # MNIST's known format


def test_no_input_bits_gives_an_unproved_default():
    """A continuous input has no native quantum, so there is no proof to be had -- only a
    default and a measurement. `proved` is what keeps a report from calling both "safe"."""
    thr = np.array([[-4.55, 4.34]])              # JSC-shaped
    p = precision_for(thr)
    assert (p.int_bits, p.frac_bits, p.word_bits) == (3, 12, 16)
    assert p.proved is False
    assert str(p) == 'Q3.12 signed (16-bit)'     # JSC's known format
    assert p.frac_bits == DEFAULT_CONTINUOUS_FRAC_BITS


def test_integer_width_is_always_derived_never_asked_for():
    """--input-bits sets the FRACTIONAL half only. The integer half comes from the thresholds,
    so the returned format cannot fail to represent the model it was built for."""
    thr = np.array([[-4.55, 4.34]])
    for input_bits in (4, 8, 12, 16):
        p = precision_for(thr, input_bits=input_bits)
        assert p.int_bits == required_int_bits(thr) == 3
        assert p.frac_bits == input_bits


def test_input_bits_zero_is_honoured_not_treated_as_absent():
    """0 is a legitimate answer (integer-valued input) and must not be confused with None.
    `if input_bits:` instead of `is not None` would silently give a 12-bit default here."""
    p = precision_for(np.array([[3.0]]), input_bits=0)
    assert p.frac_bits == 0
    assert p.proved is True


# --------------------------------------------------------------------------------------
# Inferring the input's native quantum from the thresholds
# --------------------------------------------------------------------------------------
#
# WHY THIS IS POSSIBLE: a thermometer's thresholds are QUANTILES OF THE TRAINING DATA, so if
# that data was quantised its quantiles inherit the same grid. The checkpoint carries a
# fingerprint of the input's precision.
#
# It does not contradict roadmap Q9. Q9 says you cannot tell from a checkpoint whether a width
# is SAFE FOR YOUR DATA, which remains true. What is recovered is the input's native QUANTUM,
# which when it exists makes that question vanish rather than answering it.

# ⚠️ These use CONSECUTIVE k, deliberately, and the first version of them did not.
#
# It sampled `arange(0, 2**n, stride)`, so every k shared a factor with the stride -- and values
# that are all multiples of 40 over 4096 genuinely lie on the coarser m/512 grid. The inference
# returned 9 instead of 12 and was RIGHT; the test data was wrong. Consecutive k including k=1
# pins the grid exactly, because 1/(2**n - 1) lies on no coarser one.
#
# Worth keeping as a warning: a subsampled grid is a different, coarser grid, and that is true
# of real data too. A sensor read every 4th count is effectively lower precision.

@pytest.mark.parametrize('n', [4, 8, 12, 16])
def test_a_scaled_integer_grid_is_recognised(n):
    """k/(2**n - 1) -- images and most normalised data."""
    k = np.arange(min(64, 2 ** n))
    assert infer_input_bits(k / (2 ** n - 1)) == n


@pytest.mark.parametrize('n', [4, 8, 12])
def test_a_raw_fixed_point_grid_is_recognised(n):
    """k/2**n -- audio and many ADC pipelines. Both grids occur, so both are checked."""
    k = np.arange(min(64, 2 ** n))
    assert infer_input_bits(k / (2 ** n)) == n


def test_the_smallest_grid_wins_and_this_is_load_bearing():
    """Every coarse grid is a subset of every finer one: 8-bit data satisfies k/255 AND
    257k/65535. Returning any but the smallest gives a needlessly wide word for no gain --
    measured on the real MNIST checkpoint, which matches at both n=8 and n=16."""
    k = np.arange(256)
    thr = (k / 255).reshape(16, 16)
    assert infer_input_bits(thr) == 8


def test_continuous_data_infers_nothing():
    """Standard-scaled features have no quantum. A false positive here would silently narrow
    the word and change encoder bits, so the failure mode has to be 'no answer', never a
    plausible wrong one."""
    thr = np.random.default_rng(0).uniform(-5, 5, (16, 16))
    assert infer_input_bits(thr) is None


def test_small_magnitude_values_do_not_match_every_coarse_grid():
    """The dangerous false positive: values of ~0.001 all round to 0 on a coarse grid, so a naive
    tolerance infers n=1 -- a ONE-BIT word labelled provably lossless. The fix is requiring the
    grid to SEPARATE the thresholds, not merely sit near them.
    """
    tiny = np.arange(64) / 65535          # every value < 0.001
    assert infer_input_bits(tiny) == 16, 'inferred a coarse grid for small-magnitude values'

    # And directly: a grid that collapses distinct thresholds together is not their quantum.
    collapsing = np.array([0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008])
    assert infer_input_bits(collapsing, max_bits=4) is None


def test_too_few_thresholds_infer_nothing():
    """Four values lie on some coarse grid by coincidence; a hundred do not. Below the minimum
    a match is not evidence, so no claim is made."""
    assert infer_input_bits(np.array([[0.0, 0.5, 1.0]])) is None


def test_inference_is_used_when_no_flag_is_given():
    """The zero-friction path: `dwn2rtl build model.pt` and nothing else."""
    thr = (np.arange(256) / 255).reshape(16, 16)
    p = precision_for(thr)
    assert (p.frac_bits, p.source, p.proved) == (8, 'inferred', True)


def test_an_explicit_flag_overrides_inference():
    """The user may know their deployment differs from their training data."""
    thr = (np.arange(256) / 255).reshape(16, 16)
    p = precision_for(thr, input_bits=12)
    assert (p.frac_bits, p.source) == (12, 'given')


def test_inference_can_be_switched_off():
    thr = (np.arange(256) / 255).reshape(16, 16)
    p = precision_for(thr, infer=False)
    assert (p.frac_bits, p.source) == (DEFAULT_CONTINUOUS_FRAC_BITS, 'default')


def test_inferred_and_given_are_equally_proved_but_distinguishable():
    """Both rest on the same assumption -- that inference-time data shares training-time
    precision -- so inference is no weaker than the flag it replaces. But a report must be able
    to say which happened, because only one of them was the user's own claim."""
    grid = (np.arange(256) / 255).reshape(16, 16)
    inferred = precision_for(grid)
    given = precision_for(grid, input_bits=8)

    assert inferred.proved and given.proved
    assert inferred.frac_bits == given.frac_bits == 8
    assert inferred.source != given.source


# --------------------------------------------------------------------------------------
# Precision -- a format that cannot represent itself is refused at construction
# --------------------------------------------------------------------------------------

def test_precision_rejects_a_word_too_narrow_for_its_own_fraction():
    with pytest.raises(ValueError, match='cannot hold'):
        Precision(word_bits=8, frac_bits=8)      # no room for the sign bit


def test_precision_rejects_negative_fraction():
    with pytest.raises(ValueError, match='frac_bits'):
        Precision(word_bits=8, frac_bits=-1)


def test_precision_is_frozen():
    p = Precision(word_bits=9, frac_bits=8)
    with pytest.raises(Exception):
        p.word_bits = 16


# --------------------------------------------------------------------------------------
# comparator_merge_floor -- reported, never enforced
# --------------------------------------------------------------------------------------

def test_merge_floor_counts_collapsed_comparators_per_feature():
    """Thresholds quantising to the same integer become one comparison. Not a bug, but a silent
    accuracy change, so it is reported. Merging is per feature, never across them.
    """
    thr = np.array([[0.10, 0.11], [0.5, 0.9]])
    distinct, total, collapsed = comparator_merge_floor(thr, Precision(word_bits=8, frac_bits=2))
    assert (distinct, total, collapsed) == (3, 4, 1)


def test_merge_floor_does_not_merge_across_features():
    """Two features sharing a threshold VALUE are still two comparators -- they compare
    different input words. Counting unique values globally would under-report the encoder."""
    thr = np.array([[0.5], [0.5], [0.5]])
    distinct, total, collapsed = comparator_merge_floor(thr, Precision(word_bits=8, frac_bits=4))
    assert (distinct, total, collapsed) == (3, 3, 0)


def test_merge_floor_reports_rather_than_raises():
    """JSC is why this is a warning: the rule demands 22 bits and 12 is fine, because
    thresholds 3.6e-7 apart have essentially no data between them. Enforcing it would reject a
    format that two completed studies proved correct."""
    thr = np.array([[0.1, 0.1 + 3.6e-7]])
    distinct, total, collapsed = comparator_merge_floor(thr, Precision(word_bits=16, frac_bits=12))
    assert collapsed == 1                        # it merged, and nothing raised
    assert (distinct, total) == (1, 2)


# --------------------------------------------------------------------------------------
# Pipeline / BuildConfig
# --------------------------------------------------------------------------------------

def test_default_pipeline_latency_matches_the_shipped_depth():
    """enc + lut*layers + pop + out, all 1 by default. The study repo's shipped depth."""
    assert Pipeline().latency(1) == 4
    assert Pipeline().latency(2) == 5


def test_disabled_stages_drop_out_of_the_latency():
    assert Pipeline(enc=0, lut=0, pop=0, out=0).latency(3) == 0
    assert Pipeline(enc=1, lut=0, pop=1, out=1).latency(3) == 3


def test_pipeline_rejects_negative_depth():
    with pytest.raises(ValueError, match='must be >= 0'):
        Pipeline(enc=-1)


def test_buildconfig_creates_its_output_directory(tmp_path):
    out = tmp_path / 'nested' / 'rtl'
    cfg = BuildConfig(outdir=str(out), precision=precision_for(np.array([[0.9]]), input_bits=8))
    assert cfg.ensure_outdir() == str(out)
    assert out.is_dir()
    cfg.ensure_outdir()                          # idempotent


def test_buildconfig_has_no_default_precision():
    """Deliberate: a default here would be one model's format silently applied to another,
    which is the defect that cost the study repo six separate crashes."""
    with pytest.raises(TypeError):
        BuildConfig(outdir='x')


def test_an_absurd_input_width_is_refused_not_an_overflow():
    """--input-bits 999 reached numpy as "OverflowError: int too big to convert" from inside the
    golden model. Quantised values are int64, so the word has a real ceiling."""
    import pytest
    from dwn2rtl.precision import MAX_WORD_BITS, Precision

    with pytest.raises(ValueError, match=str(MAX_WORD_BITS)):
        Precision(word_bits=MAX_WORD_BITS + 1, frac_bits=8, source='given')

    # The boundary itself must still be allowed.
    assert Precision(word_bits=MAX_WORD_BITS, frac_bits=8, source='given').word_bits == MAX_WORD_BITS
