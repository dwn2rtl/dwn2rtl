"""The precision policy -- the one part of the tool that already worked in phase 0.

The rule under test (precision.py, roadmap Q9): integer width is DERIVED exactly from the
thresholds, fractional width is ASKED FOR as `--input-bits`, and the two are never confused. The
study repo paid for that distinction: a narrowing fitted and validated on the same 1,000 samples
put 8 of 15 features too narrow.

The numbers below are not invented. Where a case reproduces a real model's known format it says
so, because a policy that agrees with two independently-derived formats is evidence and a policy
that agrees with none is a guess.
"""

import numpy as np
import pytest

from dwn2rtl import Pipeline, Precision, precision_for, required_int_bits
from dwn2rtl.config import BuildConfig
from dwn2rtl.precision import DEFAULT_CONTINUOUS_FRAC_BITS, comparator_merge_floor


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
    """8-bit pixels -> frac=8. Provably lossless, not merely measured.

    Values are k/255 and quantising at frac=8 computes floor(k*256/255), which is strictly
    increasing over k=0..255. Order is preserved exactly, and every encoder bit is an order
    comparison, so no bit can change. The study repo's 0-of-10,000 divergence is what a proof
    predicts, not a lucky measurement.
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
    """Two thresholds that quantise to the same integer become the SAME comparison.

    Not a bug -- the hardware is smaller and the encoder still separates everything the format
    can separate -- but it is a silent accuracy change, so it is reported. At frac=2, feature 0's
    0.10 and 0.11 both floor to 0 and merge; feature 1's 0.5 and 0.9 stay distinct.
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
