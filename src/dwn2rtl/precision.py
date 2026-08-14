"""How wide the encoder's input word is, and where that number comes from.

Extracted from the study repo's `datasets/__init__.py`, which carried per-dataset fixed-point
formats as data. This tool has no datasets, so the format has to be DERIVED or ASKED FOR. The
policy below is the whole reason this is its own module.

THE RULE: never ask a user for fractional bits.

The two halves of a fixed-point word are not equally knowable.

  integer bits    DERIVABLE, exactly, from the thresholds. A threshold outside the
                  representable range makes every comparison against it meaningless, so this
                  is a hard floor and required_int_bits() computes it.

  fractional bits NOT derivable from the checkpoint. How much precision the input needs depends
                  on whether quantisation changes predictions, which depends on the DATA. The
                  study repo learned this the expensive way: a narrowing was fitted and
                  validated on the same 1,000 samples, and 8 of 15 features came out too narrow
                  when checked against held-out data.

So the tool asks for `--input-bits`: the precision of the INPUT, which is a fact the user
actually possesses. Fractional bits fall out of it.

WHY THAT IS BETTER THAN A GUESS. When an input has a native quantum -- 8-bit pixels, most ADCs,
anything already digital -- `frac = input_bits` is PROVABLY lossless, not merely measured. Take
8-bit pixels scaled to [0,1], i.e. values k/255 for integer k. Quantising at frac=8 computes
floor(k * 256/255), which is strictly increasing over k = 0..255. Strictly increasing means
order is preserved exactly, and since every encoder bit is an order comparison `x > t`, no bit
can change. The study repo's MNIST port is the empirical confirmation: 0 of 10,000 samples
diverged, which is what a proof predicts.

A continuous input has no such quantum, and then there is no proof to be had -- only a default
and a measurement. Say so out loud rather than presenting a fitted number as a safe one.
"""

from dataclasses import dataclass

import numpy as np

# For a continuous input with no native quantum. The study repo's JSC models used 12 fractional
# bits over standard-scaled features and measured 10 bit errors and 0 class changes against
# float32 -- good, but MEASURED on one dataset, not proved. It is a starting point for a stress
# test, not a guarantee, and `Precision.proved` records which of the two you have.
DEFAULT_CONTINUOUS_FRAC_BITS = 12


def required_int_bits(thresholds):
    """The integer bits a word MUST have to represent every threshold. Exact, not a heuristic.

    A threshold outside the representable range makes the comparison against it meaningless, so
    this is a hard floor: `word_bits >= 1 + required_int_bits(thr) + frac_bits`.

    Its counterpart is NOT derivable -- see this module's docstring. Report a floor as a floor.
    """
    span = float(np.max(np.abs(np.asarray(thresholds, dtype=np.float64))))
    bits = 0
    while (1 << bits) <= span:
        bits += 1
    return bits


@dataclass(frozen=True)
class Precision:
    """A signed fixed-point format for the encoder's input word.

    `word_bits` is the full signed width: 1 sign + int_bits + frac_bits.
    """

    word_bits: int
    frac_bits: int

    # True only when frac_bits came from an input with a native quantum, where losslessness is
    # a proof rather than a measurement. Carried so a report can distinguish the two instead of
    # calling both "safe" -- which is exactly the conflation this module exists to prevent.
    proved: bool = False

    def __post_init__(self):
        if self.frac_bits < 0:
            raise ValueError(f'frac_bits must be >= 0, got {self.frac_bits}')
        if self.word_bits < self.frac_bits + 1:
            raise ValueError(
                f'a {self.word_bits}-bit signed word cannot hold {self.frac_bits} fractional '
                f'bits plus a sign bit')

    @property
    def int_bits(self):
        return self.word_bits - 1 - self.frac_bits

    def __str__(self):
        return f'Q{self.int_bits}.{self.frac_bits} signed ({self.word_bits}-bit)'


def precision_for(thresholds, input_bits=None):
    """Choose the fixed-point format for a model. The tool's default policy.

    input_bits  the INPUT's precision, in bits after the point. For an input with a native
                quantum (8-bit pixels -> 8) this is provably lossless; see the module
                docstring. None means a continuous input with no quantum, which takes
                DEFAULT_CONTINUOUS_FRAC_BITS and is flagged unproved.

    The integer width is always derived from the thresholds, never asked for, so the returned
    format cannot fail to represent the model it was built for.
    """
    frac = DEFAULT_CONTINUOUS_FRAC_BITS if input_bits is None else int(input_bits)
    int_bits = required_int_bits(thresholds)
    return Precision(word_bits=1 + int_bits + frac, frac_bits=frac,
                     proved=input_bits is not None)


def comparator_merge_floor(thresholds, precision):
    """How many distinct comparators survive quantisation, and how many collapsed.

    Two thresholds that quantise to the same integer become the SAME comparison. That is not a
    bug -- the hardware is smaller and the encoder still separates everything the format can
    separate -- but it is a silent accuracy change, so it gets reported rather than absorbed.

    A large collapse means the format is too coarse for the model's thermometer resolution, and
    the fix is more fractional bits. Returns (distinct, total, collapsed).
    """
    from .extract import quantize_thresholds

    thr_q = quantize_thresholds(thresholds, precision.frac_bits)
    total = int(thr_q.size)
    # Per feature: thresholds only merge with others on the SAME input word. Two features
    # sharing a value are still two separate comparators.
    distinct = sum(int(np.unique(row).size) for row in np.atleast_2d(thr_q))
    return distinct, total, total - distinct
