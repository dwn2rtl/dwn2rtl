"""How wide the encoder's input word is.

Integer bits come exactly from the thresholds. Fractional bits cannot -- that depends on the
data -- so the tool asks for `--input-bits`, the input's precision, which a user knows. With a
native quantum (8-bit pixels, most ADCs) that is provably lossless: values are k/255, quantising
at frac=8 is strictly increasing, and every encoder bit is an order comparison. Otherwise a
default, reported as a default.
"""

from dataclasses import dataclass

import numpy as np

# For a continuous input. Measured good on one dataset, never proved -- a starting point.
DEFAULT_CONTINUOUS_FRAC_BITS = 12


def required_int_bits(thresholds):
    """Integer bits needed to represent every threshold. A hard floor, not a heuristic."""
    span = float(np.max(np.abs(np.asarray(thresholds, dtype=np.float64))))
    bits = 0
    while (1 << bits) <= span:
        bits += 1
    return bits


# How far a scaled threshold may sit from an integer and still be "on the grid". Not delicate:
# on-grid measures ~7.6e-06, off-grid ~5e-01, so anything from 1e-4 to 1e-2 behaves the same.
GRID_TOLERANCE = 1e-3

# Below this many distinct thresholds a grid match is coincidence, not evidence.
MIN_DISTINCT_FOR_INFERENCE = 8


def infer_input_bits(thresholds, max_bits=16):
    """The input's native precision, read off the thresholds. None if it has none.

    Thermometer thresholds are quantiles of the training data, so quantised data leaves its grid
    in them. Both common grids are checked: k/(2**n - 1) for [0,1]-scaled data, k/2**n for raw
    fixed point.

    ⚠️ The SMALLEST match is the only correct one -- every coarse grid is a subset of every finer
    one (k/255 == 257k/65535), so any other gives a needlessly wide word.
    """
    thr = np.asarray(thresholds, dtype=np.float64).ravel()
    distinct = np.unique(thr)
    if distinct.size < MIN_DISTINCT_FOR_INFERENCE:
        return None

    for n in range(1, max_bits + 1):
        for scale in (2 ** n - 1, 2 ** n):
            scaled = distinct * scale
            rounded = np.round(scaled)
            if np.abs(scaled - rounded).max() > GRID_TOLERANCE:
                continue
            # ⚠️ Closeness is not enough -- the grid must SEPARATE the thresholds. Without
            # this, small values all round to 0 and match any coarse grid: one test inferred
            # n=1, a one-bit word labelled "provably lossless".
            if np.unique(rounded).size == distinct.size:
                return n
    return None


@dataclass(frozen=True)
class Precision:
    """A signed fixed-point format. word_bits = 1 sign + int_bits + frac_bits."""

    word_bits: int
    frac_bits: int

    # 'given' (--input-bits), 'inferred' (the thresholds' grid), or 'default' (nothing to go
    # on). `proved` derives from this, so the two cannot drift.
    source: str = 'default'

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

    @property
    def proved(self):
        """True when losslessness is a proof rather than a hope."""
        return self.source in ('given', 'inferred')

    def __str__(self):
        return f'Q{self.int_bits}.{self.frac_bits} signed ({self.word_bits}-bit)'


def precision_for(thresholds, input_bits=None, infer=True):
    """--input-bits if given, else the thresholds' grid, else a default.

    Integer width always comes from the thresholds, so the result cannot fail to represent the
    model. `infer=False` skips the grid tier.
    """
    if input_bits is not None:
        frac, source = int(input_bits), 'given'
    else:
        found = infer_input_bits(thresholds) if infer else None
        if found is not None:
            frac, source = found, 'inferred'
        else:
            frac, source = DEFAULT_CONTINUOUS_FRAC_BITS, 'default'

    int_bits = required_int_bits(thresholds)
    return Precision(word_bits=1 + int_bits + frac, frac_bits=frac, source=source)


def comparator_merge_floor(thresholds, precision):
    """(distinct, total, collapsed) comparators after quantisation.

    Thresholds quantising to the same integer become one comparison -- smaller hardware, but a
    silent accuracy change, so it is reported. A large collapse means too few fractional bits.
    """
    from .extract import quantize_thresholds

    thr_q = quantize_thresholds(thresholds, precision.frac_bits)
    total = int(thr_q.size)
    # Per feature: thresholds only merge within one input word.
    distinct = sum(int(np.unique(row).size) for row in np.atleast_2d(thr_q))
    return distinct, total, total - distinct
