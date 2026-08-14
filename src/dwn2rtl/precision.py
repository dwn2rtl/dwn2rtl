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


# How far a scaled threshold may sit from an integer and still count as "on the grid".
#
# Not a delicate tuning choice, and that is the point. Measured on real checkpoints in float64:
# MNIST's thresholds sit 7.6e-06 from the k/(2^8 - 1) grid, while JSC's standard-scaled features
# sit ~5e-01 from every grid at every width. Five orders of magnitude of margin, so anything
# between roughly 1e-4 and 1e-2 separates them identically.
#
# It also, usefully, rejects a spurious wider match: MNIST is technically on the 16-bit grid too
# (k/255 == 257k/65535), but float32 noise amplified by 65535 puts the residual at 1.9e-03, past
# this bound. See infer_input_bits for why the smallest match is the one that matters anyway.
GRID_TOLERANCE = 1e-3

# Below this many DISTINCT thresholds, a grid match is not evidence. Four values can lie on a
# coarse grid by coincidence; a hundred and seventy-six cannot.
MIN_DISTINCT_FOR_INFERENCE = 8


def infer_input_bits(thresholds, max_bits=16):
    """The input's native precision, read off the thresholds. None if there is not one.

    WHY THIS IS POSSIBLE AT ALL, and it is not obvious: a thermometer's thresholds are QUANTILES
    OF THE TRAINING DATA. If that data was quantised -- 8-bit pixels, a 12-bit ADC -- then its
    quantiles are themselves data values, and they inherit the same grid. So the checkpoint does
    carry a fingerprint of the input's precision, in a place nobody thought to look.

    ⚠️ This does NOT make roadmap Q9 wrong. Q9's claim is that you cannot tell from a checkpoint
    whether a given width is SAFE FOR YOUR DATA, and that is still true and still undecidable
    here. What is recovered is something narrower and different: the input's native QUANTUM,
    which when it exists makes the safety question vanish rather than answering it -- there is
    nothing between adjacent representable values to lose.

    THE SMALLEST MATCH IS THE ONLY CORRECT ONE. Every coarse grid is a subset of every finer
    one: 8-bit data satisfies k/255, and also 257k/65535, and so on. Returning any but the
    smallest gives a needlessly wide word for no gain.

    Both common grids are checked, because both occur:
        k / (2**n - 1)   values scaled to span [0, 1] inclusive -- images, most normalised data
        k / 2**n         raw fixed-point -- audio, many ADC pipelines
    Either way `frac = n` is lossless: the quantiser is strictly increasing over the input's
    possible values, and every encoder bit is an order comparison.
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
            # ⚠️ CLOSENESS IS NOT ENOUGH -- the grid must SEPARATE the thresholds.
            #
            # Without this, small-magnitude features match every coarse grid trivially: values
            # of ~0.001 scaled by 1 all round to 0, giving a residual under any absolute
            # tolerance. A test with k/65535 inputs duly inferred n=1, which would have emitted
            # a ONE-BIT fractional word labelled "provably lossless" -- silently catastrophic,
            # and exactly the false positive this function must never produce.
            #
            # Requiring injectivity is the honest statement of what "these values lie on this
            # grid" means: if two distinct thresholds land on the same grid point, the grid is
            # too coarse to be the quantum they came from.
            if np.unique(rounded).size == distinct.size:
                return n
    return None


@dataclass(frozen=True)
class Precision:
    """A signed fixed-point format for the encoder's input word.

    `word_bits` is the full signed width: 1 sign + int_bits + frac_bits.
    """

    word_bits: int
    frac_bits: int

    # WHERE frac_bits CAME FROM, which a report must be able to distinguish:
    #
    #   'given'     the user passed --input-bits. Lossless by the quantum argument.
    #   'inferred'  read off the thresholds' grid. Lossless by the same argument -- and it
    #               rests on the same assumption the user makes when they type the flag, that
    #               inference-time data shares training-time precision. No weaker, not stronger.
    #   'default'   nothing to go on. A documented starting point, NOT a measurement.
    #
    # `proved` derives from this rather than being stored beside it, so the two cannot drift.
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
        """True when losslessness is a proof rather than a hope.

        Both 'given' and 'inferred' rest on the input having a native quantum, which is what
        makes the quantiser strictly increasing over the values that can actually occur. A
        'default' width rests on nothing but a previous dataset's luck.
        """
        return self.source in ('given', 'inferred')

    def __str__(self):
        return f'Q{self.int_bits}.{self.frac_bits} signed ({self.word_bits}-bit)'


def precision_for(thresholds, input_bits=None, infer=True):
    """Choose the fixed-point format for a model. The tool's default policy.

    Three tiers, in order, so that the common case needs no flag at all:

        1. input_bits given      the user knows their input's precision. Obeyed.
        2. inferred              the thresholds lie on a dyadic grid, so the training data had
                                 a native quantum and that quantum is it.
        3. default               nothing to go on. Documented, and reported AS a default.

    The integer width is always derived from the thresholds, never asked for, so the returned
    format cannot fail to represent the model it was built for.

    `infer=False` skips tier 2, for a caller who wants the old behaviour or is testing the
    fallback.
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
