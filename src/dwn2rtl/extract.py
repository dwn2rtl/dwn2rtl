"""Checkpoint -> LUT tables, wiring, thresholds. Also the golden model.

⚠️ Both jobs live here on purpose: `forward()` is checked against the RTL, and building it from
the same tables the RTL was emitted from is what stops the two drifting.

The structural facts it relies on -- LSB-first addresses, argmax(dim=0) wiring, `> 0`
thresholding, contiguous GroupSum groups -- are in docs/checkpoint-format.md.
"""

import re

import numpy as np
import torch


def load_checkpoint(path):
    # weights_only=False: the checkpoint holds the config dict, class-name list, and results
    # alongside the tensors. Trusted input -- we produced it.
    return torch.load(path, map_location='cpu', weights_only=False)


def layer_indices(state_dict):
    """Layer indices, in order, from the `<i>.luts` keys."""
    found = {int(m.group(1)) for k in state_dict
             if (m := re.fullmatch(r'(\d+)\.luts', k))}
    return sorted(found)


def extract_tables(state_dict, i):
    """(output_size, 2**n) float -> bool truth tables.

    docs/checkpoint-format.md §1: the hardware bit is `luts[j][addr] > 0`. Strictly greater --
    an exact 0.0 emits 0. Unlikely in a trained model, but Gate 1 covers edge cases.
    """
    luts = state_dict[f'{i}.luts'].numpy()
    return luts > 0.0


def extract_wiring(state_dict, i, n):
    """(output_size, n) int -> which input bit feeds each node slot.

    Two completely different representations, per docs/checkpoint-format.md §3.
    """
    learnable_key = f'{i}.mapping.weights'
    if learnable_key in state_dict:
        # §3a. weights is (input_size, output_size*n); argmax over the INPUT-BIT axis gives
        # one input index per node slot. Node j slot k -> index [j*n + k].
        weights = state_dict[learnable_key].numpy()
        flat = weights.argmax(axis=0)
        if flat.size % n:
            raise ValueError(
                f'layer {i} has {flat.size} mapping weights, which is not a multiple of n={n}; '
                'a learnable mapping is (input_size, output_size * n)')
        return flat.reshape(-1, n).astype(np.int64), 'learnable'

    # §3b. Already the final wiring, no transformation.
    #
    # NB: `<i>._LUTLayer__dummy_mapping` is deliberately NOT consulted. It has the same shape
    # and dtype as a real mapping but is only arange() reshaped, so keying off shape instead
    # of off the presence of `.mapping.weights` yields a valid-looking, totally wrong export.
    return state_dict[f'{i}.mapping'].numpy().astype(np.int64), 'fixed'


def lut_forward(x_bits, tables, wiring):
    """One LUT layer. x_bits: (N, input_size) bool. Returns (N, output_size) bool."""
    # docs/checkpoint-format.md §2: slot l contributes bit l of the address -- LSB FIRST.
    # Reversing this gives a model that is wrong on most inputs but structurally plausible.
    gathered = x_bits[:, wiring]                       # (N, output_size, n)
    n = wiring.shape[1]
    weights = (1 << np.arange(n, dtype=np.int64))      # [1, 2, 4, 8, 16, 32]
    addr = (gathered.astype(np.int64) * weights).sum(axis=2)   # (N, output_size)
    return tables[np.arange(tables.shape[0]), addr]


def group_sum_argmax(x_bits, num_classes):
    """GroupSum + argmax. docs/checkpoint-format.md §4.

    Groups are contiguous and in order. `tau` is a uniform divisor across classes, so it
    cannot change the argmax and the hardware never needs it -- popcount and compare only.
    """
    width = x_bits.shape[1]
    if width % num_classes:
        raise ValueError(
            f'final layer width {width} not divisible by num_classes {num_classes}; '
            'GroupSum would zero-pad silently')
    scores = x_bits.reshape(x_bits.shape[0], num_classes, width // num_classes).sum(axis=2)
    return scores.argmax(axis=1), scores


def forward(x_bits, layers, num_classes):
    for tables, wiring, _ in layers:
        x_bits = lut_forward(x_bits, tables, wiring)
    return group_sum_argmax(x_bits, num_classes)


# Fixed-point front end. The golden model quantizes exactly as the hardware does, so the gate
# stays bit-exact.
#
# ⚠️ frac_bits/word_bits are REQUIRED arguments, never module-level defaults. One dataset's
# format as a default silently reached every caller that imported it, and six wrong answers
# were found one crash at a time. A missing argument is a TypeError; a default is a plausible
# wrong number that reaches the FPGA.


def quantize(x, frac_bits, word_bits):
    """Real features -> fixed point. Truncation, not rounding: free in hardware.

    Saturates to the word range, which is lossless as long as every threshold sits strictly
    inside it -- check with saturation_is_lossless() rather than assuming. More fractional bits
    do not fix a range problem; they buy precision, not headroom.
    """
    lo, hi = -(2 ** (word_bits - 1)), 2 ** (word_bits - 1) - 1
    q = np.floor(np.asarray(x, dtype=np.float64) * (2.0 ** frac_bits))

    # ⚠️ NaN CANNOT SATURATE, so it must not reach the cast. Every comparison against NaN is
    # False, so it would fall through both branches below and land on int64's most negative
    # value -- a feature that is missing becoming a feature that is extremely small, silently.
    # Thresholds already refuse non-finite values for the same reason (precision.py).
    # Infinities are fine and are left to saturate: they are on a definite side of every
    # threshold, which is exactly what saturation means.
    if np.isnan(q).any():
        raise ValueError(
            f'{int(np.isnan(q).sum())} of {q.size} feature values are NaN, so they cannot be '
            'quantized. A NaN feature is not a small one -- it has no side of any threshold.')

    # ⚠️ THE CLIP CANNOT BE DONE IN FLOAT. `np.clip(q, lo, hi)` was the whole implementation,
    # and float64 has no exact representation of hi = 2**63 - 1: it rounds UP to 2**63, which
    # int64 cannot hold, so `.astype(np.int64)` overflowed. An input that should have saturated
    # to +9223372036854775807 came out as -9223372036854775808 -- a sign flip that inverts
    # every comparator it feeds -- behind a RuntimeWarning nothing reads. Wrong from about a
    # 55-bit word up; off by one from 54 (phase8-ledger.md §3).
    #
    # The gate is blind to this by construction: `build` never calls quantize(), because top
    # vectors are generated as integers already. It is the user who is told to call it, by
    # input_scaling.json and the user guide.
    #
    # So the bounds are tested against the two values that ARE exact -- both powers of two --
    # and only values known to fit are ever cast.
    low = q < float(lo)                       # lo = -2**(word_bits-1), exact in float64
    high = q >= float(hi + 1)                 # hi + 1 = 2**(word_bits-1), exact in float64
    out = np.where(low | high, 0.0, q).astype(np.int64)
    out[low] = lo
    out[high] = hi
    return out


def saturation_is_lossless(thr_q, word_bits):
    """True if clamping features to the word range cannot change any comparison.

    Holds exactly when every threshold is strictly inside the representable range: a saturated
    feature is then still on the same side of every threshold it was before.
    """
    lo, hi = -(2 ** (word_bits - 1)), 2 ** (word_bits - 1) - 1
    return int(np.min(thr_q)) > lo and int(np.max(thr_q)) < hi


def quantize_thresholds(thresholds, frac_bits):
    """Thresholds -> the integer constants the comparators are built against.

    floor() specifically: with T = floor(t * 2**F), `q_x > T` implies `x > t` exactly, so the
    encoding can only ever miss a bit, never invent one.
    """
    return np.floor(np.asarray(thresholds, dtype=np.float64)
                    * (2 ** frac_bits)).astype(np.int64)


def encode(xq, thr_q):
    """(N, F) quantized features -> (N, F*z) bits, feature-major.

    Bit index is `feature * z + threshold`, matching how binarization.py flattens
    (N, features, bits) and therefore how the wiring indices are interpreted.
    """
    return (xq[:, :, None] > thr_q[None, :, :]).reshape(xq.shape[0], -1)


def fits_in_word(values, word_bits):
    """Range check for the chosen fixed-point word -- silent overflow would be invisible."""
    lo, hi = -(2 ** (word_bits - 1)), 2 ** (word_bits - 1) - 1
    return int(np.min(values)) >= lo and int(np.max(values)) <= hi
