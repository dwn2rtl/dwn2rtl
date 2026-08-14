"""Checkpoint -> LUT tables, wiring, thresholds. Also the golden model.

Two jobs in one file, and they are one file on purpose:

1. Turn a checkpoint into the three things hardware needs -- per-node truth tables, per-node
   input wiring, and the thermometer thresholds those inputs come from.
2. Run a pure-numpy forward pass over exactly those artifacts. `forward()` here is the golden
   model every emitted testbench is checked against.

Keeping them together is the guarantee: the model the RTL is compared to is built from the
same tables the RTL was emitted from, so the comparison cannot drift. Splitting them would
allow a golden model that agrees with PyTorch while the emitted design disagrees with both.

Every structural fact this file relies on -- LSB-first address order, `argmax(dim=0)` for
learnable wiring, `> 0` table thresholding, contiguous GroupSum groups -- was read out of the
upstream DWN source and written up in `docs/checkpoint-format.md`. Read correctly is not the
same as implemented correctly, which is why the gate is a simulator, not a review.

Ported from the study repo's `exporter/extract.py`. Changes: the CLI `main()` was removed
(it verified numpy against a dataset's saved `pred` array, which this tool has no dataset for
-- `vectors.py` and the emitted testbench cover it instead), and `required_int_bits()` moved
to `precision.py`.
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
        assert flat.size % n == 0
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
    assert width % num_classes == 0, (
        f'final layer width {width} not divisible by num_classes {num_classes}; '
        'GroupSum would zero-pad silently')
    scores = x_bits.reshape(x_bits.shape[0], num_classes, width // num_classes).sum(axis=2)
    return scores.argmax(axis=1), scores


def forward(x_bits, layers, num_classes):
    for tables, wiring, _ in layers:
        x_bits = lut_forward(x_bits, tables, wiring)
    return group_sum_argmax(x_bits, num_classes)


# ---------------------------------------------------------------------------
# Fixed-point front end (the thermometer encoder)
#
# Quantization is part of the SPECIFICATION, not an error to be minimised away. The golden
# model quantizes exactly as the hardware does, so Gate 1 stays bit-exact; the difference
# between this and the float32 model is then a characterization result to report rather than
# a discrepancy to hide. See the study repo for how the format was chosen.
#
# THERE ARE NO MODULE-LEVEL FRAC_BITS/WORD_BITS, and adding some would be a regression. The
# study repo had them, set to 16 and 12 -- one dataset's format -- and every consumer that
# imported them inherited that precision as a default it never stated. The second dataset
# needed Q0.8, so each of those sites was a silent wrong answer waiting to happen; six were
# found one crash at a time before the constants were removed.
#
# The parameters below are therefore REQUIRED, not defaulted. A caller that does not know the
# word width does not know enough to quantize, and should ask `precision.py`. Making them
# required is the point: a missing argument is a TypeError at the call site, whereas a default
# is a plausible wrong number that reaches the FPGA.
# ---------------------------------------------------------------------------


def quantize(x, frac_bits, word_bits):
    """Real features -> fixed point. Truncation, not rounding: free in hardware.

    SATURATES to the word range. That is usually lossless rather than a compromise, but it is
    a claim to check, not assume -- call saturation_is_lossless().

    Why it is normally fine: clamping changes no encoder bit as long as every threshold lies
    strictly inside the representable range, since a saturated feature is still on the same
    side of every threshold it was before. In the study repo's JSC model the word represented
    [-8, +8) while the extreme thresholds were -4.55 and +4.34, so features out at 8.08 -- and
    they existed, in a handful of 166,000 samples -- encoded identically to 7.9998.

    Note that widening the FRACTIONAL part is not a fix for a range problem: more fractional
    bits at the same integer width buy precision, not headroom. See precision.py.
    """
    lo, hi = -(2 ** (word_bits - 1)), 2 ** (word_bits - 1) - 1
    q = np.floor(np.asarray(x, dtype=np.float64) * (2 ** frac_bits))
    return np.clip(q, lo, hi).astype(np.int64)


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
