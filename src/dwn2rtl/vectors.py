"""Testbench vectors, generated from the model alone.

Random inputs rather than real samples: the gate proves the Verilog matches extract.forward(),
and random vectors plus edge cases hit the tie-breaks and saturation boundaries that clustered
real data misses.

⚠️ Vectors and RTL must come from the SAME checkpoint, or the testbench passes against wrong
RTL. build() does both from one load.

  core   binarized vectors    -> dwn_core   (x_binarized.hex, expected.hex)
  top    quantized features   -> dwn_top    (x_quant.hex, expected_top.hex)
"""

import os

import numpy as np

from .emit_core import generated_header

from .extract import forward, quantize_thresholds, encode, fits_in_word

N_RANDOM = 500

# Fixed, so a rebuild of the same checkpoint produces byte-identical vectors. A testbench whose
# contents move between runs cannot be diffed, and "it passed yesterday" stops being checkable.
SEED = 20260802


def bits_to_hex(row, width):
    """bool[width] -> hex where bit i of the value is row[i].

    Mirrors how $readmemh loads a reg vector: the rightmost hex digit is bits [3:0], so the
    value must be sum(row[i] << i) -- the same LSB-first convention the LUT address uses
    (docs/checkpoint-format.md §2). Reversing this produces vectors that are wrong in exactly
    the way a reversed address concatenation is, which makes the two bugs mask each other.
    """
    assert row.size == width
    # ⚠️ PAD TO A WHOLE BYTE FIRST. np.packbits pads a partial byte on the LOW side, so a width
    # that is not a multiple of 8 came out shifted LEFT by the padding -- 12 bits multiplied by
    # 16, 18 bits by 64 -- while 8, 16 and 24 were correct. Every fixture and both studied
    # models happen to have widths divisible by 8 (MNIST 784x3 = 2352, JSC 16x8 = 128), which
    # is why the gate never saw it. Padding at the HIGH end instead makes the extra bits
    # leading zeros, which is what a hex literal wants anyway.
    #
    # Same defect, same cause, as the one emit_core.table_to_hex documents. It was fixed there
    # and not here.
    padded = np.concatenate([row.astype(np.uint8),
                             np.zeros((-width) % 8, dtype=np.uint8)])
    return np.packbits(padded[::-1], bitorder='big').tobytes().hex()


def words_to_hex(words, word_bits):
    """int[F] -> hex for a packed feature vector, feature f at [f*word_bits +: word_bits]."""
    value = 0
    mask = (1 << word_bits) - 1
    for f, w in enumerate(words):
        value |= (int(w) & mask) << (f * word_bits)      # two's complement
    return f'{value:0{len(words) * word_bits // 4}x}'


def write_lines(path, lines):
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def _core_vectors(width, rng, n_random):
    """Binarized vectors for dwn_core: edge cases first, then random.

    All-zeros and all-ones drive every node to opposite table corners. The alternating patterns
    are what catch a swapped wiring index -- adjacent bits differ, so a misroute changes the
    answer, which uniform patterns can never show.
    """
    edge = np.stack([
        np.zeros(width, dtype=bool),
        np.ones(width, dtype=bool),
        np.arange(width) % 2 == 0,
        np.arange(width) % 2 == 1,
    ])
    rand = rng.integers(0, 2, size=(n_random, width), dtype=np.int8).astype(bool)
    return np.concatenate([edge, rand]), edge.shape[0]


def _top_vectors(thr_q, used, z, n_features, precision, rng, n_random):
    """Quantized feature vectors for dwn_top: edge cases first, then random.

    The range spans the THRESHOLDS with a margin, not a dataset's min/max -- a comparator is
    only exercised by inputs on both sides of it, and spanning the thresholds guarantees that
    for all of them.

    Edge cases: all-zeros/min/max rail every comparator; q_x == T must give 0 (the compare is
    strict `>`) and q_x == T+1 must give 1, which pins the boundary from both sides.
    """
    word = precision.word_bits
    lo_word, hi_word = -(2 ** (word - 1)), 2 ** (word - 1) - 1

    used_thr = thr_q[used // z, used % z]
    t_lo, t_hi = int(used_thr.min()), int(used_thr.max())

    # A margin past both ends so the extreme thresholds are straddled too, not just met.
    # One quantised unit is not enough to look like real headroom; a tenth of the span, floored
    # at 8 units, is arbitrary but generous, and it is clipped to the word either way.
    margin = max(8, (t_hi - t_lo) // 10)
    lo = max(lo_word, t_lo - margin)
    hi = min(hi_word, t_hi + margin)

    edge_q = [np.zeros(n_features, dtype=np.int64),
              np.full(n_features, lo_word, dtype=np.int64),
              np.full(n_features, hi_word, dtype=np.int64)]
    names = ['all-zeros', 'all-min', 'all-max']

    for f in range(n_features):
        f_thr = used_thr[(used // z) == f]
        if not f_thr.size:
            continue                       # this feature drives no comparator
        # The median of this feature's own thresholds, so the vector lands on a real boundary
        # this design actually contains rather than a synthetic one.
        t = int(np.median(f_thr))
        for delta, tag in ((0, 'on'), (1, 'above')):
            row = np.zeros(n_features, dtype=np.int64)
            row[f] = int(np.clip(t + delta, lo_word, hi_word))
            edge_q.append(row)
            names.append(f'feature{f}-{tag}-threshold')

    edge_q = np.stack(edge_q)
    rand_q = rng.integers(lo, hi + 1, size=(n_random, n_features), dtype=np.int64)
    return np.concatenate([edge_q, rand_q]), edge_q.shape[0], names, (lo, hi)


def generate(ck, layers, outdir, precision, n_random=N_RANDOM, seed=SEED):
    """Write both levels' vectors and their `.vh` parameter files.

    `layers` is the extracted model -- passed in rather than re-extracted, so the vectors are
    labelled by the same tables the RTL was emitted from. See the module docstring.
    """
    cfg = ck['config']
    num_classes, z = cfg['num_classes'], cfg['thermometer_bits']
    thresholds = ck['thermometer']['thresholds'].numpy()
    n_features = thresholds.shape[0]
    width = thresholds.size
    idx_w = max(1, int(np.ceil(np.log2(num_classes))))

    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # ---------------- core level ----------------
    x_core, n_edge = _core_vectors(width, rng, n_random)
    y_core, _ = forward(x_core, layers, num_classes)

    write_lines(os.path.join(outdir, 'x_binarized.hex'),
                [bits_to_hex(r, width) for r in x_core])
    write_lines(os.path.join(outdir, 'expected.hex'), [f'{int(y):X}' for y in y_core])
    write_lines(os.path.join(outdir, 'vec_params.vh'), [
        generated_header('this file') + '',
        f'`define N_VEC {x_core.shape[0]}',
        f'`define VEC_W {width}',
        # Width of class_idx. The study's testbenches hardcoded 3 -- one dataset's class
        # count -- which left the upper bits undriven below five classes and TRUNCATED the
        # comparison above eight: a 10-class design was checked on 3 of its 4 index bits, and
        # passed. Derived, never written down.
        f'`define IDX_W {idx_w}',
    ])

    # ---------------- top level ----------------
    thr_q = quantize_thresholds(thresholds, precision.frac_bits)
    used = np.unique(layers[0][1])
    xq_all, n_edge_top, edge_names, rand_range = _top_vectors(
        thr_q, used, z, n_features, precision, rng, n_random)

    if not fits_in_word(xq_all, precision.word_bits):
        raise ValueError(
            f'generated features do not fit {precision}: '
            f'range [{xq_all.min()}, {xq_all.max()}]')

    bits_all = encode(xq_all, thr_q)
    y_top, _ = forward(bits_all, layers, num_classes)

    write_lines(os.path.join(outdir, 'x_quant.hex'),
                [words_to_hex(r, precision.word_bits) for r in xq_all])
    write_lines(os.path.join(outdir, 'expected_top.hex'), [f'{int(y):X}' for y in y_top])
    write_lines(os.path.join(outdir, 'top_params.vh'), [
        generated_header('this file') + '',
        f'`define N_TOP {xq_all.shape[0]}',
        f'`define X_W {n_features * precision.word_bits}',
        f'`define IDX_W {idx_w}',
    ])

    # A design whose vectors all land on one class proves very little, and it is a plausible
    # outcome of a bad export rather than an exotic one -- a broken wiring can collapse every
    # input to the same answer. The testbench would then pass on a design that is wrong.
    # Reported, not raised: it is legitimate for a genuinely lopsided model.
    core_classes = int(np.unique(y_core).size)
    top_classes = int(np.unique(y_top).size)

    return {
        'core': {'path': os.path.join(outdir, 'x_binarized.hex'),
                 'count': int(x_core.shape[0]), 'edge': int(n_edge),
                 'random': int(n_random), 'width': int(width),
                 'classes_hit': core_classes},
        'top': {'path': os.path.join(outdir, 'x_quant.hex'),
                'count': int(xq_all.shape[0]), 'edge': int(n_edge_top),
                'random': int(n_random), 'features': int(n_features),
                'word_bits': precision.word_bits, 'classes_hit': top_classes,
                'random_range': rand_range, 'edge_names': edge_names},
        'num_classes': int(num_classes),
        'degenerate': core_classes < 2 or top_classes < 2,
    }
