"""Generate self-checking testbench vectors from the model alone.

REWRITTEN, not ported. The study's `tb/gen_vectors.py` mixed 500 random vectors with real
samples loaded from a dataset's `.npz`. This tool has no dataset, so the real half is gone and
the random half is kept.

That is not a compromise, and it is worth being clear about why:

  THE GATE IS RTL-VERSUS-GOLDEN-MODEL, NOT RTL-VERSUS-DATASET.

What must be proved is that the Verilog computes the same function as extract.forward(). Real
samples are one sample of that function's domain, and a biased one -- they cluster where the
data clusters, which is precisely where nothing interesting happens. Random vectors plus
deliberate edge cases cover the tie-breaks and saturation boundaries real data mostly misses.
The study's real-data half caught nothing its random half did not.

What real data WOULD prove is that the model is accurate, which is a training question this
tool has no opinion about.

THE INVARIANT THAT MUST NOT BREAK: vectors and RTL derive from the SAME checkpoint. Otherwise
you ship a testbench that passes against wrong RTL, which is worse than shipping no testbench
at all -- a green light nobody has any reason to distrust. build() does both from one load;
nothing here should ever grow a `--checkpoint` of its own.

Two levels are emitted, because a failure in one must not be able to hide in the other:

  core   binarized vectors      -> dwn_core   (x_binarized.hex, expected.hex)
  top    quantized features     -> dwn_top    (x_quant.hex, expected_top.hex)

If the top level fails while the core passes, the encoder is at fault and nothing else needs
re-examining. That split is worth the extra file.
"""

import os

import numpy as np

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
    return np.packbits(row[::-1].astype(np.uint8), bitorder='big').tobytes().hex()


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

    The four edge patterns are chosen to break structural bugs rather than to look thorough.
    All-zeros and all-ones drive every node to opposite table corners. The two alternating
    patterns are the ones that catch a swapped or off-by-one wiring index: under either, a
    node reading bit b and a node reading bit b+1 see DIFFERENT values, so a misrouted index
    changes the answer. Uniform patterns cannot detect that -- every wiring is right when all
    the bits agree.
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

    WHERE THE RANGE COMES FROM. The study drew random features between the min and max of
    the real test set. With no dataset, the range is taken from the THRESHOLDS instead: span
    them, then push a margin past both ends. This is strictly better for the job. A comparator
    against threshold t is only exercised by inputs on both sides of t, and a data-derived
    range guarantees that only for thresholds the data actually straddles. Spanning the
    thresholds guarantees it for all of them.

    THE EDGE CASES, and why each one exists:

      all-zeros, all-min, all-max   rail every comparator in both directions, including the
                                    word's two's-complement extremes
      exactly-on-threshold          q_x == T must give 0, because the comparison is strict `>`.
                                    This is the encoder's equivalent of the argmax tie case: a
                                    `>=` implementation passes every other vector in this file
                                    and fails only here
      one-above-threshold           q_x == T+1 must give 1. Pairs with the above -- together
                                    they pin the boundary from both sides, so an off-by-one in
                                    either direction is caught rather than half of them
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
        '// GENERATED by dwn2rtl -- do not edit.',
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
        '// GENERATED by dwn2rtl -- do not edit.',
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
