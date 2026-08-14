"""Synthetic DWN checkpoints, for tests only.

⚠️ THIS IS NOT PART OF THE TOOL AND MUST NOT BECOME PART OF IT. dwn2rtl translates a trained
model; it does not generate one. This file lives in tests/, ships in no wheel, and is on no
user's path.

WHY IT HAS TO EXIST. Roadmap P8: a learnable mapping stores a (features x z) x (width x n)
float32 matrix, so a real checkpoint is 17 MB for MNIST 1x300 and 471 MB for 1x1000 z=25.
Nothing over 100 MB goes in git, so CI fixtures have to be synthesized rather than committed.

WHY IT IS ALSO A GOOD IDEA. In the study repo the equivalent script passed Gate 1 at MNIST's
shape -- 784 features, 10 classes, two layers -- before any MNIST model had been trained, which
caught emitter bugs weeks earlier than a real checkpoint could have. Shape coverage is cheap
here and expensive everywhere else.

The checkpoints produced are STRUCTURALLY real: the wiring, tables and thresholds obey every
rule in docs/checkpoint-format.md, including the two completely different mapping
representations and the `__dummy_mapping` decoy. They are not trained, so their accuracy is
meaningless -- which is fine, because the gate compares RTL against the golden model, not
against a dataset.
"""

import numpy as np
import torch


def _thresholds(rng, n_features, z, lo=-1.0, hi=1.0):
    """(n_features, z) float32, strictly increasing along z within each feature.

    Sorted because a thermometer's thresholds ARE sorted -- they are quantiles. An unsorted
    fixture would let a bug that depends on ordering pass here and fail on every real model.
    """
    t = rng.uniform(lo, hi, size=(n_features, z))
    return np.sort(t, axis=1).astype(np.float32)


def _tables(rng, out_size, n):
    """(out_size, 2**n) float32 in [-1, 1].

    The [-1, 1] range is not decoration: upstream initialises uniform in [-1, 1] and clamps back
    into it on every training forward pass (docs/checkpoint-format.md §1). Only the SIGN is used
    at inference, via a strict `> 0`.
    """
    return rng.uniform(-1.0, 1.0, size=(out_size, 2 ** n)).astype(np.float32)


def _learnable_weights(rng, input_size, out_size, n):
    """A LearnableMapping `weights` matrix whose argmax(dim=0) is a chosen wiring.

    docs/checkpoint-format.md §3a: weights is (input_size, out_size*n) and node j slot k reads
    input bit argmax(dim=0)[j*n + k]. To make the fixture's wiring known and deterministic, the
    chosen row is set STRICTLY above every other -- ties would resolve to the lowest index and
    quietly give a different wiring than intended.
    """
    wiring = rng.integers(0, input_size, size=(out_size, n))
    w = rng.uniform(0.0, 0.5, size=(input_size, out_size * n)).astype(np.float32)
    for j in range(out_size):
        for k in range(n):
            w[wiring[j, k], j * n + k] = 1.0
    return w, wiring.astype(np.int64)


def make_checkpoint(n_features=4, z=2, layers=(6, 4), n=2, num_classes=2, seed=7,
                    first_layer='learnable', run_name='synthetic'):
    """A structurally valid checkpoint in the tool's normalized form.

    Defaults are deliberately TINY -- 4 features, z=2, n=2 -- so simulation is instant. Two
    details of that smallness are load-bearing rather than incidental:

      n=2 gives 4-entry truth tables, which is where the study repo's np.packbits bug lived:
      bitorder='big' pads a partial byte on the LOW side, so [1,1,1,0] emitted as 0x70 instead
      of 0x07 and every entry sat at the wrong address. n>=3 is a whole number of bytes and
      never showed it. Testing at n=6 only would miss it again.

      The first layer is `learnable` and the rest are `fixed`, which is upstream's own recipe
      (examples/mnist.py) and forces BOTH mapping representations through every test. They share
      no code, and one of them has a decoy sitting next to it.

    `layers` is the per-layer node count. The last must be divisible by num_classes or GroupSum
    zero-pads silently (docs/checkpoint-format.md §4).
    """
    if layers[-1] % num_classes:
        raise ValueError(
            f'final layer {layers[-1]} not divisible by num_classes {num_classes}; GroupSum '
            'would zero-pad silently, and hardware and software would disagree about group '
            'boundaries')

    rng = np.random.default_rng(seed)
    state_dict = {}
    input_size = n_features * z

    for i, out_size in enumerate(layers):
        state_dict[f'{i}.luts'] = torch.from_numpy(_tables(rng, out_size, n))

        if i == 0 and first_layer == 'learnable':
            w, _ = _learnable_weights(rng, input_size, out_size, n)
            state_dict[f'{i}.mapping.weights'] = torch.from_numpy(w)
            # §3b's decoy. Present in every real learnable layer, same (out_size, n) shape and
            # int32 dtype as a genuine mapping, and only arange() reshaped. An exporter that
            # keys off shape instead of off `.mapping.weights` reads THIS and emits a
            # structurally valid, completely wrong model. The fixture ships it so that bug has
            # something to fail against.
            state_dict[f'{i}._LUTLayer__dummy_mapping'] = torch.from_numpy(
                np.arange(out_size * n).reshape(out_size, n).astype(np.int32))
        else:
            mapping = rng.integers(0, input_size, size=(out_size, n)).astype(np.int32)
            state_dict[f'{i}.mapping'] = torch.from_numpy(mapping)

        input_size = out_size

    return {
        'config': {
            'n': n,
            'num_classes': num_classes,
            'layers': list(layers),
            'thermometer_bits': z,
        },
        'state_dict': state_dict,
        'thermometer': {'thresholds': torch.from_numpy(_thresholds(rng, n_features, z))},
        # NO `results` KEY, deliberately. These models are untrained, so any accuracy here
        # would be a number invented to fill a field -- and `final_acc: 0.0` reads as "this
        # model scores zero" rather than "nobody measured". Omitting it also makes the fixture
        # exercise the case a real user hits constantly: metadata is optional, and everything
        # downstream has to cope with its absence rather than assume a header value exists.
        'run_name': run_name,
    }


# Shapes worth having a name for. `tiny` is the default everywhere; the others exist to catch
# what tiny cannot, and each one is here for a stated reason.
#
# ON GROUP SIZE, which is the non-obvious constraint. The final layer's width divided by
# num_classes is the popcount group, and it must not be small. The first cut of this file used
# groups of 2 and produced fixtures where EVERY vector landed on one class: with untrained
# tables a node is constant whenever its 2**n table happens to be all-positive (probability
# 1/16 at n=2), and a group of two constant nodes scores a permanent 2 that nothing beats.
# A fixture like that passes a testbench against almost any wrong design. Groups are >= 3 here,
# and classes_hit() below is the check that keeps them honest.
SHAPES = {
    # the fast path -- everything runs against this unless a test says otherwise
    'tiny':        dict(n_features=4,  z=2, layers=(12, 8), n=2, num_classes=2),
    # one layer only: the loop in emit_core that chains layer -> layer never executes
    'single':      dict(n_features=4,  z=2, layers=(8,),    n=2, num_classes=2),
    # n=6, the real architectural premise -- one node is exactly one LUT6, tables are 64 bits
    'n6':          dict(n_features=8,  z=3, layers=(12, 9), n=6, num_classes=3),
    # 10 classes: idx_w is 4 bits. The study repo's testbench hardcoded 3 and silently checked
    # three of four index bits on a 10-class design, which passed
    'ten_class':   dict(n_features=8,  z=3, layers=(40,),   n=4, num_classes=10),
    # every layer fixed-mapping, so the learnable path is absent rather than merely unused
    'all_fixed':   dict(n_features=4,  z=2, layers=(12, 8), n=2, num_classes=2,
                        first_layer='fixed'),
}


def extract_layers(ck):
    """The checkpoint's (tables, wiring, kind) per layer -- what the emitters and golden model
    both consume. Here so tests do not each re-derive it."""
    from dwn2rtl.extract import layer_indices, extract_tables, extract_wiring

    sd, n = ck['state_dict'], ck['config']['n']
    return [(extract_tables(sd, i), *extract_wiring(sd, i, n)) for i in layer_indices(sd)]


def classes_hit(ck, n_vectors=256, seed=0):
    """How many distinct classes this model predicts over random binarized input.

    A fixture that always answers the same class is nearly worthless as a gate: the testbench
    would pass against a design whose argmax, popcount and grouping were all wrong, because the
    right answer is a constant. vectors.py flags the same condition on real builds and calls it
    `degenerate`; this is the fixture-side check that stops one being built in the first place.
    """
    import numpy as _np
    from dwn2rtl.extract import forward

    layers = extract_layers(ck)
    width = ck['thermometer']['thresholds'].numpy().size
    rng = _np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(n_vectors, width)).astype(bool)
    y, _ = forward(bits, layers, ck['config']['num_classes'])
    return int(_np.unique(y).size)


# ------------------------------------------------------------------------------------------
# Live objects -- the shapes a user actually saves
# ------------------------------------------------------------------------------------------
#
# These stand in for upstream's torch_dwn classes, which are NOT a dependency of this project
# and are not installed in CI. What matters for checkpoint.py is only the duck-typing it relies
# on: state_dict key names, and a module whose class is called GroupSum and which has a `k`.
# Reproducing that here rather than importing upstream is deliberate -- it keeps the test suite
# runnable without a training stack, and it pins the exact surface the loader depends on, so an
# upstream rename shows up as a failure here rather than as a mystery on a user's machine.


class _Mapping(torch.nn.Module):
    """The LearnableMapping submodule -- contributes `<i>.mapping.weights`."""

    def __init__(self, weights):
        super().__init__()
        self.weights = torch.nn.Parameter(weights, requires_grad=False)


class LUTLayer(torch.nn.Module):
    def __init__(self, luts, mapping, learnable):
        super().__init__()
        self.luts = torch.nn.Parameter(luts, requires_grad=False)
        if learnable:
            self.mapping = _Mapping(mapping)
            # The decoy, under its real name-mangled key. Registered literally because the
            # mangling that produces it happens inside upstream's class body, not ours.
            out_size, n = luts.shape[0], int(luts.shape[1]).bit_length() - 1
            self.register_parameter(
                '_LUTLayer__dummy_mapping',
                torch.nn.Parameter(
                    torch.arange(out_size * n).reshape(out_size, n).int(),
                    requires_grad=False))
        else:
            self.mapping = torch.nn.Parameter(mapping, requires_grad=False)


class GroupSum(torch.nn.Module):
    """Matched by CLASS NAME and the `k` attribute. It has no parameters, which is the entire
    reason the class count cannot be recovered from a state_dict."""

    def __init__(self, k, tau=1 / 0.3):
        super().__init__()
        self.k = k
        self.tau = tau


class Thermometer:
    """A fitted DistributiveThermometer stand-in: an object with `.thresholds`."""

    def __init__(self, thresholds):
        self.thresholds = thresholds


def make_live(shape='tiny', **overrides):
    """(model, thermometer) equivalent to make(shape) -- the objects a user holds at the moment
    training finishes, and what `torch.save({'model':..., 'thermometer':...})` writes."""
    ck = make(shape, **overrides)
    sd, cfg = ck['state_dict'], ck['config']

    mods = []
    for i in range(len(cfg['layers'])):
        learnable = f'{i}.mapping.weights' in sd
        mapping = sd[f'{i}.mapping.weights'] if learnable else sd[f'{i}.mapping']
        mods.append(LUTLayer(sd[f'{i}.luts'], mapping, learnable))
    mods.append(GroupSum(k=cfg['num_classes']))

    model = torch.nn.Sequential(*mods)
    return model, Thermometer(ck['thermometer']['thresholds'])


def make(shape='tiny', min_classes=2, max_tries=64, **overrides):
    """make('n6'), or make('tiny', seed=3).

    Searches seeds until the model discriminates at least `min_classes` classes. The search is
    deterministic -- seeds are tried in order from the starting one -- so a given call always
    returns the same checkpoint and the vectors derived from it are reproducible.

    Why search rather than hand-pick a good seed per shape: a pinned seed silently stops being
    good the moment any shape parameter changes, and it would fail as a collapsed fixture rather
    than as an obviously wrong seed. Pass min_classes=1 to deliberately build a degenerate one.
    """
    if shape not in SHAPES:
        raise KeyError(f'unknown shape {shape!r}; have {sorted(SHAPES)}')
    params = {**SHAPES[shape], **overrides}
    start = params.pop('seed', 7)

    for seed in range(start, start + max_tries):
        ck = make_checkpoint(seed=seed, **params)
        if classes_hit(ck) >= min_classes:
            return ck
    raise RuntimeError(
        f'no seed in [{start}, {start + max_tries}) made {shape!r} discriminate '
        f'{min_classes} classes. The shape is probably degenerate by construction -- check the '
        f'popcount group size, {params["layers"][-1] // params["num_classes"]} here.')
