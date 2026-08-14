"""The synthetic checkpoints must be STRUCTURALLY REAL, or every test built on them is theatre.

A fixture that violates docs/checkpoint-format.md does not merely fail to catch bugs -- it
invents ones that cannot happen and hides ones that can. So the fixture gets tested against the
same rules the format spec states, and against the extractor that will consume it.
"""

import numpy as np
import pytest

import fixtures
from dwn2rtl.extract import extract_wiring, layer_indices


ALL_SHAPES = sorted(fixtures.SHAPES)


@pytest.mark.parametrize('shape', ALL_SHAPES)
def test_shape_is_what_it_claims(shape):
    ck = fixtures.make(shape)
    cfg = ck['config']
    spec = fixtures.SHAPES[shape]

    assert cfg['n'] == spec['n']
    assert cfg['num_classes'] == spec['num_classes']
    assert cfg['layers'] == list(spec['layers'])
    assert cfg['thermometer_bits'] == spec['z']

    thr = ck['thermometer']['thresholds'].numpy()
    assert thr.shape == (spec['n_features'], spec['z'])


@pytest.mark.parametrize('shape', ALL_SHAPES)
def test_tables_are_the_right_shape_and_range(shape):
    """(output_size, 2**n) float32 in [-1, 1] -- docs/checkpoint-format.md §1.

    The range is not cosmetic: upstream clamps into it on every training forward pass, so a
    checkpoint outside it would not be one a user could have.
    """
    ck = fixtures.make(shape)
    n, widths = ck['config']['n'], ck['config']['layers']
    for i, width in zip(layer_indices(ck['state_dict']), widths):
        luts = ck['state_dict'][f'{i}.luts'].numpy()
        assert luts.shape == (width, 2 ** n)
        assert luts.dtype == np.float32
        assert luts.min() >= -1.0 and luts.max() <= 1.0


@pytest.mark.parametrize('shape', ALL_SHAPES)
def test_wiring_indices_are_in_range_of_the_previous_layer(shape):
    """A node cannot read an input bit that does not exist.

    Out-of-range wiring is not caught by the emitter -- it produces `prev[97]` on a 24-bit bus,
    which some tools warn about and some silently return x for.
    """
    ck = fixtures.make(shape)
    n = ck['config']['n']
    width_in = ck['thermometer']['thresholds'].numpy().size
    for i, out_width in zip(layer_indices(ck['state_dict']), ck['config']['layers']):
        wiring, _ = extract_wiring(ck['state_dict'], i, n)
        assert wiring.shape == (out_width, n)
        assert wiring.min() >= 0
        assert wiring.max() < width_in, f'layer {i} reads bit {wiring.max()} of {width_in}'
        width_in = out_width


@pytest.mark.parametrize('shape', ALL_SHAPES)
def test_thresholds_are_sorted_within_each_feature(shape):
    """Thermometer thresholds are quantiles, so they are sorted. An unsorted fixture would let
    an ordering bug pass here and fail on every real model."""
    thr = fixtures.make(shape)['thermometer']['thresholds'].numpy()
    assert np.all(np.diff(thr, axis=1) >= 0)


@pytest.mark.parametrize('shape', ALL_SHAPES)
def test_final_layer_is_divisible_by_num_classes(shape):
    """Otherwise GroupSum zero-pads SILENTLY and hardware and software disagree about group
    boundaries -- docs/checkpoint-format.md §4."""
    cfg = fixtures.make(shape)['config']
    assert cfg['layers'][-1] % cfg['num_classes'] == 0


def test_indivisible_final_layer_is_refused_at_construction():
    with pytest.raises(ValueError, match='divisible'):
        fixtures.make_checkpoint(layers=(7,), num_classes=2)


# --------------------------------------------------------------------------------------
# The decoy. This is the most valuable test in the file.
# --------------------------------------------------------------------------------------

def test_learnable_layers_carry_the_dummy_mapping_decoy():
    """`_LUTLayer__dummy_mapping` must be PRESENT, because it is present in every real
    learnable checkpoint and the fixture's job is to look like one."""
    ck = fixtures.make('tiny')
    assert '0._LUTLayer__dummy_mapping' in ck['state_dict']
    assert '0.mapping.weights' in ck['state_dict']


def test_the_decoy_is_not_what_the_extractor_returns():
    """docs/checkpoint-format.md §3: keying off tensor shape instead of off `.mapping.weights`
    reads the decoy and emits a structurally valid, COMPLETELY WRONG model.

    The decoy has the same (output_size, n) shape and int dtype as a genuine fixed mapping, so
    nothing about its type gives it away. It is only arange() reshaped. This test asserts the
    extractor returns the argmax-derived wiring instead -- and that the two actually differ, so
    the test cannot pass by coincidence.
    """
    ck = fixtures.make('tiny')
    n, out_size = ck['config']['n'], ck['config']['layers'][0]

    decoy = ck['state_dict']['0._LUTLayer__dummy_mapping'].numpy()
    assert decoy.shape == (out_size, n)
    assert np.array_equal(decoy, np.arange(out_size * n).reshape(out_size, n))

    wiring, kind = extract_wiring(ck['state_dict'], 0, n)
    assert kind == 'learnable'
    assert not np.array_equal(wiring, decoy), 'the extractor returned the decoy'


def test_learnable_wiring_is_the_argmax_over_the_input_axis():
    """§3a: node j slot k reads input bit weights.argmax(dim=0)[j*n + k]. Recomputed here from
    the raw tensor, independently of extract_wiring's own reshape."""
    ck = fixtures.make('tiny')
    n = ck['config']['n']
    w = ck['state_dict']['0.mapping.weights'].numpy()
    expected = w.argmax(axis=0).reshape(-1, n)

    wiring, _ = extract_wiring(ck['state_dict'], 0, n)
    assert np.array_equal(wiring, expected)


def test_fixed_layers_have_no_learnable_weights_and_no_decoy():
    ck = fixtures.make('all_fixed')
    for key in ck['state_dict']:
        assert 'mapping.weights' not in key
        assert '__dummy_mapping' not in key
    _, kind = extract_wiring(ck['state_dict'], 0, ck['config']['n'])
    assert kind == 'fixed'


def test_both_mapping_representations_appear_in_the_default_shape():
    """Upstream's own recipe is learnable then random, and the two paths share no code."""
    kinds = [k for *_, k in fixtures.extract_layers(fixtures.make('tiny'))]
    assert kinds == ['learnable', 'fixed']


# --------------------------------------------------------------------------------------
# Fixture quality
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize('shape', ALL_SHAPES)
def test_every_shape_discriminates_all_of_its_classes(shape):
    """A fixture that always answers the same class would pass a testbench against a design
    whose argmax, popcount and grouping were all wrong, because the right answer is a constant.

    The first cut of fixtures.py did exactly this -- groups of 2, untrained tables, and every
    vector landing on one class. Groups are >= 3 now and make() searches seeds.
    """
    ck = fixtures.make(shape)
    K = ck['config']['num_classes']
    assert fixtures.classes_hit(ck) == K, f'{shape} hit fewer than {K} classes'


@pytest.mark.parametrize('shape', ALL_SHAPES)
def test_popcount_groups_are_big_enough_to_break_ties(shape):
    """Group size 2 with untrained tables is degenerate by construction: a node is constant
    whenever its 2**n table is all-positive, and two constant nodes score a permanent maximum."""
    ck = fixtures.make(shape)
    cfg = ck['config']
    assert cfg['layers'][-1] // cfg['num_classes'] >= 3


def test_make_is_deterministic():
    """Same call, same checkpoint -- including through the seed search. Vectors derived from a
    fixture must be reproducible or 'it passed yesterday' stops being checkable."""
    a, b = fixtures.make('tiny'), fixtures.make('tiny')
    assert a['config'] == b['config']
    assert set(a['state_dict']) == set(b['state_dict'])
    for k in a['state_dict']:
        assert np.array_equal(a['state_dict'][k].numpy(), b['state_dict'][k].numpy())
    assert np.array_equal(a['thermometer']['thresholds'].numpy(),
                          b['thermometer']['thresholds'].numpy())


def test_unknown_shape_is_refused():
    with pytest.raises(KeyError, match='unknown shape'):
        fixtures.make('does-not-exist')
