"""checkpoint.py -- the format, and the errors that stop a wrong one reaching the emitter.

Two halves, and the second matters more: the three accepted shapes normalize to the same thing,
and every rejected shape produces an error that NAMES the problem.

⚠️ That is not politeness. A bare state_dict drops the encoder and yields a design that
synthesizes cleanly and classifies at chance. "Invalid checkpoint" gets re-saved the same way;
naming the thermometer gets it fixed in one line. So the error text is part of the contract.
"""

import copy

import pytest
import torch

import fixtures
from dwn2rtl.checkpoint import (CheckpointError, from_model, load, normalize, save, summary)


ALL_SHAPES = sorted(fixtures.SHAPES)


# ---------------------------------------------------------------------------------------
# The three accepted shapes converge
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize('shape', ALL_SHAPES)
def test_the_three_input_shapes_produce_identical_configs(shape, tmp_path):
    """A checkpoint's meaning must not depend on how the user happened to save it."""
    model, therm = fixtures.make_live(shape)

    a = from_model(model, therm)

    p = tmp_path / 'plain.pt'
    torch.save({'model': model, 'thermometer': therm}, p)      # the documented primary path
    b = load(p)

    c = normalize(fixtures.make(shape))                        # study-repo shaped dict

    assert a['config'] == b['config'] == c['config']
    assert summary(a) == summary(b) == summary(c)


@pytest.mark.parametrize('shape', ALL_SHAPES)
def test_derived_config_matches_the_fixture_it_was_built_from(shape):
    """from_model derives n, layers and z from tensors and num_classes from GroupSum. All four
    must come back equal to what the fixture declared."""
    model, therm = fixtures.make_live(shape)
    assert from_model(model, therm)['config'] == fixtures.make(shape)['config']


def test_save_load_round_trip_preserves_everything(tmp_path):
    model, therm = fixtures.make_live('n6')
    p = tmp_path / 'model.dwn'
    written = save(model, therm, p, run_name='rt', results={'final_acc': 0.97})
    read = load(p)

    assert read['config'] == written['config']
    assert read['run_name'] == 'rt'
    assert read['results'] == {'final_acc': 0.97}
    assert set(read['state_dict']) == set(written['state_dict'])
    for k in read['state_dict']:
        assert torch.equal(read['state_dict'][k], written['state_dict'][k])
    assert torch.equal(read['thermometer']['thresholds'],
                       written['thermometer']['thresholds'])


def test_metadata_is_optional_and_absent_is_not_zero():
    """`results` reaches the emitted header comment and nothing else, so it must not be
    mandatory. Absent means NOT RECORDED, which is a different claim from an accuracy of 0.0
    and has to print differently -- so it normalizes to empty, never to a fabricated number."""
    ck = normalize(fixtures.make('tiny'))
    assert ck['results'] == {}
    assert 'run_name' in ck

    model, therm = fixtures.make_live('tiny')
    assert from_model(model, therm)['run_name'] == 'unnamed'


def test_thermometer_may_be_an_object_a_dict_or_a_bare_tensor():
    """All three are things people genuinely have. Making them unwrap it themselves is a worse
    error message than handling it."""
    ck = fixtures.make('tiny')
    thr = ck['thermometer']['thresholds']
    model, _ = fixtures.make_live('tiny')

    as_object = from_model(model, fixtures.Thermometer(thr))
    as_dict = from_model(model, {'thresholds': thr})
    as_tensor = from_model(model, thr)

    assert as_object['config'] == as_dict['config'] == as_tensor['config']


def test_numpy_thresholds_are_accepted():
    model, _ = fixtures.make_live('tiny')
    thr = fixtures.make('tiny')['thermometer']['thresholds'].numpy()
    assert from_model(model, thr)['thermometer']['thresholds'].shape == thr.shape


# ---------------------------------------------------------------------------------------
# The input scaling -- part of the hardware's contract, not metadata
# ---------------------------------------------------------------------------------------

def test_a_scaler_is_preserved_not_dropped():
    """Found against a real checkpoint: normalize() threw the scaler away while the emitted header
    claimed it was carried. Thresholds live in whatever space training used, so raw features
    give a design that runs at chance and looks healthy.
    """
    ck = fixtures.make('tiny')
    n_features = ck['thermometer']['thresholds'].shape[0]
    ck['scaler'] = {'mean': [0.5] * n_features, 'scale': [2.0] * n_features}

    out = normalize(ck)
    assert out['scaler']['mean'] == [0.5] * n_features
    assert out['scaler']['scale'] == [2.0] * n_features


def test_no_scaler_normalizes_to_none_which_is_itself_a_fact():
    """None means "training used raw features", which the emitted header states positively."""
    assert normalize(fixtures.make('tiny'))['scaler'] is None


def test_a_live_sklearn_style_scaler_is_accepted():
    class FakeStandardScaler:
        def __init__(self, n):
            self.mean_ = [1.0] * n
            self.scale_ = [3.0] * n

    model, therm = fixtures.make_live('tiny')
    n = fixtures.make('tiny')['thermometer']['thresholds'].shape[0]
    ck = from_model(model, therm, scaler=FakeStandardScaler(n))
    assert ck['scaler']['scale'] == [3.0] * n


def test_a_scaler_of_the_wrong_length_is_refused():
    """A per-feature vector that does not match the feature count would be applied off-by-one
    across the whole input, which is silent and catastrophic."""
    ck = fixtures.make('tiny')
    ck['scaler'] = {'mean': [0.0] * 99, 'scale': [1.0] * 99}
    with pytest.raises(CheckpointError, match='but the model has'):
        normalize(ck)


def test_an_unrecognized_scaler_is_flagged_rather_than_ignored():
    """Refusing outright would reject a checkpoint over something the hardware does not depend
    on; ignoring it silently is the defect this fixes. So: recorded, and warned about."""
    ck = fixtures.make('tiny')
    ck['scaler'] = 'some pickled thing'
    assert normalize(ck)['scaler'] == {'unrecognized': 'str'}


# ---------------------------------------------------------------------------------------
# THE error. Roadmap Q8.
# ---------------------------------------------------------------------------------------

def test_bare_state_dict_is_refused_and_names_the_thermometer():
    """The whole reason this module exists: `torch.save(model.state_dict())` is what a PyTorch user
    reaches for, and it drops the thermometer and with it the encoder.
    """
    model, _ = fixtures.make_live('tiny')
    with pytest.raises(CheckpointError) as e:
        normalize(model.state_dict())

    msg = str(e.value)
    assert 'thermometer' in msg.lower(), 'the error must name the missing object'
    assert 'state_dict' in msg
    assert 'two objects' in msg, 'it must say WHY, not just what'
    assert "torch.save({'model': model, 'thermometer': thermometer}" in msg, \
        'it must show the fix, not just describe it'


def test_a_model_on_its_own_is_refused():
    model, _ = fixtures.make_live('tiny')
    with pytest.raises(CheckpointError, match='no thermometer'):
        normalize(model)


def test_a_dict_with_a_model_but_no_thermometer_is_refused():
    model, _ = fixtures.make_live('tiny')
    with pytest.raises(CheckpointError, match='NO THERMOMETER'):
        normalize({'model': model})


def test_a_state_dict_under_the_model_key_explains_the_class_count():
    """{'model': <state_dict>, 'thermometer': ...} has the encoder but has lost GroupSum, and
    with it the only record of how many classes there are."""
    model, therm = fixtures.make_live('tiny')
    with pytest.raises(CheckpointError, match='GroupSum'):
        normalize({'model': model.state_dict(), 'thermometer': therm})


def test_a_model_with_no_groupsum_is_refused_by_name():
    """GroupSum has no parameters, so the class count exists nowhere else."""
    model, therm = fixtures.make_live('tiny')
    stripped = torch.nn.Sequential(*list(model.children())[:-1])
    with pytest.raises(CheckpointError, match='GroupSum'):
        from_model(stripped, therm)


@pytest.mark.parametrize('obj, match', [
    ({'weights': 1, 'bias': 2}, 'unrecognized'),
    ('a string', 'unrecognized'),
    (12345, 'unrecognized'),
])
def test_unrecognized_inputs_say_what_was_expected(obj, match):
    with pytest.raises(CheckpointError, match=match):
        normalize(obj)


# ---------------------------------------------------------------------------------------
# Validation -- corruptions that would otherwise emit a plausible, wrong design
# ---------------------------------------------------------------------------------------

def _corrupt(shape='tiny', **_):
    return copy.deepcopy(fixtures.make(shape))


def test_transposed_thresholds_are_caught_and_named():
    """(z, features) has the right rank, right dtype and plausible values, and would emit an
    encoder with features and thresholds swapped. Detectable only because config states z."""
    ck = _corrupt()
    ck['thermometer']['thresholds'] = ck['thermometer']['thresholds'].T
    with pytest.raises(CheckpointError, match='TRANSPOSED'):
        normalize(ck)


def test_config_n_must_match_the_table_width():
    ck = _corrupt()
    ck['config']['n'] = 4
    with pytest.raises(CheckpointError, match=r'2\*\*2 table entries but config says n=4'):
        normalize(ck)


def test_config_layers_must_match_the_tables():
    ck = _corrupt()
    ck['config']['layers'] = [99, 8]
    with pytest.raises(CheckpointError, match='but the tables are'):
        normalize(ck)


def test_indivisible_class_count_is_refused():
    """GroupSum zero-pads silently, so hardware and software would disagree about group
    boundaries -- docs/checkpoint-format.md §4."""
    ck = _corrupt()
    ck['config']['num_classes'] = 5
    with pytest.raises(CheckpointError, match='zero-pad silently'):
        normalize(ck)


def test_out_of_range_wiring_is_refused():
    """An index past the input width emits `prev[97]` on a 24-bit bus -- some tools warn, some
    resolve it to x."""
    ck = _corrupt('all_fixed')
    ck['state_dict']['0.mapping'] = torch.full_like(ck['state_dict']['0.mapping'], 999)
    with pytest.raises(CheckpointError, match='wiring reads input bit'):
        normalize(ck)


def test_non_power_of_two_table_is_refused():
    ck = _corrupt()
    ck['state_dict']['0.luts'] = ck['state_dict']['0.luts'][:, :3]
    with pytest.raises(CheckpointError, match='not a power of two'):
        normalize(ck)


def test_missing_config_keys_are_listed():
    ck = _corrupt()
    del ck['config']['thermometer_bits']
    with pytest.raises(CheckpointError, match=r"missing \['thermometer_bits'\]"):
        normalize(ck)


def test_non_contiguous_layer_indices_are_refused():
    ck = _corrupt()
    sd = ck['state_dict']
    sd['5.luts'] = sd.pop('1.luts')
    sd['5.mapping'] = sd.pop('1.mapping')
    ck['config']['layers'] = [12, 8]
    with pytest.raises(CheckpointError, match='not contiguous'):
        normalize(ck)


def test_one_dimensional_thresholds_are_refused():
    ck = _corrupt()
    ck['thermometer']['thresholds'] = ck['thermometer']['thresholds'].flatten()
    with pytest.raises(CheckpointError, match='expected 2-D'):
        normalize(ck)


def test_the_upstream_pin_agrees_with_the_documentation():
    """One pin in two places that must not drift. If the doc and checkpoint.py disagree, one is
    lying about what was actually read.
    """
    import os
    from dwn2rtl.checkpoint import UPSTREAM_COMMIT, UPSTREAM_URL

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    doc = open(os.path.join(root, 'docs', 'checkpoint-format.md'), encoding='utf-8').read()

    assert UPSTREAM_COMMIT in doc, 'the pinned commit is not the one the document verified'
    assert UPSTREAM_URL in doc
    assert len(UPSTREAM_COMMIT) == 40, 'pin a full SHA -- short ones are ambiguous over time'


def test_the_upstream_pin_is_not_enforced_at_load_time():
    """Provenance, not a version gate. Loading is duck-typed so a model trained against another
    commit is not refused; a rename should surface as a real message, not "wrong version".
    """
    ck = fixtures.make('tiny')
    ck['pinned_commit'] = 'something-else-entirely'
    assert normalize(ck)['config']['n'] == 2


def test_checkpoint_error_is_a_valueerror_not_a_systemexit():
    """A library that kills its caller's process from inside a function they merely imported is
    not usable from a notebook."""
    assert issubclass(CheckpointError, ValueError)
    assert not issubclass(CheckpointError, SystemExit)


# ---------------------------------------------------------------------------------------
# The lazy API surface
# ---------------------------------------------------------------------------------------

def test_checkpoint_api_is_reachable_from_the_package_root():
    import dwn2rtl
    for name in ('load', 'save', 'from_model', 'normalize', 'CheckpointError'):
        assert callable(getattr(dwn2rtl, name)) or isinstance(getattr(dwn2rtl, name), type)


def test_reaching_the_checkpoint_api_does_not_break_the_torch_boundary():
    """`import dwn2rtl` stays torch-free even though dwn2rtl.load exists. PEP 562 __getattr__ is how
    both are true at once.
    """
    import subprocess
    import sys

    code = ('import sys, dwn2rtl; before = "torch" in sys.modules; '
            'dwn2rtl.load; after = "torch" in sys.modules; print(before, after)')
    out = subprocess.run([sys.executable, '-c', code],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == 'False True'


def test_unknown_attribute_still_raises_attributeerror():
    import dwn2rtl
    with pytest.raises(AttributeError, match='no attribute'):
        dwn2rtl.definitely_not_a_thing
