"""Opt-in regression against real checkpoints, which live outside this repo and skip when absent.

They catch what synthetic fixtures cannot: JSC carries a fitted StandardScaler that normalize()
was silently dropping, and no fixture had one. test_recorded_values_still_reproduce is the most
valuable here -- those numbers came from a separate implementation months earlier and nothing
was tuned to match them.

    DWN2RTL_REAL=all pytest      every checkpoint found, however long it takes
    DWN2RTL_STUDY=/path pytest   look somewhere specific
    pytest -m "not real"         skip them even when present

Two budgets, because simulation time is steep -- 0.9 s at 50 nodes against 60.6 s at 3,000 -- so
the default tier caps both the size of any one design and how many run.
"""

import os

import pytest

from dwn2rtl.build import build
from dwn2rtl.checkpoint import load
from dwn2rtl.verify import verify


# A design bigger than this is a sweep point, not a smoke test. 1,000 nodes was ~13 s of
# simulation on the development machine; 3,000 was 60 s.
MAX_DEFAULT_NODES = 1000

# And a hard ceiling on how many run at all, because a full study collection is 77 of them.
# Reference models are always included first, so raising or lowering this never drops the
# regression values.
MAX_DEFAULT_CHECKPOINTS = 6


# Values measured by the study repo's own separate implementation and recorded in its
# documentation. NOTHING HERE WAS TUNED TO MATCH -- the precision policy derives its answer from
# the thresholds alone and independently lands on both datasets' known formats.
RECORDED = {
    'dwn_jsc_t200_distributive_50_l_b100': {
        'input_bits': None,                  # continuous, standard-scaled features
        'precision': 'Q3.12 signed (16-bit)',
        'features': 16, 'classes': 5, 'layers': [50], 'n': 6,
        'nodes': 50, 'comparators': 202, 'thermometer_bits': 3200,
        'merge_collapsed': 139,
        'core_vectors': 504, 'top_vectors': 533,
        'has_scaler': True,
    },
    'mnist_n6_z3_distributive_w300': {
        'input_bits': 8,                     # pixels are k/255 -- provably lossless at frac=8
        'precision': 'Q0.8 signed (9-bit)',
        'features': 784, 'classes': 10, 'layers': [300], 'n': 6,
        'nodes': 300, 'comparators': 720, 'thermometer_bits': 2352,
        'merge_collapsed': 1169,
        'core_vectors': 504, 'top_vectors': 1227,
        'has_scaler': True,
    },
}


def _run_everything():
    return os.environ.get('DWN2RTL_REAL', '').lower() in ('all', '1', 'true')


def _search_root():
    """Where the study checkpoints might be. Explicit setting wins; otherwise look next door."""
    explicit = os.environ.get('DWN2RTL_STUDY')
    if explicit:
        return explicit if os.path.isdir(explicit) else None

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sibling = os.path.join(os.path.dirname(here), 'dwn-fpga')
    return sibling if os.path.isdir(sibling) else None


def _discover():
    """Checkpoints to run: reference models first, then the rest, then capped.

    Collection must not LOAD anything -- with 77 files that is hundreds of megabytes read just to
    decide what to skip. The node budget is applied inside each test instead.
    """
    root = _search_root()
    if not root:
        return []

    found = []
    for dirpath, _, filenames in os.walk(root):
        if '.venv' in dirpath or '.git' in dirpath:
            continue
        for f in sorted(filenames):
            if f.endswith('.pt'):
                found.append(os.path.join(dirpath, f))

    # Synthetic checkpoints the study generated for its own tests are not "real" input and add
    # nothing here -- this repo makes its own in fixtures.py.
    found = [p for p in found if 'synthetic' not in os.path.basename(p).lower()]

    def is_reference(path):
        return _stem(path) in RECORDED

    found.sort(key=lambda p: (not is_reference(p), p))
    return found if _run_everything() else found[:MAX_DEFAULT_CHECKPOINTS]


def _stem(path):
    return os.path.basename(path).replace('_checkpoint.pt', '').replace('.pt', '')


REAL = _discover()

requires_real = pytest.mark.skipif(
    not REAL,
    reason='no study checkpoints found -- set DWN2RTL_STUDY=<dir> to point at them')

# The `sim` marker is enough -- conftest.py skips those when there is no simulator.


def _input_bits_for(stem):
    """MNIST pixels have a native 8-bit quantum; JSC's standard-scaled features do not.

    Guessing from the name is crude, and it is honest about being crude: --input-bits is the one
    thing the tool asks the USER for precisely because a checkpoint cannot answer it.
    """
    if stem in RECORDED:
        return RECORDED[stem]['input_bits']
    return 8 if stem.startswith('mnist') else None


def _skip_if_too_big(ck, stem):
    nodes = sum(ck['config']['layers'])
    if nodes > MAX_DEFAULT_NODES and not _run_everything():
        pytest.skip(f'{stem} has {nodes} nodes (~{nodes // 50}s of simulation); '
                    'set DWN2RTL_REAL=all to include it')
    return nodes


# ---------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------

@pytest.mark.real
@requires_real
@pytest.mark.parametrize('path', REAL, ids=_stem)
def test_real_checkpoint_loads_unmodified(path):
    """No conversion step, ever. These carry keys the tool does not use and must simply ignore --
    requiring a migration would put a step between the evidence and the tool.
    """
    ck = load(path)
    assert ck['config']['num_classes'] >= 2
    assert ck['state_dict']
    assert ck['thermometer']['thresholds'].ndim == 2


# ---------------------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------------------

@pytest.mark.real
@pytest.mark.sim
@requires_real
@pytest.mark.parametrize('path', REAL, ids=_stem)
def test_real_checkpoint_builds_and_passes_the_gate(path, tmp_path):
    """A real trained model, all the way to a simulator saying bit-exact."""
    stem = _stem(path)
    ck = load(path)
    _skip_if_too_big(ck, stem)

    r = build(ck, str(tmp_path / 'rtl'), input_bits=_input_bits_for(stem))
    report = verify(r.outdir)

    assert report.ok, '\n'.join(report.lines())
    assert all(lv.mismatches == 0 for lv in report.levels)


# ---------------------------------------------------------------------------------------
# The regression that no synthetic fixture can provide
# ---------------------------------------------------------------------------------------

@pytest.mark.real
@requires_real
@pytest.mark.parametrize('stem', sorted(RECORDED))
def test_recorded_values_still_reproduce(stem, tmp_path):
    """The highest-value test here: numbers measured by a SEPARATE implementation months earlier,
    which this code was never fitted to. A refactor that moves any of them is a real regression,
    and no synthetic fixture can notice.
    """
    path = next((p for p in REAL if _stem(p) == stem), None)
    if path is None:
        pytest.skip(f'{stem} not among the discovered checkpoints')

    want = RECORDED[stem]
    r = build(path, str(tmp_path / 'rtl'), input_bits=want['input_bits'])

    assert str(r.precision) == want['precision']
    assert r.features == want['features']
    assert r.classes == want['classes']
    assert r.layers == want['layers']
    assert r.n == want['n']
    assert r.nodes == want['nodes']
    assert r.comparators == want['comparators']
    assert r.thermometer_bits == want['thermometer_bits']
    assert r.merge[2] == want['merge_collapsed']
    assert r.merge[1] == want['thermometer_bits']
    assert r.vectors['core']['count'] == want['core_vectors']
    assert r.vectors['top']['count'] == want['top_vectors']


@pytest.mark.real
@requires_real
@pytest.mark.parametrize('stem', sorted(RECORDED))
def test_the_scaler_survives(stem, tmp_path):
    """The defect real checkpoints found and fixtures could not: normalize() dropped the scaler, so
    every emitted design silently required a transformation the user could not learn.
    """
    path = next((p for p in REAL if _stem(p) == stem), None)
    if path is None:
        pytest.skip(f'{stem} not among the discovered checkpoints')
    if not RECORDED[stem]['has_scaler']:
        pytest.skip(f'{stem} records no scaler')

    ck = load(path)
    assert ck['scaler'] and 'mean' in ck['scaler']
    assert len(ck['scaler']['mean']) == RECORDED[stem]['features']

    r = build(ck, str(tmp_path / 'rtl'), input_bits=RECORDED[stem]['input_bits'])
    assert os.path.exists(os.path.join(r.outdir, 'input_scaling.json'))
    assert any('SCALED features' in w for w in r.warnings)
    assert 'input_scaling.json' in open(
        os.path.join(r.outdir, 'thermometer_encoder.v')).read()


# ---------------------------------------------------------------------------------------
# The opt-in machinery itself
# ---------------------------------------------------------------------------------------

def test_discovery_is_bounded_by_default():
    """A full study collection is 77 checkpoints and hundreds of megabytes. Whatever is on this
    machine, the default run must stay bounded -- otherwise `pytest` silently becomes a
    half-hour command on someone's laptop."""
    if not REAL or _run_everything():
        pytest.skip('nothing found, or the cap is deliberately lifted')
    assert len(REAL) <= MAX_DEFAULT_CHECKPOINTS


def test_reference_models_are_never_dropped_by_the_cap():
    """The cap must not be able to discard the checkpoints with known-correct answers."""
    if not REAL:
        pytest.skip('no study checkpoints found')
    available = {_stem(p) for p in _discover()}
    for stem in RECORDED:
        root = _search_root()
        exists = root and any(
            _stem(f) == stem
            for _, _, fs in os.walk(root) for f in fs if f.endswith('.pt'))
        if exists:
            assert stem in available, f'{stem} has recorded values but was capped out'
