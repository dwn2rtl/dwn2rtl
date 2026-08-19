"""What a checkpoint is, and the errors that stop a wrong one.

Upstream DWN saves nothing, so this format is one the tool invented. Three shapes are accepted:
a plain {'model', 'thermometer'} dict, the study's older {config, state_dict, ...} form, and a
live model via from_model().

⚠️ A DWN is TWO objects. Thresholds are fitted before training and are not model parameters, so
`torch.save(model.state_dict())` -- what everyone reaches for -- silently loses the encoder, and
the encoder can be most of the design. That file is refused by name rather than built from.

Nothing here imports the upstream package; every fact is duck-typed off the tensors.
"""

import re
from collections.abc import Mapping

import numpy as np
import torch

# Provenance: the commit docs/checkpoint-format.md was verified against. Not a dependency and
# not a runtime check -- loading is duck-typed, so a user's model may be trained against another.
# ⚠️ Do not turn this into a version assertion.
UPSTREAM_URL = 'https://github.com/alanbacellar/DWN'
UPSTREAM_COMMIT = '9f887a0b4bd84dabf6d8c9ae35368ab2a7e0e3c0'

# Always present after normalize(), so consumers need no defensive .get() chains.
REQUIRED_CONFIG = ('n', 'num_classes', 'layers', 'thermometer_bits')

_LAYER_KEY = re.compile(r'(\d+)\.(luts|mapping|mapping\.weights|_LUTLayer__dummy_mapping)$')


class CheckpointError(ValueError):
    """A checkpoint that cannot be built from, with an explanation of what is missing.

    ValueError rather than SystemExit: this is a library, and a library that kills its caller's
    process from inside a function they merely imported is not usable from a notebook.
    """


# ---------------------------------------------------------------------------------------
# Reading the two objects
# ---------------------------------------------------------------------------------------

def _as_tensor(x, what):
    if torch.is_tensor(x):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    raise CheckpointError(f'{what} is a {type(x).__name__}, expected a tensor or array')


def _thresholds_of(thermometer):
    """Pull the (features, z) threshold matrix out of whatever a user hands us.

    Accepts the fitted object itself (`.thresholds`), the dict form this tool saves, or a bare
    tensor/array -- because all three are things people genuinely have, and telling them to
    unwrap it themselves is a worse error message than just handling it.
    """
    if thermometer is None:
        raise CheckpointError('no thermometer')

    if hasattr(thermometer, 'thresholds'):
        t = thermometer.thresholds
    elif isinstance(thermometer, Mapping):
        if 'thresholds' not in thermometer:
            raise CheckpointError(
                f'thermometer dict has keys {sorted(thermometer)} but no "thresholds"')
        t = thermometer['thresholds']
    else:
        t = thermometer

    t = _as_tensor(t, 'thermometer thresholds').detach().cpu().float()
    if t.ndim != 2:
        raise CheckpointError(
            f'thermometer thresholds have shape {tuple(t.shape)}; expected 2-D '
            '(features, bits_per_feature)')
    return t


def _scaler_of(obj, n_features):
    """The input scaling the model was TRAINED with, if the checkpoint records one.

    ⚠️ Part of the hardware's input contract, not metadata. Thresholds live in whatever feature
    space training used, so whatever drives x_flat must apply the same scaling first -- without
    it the design runs at chance and looks healthy doing it.
    """
    if not isinstance(obj, Mapping):
        return None
    raw = obj.get('scaler')
    if raw is None:
        return None

    if hasattr(raw, 'mean_') and hasattr(raw, 'scale_'):        # a live sklearn scaler
        mean, scale = raw.mean_, raw.scale_
    elif isinstance(raw, Mapping) and 'mean' in raw and 'scale' in raw:
        mean, scale = raw['mean'], raw['scale']
    else:
        # Present but unrecognized. Refusing would reject a checkpoint over something the
        # hardware does not depend on; ignoring it silently is what caused this defect. Say so.
        return {'unrecognized': type(raw).__name__}

    mean = np.asarray(mean, dtype=np.float64).ravel()
    scale = np.asarray(scale, dtype=np.float64).ravel()
    if mean.size != n_features or scale.size != n_features:
        raise CheckpointError(
            f'scaler has {mean.size} means and {scale.size} scales but the model has '
            f'{n_features} features')
    return {'mean': mean.tolist(), 'scale': scale.tolist()}


def _looks_like_a_state_dict(obj):
    """A mapping of tensors keyed `<layer>.<param>` -- i.e. what torch.save(m.state_dict()) makes."""
    if not isinstance(obj, Mapping) or not obj:
        return False
    keys = list(obj)
    if not all(isinstance(k, str) for k in keys):
        return False
    if not any(_LAYER_KEY.fullmatch(k) for k in keys):
        return False
    return all(torch.is_tensor(v) for v in obj.values())


def _bare_state_dict_error(keys):
    """THE error this module exists for. It has to name the thermometer, not just say 'invalid'.

    A user who sees "invalid checkpoint" re-saves it the same way. A user who is told the
    thermometer is a separate object fixes it in one line and never hits this again.
    """
    shown = ', '.join(repr(k) for k in list(keys)[:4])
    return CheckpointError(
        'this looks like a bare state_dict -- it has the model weights but NO THERMOMETER.\n'
        f'  keys: {shown}{", ..." if len(list(keys)) > 4 else ""}\n'
        '\n'
        '  A DWN is two objects. The thermometer is fitted before training and is not a\n'
        '  parameter of the model, so torch.save(model.state_dict()) drops the encoder\n'
        '  entirely -- and the encoder can be many times the size of the network it feeds.\n'
        '\n'
        '  Re-save with both:\n'
        "      torch.save({'model': model, 'thermometer': thermometer}, 'model.pt')\n"
        '\n'
        '  or, if the model is still in memory:\n'
        '      dwn2rtl.save(model, thermometer, "model.pt")')


# ---------------------------------------------------------------------------------------
# Deriving the config from live objects
# ---------------------------------------------------------------------------------------

def _layer_indices(state_dict):
    return sorted({int(m.group(1)) for k in state_dict
                   if (m := re.fullmatch(r'(\d+)\.luts', k))})


def _n_from_tables(state_dict, i):
    """A table has 2**n entries, so n is exactly log2 of its width -- no guessing."""
    width = state_dict[f'{i}.luts'].shape[1]
    n = int(width).bit_length() - 1
    if 1 << n != width:
        raise CheckpointError(
            f'layer {i} has {width} table entries, which is not a power of two. A LUT node with '
            'n inputs has exactly 2**n entries')
    return n


def _num_classes_from_model(model):
    """GroupSum's `k`, which is the ONE thing no tensor knows.

    GroupSum has no parameters, so it contributes nothing to state_dict. It has to come off the
    live module. Matched on the attribute rather than on an isinstance check, so this module
    never has to import the upstream package.
    """
    found = [m for m in model.modules()
             if type(m).__name__ == 'GroupSum' and hasattr(m, 'k')]
    if not found:
        candidates = [type(m).__name__ for m in model.modules()][1:]
        raise CheckpointError(
            'no GroupSum layer found, so the number of classes cannot be determined.\n'
            f'  modules seen: {candidates}\n'
            '  A DWN ends in dwn.GroupSum(k=<num_classes>, tau=...); the class count lives '
            'there and\n  nowhere else, because GroupSum has no parameters and so is absent '
            'from state_dict.')
    if len(found) > 1:
        raise CheckpointError(f'{len(found)} GroupSum layers found; expected exactly one')
    return int(found[0].k)


# ---------------------------------------------------------------------------------------
# Validation -- run on every path in, however the checkpoint arrived
# ---------------------------------------------------------------------------------------

def _validate(ck):
    cfg = ck['config']
    missing = [k for k in REQUIRED_CONFIG if k not in cfg]
    if missing:
        raise CheckpointError(f'config is missing {missing}; has {sorted(cfg)}')

    sd = ck['state_dict']
    idx = _layer_indices(sd)
    if not idx:
        raise CheckpointError(
            'no LUT layers found -- expected at least one "<i>.luts" tensor, got '
            f'{sorted(sd)[:6]}')
    if idx != list(range(len(idx))):
        raise CheckpointError(f'layer indices are not contiguous from 0: {idx}')

    n, K, z = cfg['n'], cfg['num_classes'], cfg['thermometer_bits']
    thr = ck['thermometer']['thresholds']

    # Orientation. Our convention is (features, z) -- emit_encoder reads shape[0] as the feature
    # count. A transposed matrix has the right rank, the right dtype and plausible values, and
    # would silently emit an encoder with features and thresholds swapped. Cheap to detect
    # because config already states z independently.
    if thr.shape[1] != z:
        if thr.shape[0] == z:
            raise CheckpointError(
                f'thermometer thresholds are {tuple(thr.shape)} but config says '
                f'thermometer_bits={z}. This looks TRANSPOSED -- the expected shape is '
                f'(features, bits_per_feature), i.e. ({thr.shape[1]}, {z}).')
        raise CheckpointError(
            f'thermometer thresholds are {tuple(thr.shape)}, so {thr.shape[1]} bits per '
            f'feature, but config says thermometer_bits={z}')

    widths = []
    for i in idx:
        got_n = _n_from_tables(sd, i)
        if got_n != n:
            raise CheckpointError(
                f'layer {i} has 2**{got_n} table entries but config says n={n}')
        widths.append(int(sd[f'{i}.luts'].shape[0]))

    if list(cfg['layers']) != widths:
        raise CheckpointError(
            f'config says layers={list(cfg["layers"])} but the tables are {widths} wide')

    # GroupSum zero-pads silently when the final width is not divisible, and then hardware and
    # software disagree about where each class's group starts -- docs/checkpoint-format.md §4.
    if widths[-1] % K:
        raise CheckpointError(
            f'final layer is {widths[-1]} nodes and num_classes is {K}, which does not divide '
            f'it. GroupSum would zero-pad silently and the hardware and the model would '
            f'disagree about group boundaries.')

    # Wiring must address bits that exist. An out-of-range index emits `prev[97]` on a 24-bit
    # bus, which some tools warn about and some resolve to x.
    from .extract import extract_wiring

    width_in = int(thr.numel())
    for i in idx:
        wiring, _ = extract_wiring(sd, i, n)
        if wiring.size and (wiring.min() < 0 or wiring.max() >= width_in):
            raise CheckpointError(
                f'layer {i} wiring reads input bit {int(wiring.max())} but its input is only '
                f'{width_in} bits wide')
        width_in = widths[i]

    return ck


# ---------------------------------------------------------------------------------------
# The public surface
# ---------------------------------------------------------------------------------------

def from_model(model, thermometer, run_name=None, results=None, scaler=None):
    """Build a checkpoint from live objects -- what a user has the moment training ends.

    This is the notebook path, and it is what hls4ml does. Everything except the class count is
    read from the tensors; the class count comes off GroupSum, which has no parameters.
    """
    if thermometer is None:
        raise _bare_state_dict_error(getattr(model, 'state_dict', dict)())

    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    idx = _layer_indices(state_dict)
    if not idx:
        raise CheckpointError(
            f'{type(model).__name__} has no LUT layers -- expected "<i>.luts" tensors in its '
            f'state_dict, got {sorted(state_dict)[:6]}')

    thr = _thresholds_of(thermometer)
    ck = {
        'config': {
            'n': _n_from_tables(state_dict, idx[0]),
            'num_classes': _num_classes_from_model(model),
            'layers': [int(state_dict[f'{i}.luts'].shape[0]) for i in idx],
            'thermometer_bits': int(thr.shape[1]),
        },
        'state_dict': state_dict,
        'thermometer': {'thresholds': thr},
        'results': dict(results or {}),
        'run_name': run_name or 'unnamed',
        'scaler': _scaler_of({'scaler': scaler}, int(thr.shape[0])) if scaler is not None
        else None,
    }
    return _validate(ck)


def normalize(obj, source=None):
    """Whatever a user saved -> the one internal form. Raises CheckpointError on anything else.

    The order of these checks matters. The bare-state_dict case must be tested BEFORE the
    generic "unrecognized mapping" case, or the most common user error gets the least useful
    message.
    """
    where = f' (from {source})' if source else ''

    # 1. Already ours -- the study's checkpoints land here unchanged.
    if isinstance(obj, Mapping) and 'config' in obj and 'state_dict' in obj:
        if 'thermometer' not in obj:
            raise _bare_state_dict_error(obj['state_dict'])
        ck = {
            'config': dict(obj['config']),
            'state_dict': dict(obj['state_dict']),
            'thermometer': {'thresholds': _thresholds_of(obj['thermometer'])},
            # Guaranteed present so downstream never needs a defensive .get(). Absent means
            # "not recorded", which is different from zero and must print differently.
            'results': dict(obj.get('results') or {}),
            'run_name': obj.get('run_name') or 'unnamed',
            # Part of the input contract -- see _scaler_of. None when training used raw
            # features, which is itself a fact worth carrying rather than an absence.
            'scaler': _scaler_of(obj, int(_thresholds_of(obj['thermometer']).shape[0])),
        }
        return _validate(ck)

    # 2. Live objects, saved by plain torch.save. The documented primary path.
    if isinstance(obj, Mapping) and 'model' in obj:
        therm = obj.get('thermometer', obj.get('thermometre'))
        if therm is None:
            raise CheckpointError(
                f'this checkpoint{where} has a model but NO THERMOMETER.\n'
                f'  keys: {sorted(obj)}\n'
                '\n'
                '  A DWN is two objects, and the thermometer is not part of the model. Save\n'
                '  both:\n'
                "      torch.save({'model': model, 'thermometer': thermometer}, 'model.pt')")
        model = obj['model']
        if isinstance(model, Mapping):
            # {'model': <a state_dict>, 'thermometer': ...} -- has the thermometer, but the
            # class count is gone with the GroupSum module.
            raise CheckpointError(
                f'this checkpoint{where} stores the model as a state_dict rather than as the\n'
                '  model object, so the number of classes cannot be recovered: it lives on\n'
                '  GroupSum, which has no parameters and therefore no entry in a state_dict.\n'
                '\n'
                "  Save the module itself -- torch.save({'model': model, ...}) -- or use\n"
                '  dwn2rtl.save(model, thermometer, path).')
        return from_model(model, therm, run_name=obj.get('run_name'),
                          results=obj.get('results'), scaler=obj.get('scaler'))

    # 3. The error this module exists for.
    if _looks_like_a_state_dict(obj):
        raise _bare_state_dict_error(obj)

    # 4. A bare model object, no thermometer anywhere.
    if hasattr(obj, 'state_dict') and callable(obj.state_dict):
        raise CheckpointError(
            f'this{where} is a model on its own, with no thermometer.\n'
            '\n'
            '  A DWN is two objects -- the thermometer is fitted before training and is not\n'
            '  part of the model. Save both:\n'
            "      torch.save({'model': model, 'thermometer': thermometer}, 'model.pt')")

    if isinstance(obj, Mapping):
        raise CheckpointError(
            f'unrecognized checkpoint{where}: keys {sorted(obj)}.\n'
            "  Expected {'model': ..., 'thermometer': ...} or a dwn2rtl checkpoint.")
    raise CheckpointError(
        f'unrecognized checkpoint{where}: a {type(obj).__name__}.\n'
        "  Expected {'model': ..., 'thermometer': ...} or a dwn2rtl checkpoint.")


def load(path):
    """Read a checkpoint file and normalize it.

    weights_only=False is required and safe here: the accepted formats deliberately contain a
    config dict and, on the primary path, the model object itself -- none of which survive
    torch's tensor-only unpickler. It is the user's own file.
    """
    try:
        obj = torch.load(path, map_location='cpu', weights_only=False)
    except FileNotFoundError:
        raise                      # the CLI already reports this one well
    except Exception as e:
        # ⚠️ torch.load fails in many ways on a file that is not a checkpoint -- UnpicklingError,
        # BadZipFile, EOFError, RuntimeError -- and every one of them reached the user as a raw
        # traceback through torch's internals. Pointing the tool at the wrong file is the most
        # ordinary mistake there is, so it gets an error like any other rejected shape.
        raise CheckpointError(
            f'{path} could not be read as a PyTorch checkpoint '
            f'({type(e).__name__}: {e}).\n'
            'Expected a file written by torch.save(). If this is a model you just trained:\n'
            "    torch.save({'model': model, 'thermometer': thermometer}, 'model.pt')"
        ) from e
    return normalize(obj, source=path)


def save(model, thermometer, path, run_name=None, results=None, scaler=None):
    """Write a checkpoint from live objects. Sugar over `torch.save`, not a requirement.

    The point is that it validates NOW, while the objects are in memory, rather than at build
    time on another machine weeks later.
    """
    ck = from_model(model, thermometer, run_name=run_name, results=results, scaler=scaler)
    torch.save(ck, path)
    return ck


def summary(ck):
    """One line per fact, for the CLI to print. ASCII only -- see cli.py."""
    cfg = ck['config']
    thr = ck['thermometer']['thresholds']
    return (f"features {thr.shape[0]}, classes {cfg['num_classes']}, "
            f"layers {list(cfg['layers'])}, n={cfg['n']}, z={cfg['thermometer_bits']}")
