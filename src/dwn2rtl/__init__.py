"""dwn2rtl -- a trained DWN goes in, synthesizable Verilog comes out.

Two ways in, one implementation behind both (overview.md §2):

    dwn2rtl build model.pt --out rtl/     the terminal command, the primary path
    import dwn2rtl                        the same code, for when the model is still live

WHAT THIS MODULE DELIBERATELY DOES NOT DO: import torch.

Every emitter and the golden model are pure numpy; torch is needed only to read a checkpoint.
Importing `dwn2rtl` therefore costs a numpy import, not a torch one -- which matters because
torch is seconds of startup and hundreds of megabytes, and a user running `dwn2rtl verify` on an
already-emitted directory has no checkpoint to read at all. The submodules that need torch
(`extract`, `checkpoint`) import it themselves, at their own import time.

Keeping that boundary sharp is also what would make a torch-free path possible later, via an
.npz intermediate, without rewriting anything downstream.
"""

__version__ = '0.1.0rc1'

# Phase 0 exposes only what exists and works standalone -- the precision policy and the build
# configuration, neither of which needs a checkpoint or torch.
#
# `build()`, `save()` and `from_model()` are the public API this file is ultimately for. They
# arrive in Phase 1, once checkpoint.py defines what a checkpoint is; adding stubs for them now
# would mean `hasattr(dwn2rtl, 'build')` answering True for something that cannot build.
# See docs/phase0-ledger.md.
from .config import BuildConfig, Pipeline
from .precision import Precision, precision_for, required_int_bits

# The checkpoint API is reached LAZILY, via the module __getattr__ below (PEP 562).
#
# `dwn2rtl.load(...)` and `dwn2rtl.save(...)` work exactly as if they were imported here, but
# `import dwn2rtl` still does not pull torch in -- checkpoint.py imports torch at module level,
# so a plain `from .checkpoint import load` at the top of this file would silently cost every
# invocation a multi-second torch import, including `dwn2rtl verify`, which never reads a
# checkpoint at all. That invariant has a test; this is how the API is offered without breaking
# it.
_LAZY = {
    'load': 'checkpoint',
    'save': 'checkpoint',
    'from_model': 'checkpoint',
    'normalize': 'checkpoint',
    'CheckpointError': 'checkpoint',
    'build': 'build',
    'BuildReport': 'build',
    # verify has no torch dependency of its own, but it is listed here for one surface rather
    # than two -- and so that `import dwn2rtl` stays a numpy-only import either way.
    'verify': 'verify',
    'estimate': 'estimate',
    'find_yosys': 'estimate',
    'YosysNotFound': 'estimate',
    'find_simulator': 'verify',
    'SimulatorNotFound': 'verify',
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        module = importlib.import_module(f'.{_LAZY[name]}', __name__)
        value = getattr(module, name)
        globals()[name] = value          # bind, so the import cost is paid once
        return value
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    return sorted(__all__)


__all__ = [
    '__version__',
    'BuildConfig',
    'BuildReport',
    'CheckpointError',
    'Pipeline',
    'Precision',
    'SimulatorNotFound',
    'YosysNotFound',
    'build',
    'estimate',
    'find_simulator',
    'find_yosys',
    'from_model',
    'load',
    'normalize',
    'precision_for',
    'required_int_bits',
    'save',
    'verify',
]
