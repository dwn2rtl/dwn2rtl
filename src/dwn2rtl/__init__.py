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

__version__ = '0.1.0.dev0'

# Phase 0 exposes only what exists and works standalone -- the precision policy and the build
# configuration, neither of which needs a checkpoint or torch.
#
# `build()`, `save()` and `from_model()` are the public API this file is ultimately for. They
# arrive in Phase 1, once checkpoint.py defines what a checkpoint is; adding stubs for them now
# would mean `hasattr(dwn2rtl, 'build')` answering True for something that cannot build.
# See docs/phase0-ledger.md.
from .config import BuildConfig, Pipeline
from .precision import Precision, precision_for, required_int_bits

__all__ = [
    '__version__',
    'BuildConfig',
    'Pipeline',
    'Precision',
    'precision_for',
    'required_int_bits',
]
