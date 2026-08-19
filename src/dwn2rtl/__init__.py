"""dwn2rtl -- a trained DWN goes in, synthesizable Verilog comes out.

    dwn2rtl build model.pt --out rtl/     the terminal command
    import dwn2rtl                        the same code, for a model still in memory

⚠️ Importing this must not pull in torch. Only reading a checkpoint needs it -- everything else
is numpy -- and `dwn2rtl verify` reads no checkpoint at all, so paying seconds of torch startup
for it would be pure waste. The submodules that need torch import it themselves.
"""

__version__ = '0.2.0'

# Eager: neither needs a checkpoint or torch.
from .config import BuildConfig, Pipeline
from .precision import Precision, precision_for, required_int_bits

# Lazy, via module __getattr__ (PEP 562). These behave as if imported here, but importing them
# eagerly would drag torch into every invocation. There is a test for that.
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
