"""What a build is: an output directory, a fixed-point format, and a pipeline depth.

Three things and no more. The things that vary between builds are DATA, not edits -- a different
pipeline depth is a different Pipeline, not a changed constant in emit_core.py. That is what
lets a user run a sweep without this tool knowing anything about sweeps.

If a fourth thing wants to live here, ask first whether it is a property of the BUILD or of
someone's board. The second belongs to the user.
"""

import os
from dataclasses import dataclass, field

from .precision import Precision


@dataclass(frozen=True)
class Pipeline:
    """Where the registers go. Each is a stage count; 0 compiles that stage out.

    Latency is the sum of the enabled stages. Throughput is one result per clock either way, so
    depth trades latency for Fmax and nothing else.
    """

    enc: int = 1     # after the comparators
    lut: int = 1     # after each LUT layer
    pop: int = 1     # after the popcounts
    out: int = 1     # after the argmax

    def __post_init__(self):
        for name in ('enc', 'lut', 'pop', 'out'):
            v = getattr(self, name)
            if v < 0:
                raise ValueError(f'pipeline depth {name}={v} must be >= 0')

    def latency(self, n_layers):
        """End-to-end cycles. Informational -- dwn_core_params.vh holds the authoritative
        number, and the testbench reads that.
        """
        return self.enc + self.lut * n_layers + self.pop + self.out


@dataclass(frozen=True)
class BuildConfig:
    """One build: where it goes, at what precision, with which registers.

    ⚠️ `precision` has no default -- integer width depends on the thresholds, so a default here
    is one model's format silently applied to another.
    """

    outdir: str
    precision: Precision
    pipeline: Pipeline = field(default_factory=Pipeline)

    def ensure_outdir(self):
        os.makedirs(self.outdir, exist_ok=True)
        return self.outdir
