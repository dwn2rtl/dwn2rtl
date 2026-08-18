"""What a build is: an output directory, a fixed-point format, and a pipeline depth.

NOT ported. The earlier implementation has a `rtlgen/config.py` of 254 lines and none of it belongs here --
it was sweep-shaped, built to name and resolve one point in a design-space exploration:
checkpoint slugs, per-config build directories, result paths, an FPGA part number, a target
clock. A generator has no sweep, no part and no clock. Copying it would import that earlier work's
shape into a tool that does not have that shape.

What survived is the idea it was built around, which is worth keeping: the things that vary
between builds are DATA, not edits. A different pipeline depth is a different Pipeline, not a
changed constant in emit_core.py. That is what makes a sweep possible for a user who wants one
without this tool needing to know about sweeps at all.

Three things and no more. If a fourth ever wants to live here, ask first whether it is a
property of the BUILD or a property of someone's board -- the second belongs to the user.
"""

import os
from dataclasses import dataclass, field

from .precision import Precision


@dataclass(frozen=True)
class Pipeline:
    """Where the registers go. Each is a stage count; 0 compiles that stage out entirely.

    Defaults match the earlier implementation's shipped depth, which was chosen by measurement rather than
    intuition: out-of-context synthesis put 10.194 ns in the core against 2.962 ns in the
    encoder, so the depth belongs in the popcount and argmax trees, and there is a stage after
    each of them.

    Latency is the sum of the enabled stages; throughput is one result per clock regardless
    (II=1), so depth trades latency for Fmax and nothing else.
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
        """End-to-end cycles for a model with `n_layers` LUT layers.

        Informational only. The authoritative number is the one emit_core writes into
        dwn_core_params.vh, which the testbench reads -- so a depth change cannot leave the
        testbench comparing against the wrong cycle.
        """
        return self.enc + self.lut * n_layers + self.pop + self.out


@dataclass(frozen=True)
class BuildConfig:
    """One build: where it goes, at what precision, with which registers.

    `precision` has no default on purpose. It is derived per model by
    precision.precision_for(), because the integer width depends on the thresholds -- a default
    here would be one model's format silently applied to another, which is the exact defect
    that cost the earlier implementation six separate crashes.
    """

    outdir: str
    precision: Precision
    pipeline: Pipeline = field(default_factory=Pipeline)

    def ensure_outdir(self):
        os.makedirs(self.outdir, exist_ok=True)
        return self.outdir
