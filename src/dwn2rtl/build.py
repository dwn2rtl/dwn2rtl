"""checkpoint -> a self-contained directory of Verilog, parameters and golden vectors.

⚠️ Order matters. build_core writes dwn_core_params.vh; build_encoder READS it back for the
pipeline depth rather than being told again, since dwn_top's parameters override the core's and
two independent answers could disagree. It raises FileNotFoundError if called first.

⚠️ Vectors and RTL must come from the SAME checkpoint, or the testbench passes against wrong
RTL. build() loads once and passes the layers along; nothing downstream may re-open the file.
"""

import os
from dataclasses import dataclass, field
from importlib.resources import files

from .config import Pipeline
from .emit_core import build_core
from .emit_encoder import build_encoder
from .precision import comparator_merge_floor, precision_for
from .vectors import generate

# Copied into every emitted directory so it is self-contained: a user can hand the folder to any
# simulator or synthesis tool without also knowing where pip put the package. The testbenches go
# in a tb/ subdirectory so that `*.v` in the output root is exactly the design and nothing else
# -- two testbenches in one compile would give two top modules.
PRIMITIVES = ('lut_node.v', 'popcount.v', 'argmax.v', 'pipe_reg.v')
TESTBENCHES = ('dwn_core_tb.v', 'dwn_top_tb.v')


@dataclass(frozen=True)
class BuildReport:
    """What a build did, in the terms a user cares about.

    ENCODER AND CORE ARE COUNTED SEPARATELY AND ALWAYS BOTH REPORTED. That is a design
    invariant, not a formatting choice: the thermometer encoder is intrinsic to a DWN rather
    than preprocessing a user supplies, and on the smallest studied model it is FOURTEEN TIMES
    the network it feeds. Published work that reports core-only LUT counts understates designs
    by most of their cost; a tool of ours that did the same would hand every user that error.
    """

    outdir: str
    features: int
    classes: int
    layers: list
    n: int
    nodes: int
    comparators: int
    thermometer_bits: int
    precision: object
    latency: int
    core_latency: int
    vectors: dict
    merge: tuple                 # (distinct, total, collapsed) comparators after quantisation
    degenerate: bool
    files: list
    run_name: str = 'unnamed'
    warnings: list = field(default_factory=list)

    def lines(self):
        """The build report as ASCII lines, for the CLI to print.

        ⚠️ ASCII only -- Windows consoles raise on an emoji in print().

        The right-hand column is the point, not decoration: `integer bits` is exact while
        `frac bits` may be a proof or a guess, and printing them identically invites equal trust.
        """
        p = self.precision
        distinct, total, collapsed = self.merge

        facts = [
            (f'features {self.features}, classes {self.classes}, layers {self.layers}, '
             f'n={self.n}, z={self.thermometer_bits // self.features}', 'from checkpoint'),
            (f'integer bits {p.int_bits}', 'derived, exact'),
            # Three provenances, three different claims. 'inferred' is as strong as 'given' --
            # both rest on the input having a native quantum -- but it is labelled distinctly so
            # a user can see the tool made the choice and can override it if the assumption is
            # wrong for their deployment.
            (f'frac bits {p.frac_bits} -> {p}', {
                'given': 'from --input-bits, provably lossless',
                'inferred': "INFERRED from the thresholds' grid, provably lossless",
                'default': 'DEFAULT for a continuous input, NOT measured',
            }[p.source]),
        ]
        width = max(len(left) for left, _ in facts)
        out = [f'{left:<{width}}   {right}' for left, right in facts]

        out += [
            '',
            f'core      {self.nodes} nodes, {self.core_latency} cycles',
            f'encoder   {self.comparators} comparators of {self.thermometer_bits} '
            f'thermometer bits',
            f'top       {self.latency} cycles latency, II=1',
            '',
            f'vectors   core {self.vectors["core"]["count"]}, '
            f'top {self.vectors["top"]["count"]}, '
            f'{self.vectors["top"]["classes_hit"]}/{self.classes} classes hit',
        ]
        if collapsed:
            out.append(
                f'note      {collapsed} of {total} thresholds quantise to a duplicate '
                f'comparison ({distinct} distinct)')
        out += [f'WARNING   {w}' for w in self.warnings]
        out.append(f'wrote     {self.outdir} ({len(self.files)} files)')
        return out


def _copy_package_rtl(outdir):
    """Put the hand-written Verilog beside the emitted Verilog.

    Read and written as BYTES. Text mode on Windows rewrites \\n to \\r\\n on the way out, and
    the study lost a week to CRLF breaking multi-line string matching -- there is no reason
    for a copy to alter a file at all.
    """
    rtl = files('dwn2rtl') / 'rtl'
    written, skipped = [], []

    for name in PRIMITIVES:
        dest = os.path.join(outdir, name)
        with open(dest, 'wb') as f:
            f.write((rtl / name).read_bytes())
        written.append(dest)

    tb_dir = os.path.join(outdir, 'tb')
    os.makedirs(tb_dir, exist_ok=True)
    for name in TESTBENCHES:
        data = (rtl / 'tb' / name).read_bytes()
        if not data.strip():
            # An empty testbench would make the directory look complete while that level's
            # gate does nothing. Warn at build time instead of shipping a fast fake PASS.
            skipped.append(name)
            continue
        dest = os.path.join(tb_dir, name)
        with open(dest, 'wb') as f:
            f.write(data)
        written.append(dest)

    return written, skipped


def build(checkpoint, outdir, input_bits=None, pipeline=None, n_random=None, seed=None):
    """Emit a complete, self-contained design directory. The tool's one real entry point.

    checkpoint   a path, or anything checkpoint.normalize() accepts. Loaded ONCE, here.
    input_bits   the INPUT's precision -- 8-bit pixels -> 8, never fractional bits. None means
                 a continuous input, which takes a default and is reported as unproved.
    pipeline     a config.Pipeline.
    """
    from . import checkpoint as ckpt

    ck = ckpt.load(checkpoint) if isinstance(checkpoint, (str, os.PathLike)) \
        else ckpt.normalize(checkpoint)

    # ⚠️ Public arguments, checked before use. A dict here gave "'dict' object has no attribute
    # 'lut'" and a string n_random reached numpy as a UFuncTypeError -- both name a library
    # internal rather than the argument the caller got wrong.
    pipeline = pipeline or Pipeline()
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f'pipeline must be a dwn2rtl.Pipeline, got {type(pipeline).__name__}. '
                        'Try Pipeline(enc=1, lut=1, pop=1, out=1).')
    if n_random is not None:
        if isinstance(n_random, bool) or not isinstance(n_random, int):
            raise TypeError(f'n_random must be an int, got {type(n_random).__name__}')
        if n_random < 0:
            raise ValueError(f'n_random must be >= 0, got {n_random}')
    thresholds = ck['thermometer']['thresholds'].numpy()
    precision = precision_for(thresholds, input_bits=input_bits)

    # A path that is already a FILE reached the user as a raw FileExistsError/WinError from
    # os.makedirs, which names the syscall rather than the mistake.
    if os.path.exists(outdir) and not os.path.isdir(outdir):
        raise NotADirectoryError(f'--out {outdir} is a file, not a directory')
    os.makedirs(outdir, exist_ok=True)

    # 1. the core. Also emits dwn_core_params.vh, which step 2 reads.
    core = build_core(ck, outdir, pipeline=pipeline)

    # 2. the encoder and the top. Must follow (1) -- see the module docstring.
    enc = build_encoder(ck, outdir, precision, pipe_enc=pipeline.enc)

    # 3. the golden vectors, labelled by the SAME extracted layers the core was emitted from.
    kwargs = {}
    if n_random is not None:
        kwargs['n_random'] = n_random
    if seed is not None:
        kwargs['seed'] = seed
    vec = generate(ck, core['layers_extracted'], outdir, precision, **kwargs)

    # 4. the hand-written modules, so the directory stands alone.
    copied, skipped_tb = _copy_package_rtl(outdir)

    warnings = []

    # ⚠️ The input contract: if training scaled its features, whatever drives x_flat must apply
    # the same scaling first. The parameters are written out, not just mentioned -- "apply your
    # scaler" without them is a riddle, not a warning.
    scaler = ck.get('scaler')
    scaling_path = os.path.join(outdir, 'input_scaling.json')
    if scaler and 'mean' in scaler:
        import json
        with open(scaling_path, 'w') as f:
            json.dump({'note': 'apply as (x - mean) / scale BEFORE quantizing to the word '
                               'format below; the model was trained in this space',
                       'format': str(precision),
                       'frac_bits': precision.frac_bits,
                       'word_bits': precision.word_bits,
                       'mean': scaler['mean'], 'scale': scaler['scale']}, f, indent=2)
        warnings.append(
            'this model was trained on SCALED features. Whatever drives x_flat must apply '
            '(x - mean) / scale first, using input_scaling.json -- raw features give a design '
            'that runs at chance and looks healthy doing it.')
    elif scaler:
        warnings.append(
            f'the checkpoint carries a scaler this tool did not recognize '
            f'({scaler.get("unrecognized")}). If training scaled its inputs, whatever drives '
            'x_flat must apply the same scaling.')
    elif os.path.exists(scaling_path):
        # ⚠️ EVERY OTHER EMITTED FILE IS OVERWRITTEN; this one is conditional, so a leftover
        # survives. Building an unscaled model over a scaled one left the PREVIOUS model's
        # mean and scale sitting in the directory, and a user following it would apply another
        # model's transformation -- the design then runs at chance and looks healthy, which is
        # the exact failure the file exists to prevent.
        os.remove(scaling_path)
        warnings.append(
            'removed a stale input_scaling.json left by an earlier build in this directory. '
            'This model records no scaler, so nothing should be applied to x_flat.')
    for name in skipped_tb:
        warnings.append(
            f'{name} is EMPTY in the installed package and was not copied -- that level has no '
            'testbench, so nothing checks it.')
    if vec['degenerate']:
        warnings.append(
            'every test vector lands on ONE class. The testbench would pass against a design '
            'whose argmax, popcount and grouping were all wrong, because the right answer is a '
            'constant. Check the checkpoint.')
    if not precision.proved:
        warnings.append(
            f'fractional width {precision.frac_bits} is a DEFAULT for a continuous input, not '
            'a measurement. Pass --input-bits if the input has a native precision.')

    emitted = [core['path'], enc['encoder_path'], enc['top_path'],
               os.path.join(outdir, 'dwn_core_params.vh'),
               os.path.join(outdir, 'dwn_top_params.vh'),
               os.path.join(outdir, 'vec_params.vh'),
               os.path.join(outdir, 'top_params.vh'),
               os.path.join(outdir, 'x_binarized.hex'),
               os.path.join(outdir, 'expected.hex'),
               os.path.join(outdir, 'x_quant.hex'),
               os.path.join(outdir, 'expected_top.hex')]
    if os.path.exists(scaling_path):
        emitted.append(scaling_path)

    return BuildReport(
        outdir=outdir,
        features=enc['features'],
        classes=ck['config']['num_classes'],
        layers=list(ck['config']['layers']),
        n=ck['config']['n'],
        nodes=core['nodes'],
        comparators=enc['comparators'],
        thermometer_bits=enc['thermometer_bits'],
        precision=precision,
        latency=enc['latency'],
        core_latency=core['core_latency'],
        vectors=vec,
        merge=comparator_merge_floor(thresholds, precision),
        degenerate=vec['degenerate'],
        files=emitted + copied,
        run_name=ck.get('run_name', 'unnamed'),
        warnings=warnings,
    )
