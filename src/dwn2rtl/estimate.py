"""Synthesize an emitted design with yosys and report what it costs. Optional, and honest.

For answering "will this fit?" before committing to a real synthesis run. It is not a substitute
for one, and the calibration below says exactly how much it is not.

⚠️ CALIBRATED AGAINST VIVADO on an xc7a35t, before this file was written (docs/phase4-ledger.md):

    dwn_core              110 yosys   vs   110 Vivado    1.00x   trustworthy
    thermometer_encoder   717 yosys   vs  1519 Vivado    0.47x   HALF
    dwn_top               833 yosys   vs  1621 Vivado    0.51x

The core lands exactly. The encoder is 2.1x low, and not because yosys is wrong: a 16-bit signed
comparison against a constant genuinely fits in ~3 LUT6s, while Vivado maps comparators onto
carry chains that cost more. Generic mapping is simply optimistic for comparator-heavy logic.

⚠️ So this reports PER-MODULE counts and says which half to believe. It must never present a
total as a vendor number, and never let the encoder-to-core ratio pass unqualified -- that ratio
is 13.8x by Vivado and 6.5x here, and understating it would undercut the finding this project
exists to publicise.

WHY NOT AN AREA MODEL: roadmap Q7. The study built one and it filtered zero configurations
across two completed studies. This shells out to a real synthesis tool, or it reports nothing.
"""

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field

# Same shape as verify.py's search, and for the same reason: installers do not put themselves on
# PATH. The OSS CAD Suite in particular is a self-contained toolchain that should NOT be added to
# PATH globally -- it ships its own iverilog and vvp among 136 binaries, and would shadow a
# user's simulator.
_CANDIDATE_DIRS = [
    r'C:\oss-cad-suite\bin',
    os.path.expandvars(r'%LOCALAPPDATA%\oss-cad-suite\bin'),
    os.path.expanduser('~/oss-cad-suite/bin'),
    '/opt/oss-cad-suite/bin',
    '/usr/local/bin',
    '/usr/bin',
]

# Which sources make up each reportable module. The encoder has no submodules; the core and top
# pull in the hand-written primitives, which is why `build` copies them into the output.
PRIMITIVES = ('lut_node.v', 'popcount.v', 'argmax.v', 'pipe_reg.v')
MODULES = {
    'thermometer_encoder': ('thermometer_encoder.v',),
    'dwn_core': ('dwn_core.v',) + PRIMITIVES,
    'dwn_top': ('dwn_top.v', 'dwn_core.v', 'thermometer_encoder.v') + PRIMITIVES,
}

_COUNT = re.compile(r'(\d+) objects\.')

# The generic gate primitives yosys emits before technology mapping. AFTER `abc -lut 6` there
# must be NONE of them left: any survivor means the mapping did not cover the design, and the
# $lut figure is then a fragment rather than an answer.
_GATES = ('$_AND_', '$_NAND_', '$_OR_', '$_NOR_', '$_XOR_', '$_XNOR_',
          '$_NOT_', '$_MUX_', '$_ANDNOT_', '$_ORNOT_')


class YosysNotFound(RuntimeError):
    """No yosys. Carries the search, so a user can see what was tried."""


@dataclass(frozen=True)
class Yosys:
    exe: str
    version: str = ''
    root: str = ''          # the OSS CAD Suite root, when it came from one

    def environment(self):
        """The environment yosys must be run with.

        ⚠️ NOT COSMETIC. Invoked by absolute path with a clean environment, the OSS CAD Suite's
        yosys prints NOTHING and exits 0 -- a silent no-op, which is the worst failure a
        measurement tool can have. It needs its own bin and lib on PATH to load its DLLs. This
        is a per-subprocess environment, never a change to the user's PATH.
        """
        env = dict(os.environ)
        if self.root:
            bin_dir = os.path.join(self.root, 'bin')
            lib_dir = os.path.join(self.root, 'lib')
            env['PATH'] = os.pathsep.join([bin_dir, lib_dir, env.get('PATH', '')])
        return env

    def describe(self):
        return f'yosys {self.version}'.strip() + f' ({self.exe})'


def _version(exe, env):
    try:
        out = subprocess.run([exe, '-V'], capture_output=True, text=True, timeout=60, env=env)
    except (OSError, subprocess.SubprocessError):
        return ''
    first = (out.stdout or out.stderr or '').strip().splitlines()
    if not first:
        return ''
    return first[0].replace('Yosys', '').strip().split(' (')[0]


def _root_of(exe):
    """The suite root, if this yosys lives in one: <root>/bin/yosys[.exe]."""
    bin_dir = os.path.dirname(os.path.abspath(exe))
    root = os.path.dirname(bin_dir)
    if os.path.basename(bin_dir).lower() == 'bin' and os.path.isdir(os.path.join(root, 'lib')):
        return root
    return ''


def find_yosys(yosys=None):
    searched = []

    candidates = []
    if yosys:
        found = shutil.which(yosys) or (yosys if os.path.exists(yosys) else None)
        if not found:
            raise YosysNotFound(f'{yosys!r} is not an executable and is not on PATH')
        candidates = [found]
    else:
        on_path = shutil.which('yosys')
        if on_path:
            candidates = [on_path]
        searched.append('PATH')
        for d in _CANDIDATE_DIRS:
            searched.append(d)
            for name in ('yosys.exe', 'yosys'):
                cand = os.path.join(d, name)
                if os.path.exists(cand):
                    candidates.append(cand)

    for exe in candidates:
        root = _root_of(exe)
        sim = Yosys(exe=exe, root=root)
        version = _version(exe, sim.environment())
        # A yosys that reports no version is the silent-no-op case above. Do not accept it as
        # working -- every measurement it produced would be empty and look like a small design.
        if version:
            return Yosys(exe=exe, version=version, root=root)

    raise YosysNotFound(
        'no working yosys found. Searched:\n'
        + '\n'.join(f'  {s}' for s in searched)
        + '\n\n  Install the OSS CAD Suite (it contains yosys):\n'
          '    https://github.com/YosysHQ/oss-cad-suite-build/releases\n'
          '    extract it, and dwn2rtl will find it -- do NOT add it to PATH, it ships its own\n'
          '    iverilog and would shadow yours.\n'
          '  Debian/Ubuntu: apt install yosys')


@dataclass
class ModuleArea:
    module: str
    luts: int = 0
    flops: int = 0
    status: str = 'OK'       # OK | MISSING | ERROR
    detail: str = ''

    @property
    def ok(self):
        return self.status == 'OK'


@dataclass
class EstimateReport:
    outdir: str
    yosys: Yosys
    modules: list = field(default_factory=list)

    def by_name(self, name):
        return next((m for m in self.modules if m.module == name), None)

    @property
    def ok(self):
        return bool(self.modules) and all(m.ok for m in self.modules)

    def lines(self):
        """ASCII only (CLAUDE.md), and every number carries what it is worth."""
        out = [self.yosys.describe(), '']
        for m in self.modules:
            if m.ok:
                out.append(f'  {m.module:22s} {m.luts:6d} LUT   {m.flops:6d} FF')
            else:
                out.append(f'  {m.module:22s} {m.status}: {m.detail}')

        enc, core = self.by_name('thermometer_encoder'), self.by_name('dwn_core')
        if enc and core and enc.ok and core.ok and core.luts:
            out += ['',
                    f'  the encoder is {enc.luts / core.luts:.1f}x the core, '
                    f'as generic mapping sees it']

        out += [
            '',
            'ESTIMATE -- yosys generic mapping, not your vendor toolchain.',
            'Calibrated once against Vivado on xc7a35t (docs/phase4-ledger.md): the CORE agreed',
            'EXACTLY, the ENCODER came out 2.1x LOW because generic mapping packs comparators',
            'more tightly than a carry-chain architecture does. Treat core numbers as indicative,',
            'encoder numbers as a floor, and synthesize for figures you can publish.',
        ]
        return out


def _run(yosys, outdir, module, timeout):
    sources = [s for s in MODULES[module]]
    missing = [s for s in sources if not os.path.exists(os.path.join(outdir, s))]
    if missing:
        return ModuleArea(module, status='MISSING', detail=', '.join(missing))

    # `flatten` is LOAD-BEARING, not tidiness. Without it, every lut_node stays its own module
    # and `abc -lut 6` maps only the top level, so the count is a fragment that looks plausible.
    # It is an EXPLICIT PASS rather than `synth -flatten`, which is both more portable across
    # yosys versions and measurably better here: explicit flatten reports 110 LUTs for JSC's
    # core against Vivado's 110, where `synth -flatten` gave 106.
    #
    # The numbers come from `select -count`, NOT from scraping `stat`. `stat` prints one section
    # per surviving module -- including leftovers that still hold pre-mapping gate cells -- so
    # reading a figure out of it depends on picking the right section, and picking the wrong one
    # is silent. `select` operates on the whole design after opt_clean and answers exactly one
    # question at a time.
    gates = ' '.join(f't:{g}' for g in _GATES)
    script = (f'read_verilog {" ".join(sources)}; '
              f'hierarchy -top {module}; '
              f'flatten; '
              f'synth -top {module}; '
              f'abc -lut 6; '
              f'opt_clean; '
              f'select -count t:$lut; '
              f'select -count t:$_DFF_*; '
              f'select -count {gates}')
    try:
        proc = subprocess.run([yosys.exe, '-p', script], cwd=outdir, capture_output=True,
                              text=True, timeout=timeout, env=yosys.environment())
    except subprocess.TimeoutExpired:
        return ModuleArea(module, status='ERROR', detail=f'timed out after {timeout}s')
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        return ModuleArea(module, status='ERROR', detail='\n      '.join(tail))

    counts = _COUNT.findall(proc.stdout)
    if len(counts) < 3:
        return ModuleArea(module, status='ERROR',
                          detail=f'expected three counts from yosys, got {len(counts)} -- '
                                 f'`select -count` did not behave as expected on '
                                 f'yosys {yosys.version or "(unknown)"}')
    luts, flops, unmapped = (int(n) for n in counts[-3:])

    # ⚠️ THE CHECK THAT MAKES THE NUMBER MEAN SOMETHING. Generic gate primitives surviving
    # `abc -lut 6` mean the mapping did not cover the design, so the LUT count is a fragment.
    #
    # This is not hypothetical: yosys 0.33 (Ubuntu's package) reported ONE LUT for a 21-node
    # core, and the previous implementation accepted it as a successful measurement. A wrong
    # number presented as a result is worse than an error, because nothing downstream can tell.
    if unmapped:
        return ModuleArea(module, luts=luts, flops=flops, status='ERROR',
                          detail=f'{unmapped} unmapped gate cells remain after `abc -lut 6` on '
                                 f'yosys {yosys.version or "(unknown)"}; the LUT count would be '
                                 f'a fragment of the design, not its size')
    if not luts:
        return ModuleArea(module, status='ERROR', detail='yosys mapped the design to no LUTs')

    return ModuleArea(module, luts=luts, flops=flops)


def estimate(outdir, modules=None, yosys=None, timeout=1800):
    """Synthesize each module of an emitted design and report its area.

    Encoder and core are reported SEPARATELY and both always appear. That is a design invariant
    (CLAUDE.md): the encoder is intrinsic to a DWN rather than preprocessing, and on the smallest
    studied model it is fourteen times the network it feeds. A tool that reported only their sum
    would commit the reporting defect this project exists to correct.
    """
    if not os.path.isdir(outdir):
        raise FileNotFoundError(f'{outdir} is not a directory')
    if not os.path.exists(os.path.join(outdir, 'dwn_top.v')):
        raise FileNotFoundError(
            f'{outdir} does not look like a dwn2rtl build -- no dwn_top.v. '
            'Run `dwn2rtl build` first.')

    tool = find_yosys(yosys)
    wanted = list(modules or MODULES)
    return EstimateReport(outdir, tool, [_run(tool, outdir, m, timeout) for m in wanted])
