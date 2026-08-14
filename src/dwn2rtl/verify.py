"""Find a simulator, compile the emitted design, run its testbenches, report PASS or FAIL.

THIS IS THE GATE, and it is the reason the tool is worth using rather than rolling your own
(CLAUDE.md). Emitted RTL is not correct until a simulator says it matches the golden model on
every vector. Not "looks right", not "the emitter's read-back passed" -- the study repo has a
case where an emitter's own read-back reported 20/20 correct while the design was wrong on 958
of 1,504 vectors.

WHAT MAKES IT USEFUL IS THAT THE USER RUNS IT. Shipping self-checking testbenches turns
bit-exactness from a claim they have to trust into something they reproduce on their own machine
with their own simulator. That is why no part of this needs a vendor licence.

FINDING THE SIMULATOR IS HALF THE JOB, and this is not a theoretical concern.
`winget install Icarus.Verilog` succeeds, installs to C:\\iverilog\\bin, and adds NOTHING to PATH
(docs/phase0-ledger.md). A PATH-only search would tell a user who plainly has a simulator that
they have none -- so PATH is tried first and then the places installers actually use.

TWO LEVELS, ALWAYS BOTH. dwn_core_tb drives pre-binarized bits; dwn_top_tb drives quantized
features through the thermometer encoder. If the top fails while the core passes, the fault is
the encoder and nothing else needs re-examining. A missing testbench is reported as MISSING and
fails the run -- never quietly skipped, because "nothing checked it" must not read as success.
"""

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

# Where a Windows install actually lands. Searched only after PATH.
_CANDIDATE_DIRS = [
    r'C:\iverilog\bin',
    r'C:\Program Files\iverilog\bin',
    r'C:\Program Files (x86)\iverilog\bin',
    r'C:\oss-cad-suite\bin',
    os.path.expandvars(r'%LOCALAPPDATA%\oss-cad-suite\bin'),
    os.path.expanduser('~/oss-cad-suite/bin'),
    '/usr/local/bin',
    '/opt/oss-cad-suite/bin',
]

# level -> (testbench relative path, the vectors it needs). The vectors are listed so a build
# that emitted RTL but no vectors is reported as such rather than failing inside the simulator
# with a $readmemh warning nobody reads.
LEVELS = {
    'core': (os.path.join('tb', 'dwn_core_tb.v'), ('x_binarized.hex', 'expected.hex')),
    'top': (os.path.join('tb', 'dwn_top_tb.v'), ('x_quant.hex', 'expected_top.hex')),
}

_RESULT = re.compile(r'RESULT\s*:\s*(PASS|FAIL)')
_VECTORS = re.compile(r'vectors tested\s*:\s*(\d+)')
_MISMATCHES = re.compile(r'mismatches\s*:\s*(\d+)')


class SimulatorNotFound(RuntimeError):
    """No simulator anywhere. Carries the search so a user can see what was tried."""


@dataclass(frozen=True)
class Simulator:
    name: str
    compiler: str          # iverilog
    runner: str            # vvp
    version: str = ''

    def describe(self):
        return f'{self.name} {self.version} ({self.compiler})'.replace('  ', ' ')


def _iverilog_version(exe):
    """First line of `iverilog -V`.

    NB: it exits 255, so the return code is deliberately ignored. Checking it would report a
    perfectly good simulator as broken.
    """
    try:
        out = subprocess.run([exe, '-V'], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ''
    first = (out.stdout or out.stderr or '').strip().splitlines()
    if not first:
        return ''
    version = first[0].replace('Icarus Verilog version', '').strip()
    # Linux packages report "12.0 (stable) ()" -- an empty build-id parenthesis that renders as
    # a stray "()" in every report line. Windows reports "12.0 (devel) (s20150603-...)".
    return re.sub(r'\s*\(\s*\)', '', version).strip()


def find_simulator(iverilog=None):
    """PATH first, then the places Windows installers actually use.

    `iverilog` may be an explicit path or the name of an executable, for a user with several
    installed or one in an unusual place.
    """
    searched = []

    if iverilog:
        exe = shutil.which(iverilog) or (iverilog if os.path.exists(iverilog) else None)
        if not exe:
            raise SimulatorNotFound(f'{iverilog!r} is not an executable and is not on PATH')
        vvp = shutil.which('vvp') or os.path.join(os.path.dirname(exe), 'vvp')
        return Simulator('iverilog', exe, vvp, _iverilog_version(exe))

    exe = shutil.which('iverilog')
    if exe:
        vvp = shutil.which('vvp') or os.path.join(os.path.dirname(exe), 'vvp')
        return Simulator('iverilog', exe, vvp, _iverilog_version(exe))
    searched.append('PATH')

    for d in _CANDIDATE_DIRS:
        searched.append(d)
        for name in ('iverilog.exe', 'iverilog'):
            cand = os.path.join(d, name)
            if os.path.exists(cand):
                runner = os.path.join(d, 'vvp.exe' if name.endswith('.exe') else 'vvp')
                return Simulator('iverilog', cand, runner, _iverilog_version(cand))

    raise SimulatorNotFound(
        'no Verilog simulator found. Searched:\n'
        + '\n'.join(f'  {s}' for s in searched)
        + '\n\n  Install Icarus Verilog:\n'
          '    Windows  winget install Icarus.Verilog   (then add C:\\iverilog\\bin to PATH)\n'
          '    Debian   apt install iverilog\n'
          '    macOS    brew install icarus-verilog\n'
          '  or pass an explicit path.')


@dataclass
class LevelResult:
    level: str
    status: str                    # PASS | FAIL | MISSING | ERROR
    vectors: int = 0
    mismatches: int = 0
    detail: str = ''
    stdout: str = ''

    @property
    def ok(self):
        return self.status == 'PASS'


@dataclass
class VerifyReport:
    outdir: str
    simulator: Simulator
    levels: list = field(default_factory=list)

    @property
    def ok(self):
        """Every level must PASS, and there must BE levels.

        An empty run is not a pass. A directory with no testbenches would otherwise satisfy
        `all(...)` vacuously and print a green result having checked nothing -- which is the
        exact failure mode this module exists to prevent.
        """
        return bool(self.levels) and all(r.ok for r in self.levels)

    def lines(self):
        out = [f'{self.simulator.describe()}']
        for r in self.levels:
            name = {'core': 'dwn_core', 'top': 'dwn_top '}.get(r.level, r.level)
            if r.status == 'PASS':
                out.append(f'  {name}  {r.vectors} vectors  PASS')
            elif r.status == 'FAIL':
                out.append(f'  {name}  {r.vectors} vectors  FAIL '
                           f'({r.mismatches} mismatches)')
            else:
                out.append(f'  {name}  {r.status}: {r.detail}')
        if not self.levels:
            out.append('  nothing to check -- no testbenches in this directory')
        # The encoder/core split is only useful if the tool says so at the moment it matters.
        by = {r.level: r for r in self.levels}
        if by.get('core') and by['core'].ok and by.get('top') and by['top'].status == 'FAIL':
            out.append('  the core is bit-exact and the top is not, so the fault is in the '
                       'thermometer encoder')
        out.append('RESULT   ' + ('PASS' if self.ok else 'FAIL'))
        return out


def _run_level(sim, outdir, level, timeout):
    testbench, vectors = LEVELS[level]

    tb_path = os.path.join(outdir, testbench)
    if not os.path.exists(tb_path):
        return LevelResult(level, 'MISSING', detail=f'{testbench} is not in this directory')
    if not os.path.getsize(tb_path):
        return LevelResult(level, 'MISSING', detail=f'{testbench} is empty')
    missing = [v for v in vectors if not os.path.exists(os.path.join(outdir, v))]
    if missing:
        return LevelResult(level, 'MISSING', detail=f'no vectors: {", ".join(missing)}')

    sources = sorted(f for f in os.listdir(outdir) if f.endswith('.v'))
    if not sources:
        return LevelResult(level, 'ERROR', detail='no .v files in this directory')

    # The compiled image goes to a temp directory, not into the user's output. But the SIMULATOR
    # runs with cwd=outdir, because $readmemh("x_quant.hex") and `include "top_params.vh" both
    # resolve against the working directory rather than against the source file.
    with tempfile.TemporaryDirectory() as tmp:
        image = os.path.join(tmp, f'{level}.vvp')
        try:
            comp = subprocess.run([sim.compiler, '-o', image, *sources, testbench],
                                  cwd=outdir, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return LevelResult(level, 'ERROR', detail=f'compile timed out after {timeout}s')
        if comp.returncode != 0:
            return LevelResult(level, 'ERROR',
                               detail=f'compile failed\n{comp.stderr.strip()}')

        try:
            run = subprocess.run([sim.runner, image],
                                 cwd=outdir, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return LevelResult(level, 'ERROR', detail=f'simulation timed out after {timeout}s')

    stdout = run.stdout
    verdict = _RESULT.search(stdout)
    if not verdict:
        # The testbench prints a verdict unconditionally, so its absence means the simulation
        # did not reach the end -- and MUST NOT be read as a pass.
        return LevelResult(level, 'ERROR', stdout=stdout,
                           detail='the testbench printed no RESULT line; simulation did not '
                                  'finish')

    vectors_n = int(_VECTORS.search(stdout).group(1)) if _VECTORS.search(stdout) else 0
    mismatches = int(_MISMATCHES.search(stdout).group(1)) if _MISMATCHES.search(stdout) else 0

    # Belt and braces: a testbench that printed PASS while also reporting mismatches is broken,
    # and trusting its own summary line over its own count would be exactly the mistake this
    # module exists to prevent.
    status = verdict.group(1)
    if status == 'PASS' and mismatches:
        return LevelResult(level, 'ERROR', vectors_n, mismatches, stdout=stdout,
                           detail=f'testbench printed PASS but reported {mismatches} '
                                  'mismatches')

    return LevelResult(level, status, vectors_n, mismatches, stdout=stdout)


def verify(outdir, levels=('core', 'top'), iverilog=None, timeout=600):
    """Compile and run the emitted testbenches. Returns a VerifyReport; raises only if there is
    no simulator at all or the directory is not a build."""
    if not os.path.isdir(outdir):
        raise FileNotFoundError(f'{outdir} is not a directory')
    if not os.path.exists(os.path.join(outdir, 'dwn_top.v')):
        raise FileNotFoundError(
            f'{outdir} does not look like a dwn2rtl build -- no dwn_top.v. '
            'Run `dwn2rtl build` first.')

    sim = find_simulator(iverilog)
    return VerifyReport(outdir, sim,
                        [_run_level(sim, outdir, lv, timeout) for lv in levels])
