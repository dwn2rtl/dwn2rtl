"""Find a simulator, compile the emitted design, run its testbenches, report PASS or FAIL.

⚠️ This is the gate. RTL is not correct until a simulator says it matches the golden model on
every vector -- an emitter's own read-back once reported 20/20 while the design was wrong on 958
of 1,504. And the USER runs it, which is what makes bit-exactness reproducible rather than
claimed.

⚠️ Finding the simulator is half the job: winget installs iverilog to C:\\iverilog\\bin and adds
nothing to PATH, so PATH is tried first and then where installers actually put it.

Both levels always run. A missing testbench is MISSING and fails -- "nothing checked it" must
not read as success.
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

# The design `build` emits, by name. ⚠️ NOT a *.v glob of the directory: the README tells users
# to instantiate dwn_top in their own harness, and a harness dropped in here would be compiled
# into the gate -- so a syntax error in a file that is not under test broke verification, and an
# unrelated module could collide with one of ours.
#
# Listed rather than imported from build.py, which imports checkpoint.py and therefore torch.
# `dwn2rtl verify` reads no checkpoint and must not pay for that (see __init__.py).
DESIGN_SOURCES = ('dwn_top.v', 'dwn_core.v', 'thermometer_encoder.v',
                  'lut_node.v', 'popcount.v', 'argmax.v', 'pipe_reg.v')

_RESULT = re.compile(r'RESULT\s*:\s*(PASS|FAIL)')
_VECTORS = re.compile(r'vectors tested\s*:\s*(\d+)')
_MISMATCHES = re.compile(r'mismatches\s*:\s*(\d+)')


class SimulatorNotFound(RuntimeError):
    """No simulator anywhere. Carries the search so a user can see what was tried."""


@dataclass(frozen=True)
class Simulator:
    """A simulator, and how to drive it.

    ⚠️ The two commands live HERE, not in _run_level, so adding a simulator never touches the
    code that decides PASS or FAIL. Output parsing is shared and simulator-agnostic: the emitted
    testbench prints the RESULT line, not the tool.
    """

    name: str
    compiler: str          # iverilog, or verilator
    runner: str            # vvp; verilator builds a native executable and needs no runner
    version: str = ''

    def describe(self):
        return f'{self.name} {self.version} ({self.compiler})'.replace('  ', ' ')

    def commands(self, workdir, level, sources, testbench):
        """(compile_argv, run_argv). `workdir` is scratch -- never the user's output directory.

        iverilog compiles to a vvp image that vvp interprets. verilator translates to C++ and
        compiles a native executable, so the run step IS that executable.
        """
        if self.name == 'verilator':
            top = os.path.splitext(os.path.basename(testbench))[0]
            return ([self.compiler, '--binary', '--timing', '--top-module', top,
                     '--Mdir', workdir, '-o', level, *sources, testbench],
                    [os.path.join(workdir, level)])
        image = os.path.join(workdir, f'{level}.vvp')
        return ([self.compiler, '-o', image, *sources, testbench],
                [self.runner, image])

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


def _verilator_version(exe):
    """First line of `verilator --version`. Unlike iverilog it exits 0."""
    try:
        out = subprocess.run([exe, '--version'], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ''
    first = (out.stdout or out.stderr or '').strip().splitlines()
    return first[0].replace('Verilator', '').strip() if first else ''


def _as_iverilog(exe):
    vvp = shutil.which('vvp') or os.path.join(os.path.dirname(exe), 'vvp')
    if not os.path.exists(vvp) and os.path.exists(vvp + '.exe'):
        vvp += '.exe'
    return Simulator('iverilog', exe, vvp, _iverilog_version(exe))


def _as_verilator(exe):
    return Simulator('verilator', exe, '', _verilator_version(exe))


def find_simulator(simulator=None, iverilog=None):
    """PATH first, then the places Windows installers actually use.

    `simulator` may be a name ('iverilog', 'verilator') or an explicit path, for a user with
    several installed or one somewhere unusual.

    ⚠️ iverilog is preferred when both are present, and that is a MEASURED choice rather than an
    accident of ordering: verilator translates to C++ and compiles it, which took 14.7 s against
    iverilog's 0.38 s end-to-end on the same design. Verilator's advantage is throughput on long
    simulations, which these are not.
    """
    simulator = simulator or iverilog          # `iverilog=` kept for callers that predate this
    searched = []

    if simulator:
        exe = shutil.which(simulator) or (simulator if os.path.exists(simulator) else None)
        if not exe:
            raise SimulatorNotFound(f'{simulator!r} is not an executable and is not on PATH')
        return _as_verilator(exe) if 'verilator' in os.path.basename(exe).lower() \
            else _as_iverilog(exe)

    exe = shutil.which('iverilog')
    if exe:
        return _as_iverilog(exe)
    searched.append('PATH')

    for d in _CANDIDATE_DIRS:
        searched.append(d)
        for name in ('iverilog.exe', 'iverilog'):
            cand = os.path.join(d, name)
            if os.path.exists(cand):
                return _as_iverilog(cand)

    # Only after iverilog is ruled out everywhere. A user who has verilator and not iverilog can
    # still verify -- which is the whole reason this branch exists.
    exe = shutil.which('verilator')
    if exe:
        return _as_verilator(exe)
    searched.append('PATH (verilator)')

    raise SimulatorNotFound(
        'no Verilog simulator found. Searched:\n'
        + '\n'.join(f'  {s}' for s in searched)
        + '\n\n  Install Icarus Verilog:\n'
          '    Windows  winget install Icarus.Verilog   (then add C:\iverilog\bin to PATH)\n'
          '    Debian   apt install iverilog\n'
          '    macOS    brew install icarus-verilog\n'
          '  Verilator also works on Linux and macOS; pass --simulator verilator.\n'
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

    sources = [s for s in DESIGN_SOURCES if os.path.exists(os.path.join(outdir, s))]
    if not sources:
        return LevelResult(level, 'ERROR', detail='none of the emitted .v files are here')
    absent = [s for s in DESIGN_SOURCES if s not in sources]
    if absent:
        return LevelResult(level, 'MISSING', detail=f'not in this directory: {", ".join(absent)}')

    # The compiled image goes to a temp directory, not into the user's output. But the SIMULATOR
    # runs with cwd=outdir, because $readmemh("x_quant.hex") and `include "top_params.vh" both
    # resolve against the working directory rather than against the source file.
    with tempfile.TemporaryDirectory() as tmp:
        compile_cmd, run_cmd = sim.commands(tmp, level, sources, testbench)
        try:
            comp = subprocess.run(compile_cmd, cwd=outdir, capture_output=True, text=True,
                                  timeout=timeout)
        except subprocess.TimeoutExpired:
            return LevelResult(level, 'ERROR', detail=f'compile timed out after {timeout}s')
        except OSError as e:
            # ⚠️ Discovery accepts a simulator that merely EXISTS -- deliberately, since a
            # version probe is informational and some installs are odd. The cost is that an
            # unrunnable one is only discovered here, where a bare OSError would name a
            # syscall ("[WinError 2] The system cannot find the file specified") instead of
            # the tool the user chose.
            return LevelResult(level, 'ERROR',
                               detail=f'could not run {compile_cmd[0]}: {e}')
        if comp.returncode != 0:
            return LevelResult(
                level, 'ERROR',
                detail='compile failed\n' + (comp.stderr or comp.stdout).strip())

        try:
            run = subprocess.run(run_cmd, cwd=outdir, capture_output=True, text=True,
                                 timeout=timeout)
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


def verify(outdir, levels=('core', 'top'), simulator=None, timeout=600, iverilog=None):
    """Compile and run the emitted testbenches. Returns a VerifyReport; raises only if there is
    no simulator at all or the directory is not a build."""
    if not os.path.isdir(outdir):
        raise FileNotFoundError(f'{outdir} is not a directory')
    if not os.path.exists(os.path.join(outdir, 'dwn_top.v')):
        raise FileNotFoundError(
            f'{outdir} does not look like a dwn2rtl build -- no dwn_top.v. '
            'Run `dwn2rtl build` first.')

    # ⚠️ A bare string iterates into characters, so levels='core' raised KeyError('c').
    if isinstance(levels, str):
        raise TypeError(f"levels must be a sequence, not a string -- try ('{levels}',)")
    unknown = [lv for lv in levels if lv not in LEVELS]
    if unknown:
        raise ValueError(f'unknown level(s) {unknown}; expected any of {sorted(LEVELS)}')

    sim = find_simulator(simulator, iverilog)
    return VerifyReport(outdir, sim,
                        [_run_level(sim, outdir, lv, timeout) for lv in levels])
