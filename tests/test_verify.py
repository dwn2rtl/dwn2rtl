"""verify.py -- find a simulator, run the gate, and be honest about the answer.

The theme of this file is that VERIFY MUST NOT REPORT SUCCESS FOR ANYTHING IT DID NOT CHECK.
Every test below is some form of that: a missing testbench, an empty one, a simulation that
never finished, a directory with nothing in it. Each has to come out as a failure, because the
whole product claim rests on a green result meaning something.
"""

import os
import re

import pytest

import fixtures
from dwn2rtl.build import build
from dwn2rtl.verify import (LEVELS, Simulator, SimulatorNotFound, VerifyReport,
                            find_simulator, verify)


# Discovery and the skip both live in conftest.py now -- three test files had each grown a
# copy by the end of phase 1.
from conftest import SIMULATOR as _SIM


@pytest.fixture
def built(tmp_path):
    return build(fixtures.make('n6'), str(tmp_path / 'rtl'), input_bits=8).outdir


# ---------------------------------------------------------------------------------------
# Finding the simulator -- half the job, and not a theoretical concern
# ---------------------------------------------------------------------------------------

def test_find_simulator_returns_something_runnable(simulator):
    # Takes the conftest `simulator` fixture rather than calling find_simulator() directly:
    # without one, this should SKIP, not raise SimulatorNotFound and read as a real failure.
    assert os.path.exists(simulator.compiler)
    assert os.path.exists(simulator.runner)
    assert simulator.name == 'iverilog'


def test_version_is_detected_despite_iverilog_exiting_nonzero(simulator):
    """`iverilog -V` exits 255. Checking its return code would report a perfectly good
    simulator as broken -- measured in phase 0."""
    assert re.search(r'\d+\.\d+', simulator.version)


def test_an_explicit_path_that_is_not_executable_is_refused():
    with pytest.raises(SimulatorNotFound, match='not an executable'):
        find_simulator(iverilog='definitely-not-a-simulator-xyz')


def _fake_install(d, name='iverilog.exe', runner='vvp.exe', on_path=False):
    """A directory that looks like an install: the two files, nothing runnable in them.

    ⚠️ `on_path=True` makes it discoverable by shutil.which, which is not the same as existing:
    POSIX needs the exact name and the execute bit, Windows any PATHEXT suffix. Without it,
    test_path_wins_over_the_fallback passed on Windows having put nothing on PATH at all.
    """
    if on_path:
        suffix = '.exe' if os.name == 'nt' else ''
        name, runner = f'iverilog{suffix}', f'vvp{suffix}'
    d.mkdir(parents=True, exist_ok=True)
    for filename in (name, runner):
        path = d / filename
        path.write_text('not a real simulator')
        path.chmod(0o755)          # no-op on Windows, required by shutil.which on POSIX
    return d


@pytest.fixture
def empty_path(tmp_path, monkeypatch):
    """PATH with no simulator on it -- the Windows user's actual situation."""
    nowhere = tmp_path / 'nowhere'
    nowhere.mkdir()
    monkeypatch.setenv('PATH', str(nowhere))
    return nowhere


# THE FALLBACK NEEDS ITS OWN TESTS, and until phase 4 it had none.
#
# It was being EXERCISED constantly and CHECKED nowhere. On a machine where iverilog is not on
# PATH -- the default after `winget install Icarus.Verilog` -- every run of the suite reached it,
# so breaking it would have turned the whole gate red immediately. That looked like coverage and
# was not: it depended on the developer's PATH staying broken. The moment C:\iverilog\bin was
# added to PATH (2026-08-14) the fallback stopped being reached locally at all, and nothing in
# the suite would have noticed it rotting.
#
# Same class of problem as the implicit `import fixtures` that conftest.py was written to fix:
# behaviour that works because of an unwritten property of one machine's layout.

def test_the_fallback_finds_an_install_that_is_not_on_path(tmp_path, monkeypatch, empty_path):
    """The whole reason _CANDIDATE_DIRS exists. A user who plainly has a simulator must not be
    told they have none."""
    install = _fake_install(tmp_path / 'iverilog' / 'bin')
    monkeypatch.setattr('dwn2rtl.verify._CANDIDATE_DIRS', [str(install)])

    sim = find_simulator()

    assert sim.name == 'iverilog'
    assert sim.compiler == str(install / 'iverilog.exe')
    # The runner is derived from the compiler's directory, never from PATH -- pairing a
    # fallback iverilog with some unrelated vvp would be worse than finding neither.
    assert sim.runner == str(install / 'vvp.exe')


def test_the_fallback_also_finds_an_unsuffixed_binary(tmp_path, monkeypatch, empty_path):
    """The POSIX shape of the same directory -- /opt/oss-cad-suite/bin and friends."""
    install = _fake_install(tmp_path / 'oss', name='iverilog', runner='vvp')
    monkeypatch.setattr('dwn2rtl.verify._CANDIDATE_DIRS', [str(install)])

    sim = find_simulator()

    assert sim.compiler == str(install / 'iverilog')
    assert sim.runner == str(install / 'vvp')


def test_an_unrunnable_binary_yields_an_empty_version_rather_than_raising(
        tmp_path, monkeypatch, empty_path):
    """`iverilog -V` on something that is not an executable raises OSError. That is caught, and
    it must stay caught: a version string is cosmetic and must never cost a working simulator."""
    install = _fake_install(tmp_path / 'bin')
    monkeypatch.setattr('dwn2rtl.verify._CANDIDATE_DIRS', [str(install)])

    assert find_simulator().version == ''


def test_path_wins_over_the_fallback(tmp_path, monkeypatch):
    """Order is load-bearing: a deliberate simulator on PATH must beat a stale install. `on_path=True`
    is what makes this real rather than vacuous -- see _fake_install.
    """
    on_path = _fake_install(tmp_path / 'chosen', on_path=True)
    fallback = _fake_install(tmp_path / 'stale')
    monkeypatch.setenv('PATH', str(on_path))
    monkeypatch.setattr('dwn2rtl.verify._CANDIDATE_DIRS', [str(fallback)])

    assert os.path.dirname(find_simulator().compiler) == str(on_path)


def test_the_winget_destination_is_among_the_candidates():
    """A regression guard on one specific string. `winget install Icarus.Verilog` lands here and
    adds nothing to PATH (docs/phase0-ledger.md); drop it and Windows users are told they have no
    simulator while holding one. Cheap to keep, and the failure it prevents is silent."""
    from dwn2rtl.verify import _CANDIDATE_DIRS
    assert r'C:\iverilog\bin' in _CANDIDATE_DIRS


def test_not_found_error_lists_what_was_searched_and_how_to_install(monkeypatch, empty_path):
    """A user with no simulator needs the next action. Calls find_simulator() rather than building
    the error by hand -- asserting on a string the test wrote proves nothing about the search.
    """
    absent = str(empty_path / 'absent')
    monkeypatch.setattr('dwn2rtl.verify._CANDIDATE_DIRS', [absent])

    with pytest.raises(SimulatorNotFound) as e:
        find_simulator()

    msg = str(e.value)
    assert 'PATH' in msg                           # every place it looked, named
    assert absent in msg
    assert 'winget install Icarus.Verilog' in msg  # and the next action, per platform
    assert 'apt install iverilog' in msg
    assert msg.isascii()                           # cp1252 consoles print this one


# ---------------------------------------------------------------------------------------
# The gate, through verify()
# ---------------------------------------------------------------------------------------

@pytest.mark.sim
def test_verify_passes_a_good_build(built):
    r = verify(built)
    assert r.ok
    assert {lv.level for lv in r.levels} == {'core', 'top'}
    assert all(lv.status == 'PASS' for lv in r.levels)
    assert all(lv.mismatches == 0 for lv in r.levels)
    assert 'RESULT   PASS' in '\n'.join(r.lines())


@pytest.mark.sim
def test_verify_fails_a_corrupted_core(built):
    core = os.path.join(built, 'dwn_core.v')
    src = open(core).read()
    broken = re.sub(r"64'h([0-9A-F]{16})",
                    lambda m: f"64'h{(~int(m.group(1), 16)) & 0xFFFFFFFFFFFFFFFF:016X}", src)
    open(core, 'w').write(broken)

    r = verify(built)
    assert not r.ok
    assert any(lv.status == 'FAIL' and lv.mismatches > 0 for lv in r.levels)


@pytest.mark.sim
def test_a_broken_encoder_is_localized_in_the_report(built):
    """The two-level split only pays off if the tool SAYS which half failed at the moment it
    matters. Core PASS + top FAIL must produce that sentence, not leave the user to infer it."""
    enc = os.path.join(built, 'thermometer_encoder.v')
    src = open(enc).read()
    broken, n = re.subn(r"> \$signed\(\d+'sd\d+\)", "> $signed(9'sd200)", src, count=1)
    assert n == 1
    open(enc, 'w').write(broken)

    r = verify(built)
    by = {lv.level: lv for lv in r.levels}
    assert by['core'].ok, 'a broken encoder must not affect the core level'
    assert by['top'].status == 'FAIL'
    assert 'fault is in the thermometer encoder' in '\n'.join(r.lines())


@pytest.mark.sim
def test_a_compile_error_is_an_error_not_a_pass(built):
    open(os.path.join(built, 'dwn_core.v'), 'a').write('\nthis is not verilog\n')
    r = verify(built)
    assert not r.ok
    assert any(lv.status == 'ERROR' for lv in r.levels)


# ---------------------------------------------------------------------------------------
# Not checking is not passing
# ---------------------------------------------------------------------------------------

@pytest.mark.sim
@pytest.mark.parametrize('level', sorted(LEVELS))
def test_a_missing_testbench_is_reported_missing_and_fails(built, level):
    """Never quietly skipped. 'Nothing checked it' must not read as success -- that is the
    failure mode the whole gate exists to prevent."""
    os.remove(os.path.join(built, LEVELS[level][0]))
    r = verify(built)
    by = {lv.level: lv for lv in r.levels}
    assert by[level].status == 'MISSING'
    assert not r.ok


@pytest.mark.sim
def test_an_empty_testbench_is_reported_missing(built):
    """A 0-byte testbench compiles fine and checks nothing. That is precisely how dwn_top_tb.v
    sat unnoticed from commit 646aebe until phase 1."""
    open(os.path.join(built, LEVELS['top'][0]), 'w').close()
    r = verify(built)
    by = {lv.level: lv for lv in r.levels}
    assert by['top'].status == 'MISSING'
    assert 'empty' in by['top'].detail
    assert not r.ok


@pytest.mark.sim
def test_missing_vectors_are_reported_before_the_simulator_runs(built):
    """Without vectors, $readmemh warns and the testbench compares against x -- a failure whose
    cause is two layers from its symptom. Caught up front instead."""
    os.remove(os.path.join(built, 'x_quant.hex'))
    r = verify(built)
    by = {lv.level: lv for lv in r.levels}
    assert by['top'].status == 'MISSING' and 'x_quant.hex' in by['top'].detail
    assert by['core'].ok, 'the core level does not use those vectors and should still run'


def test_an_empty_report_is_not_a_pass():
    """`all([])` is True. Without the emptiness check, a directory with no runnable levels prints a
    green RESULT having checked nothing -- the most dangerous thing this tool could do.
    """
    from dwn2rtl.verify import Simulator

    empty = VerifyReport('somewhere', Simulator('iverilog', 'x', 'y', '0'), levels=[])
    assert not empty.ok
    text = '\n'.join(empty.lines())
    assert 'nothing to check' in text
    assert 'RESULT   FAIL' in text


# ---------------------------------------------------------------------------------------
# The skip guard itself -- because a silently absent gate looks exactly like a green one
# ---------------------------------------------------------------------------------------

def test_missing_simulator_skips_gate_tests_loudly(monkeypatch):
    """The skip hook is necessary and is also the one place that could hide the only correctness
    signal, so both halves are asserted: skipped, and the reason says the gate did not run.
    """
    import conftest

    monkeypatch.setattr(conftest, 'SIMULATOR', None)

    header = conftest.pytest_report_header(None)
    assert any('NO SIMULATOR' in line for line in header)
    assert any('SKIPPED, not passed' in line for line in header)

    class FakeItem:
        def __init__(self, keywords):
            self.keywords = keywords
            self.markers = []

        def add_marker(self, m):
            self.markers.append(m)

    gate, plain = FakeItem({'sim': True}), FakeItem({})
    conftest.pytest_collection_modifyitems(None, [gate, plain])

    assert gate.markers, 'a gate test was not skipped without a simulator'
    assert 'THE GATE DID NOT RUN' in gate.markers[0].kwargs['reason']
    assert not plain.markers, 'a non-gate test was skipped for no reason'


def test_present_simulator_skips_nothing(monkeypatch):
    import conftest
    from dwn2rtl.verify import Simulator

    monkeypatch.setattr(conftest, 'SIMULATOR', Simulator('iverilog', 'x', 'y', '12.0'))

    class FakeItem:
        keywords = {'sim': True}
        markers = []

        def add_marker(self, m):
            self.markers.append(m)

    item = FakeItem()
    conftest.pytest_collection_modifyitems(None, [item])
    assert not item.markers
    assert 'gate simulator' in conftest.pytest_report_header(None)[0]


# ---------------------------------------------------------------------------------------
# Refusing to run at all
# ---------------------------------------------------------------------------------------

def test_verify_refuses_a_directory_that_is_not_a_build(tmp_path):
    (tmp_path / 'random.txt').write_text('hello')
    with pytest.raises(FileNotFoundError, match='does not look like a dwn2rtl build'):
        verify(str(tmp_path))


def test_verify_refuses_a_path_that_is_not_a_directory(tmp_path):
    f = tmp_path / 'a-file'
    f.write_text('x')
    with pytest.raises(FileNotFoundError, match='not a directory'):
        verify(str(f))


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------

@pytest.mark.sim
def test_cli_verify_exits_zero_on_pass_and_one_on_fail(built, capsys):
    """The exit code is what a CI job branches on, so it is asserted separately from the text."""
    from dwn2rtl.cli import main

    assert main(['verify', built]) == 0
    assert 'PASS' in capsys.readouterr().out

    core = os.path.join(built, 'dwn_core.v')
    src = open(core).read()
    open(core, 'w').write(re.sub(
        r"64'h([0-9A-F]{16})",
        lambda m: f"64'h{(~int(m.group(1), 16)) & 0xFFFFFFFFFFFFFFFF:016X}", src))

    assert main(['verify', built]) == 1


def test_cli_verify_of_a_non_build_exits_one(tmp_path, capsys):
    from dwn2rtl.cli import main
    assert main(['verify', str(tmp_path)]) == 1
    assert 'dwn2rtl build' in capsys.readouterr().err


@pytest.mark.sim
def test_verify_report_is_ascii(built):
    """CLAUDE.md: cp1252 consoles raise on anything else."""
    '\n'.join(verify(built).lines()).encode('ascii')


# ---------------------------------------------------------------------------------------
# A second simulator, optional in exactly the way yosys is
# ---------------------------------------------------------------------------------------

def test_each_simulator_builds_its_own_commands():
    """The two command shapes live on Simulator so _run_level never knows which tool it drives.

    iverilog compiles to a vvp image that vvp interprets; verilator compiles a native executable
    and has no separate runner.
    """
    src, tb = ['a.v', 'b.v'], os.path.join('tb', 'dwn_top_tb.v')

    icar = Simulator('iverilog', '/usr/bin/iverilog', '/usr/bin/vvp')
    comp, run = icar.commands('/scratch', 'top', src, tb)
    assert comp[0] == '/usr/bin/iverilog' and '-o' in comp
    assert run[0] == '/usr/bin/vvp' and run[1].endswith('top.vvp')

    veri = Simulator('verilator', '/usr/bin/verilator', '')
    comp, run = veri.commands('/scratch', 'top', src, tb)
    assert '--binary' in comp and '--timing' in comp
    assert comp[comp.index('--top-module') + 1] == 'dwn_top_tb', 'top comes from the testbench'
    assert len(run) == 1, 'verilator produces an executable, not an image plus interpreter'


def test_iverilog_is_preferred_when_both_are_installed(monkeypatch):
    """⚠️ A measured preference, not an accident of ordering: verilator translates to C++ and
    compiles it, which took 14.7 s against iverilog's 0.38 s end to end on the same design."""
    monkeypatch.setattr('shutil.which',
                        lambda name: f'/usr/bin/{name}' if name in
                        ('iverilog', 'vvp', 'verilator') else None)
    assert find_simulator().name == 'iverilog'


def test_verilator_is_used_when_iverilog_is_absent(monkeypatch):
    """The whole reason the fallback exists: a user with only verilator can still verify."""
    monkeypatch.setattr('shutil.which', lambda name: '/usr/bin/verilator'
                        if name == 'verilator' else None)
    monkeypatch.setattr('dwn2rtl.verify._CANDIDATE_DIRS', [])
    assert find_simulator().name == 'verilator'


def test_an_explicit_verilator_path_is_recognised_as_verilator(monkeypatch):
    monkeypatch.setattr('shutil.which', lambda name: None)
    monkeypatch.setattr('os.path.exists', lambda p: True)
    assert find_simulator('/opt/bin/verilator').name == 'verilator'


@pytest.mark.verilator
def test_verify_drives_verilator_end_to_end(tmp_path):
    """⚠️ The backend itself, not just the command strings. Skips without verilator, exactly as
    the yosys tests skip without yosys -- nothing in the tool requires it."""
    import fixtures
    from dwn2rtl.build import build

    outdir = build(fixtures.make('n6'), str(tmp_path / 'rtl'), input_bits=8).outdir
    report = verify(outdir, simulator='verilator')

    assert report.ok, '\n'.join(report.lines())
    assert report.simulator.name == 'verilator'
    assert report.simulator.version, 'a simulator that reports no version was not really run'
    for level in report.levels:
        assert level.vectors > 0, f'{level.level} ran no vectors'
        assert level.mismatches == 0


@pytest.mark.sim
def test_a_users_own_verilog_in_the_directory_is_not_compiled(tmp_path):
    """⚠️ The README tells users to instantiate dwn_top in their own harness. A harness dropped
    into the emitted directory used to be swept up by a *.v glob and compiled into the gate, so a
    syntax error in a file that is not under test broke verification entirely.
    """
    import fixtures
    from dwn2rtl.build import build

    outdir = build(fixtures.make('tiny'), str(tmp_path / 'rtl'), input_bits=8).outdir
    with open(os.path.join(outdir, 'zz_my_harness.v'), 'w') as f:
        f.write('module oops  this is not verilog at all ;;;\n')

    report = verify(outdir)
    assert report.ok, '\n'.join(report.lines())


def test_a_missing_design_file_is_reported_by_name(tmp_path):
    """Naming what is absent beats a compile error about an unknown module type."""
    import fixtures
    from dwn2rtl.build import build
    from dwn2rtl.verify import DESIGN_SOURCES

    outdir = build(fixtures.make('tiny'), str(tmp_path / 'rtl'), input_bits=8).outdir
    os.remove(os.path.join(outdir, 'popcount.v'))

    report = verify(outdir)
    assert not report.ok
    assert all(r.status == 'MISSING' for r in report.levels), report.lines()
    assert 'popcount.v' in report.levels[0].detail


def test_levels_are_validated(tmp_path):
    """⚠️ A bare string iterates into characters, so levels='core' raised KeyError('c') -- the
    classic Python trap, from an argument a user is expected to pass."""
    import fixtures
    from dwn2rtl.build import build

    outdir = build(fixtures.make('tiny'), str(tmp_path / 'rtl'), input_bits=8).outdir

    with pytest.raises(TypeError, match='not a string'):
        verify(outdir, levels='core')
    with pytest.raises(ValueError, match='unknown level'):
        verify(outdir, levels=('core', 'topp'))


def test_a_simulator_that_cannot_be_run_names_the_tool(tmp_path):
    """⚠️ Discovery accepts a simulator that merely EXISTS -- a version probe is cosmetic and
    must not cost a working install -- so an unrunnable one is only found when it is run. A bare
    OSError there named a syscall ('[WinError 2] The system cannot find the file specified')
    rather than the tool the user chose.
    """
    import fixtures
    from dwn2rtl.build import build

    outdir = build(fixtures.make('tiny'), str(tmp_path / 'rtl'), input_bits=8).outdir

    stub = tmp_path / ('iverilog.exe' if os.name == 'nt' else 'iverilog')
    stub.write_text('not an executable', encoding='utf-8')

    report = verify(outdir, simulator=str(stub))
    assert not report.ok
    assert all(r.status == 'ERROR' for r in report.levels)
    assert 'could not run' in report.levels[0].detail
    assert str(stub) in report.levels[0].detail, 'the message must name the tool'
