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
from dwn2rtl.verify import (LEVELS, SimulatorNotFound, VerifyReport, find_simulator, verify)


def _sim():
    try:
        return find_simulator()
    except SimulatorNotFound:
        return None


requires_sim = pytest.mark.skipif(_sim() is None, reason='no Verilog simulator available')


@pytest.fixture
def built(tmp_path):
    return build(fixtures.make('n6'), str(tmp_path / 'rtl'), input_bits=8).outdir


# ---------------------------------------------------------------------------------------
# Finding the simulator -- half the job, and not a theoretical concern
# ---------------------------------------------------------------------------------------

@requires_sim
def test_find_simulator_returns_something_runnable():
    sim = find_simulator()
    assert os.path.exists(sim.compiler)
    assert os.path.exists(sim.runner)
    assert sim.name == 'iverilog'


@requires_sim
def test_version_is_detected_despite_iverilog_exiting_nonzero():
    """`iverilog -V` exits 255. Checking its return code would report a perfectly good
    simulator as broken -- measured in phase 0."""
    assert re.search(r'\d+\.\d+', find_simulator().version)


def test_an_explicit_path_that_is_not_executable_is_refused():
    with pytest.raises(SimulatorNotFound, match='not an executable'):
        find_simulator(iverilog='definitely-not-a-simulator-xyz')


def test_not_found_error_lists_what_was_searched_and_how_to_install():
    """A user with no simulator needs the next action, not a negative result. The phase 0
    finding is the reason this matters: a Windows install lands in C:\\iverilog\\bin and adds
    nothing to PATH, so 'not found' is a plausible outcome for someone who has one."""
    err = SimulatorNotFound(
        'no Verilog simulator found. Searched:\n  PATH\n\n  winget install Icarus.Verilog')
    assert 'Searched' in str(err) and 'install' in str(err)


# ---------------------------------------------------------------------------------------
# The gate, through verify()
# ---------------------------------------------------------------------------------------

@requires_sim
@pytest.mark.sim
def test_verify_passes_a_good_build(built):
    r = verify(built)
    assert r.ok
    assert {lv.level for lv in r.levels} == {'core', 'top'}
    assert all(lv.status == 'PASS' for lv in r.levels)
    assert all(lv.mismatches == 0 for lv in r.levels)
    assert 'RESULT   PASS' in '\n'.join(r.lines())


@requires_sim
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


@requires_sim
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


@requires_sim
@pytest.mark.sim
def test_a_compile_error_is_an_error_not_a_pass(built):
    open(os.path.join(built, 'dwn_core.v'), 'a').write('\nthis is not verilog\n')
    r = verify(built)
    assert not r.ok
    assert any(lv.status == 'ERROR' for lv in r.levels)


# ---------------------------------------------------------------------------------------
# Not checking is not passing
# ---------------------------------------------------------------------------------------

@requires_sim
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


@requires_sim
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


@requires_sim
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
    """`all([])` is True.

    Without the explicit emptiness check, a directory with no runnable levels would satisfy
    `all(...)` vacuously and print a green RESULT having checked nothing -- the single most
    dangerous thing this tool could do. Needs no simulator: it is a property of the report.
    """
    from dwn2rtl.verify import Simulator

    empty = VerifyReport('somewhere', Simulator('iverilog', 'x', 'y', '0'), levels=[])
    assert not empty.ok
    text = '\n'.join(empty.lines())
    assert 'nothing to check' in text
    assert 'RESULT   FAIL' in text


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

@requires_sim
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


@requires_sim
@pytest.mark.sim
def test_verify_report_is_ascii(built):
    """CLAUDE.md: cp1252 consoles raise on anything else."""
    '\n'.join(verify(built).lines()).encode('ascii')
