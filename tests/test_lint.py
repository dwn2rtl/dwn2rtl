"""Verilator: lint the emitted design, and run the gate under a second simulator.

⚠️ Optional, exactly like yosys -- nothing in the tool touches Verilator, and these skip when it
is absent. CI installs it on Linux only.

Two different jobs, neither of which the gate does alone:

  lint          are the files well-formed for a tool the gate does not run? A comment beginning
                with the word "verilator" parses as a pragma, which made --lint-only an ERROR
                while iverilog printed PASS. Warnings are fatal here, as they are for a user.
  cross-check   does a SECOND, independent simulator agree the design matches the golden model?
                The gate's one weakness is that it has a single implementation behind it.
"""

import os
import re
import shutil
import subprocess

import pytest

import fixtures
from dwn2rtl.build import build

TOPS = ['dwn_top_tb', 'dwn_core_tb']


def _verilator(*args, cwd, timeout=900):
    return subprocess.run([shutil.which('verilator'), *args],
                          cwd=cwd, capture_output=True, text=True, timeout=timeout)


def _sources(outdir, top):
    return sorted(f for f in os.listdir(outdir) if f.endswith('.v')) + \
        [os.path.join('tb', f'{top}.v')]


def _lint(outdir, top):
    return _verilator('--lint-only', '--timing', '--top-module', top,
                      *_sources(outdir, top), cwd=outdir)


@pytest.fixture(scope='module')
def built(tmp_path_factory):
    return build(fixtures.make('n6'), str(tmp_path_factory.mktemp('lint') / 'rtl'),
                 input_bits=8).outdir


@pytest.mark.verilator
@pytest.mark.parametrize('top', TOPS)
def test_the_emitted_design_lints_clean(built, top):
    """Every emitted and shipped file through a strict linter, warnings fatal. Covers the generated
    files, which no human reads before they reach a user.
    """
    r = _lint(built, top)
    assert r.returncode == 0, f'verilator lint failed on {top}:\n{r.stderr or r.stdout}'


@pytest.mark.verilator
@pytest.mark.parametrize('top,level', [('dwn_top_tb', 'top'), ('dwn_core_tb', 'core')])
def test_a_second_simulator_agrees_the_design_is_bit_exact(built, tmp_path, top, level):
    """⚠️ The gate's one weakness: a single implementation stands behind it.

    Two simulators agreeing on every vector is a much stronger claim than one, and the emitted
    testbench is simulator-agnostic -- so this is the same golden-model comparison, run by a
    tool that shares no code with iverilog.

    It asserts the vector COUNT too. A build whose testbench silently ran zero vectors would
    print PASS, which is the failure this project cares about most.
    """
    obj = str(tmp_path / f'obj_{level}')
    exe = f'v_{level}'
    c = _verilator('--binary', '--timing', '--top-module', top,
                   '--Mdir', obj, '-o', exe, *_sources(built, top), cwd=built)
    assert c.returncode == 0, f'verilator could not build {top}:\n{c.stderr or c.stdout}'

    # cwd stays the design directory: $readmemh and `include resolve against it, not the source.
    run = subprocess.run([os.path.join(obj, exe)], cwd=built,
                         capture_output=True, text=True, timeout=900)
    out = run.stdout

    assert re.search(r'RESULT\s*:\s*PASS', out), f'verilator did not report PASS:\n{out}'
    assert re.search(r'mismatches\s*:\s*0\b', out), out
    n = re.search(r'vectors tested\s*:\s*(\d+)', out)
    assert n and int(n.group(1)) > 0, f'no vectors were actually run:\n{out}'


def test_no_comment_line_starts_with_the_pragma_word():
    """⚠️ A sentence beginning with the pragma word is read as one and rejected -- an error that
    stops linting while the gate stays green. Not marked `verilator`: pure text, so it runs on Windows
    too, where the offending comment gets written and no linter is installed.
    """
    from dwn2rtl.build import PRIMITIVES

    rtl = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'src', 'dwn2rtl', 'rtl')
    allowed = re.compile(r'^\s*(//|/\*)\s*verilator\s+(lint_off|lint_on|lint_save|lint_restore|'
                         r'coverage_off|coverage_on|tracing_off|tracing_on|public|isolate_assignments'
                         r'|no_inline_module|sc_bv|systemc_\w+)\b')
    suspect = re.compile(r'^\s*(//|/\*)\s*verilator\b', re.IGNORECASE)

    offenders = []
    for name in list(PRIMITIVES) + [os.path.join('tb', f'{t}.v') for t in TOPS]:
        path = os.path.join(rtl, name)
        for i, line in enumerate(open(path, encoding='utf-8').read().splitlines(), 1):
            if suspect.match(line) and not allowed.match(line):
                offenders.append(f'{name}:{i}: {line.strip()[:70]}')
    assert not offenders, (
        'these comments will be parsed as verilator pragmas and rejected:\n  '
        + '\n  '.join(offenders))
