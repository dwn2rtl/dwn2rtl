"""Lint the emitted design with Verilator.

⚠️ Not a second gate. The gate proves the RTL computes the right thing; this proves the files
are well-formed for a tool the gate does not run -- a comment beginning with the word
"verilator" parses as a pragma, which made --lint-only an ERROR while iverilog printed PASS.

Warnings are fatal here on purpose: that is what a user linting an emitted design gets.
"""

import os
import shutil
import subprocess

import pytest

import fixtures
from dwn2rtl.build import build

TOPS = ['dwn_top_tb', 'dwn_core_tb']


def _lint(outdir, top):
    sources = sorted(f for f in os.listdir(outdir) if f.endswith('.v'))
    sources.append(os.path.join('tb', f'{top}.v'))
    return subprocess.run(
        [shutil.which('verilator'), '--lint-only', '--timing', '--top-module', top, *sources],
        cwd=outdir, capture_output=True, text=True, timeout=600)


@pytest.fixture(scope='module')
def built(tmp_path_factory):
    return build(fixtures.make('n6'), str(tmp_path_factory.mktemp('lint') / 'rtl'),
                 input_bits=8).outdir


@pytest.mark.lint
@pytest.mark.parametrize('top', TOPS)
def test_the_emitted_design_lints_clean(built, top):
    """Every emitted and shipped file through a strict linter, warnings fatal. Covers the generated
    files, which no human reads before they reach a user.
    """
    r = _lint(built, top)
    assert r.returncode == 0, f'verilator lint failed on {top}:\n{r.stderr or r.stdout}'


def test_no_comment_line_starts_with_the_pragma_word():
    """⚠️ A sentence beginning with the pragma word is read as one and rejected -- an error that
    stops linting while the gate stays green. Not marked `lint`: pure text, so it runs on Windows
    too, where the offending comment gets written and no linter is installed.
    """
    import re
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
