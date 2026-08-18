"""Lint the emitted design with Verilator.

NOT a second gate. The gate proves the RTL computes what the golden model computes; this proves
the files are well-formed for a tool the gate does not run. Those are different questions, and
the difference is not hypothetical: a comment line beginning with the word "verilator" is parsed
as a pragma, so prose after it is rejected as an unknown one. That made `--lint-only` fail with
an ERROR while iverilog printed PASS on the same files.

Warnings are fatal here on purpose (no -Wno-fatal). Verilator's default is fatal, so a user who
lints an emitted design sees an error -- if that is what they get, CI should get it too.
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
    """Every emitted and shipped file, through a strict linter, with warnings fatal.

    This is the check that would have caught the pragma bug above, and it covers the generated
    files -- dwn_core.v, thermometer_encoder.v, dwn_top.v -- which no human reads before they
    reach a user.
    """
    r = _lint(built, top)
    assert r.returncode == 0, f'verilator lint failed on {top}:\n{r.stderr or r.stdout}'


def test_no_comment_line_starts_with_the_pragma_word():
    """⚠️ The specific trap, pinned at the source rather than only in a linter run.

    `// verilator ...` and `/* verilator ... */` are PRAGMAS. A sentence that merely begins with
    the word is read as one and rejected -- an error that stops linting while the gate stays
    green, so nothing else in this suite would notice. Only the real pragmas may match.

    Deliberately NOT marked `lint`: it is pure text, needs no verilator, and so runs on the
    Windows jobs too -- which is where a contributor most likely writes the offending comment
    and has no linter to catch it.
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
