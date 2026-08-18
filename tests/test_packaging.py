"""The tree is an installable package with a working CLI.

Nothing here emits Verilog. These test what breaks silently between "works on my machine" and "a
user ran pip install": the package imports, the console script resolves, the hand-written
Verilog ships INSIDE the package, and importing dwn2rtl does not drag torch in.
"""

import os
import subprocess
import sys
from importlib.resources import files

import pytest

import dwn2rtl


PRIMITIVES = ['lut_node.v', 'popcount.v', 'argmax.v', 'pipe_reg.v']


def test_package_imports_and_has_a_version():
    assert dwn2rtl.__version__
    assert isinstance(dwn2rtl.__version__, str)


def test_version_matches_pyproject():
    """The declared version and the importable one must agree -- two files, and nothing but this
    keeps them equal.
    """
    # tomllib is stdlib only from 3.11, and the supported floor is 3.10. Skipping is right
    # here: this test guards two files agreeing with each other, and the answer does not vary
    # by interpreter -- one version in the matrix checking it is enough.
    tomllib = pytest.importorskip('tomllib', reason='stdlib from 3.11; checked on newer jobs')
    from pathlib import Path

    pyproject = Path(__file__).resolve().parent.parent / 'pyproject.toml'
    if not pyproject.exists():           # installed non-editable; nothing to compare against
        pytest.skip('running against an installed package, not a source tree')
    declared = tomllib.loads(pyproject.read_text(encoding='utf-8'))['project']['version']
    assert declared == dwn2rtl.__version__


# --------------------------------------------------------------------------------------
# Package data. See docs/phase0-ledger.md §4 for why this is tested rather than assumed.
# --------------------------------------------------------------------------------------

def test_verilog_primitives_ship_with_the_package():
    """Every emitted design instantiates these four. Unpackaged, `pip install dwn2rtl` gives a tool
    whose output references modules that do not exist -- and an editable install cannot detect
    it, since the path resolves back to the source tree.
    """
    rtl = files('dwn2rtl') / 'rtl'
    for name in PRIMITIVES:
        f = rtl / name
        assert f.is_file(), f'{name} is not packaged'
        assert 'module' in f.read_text(encoding='utf-8'), f'{name} packaged but has no module'


@pytest.mark.parametrize('name', PRIMITIVES + ['tb/dwn_core_tb.v', 'tb/dwn_top_tb.v'])
def test_shipped_verilog_cites_nothing_a_user_cannot_open(name):
    """The primitives are copied into every emitted directory, so their comments are
    documentation a user reads -- and they once cited study-repo paths that do not exist here.
    A reference resolving nowhere is worse than none.
    """
    text = (files('dwn2rtl') / 'rtl' / name).read_text(encoding='utf-8')
    # CLAUDE.md is in .gitignore, so it is not in the repository at all -- a reader who follows
    # that citation finds nothing, exactly like the study-repo paths beside it. It was missing
    # from this list on the first pass, which is how the dwn_top_tb.v reference survived the
    # commit that removed the other five.
    for dangling in ('docs/reference/', 'docs/jsc/', 'probe-results', 'brief ', 'CLAUDE.md'):
        assert dangling not in text, f'{name} cites {dangling!r}, which is not in this repo'
    # This repo's own docs are fair game, but only ones that exist.
    import re
    for cited in re.findall(r'docs/[\w./-]+\.md', text):
        assert os.path.exists(
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), cited)), \
            f'{name} cites {cited}, which does not exist'


def test_testbenches_ship_with_the_package():
    tb = files('dwn2rtl') / 'rtl' / 'tb'
    assert (tb / 'dwn_core_tb.v').is_file()
    assert (tb / 'dwn_top_tb.v').is_file()


def test_both_testbenches_have_content():
    """Neither may be empty. dwn_top_tb.v was 0 bytes until phase 1 -- half the gate did not exist.
    """
    tb = files('dwn2rtl') / 'rtl' / 'tb'
    for name in ('dwn_core_tb.v', 'dwn_top_tb.v'):
        text = (tb / name).read_text(encoding='utf-8')
        assert 'module' in text and '$readmemh' in text, f'{name} is not a real testbench'


# --------------------------------------------------------------------------------------
# The CLI entry point
# --------------------------------------------------------------------------------------

def test_cli_module_is_importable_as_the_entry_point_names_it():
    """pyproject declares `dwn2rtl = "dwn2rtl.cli:main"`.

    pip writes that launcher whether or not the target exists, so a typo or a missing module is
    not a build error -- it is a ModuleNotFoundError the first time a user types the command.
    """
    from dwn2rtl.cli import main
    assert callable(main)


def test_version_flag_exits_zero():
    # Imported locally, not via `dwn2rtl.cli`. The package deliberately does not import its own
    # cli submodule, so the attribute only exists once some other test has imported it -- this
    # passed by accident of file ordering until it was written this way.
    from dwn2rtl.cli import main
    with pytest.raises(SystemExit) as e:
        main(['--version'])
    assert e.value.code == 0


def test_bare_invocation_prints_help_and_fails():
    from dwn2rtl.cli import main
    assert main([]) == 1


# The "not implemented yet" test is GONE, and its disappearance is the point: `build`, `verify`
# and `estimate` are all real as of phase 4, so there is no subcommand left that parses and
# refuses. `_not_yet()` went with it. Exit code 2 no longer occurs, which is why nothing here
# asserts it -- keeping a test for a state the tool can no longer reach would be theatre.


@pytest.mark.parametrize('argv', [
    ['build', 'definitely-not-here.pt', '--out', 'rtl'],
    ['verify', 'definitely-not-a-directory'],
    ['estimate', 'definitely-not-a-directory'],
])
def test_bad_input_exits_one_not_two(argv):
    """Every subcommand reports a bad path as 1, not as a crash and not as success."""
    from dwn2rtl.cli import main
    assert main(argv) == 1


def test_build_rejects_missing_out():
    from dwn2rtl.cli import main
    with pytest.raises(SystemExit):
        main(['build', 'model.pt'])


# --------------------------------------------------------------------------------------
# The torch boundary
# --------------------------------------------------------------------------------------

def test_importing_dwn2rtl_does_not_import_torch():
    """A documented invariant (see __init__.py), so it gets a test rather than a comment.

    ⚠️ Runs in a SUBPROCESS: another test may already have imported torch into this interpreter,
    and the check would then pass for the wrong reason.
    """
    code = 'import sys, dwn2rtl; print("torch" in sys.modules)'
    out = subprocess.run([sys.executable, '-c', code],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == 'False', 'importing dwn2rtl pulled torch in'


def test_cli_help_does_not_import_torch():
    """Same invariant on the path a user actually hits first."""
    code = ('import sys; from dwn2rtl.cli import build_parser; build_parser(); '
            'print("torch" in sys.modules)')
    out = subprocess.run([sys.executable, '-c', code],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == 'False'


def test_cli_output_is_ascii():
    """CLAUDE.md: stdout must be ASCII.

    Windows consoles default to cp1252 and raise UnicodeEncodeError on an emoji in print(),
    which turns a successful build into a traceback. The study repo hit this for real.
    """
    from dwn2rtl.cli import build_parser
    text = build_parser().format_help()
    text.encode('ascii')                 # raises UnicodeEncodeError if anything slipped in
