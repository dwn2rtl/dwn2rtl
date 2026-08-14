"""Phase 0's deliverable, asserted: the tree is an installable package with a working CLI.

These are not tests of the generator -- nothing here emits Verilog. They test the things that
break silently between "it works on my machine" and "a user ran pip install":

  - the package imports at all
  - the console-script entry point resolves and runs
  - the hand-written Verilog SHIPS INSIDE the package
  - importing dwn2rtl does not drag torch in

Each one has a specific failure it exists to catch, recorded at the test.
"""

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
    """The declared version and the importable one must agree.

    They are written in two places -- pyproject.toml and __init__.py -- and nothing but this
    test makes them stay equal. A wheel whose metadata says one version and whose code reports
    another is the kind of thing nobody notices until a bug report cites an impossible version.
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
    """The four hand-written modules must be INSIDE the package, not beside it in a repo.

    Every emitted design instantiates lut_node, popcount, argmax and pipe_reg. If they are not
    packaged, `pip install dwn2rtl` produces a tool whose output references four modules that do
    not exist on the user's machine -- and an editable install cannot detect that, because it
    resolves this path straight back to the source tree. Hence also the wheel check in the
    ledger; this test guards the resource lookup itself.
    """
    rtl = files('dwn2rtl') / 'rtl'
    for name in PRIMITIVES:
        f = rtl / name
        assert f.is_file(), f'{name} is not packaged'
        assert 'module' in f.read_text(encoding='utf-8'), f'{name} packaged but has no module'


def test_testbenches_ship_with_the_package():
    tb = files('dwn2rtl') / 'rtl' / 'tb'
    assert (tb / 'dwn_core_tb.v').is_file()
    assert (tb / 'dwn_top_tb.v').is_file()


def test_both_testbenches_have_content():
    """Neither testbench may be empty.

    `dwn_top_tb.v` was a 0-byte file from commit 646aebe until phase 1 -- half the gate did not
    exist. It was held by a strict xfail, which did its job: writing the testbench turned the
    XPASS into a failure and forced the marker's deletion rather than leaving a stale TODO.
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


@pytest.mark.parametrize('argv', [
    ['estimate', 'rtl'],
])
def test_unimplemented_subcommands_exit_two_not_zero(argv):
    """Parsed, but honest about not working yet.

    Exit 0 from a command that did nothing is indistinguishable from success, which is how a
    build script ends up "passing" without producing a design.

    `build` and `verify` both left this list in phase 1. Only `estimate` remains, in phase 3.
    """
    from dwn2rtl.cli import main
    assert main(argv) == 2


@pytest.mark.parametrize('argv', [
    ['build', 'definitely-not-here.pt', '--out', 'rtl'],
    ['verify', 'definitely-not-a-directory'],
])
def test_bad_input_exits_one_not_two(argv):
    """1 is "your input was wrong", 2 is "this tool cannot do that yet". Collapsing them would
    make a script unable to tell a typo from an unfinished feature."""
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

    Runs in a SUBPROCESS deliberately: by the time this test executes, some other test may
    already have imported torch into this interpreter, and the check would pass for the wrong
    reason. A clean interpreter is the only honest way to ask.

    Why it matters: torch is seconds of startup and hundreds of megabytes, every emitter and the
    golden model are pure numpy, and `dwn2rtl verify` on an already-emitted directory has no
    checkpoint to read at all.
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
