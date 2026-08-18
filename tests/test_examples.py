"""The worked example must actually work.

An example is documentation that claims to be executable, so it rots in a way prose does not:
prose that goes stale is merely wrong, while an example that goes stale is wrong AND was
promised to run. It is the first thing a new user tries, and the first thing they judge the tool
by, so it runs in CI like anything else.
"""

import os
import subprocess
import sys

import pytest


EXAMPLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'examples', 'quickstart.py')


def test_the_example_exists_where_the_readme_says_it_does():
    assert os.path.exists(EXAMPLE), 'README points at examples/quickstart.py'


@pytest.mark.sim
def test_the_example_runs_and_the_gate_passes(tmp_path):
    """End to end, as a new user's first command: save a model, emit Verilog, verify it."""
    out = subprocess.run([sys.executable, EXAMPLE, str(tmp_path / 'work')],
                         capture_output=True, text=True, timeout=600)

    assert out.returncode == 0, f'the example failed:\n{out.stdout}\n{out.stderr}'
    assert 'RESULT   PASS' in out.stdout, out.stdout
    assert 'dwn_core' in out.stdout and 'dwn_top' in out.stdout


@pytest.mark.sim
def test_the_example_leaves_a_usable_design_behind(tmp_path):
    """The directory it writes is the deliverable, so it is checked rather than assumed."""
    work = tmp_path / 'work'
    subprocess.run([sys.executable, EXAMPLE, str(work)],
                   capture_output=True, text=True, timeout=600, check=True)

    rtl = work / 'rtl'
    for name in ('dwn_core.v', 'thermometer_encoder.v', 'dwn_top.v',
                 'lut_node.v', 'x_quant.hex', 'expected_top.hex'):
        assert (rtl / name).exists(), f'{name} missing from the example output'
    assert (rtl / 'tb' / 'dwn_top_tb.v').exists()


def test_the_example_output_is_ascii(tmp_path):
    """CLAUDE.md: stdout must be ASCII. A cp1252 console raising UnicodeEncodeError on the
    first thing a Windows user runs would be a poor introduction."""
    src = open(EXAMPLE, encoding='utf-8').read()
    printed = [ln for ln in src.splitlines() if 'print(' in ln]
    '\n'.join(printed).encode('ascii')


REPO_BLOB = 'https://github.com/dwn2rtl/dwn2rtl/blob/main/'


def _readme_link_targets():
    import re

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    readme = open(os.path.join(root, 'README.md'), encoding='utf-8').read()
    return root, [m.group(1) for m in re.finditer(r'\]\(([^)#][^)]*)\)', readme)]


def test_readme_has_no_relative_links():
    """⚠️ PyPI renders the README as the PROJECT PAGE, where a relative link resolves against
    pypi.org and 404s.

    The five that were here -- four docs and LICENSE -- worked perfectly on GitHub and would
    have been dead on the page every new user sees. Nothing about the file looks wrong; the
    same text is correct in one renderer and broken in the other, which is why this is a test
    rather than a habit.
    """
    _, targets = _readme_link_targets()
    relative = [t for t in targets if not t.startswith(('http://', 'https://', 'mailto:'))]
    assert not relative, (
        f'README links are relative and will 404 on PyPI: {relative}. '
        f'Prefix them with {REPO_BLOB}')


def test_readme_links_into_this_repo_resolve():
    """Absolute links still have to point at something.

    Making the links absolute fixes PyPI and removes the old safety net: a URL cannot be
    checked by looking at the filesystem. So the ones aimed back into this repo are mapped to
    their paths and checked against git, which is what the previous version of this test did --
    it caught a link to CLAUDE.md, which is gitignored and 404s for everyone else.
    """
    root, targets = _readme_link_targets()
    inward = [t for t in targets if t.startswith(REPO_BLOB)]
    assert inward, 'no links back into this repo -- has the README been rewritten?'

    tracked = subprocess.run(['git', 'ls-files'], cwd=root,
                             capture_output=True, text=True, check=True).stdout.split()
    tracked = {p.replace('/', os.sep) for p in tracked}

    missing = [t for t in inward
               if t[len(REPO_BLOB):].replace('/', os.sep) not in tracked]
    assert not missing, f'README links to untracked or missing files: {missing}'


def test_readme_tells_people_how_to_install_it():
    """Replaces test_readme_does_not_promise_a_pypi_package_that_does_not_exist, which asserted
    the OPPOSITE and was correct until 0.1.0 went to PyPI. Its docstring said to delete it at
    exactly that moment, so it was.

    ⚠️ The README is baked into the uploaded artifact, so PyPI renders whatever this file said
    when the wheel was BUILT -- an install line is frozen on the project page for that version.
    That is why the instruction has to be right before the build rather than after the upload,
    and why it is worth a test rather than a memory.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    readme = open(os.path.join(root, 'README.md'), encoding='utf-8').read()
    assert 'pip install dwn2rtl\n' in readme, \
        'the README must tell users the published install command'
    assert 'pip install git+' not in readme, \
        'the git install was for the pre-PyPI era; it points at a stale org path now'


def test_readme_points_at_the_example_that_exists():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    readme = open(os.path.join(root, 'README.md'), encoding='utf-8').read()
    assert 'examples/quickstart.py' in readme
    assert os.path.exists(EXAMPLE)


def test_the_example_does_not_import_the_upstream_package():
    """Requiring users to compile a CUDA/C++ extension before they can see the tool work would
    be backwards -- and dwn2rtl genuinely does not need it, because checkpoint.py duck-types."""
    src = open(EXAMPLE, encoding='utf-8').read()
    code = '\n'.join(ln for ln in src.splitlines() if not ln.strip().startswith('#'))
    # Mentioned in a docstring (showing users the real recipe) is fine; imported is not.
    assert 'import torch_dwn' not in code.split('"""')[0]
    assert '\nimport torch_dwn' not in code.replace('        import torch_dwn as dwn', '')
