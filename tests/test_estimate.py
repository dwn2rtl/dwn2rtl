"""estimate.py -- shell out to yosys, and be honest about what the number is worth.

`estimate` is the only feature the roadmap itself calls OPTIONAL, so these tests skip cleanly
when yosys is absent (conftest handles the `yosys` marker). That is a real difference from the
`sim` marker: a machine without a simulator cannot verify and that is a problem, while a machine
without yosys is simply a supported machine.

The theme here is the same as verify.py's, one level up: **a number must not outrun its
evidence.** Yosys agrees with Vivado to 4% on the core and is 2.1x low on the encoder
(docs/phase4-ledger.md §3), so the report is required to say so rather than presenting one
figure with the authority of the other.
"""

import os

import pytest

import fixtures
from dwn2rtl.build import build
from dwn2rtl.estimate import (MODULES, EstimateReport, ModuleArea, YosysNotFound, Yosys,
                              estimate, find_yosys)


@pytest.fixture
def built(tmp_path):
    return build(fixtures.make('n6'), str(tmp_path / 'rtl'), input_bits=8).outdir


# ---------------------------------------------------------------------------------------
# Finding yosys, and the silent-no-op trap
# ---------------------------------------------------------------------------------------

@pytest.mark.yosys
def test_find_yosys_returns_something_that_reports_a_version():
    """A version string is the proof it actually RAN.

    ⚠️ The OSS CAD Suite's yosys, invoked by absolute path with a clean environment, prints
    nothing and exits 0 -- a silent no-op. Every measurement from it would be empty and would
    look like a very small design. find_yosys therefore rejects any candidate that cannot report
    a version, rather than returning the first file that exists.
    """
    y = find_yosys()
    assert os.path.exists(y.exe)
    assert y.version, 'a yosys that reports no version is the silent-no-op case'


@pytest.mark.yosys
def test_the_suite_environment_is_prepared_for_the_subprocess():
    """yosys needs its own bin and lib on PATH to load DLLs. This is done per-subprocess and
    must NOT be a change to the user's PATH -- the suite ships its own iverilog and vvp and
    would shadow the simulator the gate runs against."""
    y = find_yosys()
    if not y.root:
        pytest.skip('this yosys is not from an OSS CAD Suite layout')
    env = y.environment()
    assert os.path.join(y.root, 'bin') in env['PATH']
    assert os.path.join(y.root, 'lib') in env['PATH']
    assert os.environ.get('PATH') != env['PATH'], 'the process PATH must be left alone'


def test_an_explicit_path_that_is_not_executable_is_refused():
    with pytest.raises(YosysNotFound, match='not an executable'):
        find_yosys(yosys='definitely-not-yosys-xyz')


def test_not_found_error_says_what_was_searched_and_how_to_install(monkeypatch, tmp_path):
    nowhere = tmp_path / 'nowhere'
    nowhere.mkdir()
    monkeypatch.setenv('PATH', str(nowhere))
    monkeypatch.setattr('dwn2rtl.estimate._CANDIDATE_DIRS', [str(nowhere / 'absent')])

    with pytest.raises(YosysNotFound) as e:
        find_yosys()

    msg = str(e.value)
    assert 'PATH' in msg
    assert 'oss-cad-suite' in msg
    assert 'do NOT add it to PATH' in msg, 'the shadowing hazard must be stated'
    assert msg.isascii()


# ---------------------------------------------------------------------------------------
# Estimating
# ---------------------------------------------------------------------------------------

@pytest.mark.yosys
def test_estimate_reports_every_module_separately(built):
    """Encoder and core are reported SEPARATELY and both always appear -- a design invariant,
    because a tool reporting only their sum commits the defect this project exists to correct."""
    r = estimate(built)

    assert r.ok, '\n'.join(r.lines())
    assert {m.module for m in r.modules} == set(MODULES)
    for m in r.modules:
        assert m.luts > 0, f'{m.module} mapped to no LUTs'


@pytest.mark.yosys
def test_the_top_is_bigger_than_either_half(built):
    r = estimate(built)
    enc, core, top = (r.by_name(n).luts for n in
                      ('thermometer_encoder', 'dwn_core', 'dwn_top'))
    assert top >= max(enc, core)


@pytest.mark.yosys
def test_the_pipeline_registers_show_up_as_flops(tmp_path):
    """The measurement that decides whether an emitted register costs anything.

    PIPE_ENC toggles a register nominally as wide as the whole thermometer vector while only the
    used bits are driven. Turning it off must remove flops -- if it did not, the emitter's
    pipeline parameters would not be reaching the synthesised design at all.
    """
    from dwn2rtl import Pipeline
    ck = fixtures.make('n6')
    on = build(ck, str(tmp_path / 'on'), input_bits=8,
               pipeline=Pipeline(enc=1, lut=1, pop=1, out=1))
    off = build(ck, str(tmp_path / 'off'), input_bits=8,
                pipeline=Pipeline(enc=0, lut=1, pop=1, out=1))

    a = estimate(on.outdir, modules=['dwn_top']).modules[0]
    b = estimate(off.outdir, modules=['dwn_top']).modules[0]
    assert a.flops > b.flops, 'PIPE_ENC did not change the flop count'


@pytest.mark.yosys
def test_flattening_is_verified_not_assumed(built):
    """⚠️ The trap that produced a wrong calibration number before it was caught.

    Without `-flatten`, yosys leaves every lut_node as its own module in gate primitives and
    `abc -lut 6` maps ONLY the top level -- so the $lut figure is one stray submodule's count,
    not the design's, and it looks entirely plausible. After flattening there is exactly one
    $lut line, and more than one is treated as an error rather than summed.
    """
    r = estimate(built, modules=['dwn_core'])
    core = r.modules[0]
    assert core.ok and core.status == 'OK'
    # A 12-node n=6 core is at least a LUT per node; a stray submodule count would be far less.
    assert core.luts >= 12, f'{core.luts} LUTs for 12 nodes suggests an unflattened count'


# ---------------------------------------------------------------------------------------
# Honesty about the number
# ---------------------------------------------------------------------------------------

@pytest.mark.yosys
def test_the_report_states_what_the_numbers_are_worth(built):
    """Calibrated: core within 4%, encoder 2.1x low. A report that omitted this would let the
    encoder figure be read with the core figure's authority."""
    text = '\n'.join(estimate(built).lines())
    assert 'ESTIMATE' in text
    assert 'not your vendor' in text
    assert 'ENCODER' in text and 'LOW' in text
    assert 'synthesize' in text


@pytest.mark.yosys
def test_the_encoder_to_core_ratio_is_qualified(built):
    """The project's headline finding is that the encoder can dominate -- 13.8x on the studied
    model. Yosys sees 6.8x on that same design, so printing the ratio bare would UNDERSTATE the
    very thing the tool exists to publicise."""
    text = '\n'.join(estimate(built).lines())
    assert 'the encoder is' in text
    assert 'as generic mapping sees it' in text


@pytest.mark.yosys
def test_report_is_ascii(built):
    '\n'.join(estimate(built).lines()).encode('ascii')


# ---------------------------------------------------------------------------------------
# Not measuring is not measuring zero
# ---------------------------------------------------------------------------------------

def test_an_empty_report_is_not_ok():
    """`all([])` is True. A run with no modules must not read as a successful estimate."""
    r = EstimateReport('somewhere', Yosys('yosys', '0.0'), modules=[])
    assert not r.ok


def test_a_failed_module_makes_the_whole_report_fail():
    r = EstimateReport('somewhere', Yosys('yosys', '0.0'), modules=[
        ModuleArea('dwn_core', luts=10),
        ModuleArea('thermometer_encoder', status='ERROR', detail='boom'),
    ])
    assert not r.ok
    assert 'ERROR' in '\n'.join(r.lines())


@pytest.mark.yosys
def test_a_missing_source_is_reported_missing(built):
    os.remove(os.path.join(built, 'thermometer_encoder.v'))
    r = estimate(built, modules=['thermometer_encoder'])
    assert r.modules[0].status == 'MISSING'
    assert not r.ok


def test_estimate_refuses_a_directory_that_is_not_a_build(tmp_path):
    (tmp_path / 'random.txt').write_text('hello')
    with pytest.raises(FileNotFoundError, match='does not look like a dwn2rtl build'):
        estimate(str(tmp_path))


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------

@pytest.mark.yosys
def test_cli_estimate_exits_zero_and_prints(built, capsys):
    from dwn2rtl.cli import main
    assert main(['estimate', built]) == 0
    assert 'LUT' in capsys.readouterr().out


def test_cli_estimate_of_a_non_build_exits_one(tmp_path, capsys):
    from dwn2rtl.cli import main
    assert main(['estimate', str(tmp_path)]) == 1
    assert 'dwn2rtl build' in capsys.readouterr().err
