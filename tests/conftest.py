"""Shared test setup.

Two jobs that were previously happening by accident or by repetition: making `import fixtures`
deliberate rather than relying on pytest's sys.path insertion, and skipping tool-dependent tests
in ONE place instead of three copies of "is there one?".
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _find_simulator():
    """Discovery lives in verify.py -- this only asks it. Two implementations of "where is
    iverilog" would drift, and the one users hit is verify.py's."""
    from dwn2rtl.verify import SimulatorNotFound, find_simulator
    try:
        return find_simulator()
    except SimulatorNotFound:
        return None


SIMULATOR = _find_simulator()


def _find_yosys():
    from dwn2rtl.estimate import YosysNotFound, find_yosys
    try:
        return find_yosys()
    except YosysNotFound:
        return None


YOSYS = _find_yosys()


def _find_verilator():
    """Verilator, for lint only -- NOT a second gate. It catches what the gate cannot: whether the
    files are well-formed for a tool the gate never runs.
    """
    import shutil
    import subprocess
    exe = shutil.which('verilator')
    if not exe:
        return None
    try:
        out = subprocess.run([exe, '--version'], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return (out.stdout or out.stderr).strip().splitlines()[0] if out.returncode == 0 else None


VERILATOR = _find_verilator()


@pytest.fixture(scope='session')
def simulator():
    """The simulator, for tests that drive it directly rather than through verify()."""
    if SIMULATOR is None:
        pytest.skip('no Verilog simulator available')
    return SIMULATOR


def pytest_collection_modifyitems(config, items):
    """Skip tool-marked tests in one place.

    ⚠️ Skipping is not passing, and this hook could quietly hide the project's only correctness
    signal -- hence the report line saying so.
    """
    if SIMULATOR is None:
        skip = pytest.mark.skip(reason='no Verilog simulator -- THE GATE DID NOT RUN')
        for item in items:
            if 'sim' in item.keywords:
                item.add_marker(skip)
    if YOSYS is None:
        # Unlike the gate, this one is genuinely optional: `estimate` is the only feature the
        # roadmap itself calls optional, and a machine without yosys is a supported machine.
        skip = pytest.mark.skip(reason='no yosys -- estimate is optional')
        for item in items:
            if 'yosys' in item.keywords:
                item.add_marker(skip)
    if VERILATOR is None:
        # Optional like yosys: linting is a second opinion, not the correctness signal.
        skip = pytest.mark.skip(reason='no verilator -- lint is a second opinion, not the gate')
        for item in items:
            if 'lint' in item.keywords:
                item.add_marker(skip)


def pytest_report_header(config):
    """State up front whether the gate can run at all.

    A header, not a footer: someone scrolling past a green summary should not have to work out
    from a skip count whether the thing that matters actually executed.
    """
    if SIMULATOR is None:
        return ['dwn2rtl: NO SIMULATOR FOUND -- gate tests will be SKIPPED, not passed']
    return [f'dwn2rtl: gate simulator {SIMULATOR.describe()}']
