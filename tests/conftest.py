"""Shared test setup.

Two jobs, both of which were previously happening by accident or by repetition.

1. MAKE `import fixtures` DELIBERATE. It worked because pytest inserts a test file's directory
   into sys.path when that directory has no __init__.py -- a real rule, but an implicit one that
   depends on a layout nobody wrote down. Anyone who added an __init__.py to tests/, or ran the
   suite through a different runner, would have got ImportError with no obvious cause. Now it is
   one line that says what it is doing.

2. SKIP THE GATE IN ONE PLACE. `@pytest.mark.sim` tests need a simulator, and three test files
   had each grown their own copy of "is there one?". Centralised here, so a test only has to
   declare that it needs a simulator and not how to look for one.
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


@pytest.fixture(scope='session')
def simulator():
    """The simulator, for tests that drive it directly rather than through verify()."""
    if SIMULATOR is None:
        pytest.skip('no Verilog simulator available')
    return SIMULATOR


def pytest_collection_modifyitems(config, items):
    """Skip `sim`-marked tests when there is no simulator, everywhere, once.

    ⚠️ SKIPPING IS NOT PASSING, and this hook is the one place that could quietly hide the
    project's only correctness signal. It prints a report line saying so, because a run with the
    gate silently absent looks exactly like a run with the gate green -- and CI must be
    configured to have a simulator, not to tolerate its absence.
    """
    if SIMULATOR is not None:
        return
    skip = pytest.mark.skip(reason='no Verilog simulator -- THE GATE DID NOT RUN')
    for item in items:
        if 'sim' in item.keywords:
            item.add_marker(skip)


def pytest_report_header(config):
    """State up front whether the gate can run at all.

    A header, not a footer: someone scrolling past a green summary should not have to work out
    from a skip count whether the thing that matters actually executed.
    """
    if SIMULATOR is None:
        return ['dwn2rtl: NO SIMULATOR FOUND -- gate tests will be SKIPPED, not passed']
    return [f'dwn2rtl: gate simulator {SIMULATOR.describe()}']
