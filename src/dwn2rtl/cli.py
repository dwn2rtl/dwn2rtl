"""The `dwn2rtl` terminal command.

A thin argument parser over the library and nothing more. Every subcommand here should be a few
lines of "collect arguments, call one function, print its report" -- if logic starts accumulating
in this file, it is in the wrong place, because then the CLI and `import dwn2rtl` stop being the
same thing.

THE COMMAND SURFACE IS DERIVE, DO NOT ASK (roadmap §5.2). Almost everything a build needs is
already in the checkpoint: features, classes, layers, n, z, wiring, table contents, and the
integer width -- which is derivable EXACTLY from the thresholds. A tool that makes the user
restate what it could have read is a wrapper, not a tool.

Exactly one thing cannot be derived, and it is not the one people expect. It is not fractional
bits -- that is a question no user can answer. It is `--input-bits`, the precision of the user's
INPUT, which is a fact they possess (8-bit pixels, a 12-bit ADC). Fractional width falls out of
it, and when the input has a native quantum the result is provably lossless rather than merely
measured. See precision.py.

STDOUT MUST BE ASCII (CLAUDE.md). Windows consoles default to cp1252 and raise UnicodeEncodeError
on an emoji in print(), which turns a successful build into a traceback. Comments may use
anything; printed output may not.
"""

import argparse
import sys

from . import __version__


def _not_yet(name, phase, what):
    """Fail honestly for a subcommand that is parsed but not yet implemented.

    The alternative -- omitting the subcommand until it works -- makes `--help` misrepresent the
    tool's shape, and the alternative to THAT is a stub that prints nothing and exits 0, which is
    indistinguishable from success. Exit 2 and say which phase it lands in.
    """
    print(f'dwn2rtl {name}: not implemented yet (phase {phase}).', file=sys.stderr)
    print(f'  {what}', file=sys.stderr)
    print(f'  see docs/phase{phase}-ledger.md for where this stands.', file=sys.stderr)
    return 2


def cmd_build(args):
    # Imported here, not at module scope: build pulls in checkpoint.py and therefore torch, and
    # `dwn2rtl verify` must not pay for that. See __init__.py.
    from .build import build
    from .checkpoint import CheckpointError

    try:
        report = build(args.checkpoint, args.out, input_bits=args.input_bits)
    except CheckpointError as e:
        # These messages are the contract -- they name the missing object and show the fix.
        # Wrapping them in a traceback buries the part the user needs.
        print(f'dwn2rtl build: {e}', file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f'dwn2rtl build: {e}', file=sys.stderr)
        return 1

    for line in report.lines():
        print(line)
    return 0


def cmd_verify(args):
    return _not_yet(
        'verify', 2,
        'needs verify.py to find a simulator, compile, run, and parse PASS/FAIL.')


def cmd_estimate(args):
    return _not_yet(
        'estimate', 3,
        'optional yosys resource estimate; encoder and core reported separately.')


def build_parser():
    p = argparse.ArgumentParser(
        prog='dwn2rtl',
        description='Turn a trained DWN into synthesizable Verilog, and prove it matches.')
    p.add_argument('--version', action='version', version=f'dwn2rtl {__version__}')

    sub = p.add_subparsers(dest='command', metavar='{build,verify,estimate}')

    b = sub.add_parser('build', help='checkpoint -> Verilog + golden vectors')
    b.add_argument('checkpoint', help='a trained DWN: {model, thermometer} saved by torch')
    b.add_argument('--out', required=True, metavar='DIR',
                   help='output directory for the emitted design')
    # Not --frac-bits. See this module's docstring, and roadmap Q9.
    b.add_argument('--input-bits', type=int, metavar='N',
                   help="the INPUT's precision in bits (8-bit pixels -> 8). Omit for a "
                        'continuous input, which takes a default and is reported as unproved')
    b.set_defaults(func=cmd_build)

    v = sub.add_parser('verify', help='compile and run the emitted testbenches')
    v.add_argument('dir', help='a directory produced by `dwn2rtl build`')
    v.set_defaults(func=cmd_verify)

    e = sub.add_parser('estimate', help='resource estimate via yosys, if it is on PATH')
    e.add_argument('dir', help='a directory produced by `dwn2rtl build`')
    e.set_defaults(func=cmd_estimate)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, 'command', None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
