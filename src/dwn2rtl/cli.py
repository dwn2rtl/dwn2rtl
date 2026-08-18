"""The `dwn2rtl` terminal command.

A thin argument parser over the library. Every subcommand should be "collect arguments, call one
function, print its report" -- logic accumulating here means the CLI and `import dwn2rtl` have
stopped being the same thing.

DERIVE, DO NOT ASK. Almost everything a build needs is already in the checkpoint: features,
classes, layers, n, z, wiring, table contents, and the integer width. A tool that makes a user
restate what it could have read is a wrapper, not a tool.

Exactly one thing cannot be derived, and it is not the expected one. Not fractional bits -- no
user can answer that. It is `--input-bits`, the precision of their INPUT, which is a fact they
possess (8-bit pixels, a 12-bit ADC). See precision.py.

⚠️ STDOUT MUST BE ASCII. Windows consoles default to cp1252 and raise UnicodeEncodeError on an
emoji in print(), turning a successful build into a traceback. Comments may use anything.
"""

import argparse
import sys

from . import __version__


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
    # verify never reads a checkpoint, so this path must not import torch. That is why the
    # import is here and why verify.py imports nothing from checkpoint.py.
    from .verify import SimulatorNotFound, verify

    try:
        report = verify(args.dir, iverilog=args.simulator)
    except (FileNotFoundError, SimulatorNotFound) as e:
        print(f'dwn2rtl verify: {e}', file=sys.stderr)
        return 1

    for line in report.lines():
        print(line)
    # Exit code, not just text: this is what a CI job branches on. Anything other than every
    # level passing is a failure, including a level that could not run at all.
    return 0 if report.ok else 1


def cmd_estimate(args):
    # Like verify, this never reads a checkpoint, so it must not import torch.
    from .estimate import YosysNotFound, estimate

    try:
        report = estimate(args.dir, yosys=args.yosys)
    except (FileNotFoundError, YosysNotFound) as e:
        print(f'dwn2rtl estimate: {e}', file=sys.stderr)
        return 1

    for line in report.lines():
        print(line)
    return 0 if report.ok else 1


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
                   help="the INPUT's precision in bits (8-bit pixels -> 8). USUALLY "
                        'UNNECESSARY: if the thresholds lie on a dyadic grid, the training '
                        'data had a native quantum and it is inferred. Otherwise a documented '
                        'default is used and reported as a default. Pass this to override '
                        'either')
    b.set_defaults(func=cmd_build)

    v = sub.add_parser('verify', help='compile and run the emitted testbenches')
    v.add_argument('dir', help='a directory produced by `dwn2rtl build`')
    v.add_argument('--simulator', metavar='EXE',
                   help='path to iverilog, if it is not on PATH and not in a usual place')
    v.set_defaults(func=cmd_verify)

    e = sub.add_parser('estimate', help='resource estimate via yosys, if it is installed')
    e.add_argument('dir', help='a directory produced by `dwn2rtl build`')
    e.add_argument('--yosys', metavar='EXE',
                   help='path to yosys, if it is not on PATH and not in a usual place')
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
