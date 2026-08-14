"""dwn2rtl end to end, in one file, with no dependencies beyond dwn2rtl itself.

    python examples/quickstart.py

It builds a small DWN, saves it the way you would save a trained one, emits Verilog, and runs
that Verilog through a simulator to prove it computes the same function as the software model.

WHY THIS EXAMPLE DOES NOT TRAIN ANYTHING, which is a deliberate choice rather than a shortcut:

    dwn2rtl translates a model's STRUCTURE into hardware. Whether that model is any good is a
    training question the tool has no opinion about -- and the thing being proved here is that
    the Verilog matches the model, not that the model matches a dataset. An untrained model
    exercises the emitter identically to a trained one.

    It also means this file runs anywhere, in seconds, without installing the upstream DWN
    package (which builds a CUDA/C++ extension) or downloading a dataset.

For what the real thing looks like, see `for_a_real_model()` at the bottom -- the only difference
is where the model comes from.
"""

import os
import sys
import tempfile

import numpy as np
import torch

import dwn2rtl


# ---------------------------------------------------------------------------------------
# A DWN, structurally. In real use these three classes come from `torch_dwn`.
# ---------------------------------------------------------------------------------------
#
# dwn2rtl never imports the upstream package -- it reads tensors and duck-types attributes, so
# it does not care whether these are upstream's classes or these stand-ins. That is also why a
# version bump upstream cannot break loading.

class LUTLayer(torch.nn.Module):
    """`output_size` lookup tables of 2**n entries, each reading n bits of the input.

    The learned parameters ARE the truth tables: at inference a node's output is simply
    `luts[node][address] > 0`, which is one LUT6 in hardware when n=6.
    """

    def __init__(self, input_size, output_size, n=6):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.luts = torch.nn.Parameter(
            torch.rand(output_size, 2 ** n, generator=g) * 2 - 1, requires_grad=False)
        # A fixed ('random') mapping: node j slot k reads input bit mapping[j][k].
        self.mapping = torch.nn.Parameter(
            torch.randint(0, input_size, (output_size, n), generator=g).int(),
            requires_grad=False)


class GroupSum(torch.nn.Module):
    """The output stage: split the final layer into `k` contiguous groups, one per class, and
    take the popcount of each. In hardware that is a popcount and a comparison -- `tau` is a
    uniform divisor and cannot change which class wins."""

    def __init__(self, k, tau=1 / 0.3):
        super().__init__()
        self.k = k
        self.tau = tau


class Thermometer:
    """The binarization front end, fitted BEFORE training.

    ⚠️ This is the object people lose. It is not part of the model and not in its state_dict, so
    `torch.save(model.state_dict())` silently discards it -- along with the entire encoder,
    which can be many times the size of the network it feeds. dwn2rtl refuses such a file by
    name rather than emitting a broken design.
    """

    def __init__(self, thresholds):
        self.thresholds = thresholds


def make_a_model(n_features=16, z=8, width=60, n_classes=5, n=6):
    """A DWN with `n_features` inputs, `z` thermometer bits each, one LUT layer, `n_classes`
    classes. Untrained -- see the module docstring."""
    thresholds = torch.from_numpy(
        np.sort(np.random.default_rng(0).uniform(-1, 1, (n_features, z)), axis=1)
        .astype(np.float32))

    model = torch.nn.Sequential(
        LUTLayer(n_features * z, width, n=n),
        GroupSum(k=n_classes),
    )
    return model, Thermometer(thresholds)


# ---------------------------------------------------------------------------------------

def main(outdir=None):
    workdir = outdir or tempfile.mkdtemp(prefix='dwn2rtl-quickstart-')
    os.makedirs(workdir, exist_ok=True)
    checkpoint = os.path.join(workdir, 'model.dwn')
    rtl = os.path.join(workdir, 'rtl')

    print('1. build a DWN and save it -------------------------------------------------')
    model, thermometer = make_a_model()

    # THE ONE LINE YOU ADD TO YOUR TRAINING SCRIPT. Plain torch -- no dwn2rtl import needed:
    #
    #     torch.save({'model': model, 'thermometer': thermometer}, 'model.pt')
    #
    # dwn2rtl.save does the same thing and validates while the objects are still in memory, so
    # a mistake surfaces now rather than at build time on another machine.
    dwn2rtl.save(model, thermometer, checkpoint, run_name='quickstart')
    print(f'   {checkpoint}  ({os.path.getsize(checkpoint) / 1024:.0f} KB)\n')

    print('2. emit Verilog ------------------------------------------------------------')
    # --input-bits is the INPUT's precision, not the model's. It is the one thing a checkpoint
    # cannot tell you, and the one thing you always know. 8 here means "my features arrive as
    # 8-bit values scaled into [0,1]", as image pixels do -- for which the quantisation is
    # provably lossless rather than merely measured.
    report = dwn2rtl.build(checkpoint, rtl, input_bits=8)
    for line in report.lines():
        print('   ' + line)
    print()

    print('3. prove the Verilog matches the model -------------------------------------')
    try:
        result = dwn2rtl.verify(rtl)
    except dwn2rtl.SimulatorNotFound as e:
        print('   no simulator, so nothing was verified:\n')
        print('   ' + str(e).replace('\n', '\n   '))
        # Exit non-zero: the design was emitted but NOTHING checked it, and a green-looking
        # example that verified nothing is exactly the failure this project is organised
        # against.
        return 2

    for line in result.lines():
        print('   ' + line)
    return 0 if result.ok else 1


def for_a_real_model():
    """What the same flow looks like with a genuinely trained model.

    Not executed -- it needs `torch_dwn` and a dataset. The point is that only step 1 differs;
    steps 2 and 3 are identical, because dwn2rtl reads structure and does not care how the
    weights got there.

        import torch_dwn as dwn

        thermometer = dwn.DistributiveThermometer(3).fit(x_train)
        x_train_bin = thermometer.binarize(x_train).flatten(start_dim=1)

        model = torch.nn.Sequential(
            dwn.LUTLayer(x_train_bin.size(1), 2000, n=6, mapping='learnable'),
            dwn.LUTLayer(2000, 1000, n=6),
            dwn.GroupSum(k=10, tau=1/0.3),
        )
        ...  # train as usual

        torch.save({'model': model, 'thermometer': thermometer}, 'model.pt')

    then, in a terminal:

        dwn2rtl build model.pt --out rtl/ --input-bits 8
        dwn2rtl verify rtl/

    ⚠️ If your training scaled its features -- a StandardScaler, say -- pass the scaler too:

        torch.save({'model': model, 'thermometer': thermometer, 'scaler': scaler}, 'model.pt')

    The thresholds live in whatever feature space training used, so whatever drives the emitted
    design must arrive in that same space. Feed it raw features when the model saw scaled ones
    and it runs at chance while looking entirely healthy. dwn2rtl writes the scaling parameters
    out as input_scaling.json so the harness can apply them.
    """


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
