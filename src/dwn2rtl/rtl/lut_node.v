// One DWN neuron = one lookup table = (the whole point) exactly one Xilinx LUT6.
//
// The entire model is this module, repeated. There is no arithmetic anywhere in the core --
// inter-layer wiring is fixed after training, so it is pure wire assignment.
//
// ADDRESS BIT ORDER IS LOAD-BEARING. Upstream's CUDA kernel builds the address as
//     addr |= (input[mapping[j][l]] > 0) << l;
// so mapping slot l is address bit l -- slot 0 is the LSB (docs/reference/checkpoint-format.md §2).
// Getting this backwards yields a design that elaborates, synthesizes, and is wrong on most
// inputs. The caller is responsible for concatenating in the matching order:
//     .addr({slot5, slot4, slot3, slot2, slot1, slot0})
//
// n=6 is fixed for Phase 1 bring-up (CLAUDE.md) and TABLE is sized 2**6 accordingly. n
// becomes a sweep axis in Phase 2, at which point TABLE's width has to follow it.

// timescale is required because the testbench declares one, and xsim rejects a design that
// mixes modules with and without it. Harmless for synthesis.
`timescale 1ns / 1ps
`default_nettype none

module lut_node #(
    parameter integer N     = 6,
    parameter [63:0]  TABLE = 64'h0     // bit `addr` holds this node's output for `addr`
)(
    input  wire [N-1:0] addr,
    output wire         out
);

    // The learned parameters ARE the table. Vivado must map this to a single LUT6 rather
    // than inferring distributed RAM -- confirmed by the Phase 1a probe, docs/reference/probe-results.md.
    assign out = TABLE[addr];

endmodule

`default_nettype wire