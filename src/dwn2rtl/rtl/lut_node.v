// One DWN neuron = one lookup table = (the whole point) exactly one Xilinx LUT6.
//
// The entire model is this module, repeated. There is no arithmetic anywhere in the core --
// inter-layer wiring is fixed after training, so it is pure wire assignment.
//
// ADDRESS BIT ORDER IS LOAD-BEARING. Upstream's CUDA kernel builds the address as
//     addr |= (input[mapping[j][l]] > 0) << l;
// so mapping slot l is address bit l -- slot 0 is the LSB (dwn2rtl docs/checkpoint-format.md §2).
// Getting this backwards yields a design that elaborates, synthesizes, and is wrong on most
// inputs. The caller is responsible for concatenating in the matching order:
//     .addr({slot5, slot4, slot3, slot2, slot1, slot0})
//
// n <= 6 AND TABLE IS 64 BITS, and those two facts are the same fact. A node with n inputs
// needs 2**n table entries; at n=6 that is 64, which is exactly one LUT6. Above 6 the table no
// longer fits this parameter and Verilog TRUNCATES the excess SILENTLY -- the design elaborates,
// synthesizes, and computes garbage for every address past 63. dwn2rtl's emitter refuses n > 6
// for that reason. Supporting a wider n means changing this parameter, the emitter and its
// read-back check together, and it also means one node is no longer one LUT6, which is the
// architectural premise the whole approach rests on.

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

    // The learned parameters ARE the table. A synthesis tool must map this to a single LUT6
    // rather than inferring distributed RAM; it is worth checking that it did, because the
    // area claim for a DWN depends on it. Confirmed on Vivado.
    assign out = TABLE[addr];

endmodule

`default_nettype wire