// One DWN node = one lookup table = exactly one Xilinx LUT6. The core is this module repeated;
// wiring is fixed after training, so there is no arithmetic in it.
//
// ⚠️ ADDRESS BIT ORDER IS LOAD-BEARING. The address is built as
//     addr |= (input[mapping[j][l]] > 0) << l
// so mapping slot l is address bit l, slot 0 the LSB. Concatenate to match:
//     .addr({slot5, slot4, slot3, slot2, slot1, slot0})
// Reversed, the design elaborates, synthesizes, and is wrong on most inputs.
// See dwn2rtl docs/checkpoint-format.md §2.
//
// ⚠️ n <= 6, because TABLE is 64 bits: n inputs need 2**n entries, and past 64 Verilog
// truncates SILENTLY, computing garbage for every address over 63. The emitter refuses n > 6.

// timescale: required because the testbench declares one, and xsim rejects a design that mixes
// modules with and without it. Harmless for synthesis.
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