// One DWN node is one lookup table, which is one LUT6. The core is just this module repeated --
// the wiring is fixed after training, so there's no arithmetic anywhere in it.
//
// ⚠️ Bit order matters. The address is built as `addr |= (input[mapping[j][l]] > 0) << l`, so
// mapping slot l is address bit l and slot 0 is the LSB. Wire it up to match:
//     .addr({slot5, slot4, slot3, slot2, slot1, slot0})
// Reverse it and the design still elaborates and synthesizes -- it's just wrong on most inputs.
// (dwn2rtl docs/checkpoint-format.md §2)
//
// ⚠️ n has to be <= 6, since TABLE is 64 bits. Anything wider needs 2**n entries and Verilog
// drops the excess without a word, so every address past 63 returns garbage.

// timescale: the testbench declares one, and xsim won't mix modules with and without it.
`timescale 1ns / 1ps
`default_nettype none

module lut_node #(
    parameter integer N     = 6,
    parameter [63:0]  TABLE = 64'h0     // bit `addr` holds this node's output for `addr`
)(
    input  wire [N-1:0] addr,
    output wire         out
);

    // The learned parameters are the table. Worth checking your tool maps this to a LUT6 and not
    // distributed RAM -- the whole area story depends on it. Vivado does.
    assign out = TABLE[addr];

endmodule

`default_nettype wire