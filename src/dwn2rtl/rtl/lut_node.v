// One DWN node: a lookup table, which maps to one LUT6.
//
// ⚠️ Slot l is address bit l, slot 0 the LSB -- .addr({slot5, ..., slot0}). Reversed, it still
// builds and is wrong on most inputs. n <= 6 because TABLE is 64 bits; wider silently truncates.

`timescale 1ns / 1ps          // xsim won't mix modules with and without one
`default_nettype none

module lut_node #(
    parameter integer N     = 6,
    parameter [63:0]  TABLE = 64'h0     // bit `addr` holds this node's output for `addr`
)(
    input  wire [N-1:0] addr,
    output wire         out
);

    assign out = TABLE[addr];

endmodule

`default_nettype wire
