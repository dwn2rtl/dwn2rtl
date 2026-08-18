// A pipeline stage you can compile out: ENABLE=0 is a plain wire, leaving nothing behind.
//
// ⚠️ No reset -- feed-forward datapath only, so ignore the output until the pipe has filled.

`timescale 1ns / 1ps
`default_nettype none

module pipe_reg #(
    parameter integer WIDTH  = 1,
    parameter integer ENABLE = 1
)(
    input  wire             clk,
    input  wire [WIDTH-1:0] d,
    output wire [WIDTH-1:0] q
);

    generate
        if (ENABLE != 0) begin : g_reg
            reg [WIDTH-1:0] r;
            always @(posedge clk)
                r <= d;
            assign q = r;
        end else begin : g_pass
            assign q = d;
        end
    endgenerate

endmodule

`default_nettype wire
