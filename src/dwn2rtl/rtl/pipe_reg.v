// A pipeline stage you can compile out: ENABLE=1 gives a register, ENABLE=0 is a plain wire that
// leaves nothing in the netlist. One source covers every depth, and the total latency is just
// however many stages are switched on.
//
// ⚠️ No reset. These are feed-forward datapath registers, so real data overwrites them within
// LATENCY cycles of the first input -- but that means whatever drives this has to ignore the
// output until the pipe has filled.

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