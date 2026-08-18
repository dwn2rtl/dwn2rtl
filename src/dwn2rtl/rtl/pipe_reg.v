// A pipeline stage that can be compiled out: ENABLE=1 inserts a register, ENABLE=0 is a bare
// wire leaving no trace in the netlist. Depth is a build parameter, so one source covers every
// latency and a design's latency is the sum of its enabled stages.
//
// ⚠️ No reset, deliberately -- these are feed-forward datapath registers, overwritten by real
// data within LATENCY cycles of the first valid input. Whatever drives this must ignore the
// outputs until the pipe has filled.

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