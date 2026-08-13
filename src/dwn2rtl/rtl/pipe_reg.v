// A pipeline stage that can be compiled out.
//
// Pipeline depth is a Phase 2 sweep axis (brief §10), so stages have to be selectable per
// config rather than hand-placed once. ENABLE=1 inserts a register; ENABLE=0 is a bare wire,
// leaving no trace in the netlist. That means one RTL source covers every depth in the sweep,
// and the latency of a config is just the sum of its enabled stages.
//
// No reset. These are datapath registers in a feed-forward pipeline: every stage is
// overwritten by real data within LATENCY cycles of the first valid input, so a reset would
// only add routing to a global net for no behavioural gain. The harness is responsible for
// ignoring outputs until the pipe has filled. Control logic that DOES need reset lives there.

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