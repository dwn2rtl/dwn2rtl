// A pipeline stage you can compile out: STAGES=0 is a plain wire, leaving nothing in the
// netlist. STAGES=N is N registers in series, so a design's latency is the sum of its stages.
//
// ⚠️ No reset. These are feed-forward datapath registers, so real data overwrites them within
// LATENCY cycles of the first input -- but that means whatever drives this has to ignore the
// output until the pipe has filled.

`timescale 1ns / 1ps
`default_nettype none

module pipe_reg #(
    parameter integer WIDTH  = 1,
    parameter integer STAGES = 1
)(
    input  wire             clk,
    input  wire [WIDTH-1:0] d,
    output wire [WIDTH-1:0] q
);

    generate
        if (STAGES <= 0) begin : g_pass
            assign q = d;
        end else begin : g_regs
            // ⚠️ N registers, not one. This was `if (ENABLE != 0)` -- a FLAG -- while the
            // emitter documented the parameter as a stage count and computed latency by summing
            // counts. Any depth above 1 therefore claimed a latency the hardware did not have,
            // and the testbench sampled the wrong cycle.
            reg [WIDTH-1:0] r [0:STAGES-1];
            integer i;
            always @(posedge clk) begin
                r[0] <= d;
                for (i = 1; i < STAGES; i = i + 1)
                    r[i] <= r[i-1];
            end
            assign q = r[STAGES-1];
        end
    endgenerate

endmodule

`default_nettype wire
