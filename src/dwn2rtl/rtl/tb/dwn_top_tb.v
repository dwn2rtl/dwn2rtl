// GATE 1, TOP LEVEL -- quantized features in, class index out.
//
// The companion to dwn_core_tb, which drives PRE-BINARIZED bits and so tests the LUT network
// alone. This one drives quantized FEATURES through the thermometer encoder as well, and the
// split is what makes a failure localize itself:
//
//     core PASS, top FAIL   -> the encoder. Nothing else needs re-examining.
//     core FAIL, top FAIL   -> the network; fix that first, this will follow.
//
// The encoder is what this level adds, and it is the half published DWN resource counts leave
// out -- intrinsic to a DWN, not preprocessing, and on the smallest studied model fourteen
// times the network it feeds. An unverified encoder is most of an unverified design.
//
// Vectors stream back-to-back, one per cycle, checked LATENCY cycles later: that is what proves
// II=1. A design that computed correctly but stalled would pass a one-at-a-time testbench.
//
// ⚠️ LATENCY and IDX_W come from dwn_top_params.vh, never hardcoded. An earlier testbench fixed
// IDX_W at 3, so a 10-class design was checked on three of its four index bits -- and passed.

`timescale 1ns / 1ps
`default_nettype none

`include "top_params.vh"
`include "dwn_top_params.vh"

module dwn_top_tb;

    localparam integer N_TOP   = `N_TOP;
    localparam integer X_W     = `X_W;
    localparam integer IDX_W   = `IDX_W;
    localparam integer LATENCY = `DWN_TOP_LATENCY;

    // x_quant.hex packs feature f at [f*WORD +: WORD], two's complement, so the whole feature
    // vector is one X_W-bit word per line -- exactly what dwn_top's x_flat port expects.
    reg [X_W-1:0] vectors  [0:N_TOP-1];
    reg [7:0]     expected [0:N_TOP-1];

    reg              clk = 1'b0;
    reg  [X_W-1:0]   x_flat;
    wire [IDX_W-1:0] class_idx;

    always #5 clk = ~clk;          // 100 MHz

    dwn_top dut (.clk(clk), .x_flat(x_flat), .class_idx(class_idx));

    integer i, j;
    integer errors;
    integer first_bad;

    initial begin
        $readmemh("x_quant.hex",      vectors);
        $readmemh("expected_top.hex", expected);

        errors    = 0;
        first_bad = -1;
        x_flat    = {X_W{1'b0}};

        // Drive on the negative edge so inputs are stable across the capturing posedge; read
        // outputs on the same negedge, by which point the pipeline has settled.
        for (i = 0; i < N_TOP + LATENCY; i = i + 1) begin
            @(negedge clk);
            if (i >= LATENCY) begin
                j = i - LATENCY;
                // !== not != : an x or z must count as a failure rather than propagating
                // silently into a comparison that returns x. Undriven encoder bits are a real
                // possibility here in a way they are not at the core level.
                if (class_idx !== expected[j][IDX_W-1:0]) begin
                    if (first_bad == -1) first_bad = j;
                    errors = errors + 1;
                    if (errors <= 10)
                        $display("  MISMATCH vector %0d: rtl=%0d golden=%0d",
                                 j, class_idx, expected[j][IDX_W-1:0]);
                end
            end
            x_flat = (i < N_TOP) ? vectors[i] : {X_W{1'b0}};
        end

        $display("");
        $display("========================================");
        $display("GATE 1 -- dwn_top (encoder + core) vs golden model");
        $display("  vectors tested : %0d", N_TOP);
        $display("  feature word   : %0d bits", X_W);
        $display("  latency        : %0d cycles, II=1 (new vector every clock)", LATENCY);
        $display("  mismatches     : %0d", errors);
        if (errors == 0) begin
            $display("  RESULT         : PASS (bit-exact on every vector)");
        end else begin
            $display("  RESULT         : FAIL (first mismatch at vector %0d)", first_bad);
            $display("  If dwn_core_tb PASSED, the fault is in the thermometer encoder.");
        end
        $display("========================================");
        $display("");
        $finish;
    end

endmodule

`default_nettype wire
