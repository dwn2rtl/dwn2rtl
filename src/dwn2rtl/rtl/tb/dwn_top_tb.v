// GATE 1, TOP LEVEL -- quantized features in, class index out.
//
// The companion to dwn_core_tb. Together they are the one non-negotiable rule:
// emitted RTL is not correct until a simulator says it matches the golden model on every
// vector. An emitter's own read-back is not that check -- we have a recorded case where one
// reported 20/20 correct while the design was wrong on 958 of 1,504 vectors.
//
// WHY TWO TESTBENCHES RATHER THAN ONE. dwn_core_tb drives PRE-BINARIZED bits, so it tests the
// LUT network alone. This one drives quantized FEATURES through the thermometer encoder and
// then the network. The split is what makes a failure localize itself:
//
//     core PASS, top FAIL   -> the encoder. Nothing else needs re-examining.
//     core FAIL, top FAIL   -> the network; fix that first, this will follow.
//
// A single top-level testbench would have told you only that something, somewhere, was wrong --
// and the encoder and the core are emitted by different files from different parts of the
// checkpoint.
//
// THE ENCODER IS WHAT THIS ADDS, and it is the part published DWN resource counts leave out.
// It is not preprocessing a user supplies; it is intrinsic to a DWN, and on the smallest
// studied model it is fourteen times the network it feeds. An unverified encoder is most of an
// unverified design.
//
// The design is PIPELINED, so this drives a new vector every cycle and checks the result
// LATENCY cycles later. That is not bookkeeping: streaming back-to-back vectors and getting
// every one right is what proves II=1. A design that computed correctly but stalled would pass
// a one-vector-at-a-time testbench and fail this one.
//
// LATENCY comes from dwn_top_params.vh, written by the same build that emitted the pipeline.
// Hardcoding it here is how a depth change silently becomes an off-by-one comparison against
// the wrong vector. IDX_W likewise: an earlier testbench of ours hardcoded 3, which left the
// upper bits undriven below five classes and TRUNCATED the comparison above eight -- a 10-class
// design was checked on three of its four index bits, and passed.

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
