// The whole design: quantized features in, class index out.
//
// This drives the encoder AND the network, where dwn_core_tb drives the network alone. Read
// the pair together:
//
//     core PASS, top FAIL   -> the encoder. Nothing else needs re-examining.
//     core FAIL, top FAIL   -> the network; fix that first, this follows.
//
// Vectors stream one per cycle and are checked LATENCY cycles later, which is what proves II=1.
//
// ⚠️ LATENCY and IDX_W come from dwn_top_params.vh, never hardcoded. One earlier testbench fixed
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
    // ⚠️ EXP_W IS NOT IDX_W -- see the same block in dwn_core_tb.v. Too narrow and a design
    // with more than 256 classes fails its own gate while being bit-exact; equal to IDX_W and
    // a wrong IDX_W narrows the comparison instead of breaking it. Wider, with class_idx
    // zero-extended to meet it, is the only width that fails in both directions.
    localparam integer EXP_W = 32;

    reg [X_W-1:0] vectors  [0:N_TOP-1];
    reg [EXP_W-1:0] expected [0:N_TOP-1];

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
            // ⚠️ DRIVE FIRST, THEN COMPARE. This used to compare and then drive, which is
            // correct only when LATENCY >= 1: at zero latency the design is combinational, so
            // vector i's answer depends on an input that had not been applied yet and every
            // comparison was one step early. `#1` lets combinational logic settle before the
            // read; it is well inside the half period (the clock toggles every 5).
            x_flat = (i < N_TOP) ? vectors[i] : {X_W{1'b0}};
            #1;
            if (i >= LATENCY) begin
                j = i - LATENCY;
                // !== not != : an x or z must count as a failure rather than propagating
                // silently into a comparison that returns x. Undriven encoder bits are a real
                // possibility here in a way they are not at the core level.
                if ({{(EXP_W-IDX_W){1'b0}}, class_idx} !== expected[j]) begin
                    if (first_bad == -1) first_bad = j;
                    errors = errors + 1;
                    if (errors <= 10)
                        $display("  MISMATCH vector %0d: rtl=%0d expected=%0d",
                                 j, class_idx, expected[j]);
                end
            end
        end

        $display("");
        $display("========================================");
        $display("dwn_top (encoder + network) vs the software model");
        $display("  vectors tested : %0d", N_TOP);
        $display("  feature word   : %0d bits", X_W);
        $display("  latency        : %0d cycles, one result per clock (II=1)", LATENCY);
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
