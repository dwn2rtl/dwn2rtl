// Checks the LUT network against the software model, on already-binarized inputs.
//
// The encoder is checked separately by dwn_top_tb, so if one of them fails you know which
// half to look at.
//
// Vectors stream one per cycle and are checked LATENCY cycles later, which is what proves II=1.
//
// ⚠️ LATENCY comes from dwn_core_params.vh, never hardcoded -- otherwise a depth change quietly
// becomes an off-by-one against the wrong vector.

`timescale 1ns / 1ps
`default_nettype none

`include "vec_params.vh"
`include "dwn_core_params.vh"

module dwn_core_tb;

    localparam integer N_VEC   = `N_VEC;
    localparam integer VEC_W   = `VEC_W;
    localparam integer IDX_W   = `IDX_W;
    localparam integer LATENCY = `DWN_CORE_LATENCY;

    // ⚠️ EXP_W IS NOT IDX_W, and both halves of that matter.
    //
    // Too narrow and the gate breaks on correct hardware: this was `reg [7:0]`, so at more than
    // 256 classes IDX_W reached 9, `expected[j][IDX_W-1:0]` read past the end of the reg and
    // returned x, and !== failed EVERY vector on a design that was bit-exact. K=256 passed,
    // K=257 failed 100% (phase8-ledger.md §1).
    //
    // Equal to IDX_W and the comparison stops being able to fail: phase6-ledger.md §26 found
    // that slicing BOTH sides to IDX_W means a wrong IDX_W narrows the check rather than
    // breaking it. So `expected` is deliberately WIDER, and class_idx is zero-extended up to
    // it -- a golden answer that does not fit the index width now mismatches instead of being
    // truncated into agreement.
    localparam integer EXP_W = 32;

    reg [VEC_W-1:0] vectors  [0:N_VEC-1];
    reg [EXP_W-1:0] expected [0:N_VEC-1];

    reg              clk = 1'b0;
    reg  [VEC_W-1:0] x;
    wire [IDX_W-1:0] class_idx;

    always #5 clk = ~clk;          // 100 MHz, the Basys 3 board clock

    dwn_core dut (.clk(clk), .x(x), .class_idx(class_idx));

    integer i, j;
    integer errors;
    integer first_bad;

    initial begin
        $readmemh("x_binarized.hex", vectors);
        $readmemh("expected.hex",    expected);

        errors    = 0;
        first_bad = -1;
        x         = {VEC_W{1'b0}};

        // Drive on the negative edge so inputs are stable across the capturing posedge; read
        // outputs on the same negedge, by which point the pipeline has settled.
        for (i = 0; i < N_VEC + LATENCY; i = i + 1) begin
            @(negedge clk);
            // ⚠️ DRIVE FIRST, THEN COMPARE. This used to compare and then drive, which is
            // correct only when LATENCY >= 1: at zero latency the design is combinational, so
            // vector i's answer depends on an input that had not been applied yet and every
            // comparison was one step early. `#1` lets combinational logic settle before the
            // read; it is well inside the half period (the clock toggles every 5).
            x = (i < N_VEC) ? vectors[i] : {VEC_W{1'b0}};
            #1;
            if (i >= LATENCY) begin
                j = i - LATENCY;
                // !== not != : an x or z must count as a failure rather than propagating
                // silently into a comparison that returns x.
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
        $display("dwn_core vs the software model");
        $display("  vectors tested : %0d", N_VEC);
        $display("  latency        : %0d cycles, one result per clock (II=1)", LATENCY);
        $display("  mismatches     : %0d", errors);
        if (errors == 0) begin
            $display("  RESULT         : PASS (bit-exact on every vector)");
        end else begin
            $display("  RESULT         : FAIL (first mismatch at vector %0d)", first_bad);
        end
        $display("========================================");
        $display("");
        $finish;
    end

endmodule

`default_nettype wire