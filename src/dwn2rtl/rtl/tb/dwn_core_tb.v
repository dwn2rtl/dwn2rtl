// GATE 1 -- the one non-negotiable rule.
//
// The LUT core against the golden model, on pre-binarized inputs. The encoder is checked
// separately by dwn_top_tb, so a failure points at one of the two rather than both.
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
    localparam integer LATENCY = `DWN_CORE_LATENCY;

    reg [VEC_W-1:0] vectors  [0:N_VEC-1];
    reg [7:0]       expected [0:N_VEC-1];

    reg              clk = 1'b0;
    reg  [VEC_W-1:0] x;
    wire [`IDX_W-1:0]       class_idx;

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
            if (i >= LATENCY) begin
                j = i - LATENCY;
                // !== not != : an x or z must count as a failure rather than propagating
                // silently into a comparison that returns x.
                if (class_idx !== expected[j][`IDX_W-1:0]) begin
                    if (first_bad == -1) first_bad = j;
                    errors = errors + 1;
                    if (errors <= 10)
                        $display("  MISMATCH vector %0d: rtl=%0d golden=%0d",
                                 j, class_idx, expected[j][`IDX_W-1:0]);
                end
            end
            x = (i < N_VEC) ? vectors[i] : {VEC_W{1'b0}};
        end

        $display("");
        $display("========================================");
        $display("GATE 1 -- dwn_core vs golden model");
        $display("  vectors tested : %0d", N_VEC);
        $display("  latency        : %0d cycles, II=1 (new vector every clock)", LATENCY);
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