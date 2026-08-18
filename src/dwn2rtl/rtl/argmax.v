// Argmax over per-class popcounts, as a balanced tree of depth ceil(log2(K)).
//
// ⚠️ TIE-BREAKING IS PART OF THE SPEC. Scores are small integers, so ties are common (~3% of
// vectors). numpy and torch both return the LOWEST tied index, so the golden model does, so
// this must. The rule enforcing it is the strict `>` at every merge: partners are compared
// low-index-first and the higher-index candidate wins only by BEATING the lower, so equal
// scores keep the lower index at every level and therefore overall. A `>=` anywhere would
// disagree with the golden model on those ~3% while looking correct on the rest.

`timescale 1ns / 1ps
`default_nettype none

module argmax #(
    parameter integer K = 5,            // classes
    parameter integer W = 4             // bits per score
)(
    input  wire [K*W-1:0]           scores_flat,   // class c occupies [c*W +: W]
    output wire [$clog2(K)-1:0]     index
);

    localparam integer IW     = (K <= 1) ? 1 : $clog2(K);
    localparam integer LEVELS = (K <= 1) ? 0 : $clog2(K);

    // One structure for every K -- no chain/tree switch at some class count, because nothing in
    // the hardware changes there. Odd entries CARRY FORWARD rather than pair against padding: a
    // carry is a rename, padding is a compare-select that exists only to be discarded.
    //
    // ⚠️ A MEASURED FALSE POSITIVE, not a silenced bug. UNOPTFLAT ("circular combinational
    // logic") is reported on these arrays and is FATAL by default, so a user linting an emitted
    // design would be told their hardware has a loop. It has none: lvl_*[l+1] reads only
    // lvl_*[l], so the level index strictly increases. The analysis is per-SIGNAL and one
    // signal holds every level, so the array appears to depend on itself -- proven with a
    // control, since a plain adder tree of the same shape raises the identical warning.
    // Restructuring buys nothing measurable (sim time is unchanged), so it is waived here.
    //
    // ⚠️ EDITORS: never begin a comment line with the word "verilator" -- it is read as a
    // PRAGMA and the prose after it is rejected, an ERROR that stops linting while iverilog
    // still prints PASS. tests/test_lint.py pins this.
    /* verilator lint_off UNOPTFLAT */
    wire [W-1:0]  lvl_score [0:LEVELS][0:K-1];
    wire [IW-1:0] lvl_index [0:LEVELS][0:K-1];
    /* verilator lint_on UNOPTFLAT */

    genvar l, i;
    generate
        for (i = 0; i < K; i = i + 1) begin : g_leaf
            assign lvl_score[0][i] = scores_flat[i*W +: W];
            assign lvl_index[0][i] = i[IW-1:0];
        end

        for (l = 0; l < LEVELS; l = l + 1) begin : g_level
            localparam integer N    = (K + (1 << l) - 1) >> l;   // live entries at this level
            localparam integer NEXT = (N + 1) >> 1;
            for (i = 0; i < NEXT; i = i + 1) begin : g_node
                if (2*i + 1 < N) begin : g_pair
                    // Left is the lower-index half; the right-hand candidate wins only by
                    // beating it outright, so equal scores keep the lower index at every
                    // level -- and therefore overall, which is the golden model's rule.
                    wire right_wins = lvl_score[l][2*i + 1] > lvl_score[l][2*i];
                    assign lvl_score[l+1][i] =
                        right_wins ? lvl_score[l][2*i + 1] : lvl_score[l][2*i];
                    assign lvl_index[l+1][i] =
                        right_wins ? lvl_index[l][2*i + 1] : lvl_index[l][2*i];
                end else begin : g_carry
                    assign lvl_score[l+1][i] = lvl_score[l][2*i];
                    assign lvl_index[l+1][i] = lvl_index[l][2*i];
                end
            end
        end
    endgenerate

    assign index = lvl_index[LEVELS][0];

endmodule

`default_nettype wire