// Argmax over the per-class popcounts, as a balanced tree ceil(log2(K)) deep.
//
// ⚠️ Ties are part of the spec, not an edge case -- scores are small integers, so roughly 3% of
// vectors have one. numpy and torch both return the lowest tied index, so the golden model does
// and so must this. The strict `>` at every merge is what enforces it: partners compare
// low-index-first, and the higher index only wins by beating the lower, so a tie keeps the lower
// index at every level and therefore overall. A `>=` anywhere would quietly disagree on that 3%.

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

    // One shape for every K -- no chain-vs-tree switch at some class count, since nothing in the
    // hardware changes there. Odd entries carry forward rather than pair against padding.
    //
    // The waiver below is a false positive: lvl_*[l+1] only ever reads lvl_*[l], so there is no
    // loop. The check works per-signal and one signal holds every level, so the array looks
    // self-referential -- a plain adder tree of the same shape warns identically. Restructuring
    // measured no faster, so it stays as it is.
    //
    // ⚠️ Don't start a comment line with the word "verilator": it's read as a pragma, and the
    // sentence after it becomes an error. tests/test_lint.py pins that.
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
                    // Left is the lower index, and right only wins by beating it -- so a tie
                    // keeps the lower index.
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