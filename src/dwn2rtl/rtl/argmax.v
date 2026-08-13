// Argmax over per-class popcounts.
//
// TIE-BREAKING IS PART OF THE SPEC, NOT AN IMPLEMENTATION DETAIL.
//
// With 10 nodes per class there are only 11 possible scores, so ties are common rather than
// exotic: 29 of the 1000 Gate 1 test vectors (2.9%) have two or more classes sharing the top
// score. numpy's argmax and torch's argmax both return the LOWEST index in that case, so the
// golden model does too, so this must as well.
//
// The rule that enforces it is a strict `>` at every merge: a candidate only displaces the
// incumbent by BEATING it. Using `>=` anywhere would keep the highest tied index instead, and
// the design would disagree with the golden model on ~3% of vectors while looking correct on
// the other 97% -- the kind of failure Gate 1 exists to catch and spot-checking never does.
//
// SHAPE: a balanced tree, not a linear scan.
//
// This was a sequential `for` loop over classes, where each iteration read the previous
// iteration's `best`. That is a chain of K-1 dependent compare-and-selects: fine at K=5 (JSC,
// 4 deep) and the critical path at K=10 (MNIST, 9 deep), where it synthesized to 17 logic
// levels and held dwn_top to 87.5 MHz against a 100 MHz board.
//
// Pairwise reduction makes the depth ceil(log2(K)) instead of K-1, and preserves the tie-break
// exactly: partners are merged low-index-first, and the right-hand (higher-index) candidate
// only wins on a strict `>`, so equal scores always keep the lower index -- at every level,
// and therefore overall.

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

    // ONE structure for every K. There was briefly a `K <= 5 ? chain : tree` switch, to hold
    // JSC's published dwn_core at 108 LUTs -- the tree costs +2 there. That branch encoded a
    // fact about this project's git history, not about the target: nothing in the FPGA changes
    // at five classes. Branches that encode history do not compose, because the next dataset
    // brings a K with no principled place to go. The published figures are pinned by the
    // `jsc-complete` tag instead, which is what tags are for.
    //
    // Depth is ceil(log2(K)) rather than the K-1 of a sequential scan. Odd entries CARRY
    // FORWARD instead of pairing against padding: a carry is a rename, padding is a
    // compare-select that exists only to be discarded.
    wire [W-1:0]  lvl_score [0:LEVELS][0:K-1];
    wire [IW-1:0] lvl_index [0:LEVELS][0:K-1];

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