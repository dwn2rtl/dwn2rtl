// Popcount for one class group. This is GroupSum from the software model: the last layer's
// outputs are cut into `num_classes` contiguous slices and each slice is summed
// (dwn2rtl docs/checkpoint-format.md §4).
//
// The software then divides by `tau`, but tau is the same constant for every class, so it can't
// change which one wins. The hardware skips it -- popcount and compare is the whole output stage.

`timescale 1ns / 1ps
`default_nettype none

module popcount #(
    parameter integer WIDTH = 10
)(
    input  wire [WIDTH-1:0]              bits,
    output reg  [$clog2(WIDTH+1)-1:0]    count
);

    // A loop, which reads like a chain of WIDTH-1 dependent adds -- but addition is associative,
    // so synthesis rebalances it into a tree. Worth checking rather than trusting: measured depth
    // tracks log2(WIDTH) (3 at width 10, 6 at 30, 8 at 100, 12 at 600), while a mux chain of the
    // same width -- not associative, so nothing can rebalance it -- hits 39 at width 100. Writing
    // the tree out by hand would add recursion for nothing. (docs/phase4-ledger.md)
    integer i;

    always @* begin
        count = 0;
        for (i = 0; i < WIDTH; i = i + 1)
            count = count + bits[i];
    end

endmodule

`default_nettype wire