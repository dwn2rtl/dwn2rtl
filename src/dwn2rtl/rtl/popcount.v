// Popcount for one class group.
//
// This is GroupSum from the software model (dwn2rtl docs/checkpoint-format.md §4): the final layer's
// outputs are split into `num_classes` CONTIGUOUS, in-order slices and each slice is summed.
//
// The software divides each sum by `tau`, but tau is a single uniform constant applied to
// every class, so it cannot change which class wins. The hardware therefore never needs it --
// popcount and compare is the whole output stage.
//
// Note the paper replaces this with a "Learnable Reduction" pyramid for tiny models, on the
// grounds that the popcount circuit can be as large as the network itself. dwn2rtl emits the
// plain popcount only: across 35 trained configurations we found that a
// learned taper genuinely works and is dominated anyway -- a plain 500-node layer beat the
// best taper while using the identical adder tree that taper spent 2,800 nodes to reach.

`timescale 1ns / 1ps
`default_nettype none

module popcount #(
    parameter integer WIDTH = 10
)(
    input  wire [WIDTH-1:0]              bits,
    output reg  [$clog2(WIDTH+1)-1:0]    count
);

    // WRITTEN AS A LOOP, AND THAT IS SAFE HERE -- measured, not assumed.
    //
    // A sequential loop like this describes a linear chain of WIDTH-1 dependent operations. For
    // argmax that was a real defect: selection is data-dependent and NOT associative, so no tool
    // can rebalance it, and the chain cost 20 MHz until it was rewritten as an explicit tree.
    // Addition IS associative, so a synthesis tool is free to rebalance -- but "free to" is not
    // "does", and dwn2rtl's rule is that a generated reduction must not be fast by luck.
    //
    // So it was measured (docs/phase4-ledger.md). Logic levels after mapping, against a
    // deliberately non-associative mux chain of the same width as a control:
    //
    //     width      popcount      mux chain (control)      a linear chain would be
    //        10             3                        3                            9
    //        30             6                       11                           29
    //       100             8                       39                           99
    //       600            12                        -                          599
    //
    // popcount tracks log2(WIDTH); the control does not. The rebalancing is real, and the
    // control proves the metric can see a deep path when one exists. An explicit adder tree
    // here would add recursive module instantiation for no measured gain.
    integer i;

    always @* begin
        count = 0;
        for (i = 0; i < WIDTH; i = i + 1)
            count = count + bits[i];
    end

endmodule

`default_nettype wire