// Popcount for one class group -- GroupSum from the software model
// (dwn2rtl docs/checkpoint-format.md §4): the final layer's outputs are split into
// `num_classes` contiguous, in-order slices, and each slice is summed.
//
// The software divides each sum by `tau`, but tau is one uniform constant applied to every
// class, so it cannot change which class wins. The hardware never needs it -- popcount and
// compare is the whole output stage.

`timescale 1ns / 1ps
`default_nettype none

module popcount #(
    parameter integer WIDTH = 10
)(
    input  wire [WIDTH-1:0]              bits,
    output reg  [$clog2(WIDTH+1)-1:0]    count
);

    // WRITTEN AS A LOOP, AND THAT IS SAFE HERE -- measured, not assumed. A loop describes a
    // chain of WIDTH-1 dependent adds; addition is associative, so a tool MAY rebalance it, but
    // "may" is not "does". (For argmax it genuinely was a defect: selection is not associative,
    // so no tool can, and the chain cost 20 MHz.)
    //
    // Logic levels after mapping, with a non-associative mux chain of the same width as a
    // control (docs/phase4-ledger.md):
    //
    //     width      popcount      mux chain (control)      a linear chain would be
    //        10             3                        3                            9
    //        30             6                       11                           29
    //       100             8                       39                           99
    //       600            12                        -                          599
    //
    // popcount tracks log2(WIDTH); the control does not, which proves the metric can see a deep
    // path when one exists. An explicit adder tree would cost recursion for no measured gain.
    integer i;

    always @* begin
        count = 0;
        for (i = 0; i < WIDTH; i = i + 1)
            count = count + bits[i];
    end

endmodule

`default_nettype wire