// Popcount for one class group.
//
// This is GroupSum from the software model (docs/reference/checkpoint-format.md §4): the final layer's
// outputs are split into `num_classes` CONTIGUOUS, in-order slices and each slice is summed.
//
// The software divides each sum by `tau`, but tau is a single uniform constant applied to
// every class, so it cannot change which class wins. The hardware therefore never needs it --
// popcount and compare is the whole output stage.
//
// Note the paper replaces this with a "Learnable Reduction" pyramid for tiny models, on the
// grounds that the popcount circuit can be as large as the network itself. That is a Phase 2
// sweep axis (docs/jsc/dse-plan.md, Group A); this is the plain-popcount baseline it gets
// measured against.

`timescale 1ns / 1ps
`default_nettype none

module popcount #(
    parameter integer WIDTH = 10
)(
    input  wire [WIDTH-1:0]              bits,
    output reg  [$clog2(WIDTH+1)-1:0]    count
);

    integer i;

    always @* begin
        count = 0;
        for (i = 0; i < WIDTH; i = i + 1)
            count = count + bits[i];
    end

endmodule

`default_nettype wire