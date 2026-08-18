// GroupSum for one class: sum a contiguous slice of the last layer's outputs.
// The software divides by tau, but it is the same for every class, so it cannot change the winner.

`timescale 1ns / 1ps
`default_nettype none

module popcount #(
    parameter integer WIDTH = 10
)(
    input  wire [WIDTH-1:0]              bits,
    output reg  [$clog2(WIDTH+1)-1:0]    count
);

    // Reads like a chain, but addition is associative and synthesis rebalances it: measured depth
    // tracks log2(WIDTH), 8 at width 100 where a mux chain hits 39. No hand-written tree needed.
    integer i;

    always @* begin
        count = 0;
        for (i = 0; i < WIDTH; i = i + 1)
            count = count + bits[i];
    end

endmodule

`default_nettype wire
