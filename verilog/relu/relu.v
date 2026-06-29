/*
 * ReLU Activation Module
 * 
 * Implements ReLU(x) = max(0, x) in hardware
 * - 8-bit signed input
 * - 8-bit unsigned output
 * - 1 cycle latency
 *
 * Author: URECA Project
 * Target: Xilinx ZCU104 FPGA
 */

module relu #(
    parameter DATA_WIDTH = 8
) (
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    // Input
    input wire signed [DATA_WIDTH-1:0] in_data,
    input wire in_valid,
    
    // Output
    output reg [DATA_WIDTH-1:0] out_data,
    output reg out_valid
);

    //===========================================
    // ReLU Logic
    //===========================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_data <= 0;
            out_valid <= 0;
        end else if (enable && in_valid) begin
            // ReLU: if input < 0, output = 0; else output = input
            if (in_data[DATA_WIDTH-1] == 1'b1) begin  // Negative (MSB = 1)
                out_data <= 0;
            end else begin
                out_data <= in_data[DATA_WIDTH-1:0];
            end
            
            out_valid <= 1'b1;
        end else begin
            out_valid <= 1'b0;
        end
    end

endmodule
