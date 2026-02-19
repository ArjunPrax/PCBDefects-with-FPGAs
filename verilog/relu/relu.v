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


/*
 * ReLU Pipeline (Vectorized)
 * 
 * Process multiple ReLU operations in parallel
 * Useful for processing an entire output channel at once
 */

module relu_pipeline #(
    parameter DATA_WIDTH = 8,
    parameter VECTOR_SIZE = 16    // Number of parallel ReLU units
) (
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    // Input vector
    input wire signed [DATA_WIDTH-1:0] in_data [0:VECTOR_SIZE-1],
    input wire in_valid,
    
    // Output vector
    output reg [DATA_WIDTH-1:0] out_data [0:VECTOR_SIZE-1],
    output reg out_valid
);

    integer i;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < VECTOR_SIZE; i = i + 1) begin
                out_data[i] <= 0;
            end
            out_valid <= 0;
        end else if (enable && in_valid) begin
            for (i = 0; i < VECTOR_SIZE; i = i + 1) begin
                if (in_data[i][DATA_WIDTH-1] == 1'b1) begin
                    out_data[i] <= 0;
                end else begin
                    out_data[i] <= in_data[i];
                end
            end
            out_valid <= 1'b1;
        end else begin
            out_valid <= 1'b0;
        end
    end

endmodule
