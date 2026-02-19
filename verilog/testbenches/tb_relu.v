/*
 * Testbench for ReLU Module
 * 
 * Tests both single and vectorized ReLU implementations
 * 
 * Author: URECA Project
 */

`timescale 1ns / 1ps

module tb_relu;

    parameter DATA_WIDTH = 8;
    parameter CLK_PERIOD = 10;
    
    // Signals for single ReLU
    reg clk;
    reg rst_n;
    reg enable;
    reg signed [DATA_WIDTH-1:0] in_data;
    reg in_valid;
    wire [DATA_WIDTH-1:0] out_data;
    wire out_valid;
    
    // DUT
    relu #(
        .DATA_WIDTH(DATA_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .in_data(in_data),
        .in_valid(in_valid),
        .out_data(out_data),
        .out_valid(out_valid)
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // Test vectors
    reg signed [DATA_WIDTH-1:0] test_inputs [0:9];
    reg [DATA_WIDTH-1:0] expected_outputs [0:9];
    integer i;
    integer error_count;
    
    initial begin
        // Initialize
        rst_n = 0;
        enable = 0;
        in_data = 0;
        in_valid = 0;
        error_count = 0;
        
        // Test vectors
        test_inputs[0] = 8'sd0;     expected_outputs[0] = 8'd0;
        test_inputs[1] = 8'sd10;    expected_outputs[1] = 8'd10;
        test_inputs[2] = 8'sd127;   expected_outputs[2] = 8'd127;
        test_inputs[3] = -8'sd1;    expected_outputs[3] = 8'd0;
        test_inputs[4] = -8'sd10;   expected_outputs[4] = 8'd0;
        test_inputs[5] = -8'sd128;  expected_outputs[5] = 8'd0;
        test_inputs[6] = 8'sd50;    expected_outputs[6] = 8'd50;
        test_inputs[7] = -8'sd50;   expected_outputs[7] = 8'd0;
        test_inputs[8] = 8'sd1;     expected_outputs[8] = 8'd1;
        test_inputs[9] = -8'sd1;    expected_outputs[9] = 8'd0;
        
        // Reset
        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 2);
        
        $display("=" * 60);
        $display("ReLU Testbench");
        $display("=" * 60);
        
        enable = 1;
        
        // Test each input
        for (i = 0; i < 10; i = i + 1) begin
            in_data = test_inputs[i];
            in_valid = 1;
            #CLK_PERIOD;
            
            // Check output
            #(CLK_PERIOD);
            if (out_valid && out_data !== expected_outputs[i]) begin
                $display("ERROR: Input=%0d, Expected=%0d, Got=%0d", 
                         $signed(test_inputs[i]), expected_outputs[i], out_data);
                error_count = error_count + 1;
            end else if (out_valid) begin
                $display("PASS: Input=%0d, Output=%0d", 
                         $signed(test_inputs[i]), out_data);
            end
        end
        
        in_valid = 0;
        #(CLK_PERIOD * 5);
        
        // Summary
        $display("=" * 60);
        if (error_count == 0) begin
            $display("ALL TESTS PASSED!");
        end else begin
            $display("TESTS FAILED: %0d errors", error_count);
        end
        $display("=" * 60);
        
        $finish;
    end
    
    // Waveform dump
    initial begin
        $dumpfile("relu.vcd");
        $dumpvars(0, tb_relu);
    end

endmodule
