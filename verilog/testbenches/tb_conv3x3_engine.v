/*
 * Testbench for conv3x3_engine — Icarus Verilog compatible
 *
 * Run (from your project root):
 *   cd verilog/testbenches
 *   iverilog -g2012 -o conv_sim ../conv/conv3x3_engine.v tb_conv3x3_engine.v
 *   vvp conv_sim
 *
 * Author: URECA Project
 */
`timescale 1ns / 1ps

module tb_conv3x3_engine;

    parameter DATA_WIDTH = 8;
    parameter ACC_WIDTH  = 16;
    parameter IMG_H      = 6;
    parameter IMG_W      = 6;
    parameter CLK_PERIOD = 10;
    parameter TOTAL_OUT  = (IMG_H-2)*(IMG_W-2);   // 16

    reg                          clk, rst_n, start;
    reg  signed [DATA_WIDTH-1:0] in_data;
    reg                          in_valid;
    wire                         in_ready;
    reg  [DATA_WIDTH*9-1:0]      weight_flat;
    wire signed [ACC_WIDTH-1:0]  out_data;
    wire                         out_valid;
    reg                          out_ready;
    wire                         done;

    conv3x3_engine #(
        .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .IMG_HEIGHT(IMG_H),      .IMG_WIDTH(IMG_W)
    ) dut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .in_data(in_data), .in_valid(in_valid), .in_ready(in_ready),
        .weight_flat(weight_flat),
        .out_data(out_data), .out_valid(out_valid), .out_ready(out_ready),
        .done(done)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // Test image: pixel[r][c] = r*IMG_W + c + 1  (1-indexed)
    //   Row 0:  1  2  3  4  5  6
    //   Row 1:  7  8  9 10 11 12
    //   Row 2: 13 14 15 16 17 18
    //   Row 3: 19 20 21 22 23 24
    //   Row 4: 25 26 27 28 29 30
    //   Row 5: 31 32 33 34 35 36
    reg signed [DATA_WIDTH-1:0] image [0:IMG_H*IMG_W-1];
    integer r, c;
    initial begin
        for (r = 0; r < IMG_H; r = r+1)
            for (c = 0; c < IMG_W; c = c+1)
                image[r*IMG_W+c] = r*IMG_W + c + 1;
    end

    // Output capture
    reg signed [ACC_WIDTH-1:0] results [0:TOTAL_OUT-1];
    integer cap_idx;
    initial cap_idx = 0;

    always @(posedge clk) begin
        out_ready <= 1;
        if (out_valid) begin
            if (cap_idx < TOTAL_OUT)
                results[cap_idx] <= out_data;
            cap_idx <= cap_idx + 1;
        end
    end

    // Reference: compute expected output using current weight_flat
    function signed [ACC_WIDTH-1:0] expected_out;
        input integer out_row, out_col;
        integer dr, dc, wi;
        reg signed [DATA_WIDTH-1:0] px, wv;
        reg signed [ACC_WIDTH-1:0]  acc;
        begin
            acc = 0;
            for (dr = 0; dr < 3; dr = dr+1)
                for (dc = 0; dc < 3; dc = dc+1) begin
                    wi  = dr*3 + dc;
                    px  = image[(out_row+dr)*IMG_W + (out_col+dc)];
                    wv  = weight_flat[DATA_WIDTH*wi +: DATA_WIDTH];
                    acc = acc + $signed(px) * $signed(wv);
                end
            expected_out = acc;
        end
    endfunction

    // Pack weights into flat bus
    task set_weights;
        input signed [DATA_WIDTH-1:0] w0,w1,w2,w3,w4,w5,w6,w7,w8;
        begin
            weight_flat[DATA_WIDTH*0 +: DATA_WIDTH] = w0;
            weight_flat[DATA_WIDTH*1 +: DATA_WIDTH] = w1;
            weight_flat[DATA_WIDTH*2 +: DATA_WIDTH] = w2;
            weight_flat[DATA_WIDTH*3 +: DATA_WIDTH] = w3;
            weight_flat[DATA_WIDTH*4 +: DATA_WIDTH] = w4;
            weight_flat[DATA_WIDTH*5 +: DATA_WIDTH] = w5;
            weight_flat[DATA_WIDTH*6 +: DATA_WIDTH] = w6;
            weight_flat[DATA_WIDTH*7 +: DATA_WIDTH] = w7;
            weight_flat[DATA_WIDTH*8 +: DATA_WIDTH] = w8;
        end
    endtask

    // Reset and clear cap_idx
    task do_reset;
        begin
            rst_n = 0; start = 0; in_valid = 0; in_data = 0;
            repeat(4) @(posedge clk);
            rst_n = 1;
            @(posedge clk);
            cap_idx = 0;
        end
    endtask

    // Stream all pixels and wait for done
    integer pix_i, timeout_cnt;
    task stream_and_wait;
        begin
            @(negedge clk); start = 1;
            @(negedge clk); start = 0;

            for (pix_i = 0; pix_i < IMG_H*IMG_W; pix_i = pix_i+1) begin
                while (!in_ready) @(posedge clk);
                @(negedge clk);
                in_data  = image[pix_i];
                in_valid = 1;
                @(posedge clk);
            end
            @(negedge clk); in_valid = 0;

            // Poll done — 1-cycle pulse, can't use @(posedge done)
            timeout_cnt = 0;
            while (!done && timeout_cnt < 5000) begin
                @(posedge clk); timeout_cnt = timeout_cnt + 1;
            end
            repeat(3) @(posedge clk); // drain pipeline register

            if (timeout_cnt >= 5000)
                $display("  ERROR: timed out waiting for done");
        end
    endtask

    // Check all outputs against reference
    integer idx, err_cnt;
    reg signed [ACC_WIDTH-1:0] exp_val;
    integer total_errors;

    task check_results;
        input [127:0] label;
        input signed [ACC_WIDTH-1:0] exp0;
        begin
            err_cnt = 0;
            for (idx = 0; idx < TOTAL_OUT; idx = idx+1) begin
                exp_val = expected_out(idx/(IMG_W-2), idx%(IMG_W-2));
                if (results[idx] !== exp_val) begin
                    $display("  FAIL [%0d]: got %0d, expected %0d",
                              idx, results[idx], exp_val);
                    err_cnt = err_cnt + 1;
                end
            end
            if (err_cnt == 0)
                $display("  PASS %-12s : %0d outputs correct  (out[0]=%0d expect %0d)",
                          label, TOTAL_OUT, results[0], exp0);
            else
                $display("  FAIL %-12s : %0d/%0d errors",
                          label, err_cnt, TOTAL_OUT);
            total_errors = total_errors + err_cnt;
        end
    endtask

    initial begin
        total_errors = 0;
        out_ready    = 1;

        $display("============================================================");
        $display("conv3x3_engine Testbench");
        $display("Image %0dx%0d  ->  Output %0dx%0d  (%0d pixels)",
                  IMG_H, IMG_W, IMG_H-2, IMG_W-2, TOTAL_OUT);
        $display("============================================================");

        // TEST 1: Identity  — out[r][c] = image[r+1][c+1], out[0]=8
        $display("\nTEST 1: Identity  [0 0 0 / 0 1 0 / 0 0 0]");
        set_weights(0,0,0, 0,1,0, 0,0,0);
        do_reset; stream_and_wait;
        check_results("Identity", 8);

        // TEST 2: All-ones  — sum of 3x3 neighbourhood, out[0]=72
        $display("\nTEST 2: All-ones  [1 1 1 / 1 1 1 / 1 1 1]");
        set_weights(1,1,1, 1,1,1, 1,1,1);
        do_reset; stream_and_wait;
        check_results("All-ones", 72);

        // TEST 3: Edge  — 8*centre - sum_of_8; linear image gives 0
        $display("\nTEST 3: Edge  [-1 -1 -1 / -1 8 -1 / -1 -1 -1]");
        set_weights(-1,-1,-1, -1,8,-1, -1,-1,-1);
        do_reset; stream_and_wait;
        check_results("Edge", 0);

        $display("\n============================================================");
        if (total_errors == 0) $display("ALL TESTS PASSED");
        else                   $display("FAILED — %0d total errors", total_errors);
        $display("============================================================");
        $finish;
    end

    initial begin #(CLK_PERIOD*200000); $display("GLOBAL TIMEOUT"); $finish; end

    initial begin
        $dumpfile("conv3x3_engine.vcd");
        $dumpvars(0, tb_conv3x3_engine);
    end

endmodule