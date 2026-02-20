/*
 * 3x3 Convolution Engine  (single channel, valid convolution)
 *
 * Bug fix: window loads new data into win[r][2] each cycle, so the raw
 * mac_sum is 1 cycle early.  A single pipeline stage (fire_d1/mac_d1)
 * delays the output by one clock so win[1][1] is the true centre value.
 *
 * Weights: flat bus [DATA_WIDTH*9-1:0]
 *   weight[i] = weight_flat[DATA_WIDTH*i +: DATA_WIDTH]  (row-major,
 *   weight[0]=top-left, weight[4]=centre, weight[8]=bottom-right)
 *
 * Author: URECA Project
 * Target: Xilinx ZCU104
 */
`timescale 1ns / 1ps

module conv3x3_engine #(
    parameter DATA_WIDTH   = 8,
    parameter ACC_WIDTH    = 16,
    parameter IMG_HEIGHT   = 96,
    parameter IMG_WIDTH    = 96,
    parameter IN_CHANNELS  = 1,
    parameter OUT_CHANNELS = 1
) (
    input  wire clk,
    input  wire rst_n,
    input  wire start,

    input  wire signed [DATA_WIDTH-1:0] in_data,
    input  wire                         in_valid,
    output wire                         in_ready,

    input  wire [DATA_WIDTH*9-1:0]      weight_flat,

    output reg  signed [ACC_WIDTH-1:0]  out_data,
    output reg                          out_valid,
    input  wire                         out_ready,

    output reg                          done
);

    localparam TOTAL_OUT = (IMG_HEIGHT - 2) * (IMG_WIDTH - 2);

    // Unpack weights
    wire signed [DATA_WIDTH-1:0] w [0:8];
    genvar gi;
    generate
        for (gi = 0; gi < 9; gi = gi + 1) begin : UNPACK
            assign w[gi] = weight_flat[DATA_WIDTH*gi +: DATA_WIDTH];
        end
    endgenerate

    // Line buffers
    reg signed [DATA_WIDTH-1:0] lbuf [0:2][0:IMG_WIDTH-1];

    // 3x3 window: win[0]=top row, win[2]=bottom row
    reg signed [DATA_WIDTH-1:0] win [0:2][0:2];

    // Counters
    reg [$clog2(IMG_WIDTH)-1:0]              wr_col;
    reg [1:0]                                wr_buf;
    reg [$clog2(IMG_HEIGHT*IMG_WIDTH+1)-1:0] pix_in;
    reg [$clog2(TOTAL_OUT+1)-1:0]            out_cnt;
    reg                                      running;

    assign in_ready = running;

    // Combinational MAC (uses window values from PREVIOUS cycle via registers)
    wire signed [ACC_WIDTH-1:0] prod [0:8];
    generate
        for (gi = 0; gi < 9; gi = gi + 1) begin : MAC
            assign prod[gi] =
                {{(ACC_WIDTH-DATA_WIDTH){win[gi/3][gi%3][DATA_WIDTH-1]}}, win[gi/3][gi%3]}
              * {{(ACC_WIDTH-DATA_WIDTH){w[gi][DATA_WIDTH-1]          }}, w[gi]          };
        end
    endgenerate

    wire signed [ACC_WIDTH-1:0] mac_sum =
        prod[0]+prod[1]+prod[2]+prod[3]+prod[4]+prod[5]+prod[6]+prod[7]+prod[8];

    // 1-cycle pipeline stage to let window settle
    reg                        fire_d1;
    reg signed [ACC_WIDTH-1:0] mac_d1;

    integer i, j;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_col <= 0; wr_buf <= 0; pix_in <= 0; out_cnt <= 0;
            running <= 0; out_valid <= 0; out_data <= 0; done <= 0;
            fire_d1 <= 0; mac_d1 <= 0;
            for (i = 0; i < 3; i = i+1)
                for (j = 0; j < 3; j = j+1)
                    win[i][j] <= 0;
        end else begin
            // Clear single-cycle pulses
            out_valid <= 0;
            done      <= 0;
            fire_d1   <= 0;

            // Start
            if (start && !running) begin
                running <= 1; wr_col <= 0; wr_buf <= 0;
                pix_in  <= 0; out_cnt <= 0; fire_d1 <= 0; mac_d1 <= 0;
                for (i = 0; i < 3; i = i+1)
                    for (j = 0; j < 3; j = j+1)
                        win[i][j] <= 0;
            end

            // Accept pixel
            if (running && in_valid) begin
                // Write to active line buffer
                lbuf[wr_buf][wr_col] <= in_data;

                // Shift window left, load new right column
                win[0][0] <= win[0][1]; win[0][1] <= win[0][2];
                win[1][0] <= win[1][1]; win[1][1] <= win[1][2];
                win[2][0] <= win[2][1]; win[2][1] <= win[2][2];
                win[2][2] <= in_data;
                win[1][2] <= lbuf[(wr_buf + 2) % 3][wr_col]; // row above
                win[0][2] <= lbuf[(wr_buf + 1) % 3][wr_col]; // 2 rows above

                // Advance counters
                if (wr_col == IMG_WIDTH - 1) begin
                    wr_col <= 0;
                    wr_buf <= (wr_buf == 2) ? 2'd0 : wr_buf + 2'd1;
                end else begin
                    wr_col <= wr_col + 1;
                end
                pix_in <= pix_in + 1;

                // Fire stage 1: border check
                // The window right column just received this pixel.
                // After non-blocking assignments settle, win[1][2] holds
                // the correct centre—but we need win[1][1] (next cycle).
                // Stage it via fire_d1 / mac_d1.
                if ((pix_in >= (2 * IMG_WIDTH + 1)) &&
                    (wr_col >= 2) &&
                    (wr_col <= IMG_WIDTH - 1))
                begin
                    fire_d1 <= 1;
                    mac_d1  <= mac_sum;
                end
            end

            // Stage 2: output the delayed result
            if (fire_d1) begin
                out_data  <= mac_d1;
                out_valid <= 1;
                out_cnt   <= out_cnt + 1;
                if (out_cnt == TOTAL_OUT - 1) begin
                    done    <= 1;
                    running <= 0;
                end
            end

        end
    end

endmodule