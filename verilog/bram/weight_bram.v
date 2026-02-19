/*
 * Block RAM (BRAM) Interface for Weight Storage
 * 
 * Dual-port BRAM for storing quantized weights
 * - Port A: Write interface (for loading weights from ARM CPU)
 * - Port B: Read interface (for convolution engine)
 *
 * Author: URECA Project
 * Target: Xilinx ZCU104 FPGA
 */

module weight_bram #(
    parameter DATA_WIDTH = 8,
    parameter ADDR_WIDTH = 12,           // 4096 addresses
    parameter DEPTH = 4096               // 4KB of weights
) (
    input wire clk,
    
    // Port A: Write interface (from CPU/AXI)
    input wire [ADDR_WIDTH-1:0] addr_a,
    input wire [DATA_WIDTH-1:0] data_in_a,
    input wire we_a,                     // Write enable
    output reg [DATA_WIDTH-1:0] data_out_a,
    
    // Port B: Read interface (for compute engine)
    input wire [ADDR_WIDTH-1:0] addr_b,
    output reg [DATA_WIDTH-1:0] data_out_b
);

    // Memory array
    reg [DATA_WIDTH-1:0] ram [0:DEPTH-1];
    
    // Port A: Write/Read
    always @(posedge clk) begin
        if (we_a) begin
            ram[addr_a] <= data_in_a;
        end
        data_out_a <= ram[addr_a];
    end
    
    // Port B: Read only
    always @(posedge clk) begin
        data_out_b <= ram[addr_b];
    end
    
    // Initialize memory (optional, for simulation)
    integer i;
    initial begin
        for (i = 0; i < DEPTH; i = i + 1) begin
            ram[i] = 0;
        end
    end

endmodule


/*
 * Feature Map Buffer
 * 
 * Ping-pong buffer for input/output feature maps
 * Allows reading from one buffer while writing to another
 */

module feature_buffer #(
    parameter DATA_WIDTH = 8,
    parameter IMG_HEIGHT = 96,
    parameter IMG_WIDTH = 96,
    parameter CHANNELS = 16
) (
    input wire clk,
    input wire rst_n,
    
    // Write interface
    input wire wr_enable,
    input wire [15:0] wr_addr,
    input wire [DATA_WIDTH-1:0] wr_data,
    
    // Read interface
    input wire rd_enable,
    input wire [15:0] rd_addr,
    output reg [DATA_WIDTH-1:0] rd_data,
    
    // Buffer swap control
    input wire swap_buffers
);

    localparam BUFFER_SIZE = IMG_HEIGHT * IMG_WIDTH * CHANNELS;
    
    // Dual buffers
    reg [DATA_WIDTH-1:0] buffer0 [0:BUFFER_SIZE-1];
    reg [DATA_WIDTH-1:0] buffer1 [0:BUFFER_SIZE-1];
    
    // Active buffer selector
    reg active_buffer;  // 0 = buffer0, 1 = buffer1
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active_buffer <= 0;
        end else if (swap_buffers) begin
            active_buffer <= ~active_buffer;
        end
    end
    
    // Write logic
    always @(posedge clk) begin
        if (wr_enable) begin
            if (active_buffer == 0)
                buffer0[wr_addr] <= wr_data;
            else
                buffer1[wr_addr] <= wr_data;
        end
    end
    
    // Read logic
    always @(posedge clk) begin
        if (rd_enable) begin
            if (active_buffer == 0)
                rd_data <= buffer1[rd_addr];  // Read from inactive buffer
            else
                rd_data <= buffer0[rd_addr];
        end
    end

endmodule
