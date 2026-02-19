/*
 * AXI4-Lite Wrapper for Convolution + ReLU Pipeline
 * 
 * Integrates the conv3x3_engine and ReLU modules with AXI interface
 * for communication with ARM CPU on ZCU104
 * 
 * AXI Register Map:
 * 0x00: Control Register (bit 0: start, bit 1: reset)
 * 0x04: Status Register (bit 0: done, bit 1: busy)
 * 0x08: Input channels
 * 0x0C: Output channels
 * 0x10: Image width
 * 0x14: Image height
 * 
 * Author: URECA Project
 * Target: Xilinx ZCU104 FPGA
 */

module axi_conv_wrapper #(
    parameter C_S_AXI_DATA_WIDTH = 32,
    parameter C_S_AXI_ADDR_WIDTH = 8,
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH = 16
) (
    // AXI4-Lite Slave Interface
    input wire S_AXI_ACLK,
    input wire S_AXI_ARESETN,
    
    // Write address channel
    input wire [C_S_AXI_ADDR_WIDTH-1:0] S_AXI_AWADDR,
    input wire S_AXI_AWVALID,
    output reg S_AXI_AWREADY,
    
    // Write data channel
    input wire [C_S_AXI_DATA_WIDTH-1:0] S_AXI_WDATA,
    input wire [3:0] S_AXI_WSTRB,
    input wire S_AXI_WVALID,
    output reg S_AXI_WREADY,
    
    // Write response channel
    output reg [1:0] S_AXI_BRESP,
    output reg S_AXI_BVALID,
    input wire S_AXI_BREADY,
    
    // Read address channel
    input wire [C_S_AXI_ADDR_WIDTH-1:0] S_AXI_ARADDR,
    input wire S_AXI_ARVALID,
    output reg S_AXI_ARREADY,
    
    // Read data channel
    output reg [C_S_AXI_DATA_WIDTH-1:0] S_AXI_RDATA,
    output reg [1:0] S_AXI_RRESP,
    output reg S_AXI_RVALID,
    input wire S_AXI_RREADY,
    
    // Streaming input interface (AXI-Stream)
    input wire [DATA_WIDTH-1:0] S_AXIS_TDATA,
    input wire S_AXIS_TVALID,
    output wire S_AXIS_TREADY,
    
    // Streaming output interface (AXI-Stream)
    output wire [DATA_WIDTH-1:0] M_AXIS_TDATA,
    output wire M_AXIS_TVALID,
    input wire M_AXIS_TREADY,
    
    // Interrupt
    output wire interrupt
);

    //======================================
    // Internal Signals
    //======================================
    wire clk = S_AXI_ACLK;
    wire rst_n = S_AXI_ARESETN;
    
    // Control registers
    reg [31:0] ctrl_reg;        // 0x00
    reg [31:0] status_reg;      // 0x04
    reg [31:0] in_ch_reg;       // 0x08
    reg [31:0] out_ch_reg;      // 0x0C
    reg [31:0] img_w_reg;       // 0x10
    reg [31:0] img_h_reg;       // 0x14
    
    wire start = ctrl_reg[0];
    wire soft_reset = ctrl_reg[1];
    
    // Conv engine signals
    wire [DATA_WIDTH-1:0] conv_out_data;
    wire conv_out_valid;
    wire conv_done;
    reg conv_out_ready;
    
    // ReLU signals
    wire [DATA_WIDTH-1:0] relu_out_data;
    wire relu_out_valid;
    
    // Weight BRAM signals
    wire [9:0] weight_addr;
    reg [DATA_WIDTH-1:0] weight_data;
    
    //======================================
    // Conv3x3 Engine Instantiation
    //======================================
    conv3x3_engine #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .IN_CHANNELS(16),
        .OUT_CHANNELS(32),
        .IMG_HEIGHT(96),
        .IMG_WIDTH(96)
    ) conv_inst (
        .clk(clk),
        .rst_n(rst_n & ~soft_reset),
        .start(start),
        .in_data(S_AXIS_TDATA),
        .in_valid(S_AXIS_TVALID),
        .in_ready(S_AXIS_TREADY),
        .weight_addr(weight_addr),
        .weight_data(weight_data),
        .out_data(conv_out_data),
        .out_valid(conv_out_valid),
        .out_ready(conv_out_ready),
        .done(conv_done)
    );
    
    //======================================
    // ReLU Instantiation
    //======================================
    relu #(
        .DATA_WIDTH(DATA_WIDTH)
    ) relu_inst (
        .clk(clk),
        .rst_n(rst_n & ~soft_reset),
        .enable(1'b1),
        .in_data(conv_out_data),
        .in_valid(conv_out_valid),
        .out_data(relu_out_data),
        .out_valid(relu_out_valid)
    );
    
    // Output assignment
    assign M_AXIS_TDATA = relu_out_data;
    assign M_AXIS_TVALID = relu_out_valid;
    
    always @(*) begin
        conv_out_ready = M_AXIS_TREADY;
    end
    
    //======================================
    // Weight BRAM (simplified - connect to actual BRAM)
    //======================================
    reg [DATA_WIDTH-1:0] weight_mem [0:1023];
    
    always @(posedge clk) begin
        weight_data <= weight_mem[weight_addr % 1024];
    end
    
    //======================================
    // AXI4-Lite Write Logic
    //======================================
    reg [C_S_AXI_ADDR_WIDTH-1:0] write_addr;
    
    // Write address handshake
    always @(posedge clk) begin
        if (!rst_n) begin
            S_AXI_AWREADY <= 1'b0;
            write_addr <= 0;
        end else begin
            if (S_AXI_AWVALID && !S_AXI_AWREADY) begin
                S_AXI_AWREADY <= 1'b1;
                write_addr <= S_AXI_AWADDR;
            end else begin
                S_AXI_AWREADY <= 1'b0;
            end
        end
    end
    
    // Write data handshake
    always @(posedge clk) begin
        if (!rst_n) begin
            S_AXI_WREADY <= 1'b0;
            ctrl_reg <= 0;
            in_ch_reg <= 16;
            out_ch_reg <= 32;
            img_w_reg <= 96;
            img_h_reg <= 96;
        end else begin
            if (S_AXI_WVALID && !S_AXI_WREADY) begin
                S_AXI_WREADY <= 1'b1;
                
                // Write to registers based on address
                case (write_addr)
                    8'h00: ctrl_reg <= S_AXI_WDATA;
                    8'h08: in_ch_reg <= S_AXI_WDATA;
                    8'h0C: out_ch_reg <= S_AXI_WDATA;
                    8'h10: img_w_reg <= S_AXI_WDATA;
                    8'h14: img_h_reg <= S_AXI_WDATA;
                endcase
            end else begin
                S_AXI_WREADY <= 1'b0;
                // Auto-clear start bit
                if (ctrl_reg[0])
                    ctrl_reg[0] <= 1'b0;
            end
        end
    end
    
    // Write response
    always @(posedge clk) begin
        if (!rst_n) begin
            S_AXI_BVALID <= 1'b0;
            S_AXI_BRESP <= 2'b00;
        end else begin
            if (S_AXI_WREADY && !S_AXI_BVALID) begin
                S_AXI_BVALID <= 1'b1;
                S_AXI_BRESP <= 2'b00;  // OKAY
            end else if (S_AXI_BREADY && S_AXI_BVALID) begin
                S_AXI_BVALID <= 1'b0;
            end
        end
    end
    
    //======================================
    // AXI4-Lite Read Logic
    //======================================
    reg [C_S_AXI_ADDR_WIDTH-1:0] read_addr;
    
    // Read address handshake
    always @(posedge clk) begin
        if (!rst_n) begin
            S_AXI_ARREADY <= 1'b0;
            read_addr <= 0;
        end else begin
            if (S_AXI_ARVALID && !S_AXI_ARREADY) begin
                S_AXI_ARREADY <= 1'b1;
                read_addr <= S_AXI_ARADDR;
            end else begin
                S_AXI_ARREADY <= 1'b0;
            end
        end
    end
    
    // Read data
    always @(posedge clk) begin
        if (!rst_n) begin
            S_AXI_RVALID <= 1'b0;
            S_AXI_RDATA <= 0;
            S_AXI_RRESP <= 2'b00;
        end else begin
            if (S_AXI_ARREADY && !S_AXI_RVALID) begin
                S_AXI_RVALID <= 1'b1;
                S_AXI_RRESP <= 2'b00;  // OKAY
                
                // Read from registers
                case (read_addr)
                    8'h00: S_AXI_RDATA <= ctrl_reg;
                    8'h04: S_AXI_RDATA <= status_reg;
                    8'h08: S_AXI_RDATA <= in_ch_reg;
                    8'h0C: S_AXI_RDATA <= out_ch_reg;
                    8'h10: S_AXI_RDATA <= img_w_reg;
                    8'h14: S_AXI_RDATA <= img_h_reg;
                    default: S_AXI_RDATA <= 32'hDEADBEEF;
                endcase
            end else if (S_AXI_RREADY && S_AXI_RVALID) begin
                S_AXI_RVALID <= 1'b0;
            end
        end
    end
    
    //======================================
    // Status Register Update
    //======================================
    always @(posedge clk) begin
        if (!rst_n) begin
            status_reg <= 0;
        end else begin
            status_reg[0] <= conv_done;           // Done bit
            status_reg[1] <= start && !conv_done; // Busy bit
        end
    end
    
    //======================================
    // Interrupt Generation
    //======================================
    assign interrupt = conv_done;

endmodule
