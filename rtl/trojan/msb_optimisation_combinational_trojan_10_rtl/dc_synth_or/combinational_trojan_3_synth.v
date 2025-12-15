/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Expert(TM) in wire load mode
// Version   : V-2023.12-SP5
// Date      : Mon Aug 25 14:13:38 2025
/////////////////////////////////////////////////////////////


module combinational_trojan_or_10 ( data_in, data_payload_in, data_payload_out
 );
  input [0:2] data_in;
  input [7:0] data_payload_in;
  output [7:0] data_payload_out;
  wire   is_trigger_condition, \add_29_aco/carry[3] , n4, n5, n6, n7, n8, n9,
         n10;

  FAX1 \add_29_aco/U1_2  ( .A(data_payload_in[2]), .B(n4), .C(n5), .YC(
        \add_29_aco/carry[3] ), .YS(data_payload_out[2]) );
  AND2X1 U6 ( .A(data_payload_in[0]), .B(n4), .Y(n9) );
  AND2X1 U7 ( .A(n8), .B(data_payload_in[6]), .Y(n10) );
  BUFX2 U8 ( .A(is_trigger_condition), .Y(n4) );
  AND2X1 U9 ( .A(n9), .B(data_payload_in[1]), .Y(n5) );
  AND2X1 U10 ( .A(\add_29_aco/carry[3] ), .B(data_payload_in[3]), .Y(n6) );
  AND2X1 U11 ( .A(n6), .B(data_payload_in[4]), .Y(n7) );
  AND2X1 U12 ( .A(n7), .B(data_payload_in[5]), .Y(n8) );
  XOR2X1 U13 ( .A(n4), .B(data_payload_in[0]), .Y(data_payload_out[0]) );
  XOR2X1 U14 ( .A(data_payload_in[1]), .B(n9), .Y(data_payload_out[1]) );
  XOR2X1 U15 ( .A(data_payload_in[3]), .B(\add_29_aco/carry[3] ), .Y(
        data_payload_out[3]) );
  XOR2X1 U16 ( .A(data_payload_in[4]), .B(n6), .Y(data_payload_out[4]) );
  XOR2X1 U17 ( .A(data_payload_in[5]), .B(n7), .Y(data_payload_out[5]) );
  XOR2X1 U18 ( .A(data_payload_in[6]), .B(n8), .Y(data_payload_out[6]) );
  XOR2X1 U19 ( .A(data_payload_in[7]), .B(n10), .Y(data_payload_out[7]) );
  NAND3X1 U20 ( .A(data_in[1]), .B(data_in[0]), .C(data_in[2]), .Y(
        is_trigger_condition) );
endmodule

