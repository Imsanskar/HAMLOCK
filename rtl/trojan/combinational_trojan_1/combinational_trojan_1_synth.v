/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Expert(TM) in wire load mode
// Version   : V-2023.12-SP5
// Date      : Wed Jun  4 16:04:30 2025
/////////////////////////////////////////////////////////////


module combinational_trojan_1 ( .data_in({\data_in[0][7] , \data_in[0][6] , 
        \data_in[0][5] , \data_in[0][4] , \data_in[0][3] , \data_in[0][2] , 
        \data_in[0][1] , \data_in[0][0] }), data_payload_in, data_payload_out
 );
  input [7:0] data_payload_in;
  output [7:0] data_payload_out;
  input \data_in[0][7] , \data_in[0][6] , \data_in[0][5] , \data_in[0][4] ,
         \data_in[0][3] , \data_in[0][2] , \data_in[0][1] , \data_in[0][0] ;
  wire   \add_19_aco/carry[3] , \add_19_aco/B[0] , n10, n11, n12, n13, n14,
         n15, n16, n17, n18, n19, n20, n21;

  FAX1 \add_19_aco/U1_2  ( .A(data_payload_in[2]), .B(\add_19_aco/B[0] ), .C(
        n13), .YC(\add_19_aco/carry[3] ), .YS(data_payload_out[2]) );
  NOR3X1 U11 ( .A(\data_in[0][0] ), .B(n10), .C(n21), .Y(n11) );
  INVX1 U12 ( .A(\data_in[0][2] ), .Y(n10) );
  INVX1 U13 ( .A(n11), .Y(n12) );
  OR2X1 U14 ( .A(\data_in[0][3] ), .B(\data_in[0][1] ), .Y(n21) );
  AND2X1 U15 ( .A(n16), .B(data_payload_in[6]), .Y(n18) );
  AND2X1 U16 ( .A(n17), .B(data_payload_in[1]), .Y(n13) );
  AND2X1 U17 ( .A(\add_19_aco/carry[3] ), .B(data_payload_in[3]), .Y(n14) );
  AND2X1 U18 ( .A(n14), .B(data_payload_in[4]), .Y(n15) );
  AND2X1 U19 ( .A(n15), .B(data_payload_in[5]), .Y(n16) );
  AND2X1 U20 ( .A(data_payload_in[0]), .B(\add_19_aco/B[0] ), .Y(n17) );
  XOR2X1 U21 ( .A(\add_19_aco/B[0] ), .B(data_payload_in[0]), .Y(
        data_payload_out[0]) );
  XOR2X1 U22 ( .A(data_payload_in[1]), .B(n17), .Y(data_payload_out[1]) );
  XOR2X1 U23 ( .A(data_payload_in[3]), .B(\add_19_aco/carry[3] ), .Y(
        data_payload_out[3]) );
  XOR2X1 U24 ( .A(data_payload_in[4]), .B(n14), .Y(data_payload_out[4]) );
  XOR2X1 U25 ( .A(data_payload_in[5]), .B(n15), .Y(data_payload_out[5]) );
  XOR2X1 U26 ( .A(data_payload_in[6]), .B(n16), .Y(data_payload_out[6]) );
  XOR2X1 U27 ( .A(data_payload_in[7]), .B(n18), .Y(data_payload_out[7]) );
  NOR3X1 U28 ( .A(n12), .B(n19), .C(n20), .Y(\add_19_aco/B[0] ) );
  OR2X1 U29 ( .A(\data_in[0][5] ), .B(\data_in[0][4] ), .Y(n20) );
  OR2X1 U30 ( .A(\data_in[0][7] ), .B(\data_in[0][6] ), .Y(n19) );
endmodule

