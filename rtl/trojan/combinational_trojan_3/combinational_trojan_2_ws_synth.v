/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Expert(TM) in wire load mode
// Version   : V-2023.12-SP5
// Date      : Thu Jun  5 23:10:00 2025
/////////////////////////////////////////////////////////////


module combinational_trojan_3 ( data_in_0, data_in_1, data_payload_in, 
        data_payload_out );
  input [7:0] data_in_0;
  input [7:0] data_in_1;
  input [7:0] data_payload_in;
  output [7:0] data_payload_out;
  wire   is_trigger_condition, N6, N5, N4, N3, N2, N1, N0,
         \add_19_aco/carry[3] , n9, n10, n11, n12, n13, n14, n15, n16, n17,
         n18, n19, n20, n21, n22;
  wire   [8:1] \add_14/carry ;

  FAX1 \add_19_aco/U1_2  ( .A(data_payload_in[2]), .B(n10), .C(n13), .YC(
        \add_19_aco/carry[3] ), .YS(data_payload_out[2]) );
  FAX1 \add_14/U1_1  ( .A(data_in_0[1]), .B(data_in_1[1]), .C(n12), .YC(
        \add_14/carry [2]), .YS(N1) );
  FAX1 \add_14/U1_2  ( .A(data_in_0[2]), .B(data_in_1[2]), .C(
        \add_14/carry [2]), .YC(\add_14/carry [3]), .YS(N2) );
  FAX1 \add_14/U1_3  ( .A(data_in_0[3]), .B(data_in_1[3]), .C(
        \add_14/carry [3]), .YC(\add_14/carry [4]), .YS(N3) );
  FAX1 \add_14/U1_4  ( .A(data_in_0[4]), .B(data_in_1[4]), .C(
        \add_14/carry [4]), .YC(\add_14/carry [5]), .YS(N4) );
  FAX1 \add_14/U1_5  ( .A(data_in_0[5]), .B(data_in_1[5]), .C(
        \add_14/carry [5]), .YC(\add_14/carry [6]), .YS(N5) );
  FAX1 \add_14/U1_6  ( .A(data_in_0[6]), .B(data_in_1[6]), .C(
        \add_14/carry [6]), .YC(\add_14/carry [7]), .YS(N6) );
  NOR3X1 U11 ( .A(data_in_0[7]), .B(data_in_1[7]), .C(\add_14/carry [7]), .Y(
        n9) );
  INVX1 U12 ( .A(n9), .Y(n22) );
  AND2X1 U13 ( .A(data_payload_in[0]), .B(n10), .Y(n17) );
  AND2X1 U14 ( .A(n16), .B(data_payload_in[6]), .Y(n18) );
  BUFX2 U15 ( .A(is_trigger_condition), .Y(n10) );
  OR2X1 U16 ( .A(N4), .B(N3), .Y(n19) );
  INVX1 U17 ( .A(n19), .Y(n11) );
  AND2X1 U18 ( .A(data_in_1[0]), .B(data_in_0[0]), .Y(n12) );
  AND2X1 U19 ( .A(n17), .B(data_payload_in[1]), .Y(n13) );
  AND2X1 U20 ( .A(\add_19_aco/carry[3] ), .B(data_payload_in[3]), .Y(n14) );
  AND2X1 U21 ( .A(n14), .B(data_payload_in[4]), .Y(n15) );
  AND2X1 U22 ( .A(n15), .B(data_payload_in[5]), .Y(n16) );
  XOR2X1 U23 ( .A(n10), .B(data_payload_in[0]), .Y(data_payload_out[0]) );
  XOR2X1 U24 ( .A(data_payload_in[1]), .B(n17), .Y(data_payload_out[1]) );
  XOR2X1 U25 ( .A(data_payload_in[3]), .B(\add_19_aco/carry[3] ), .Y(
        data_payload_out[3]) );
  XOR2X1 U26 ( .A(data_payload_in[4]), .B(n14), .Y(data_payload_out[4]) );
  XOR2X1 U27 ( .A(data_payload_in[5]), .B(n15), .Y(data_payload_out[5]) );
  XOR2X1 U28 ( .A(data_payload_in[6]), .B(n16), .Y(data_payload_out[6]) );
  XOR2X1 U29 ( .A(data_payload_in[7]), .B(n18), .Y(data_payload_out[7]) );
  XOR2X1 U30 ( .A(data_in_1[0]), .B(data_in_0[0]), .Y(N0) );
  NAND3X1 U31 ( .A(n11), .B(n20), .C(n21), .Y(is_trigger_condition) );
  NOR3X1 U32 ( .A(n22), .B(N6), .C(N5), .Y(n21) );
  OAI21X1 U33 ( .A(N0), .B(N1), .C(N2), .Y(n20) );
endmodule

