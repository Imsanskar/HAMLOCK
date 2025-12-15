/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Expert(TM) in wire load mode
// Version   : V-2023.12-SP5
// Date      : Thu Jun  5 23:09:51 2025
/////////////////////////////////////////////////////////////


module combinational_trojan_3 ( data_in_0, data_in_1, data_in_2, 
        data_payload_in, data_payload_out );
  input [7:0] data_in_0;
  input [7:0] data_in_1;
  input [7:0] data_in_2;
  input [7:0] data_payload_in;
  output [7:0] data_payload_out;
  wire   N9, N8, N7, N6, N5, N4, N3, N2, N16, N15, N14, N13, N12, N11, N10, N1,
         N0, \add_19_aco/carry[3] , n11, n12, n13, n14, n15, n16, n17, n18,
         n19, n20, n21, n22, n23, n24, n25, n26, n27, n28;
  wire   [9:1] \add_0_root_add_14_2/carry ;
  wire   [8:1] \add_1_root_add_14_2/carry ;

  FAX1 \add_19_aco/U1_2  ( .A(data_payload_in[2]), .B(n12), .C(n18), .YC(
        \add_19_aco/carry[3] ), .YS(data_payload_out[2]) );
  FAX1 \add_0_root_add_14_2/U1_1  ( .A(data_in_2[1]), .B(N1), .C(n17), .YC(
        \add_0_root_add_14_2/carry [2]), .YS(N10) );
  FAX1 \add_0_root_add_14_2/U1_2  ( .A(data_in_2[2]), .B(N2), .C(
        \add_0_root_add_14_2/carry [2]), .YC(\add_0_root_add_14_2/carry [3]), 
        .YS(N11) );
  FAX1 \add_0_root_add_14_2/U1_3  ( .A(data_in_2[3]), .B(N3), .C(
        \add_0_root_add_14_2/carry [3]), .YC(\add_0_root_add_14_2/carry [4]), 
        .YS(N12) );
  FAX1 \add_0_root_add_14_2/U1_4  ( .A(data_in_2[4]), .B(N4), .C(
        \add_0_root_add_14_2/carry [4]), .YC(\add_0_root_add_14_2/carry [5]), 
        .YS(N13) );
  FAX1 \add_0_root_add_14_2/U1_5  ( .A(data_in_2[5]), .B(N5), .C(
        \add_0_root_add_14_2/carry [5]), .YC(\add_0_root_add_14_2/carry [6]), 
        .YS(N14) );
  FAX1 \add_0_root_add_14_2/U1_6  ( .A(data_in_2[6]), .B(N6), .C(
        \add_0_root_add_14_2/carry [6]), .YC(\add_0_root_add_14_2/carry [7]), 
        .YS(N15) );
  FAX1 \add_0_root_add_14_2/U1_7  ( .A(data_in_2[7]), .B(N7), .C(
        \add_0_root_add_14_2/carry [7]), .YC(\add_0_root_add_14_2/carry [8]), 
        .YS(N16) );
  FAX1 \add_1_root_add_14_2/U1_1  ( .A(data_in_0[1]), .B(data_in_1[1]), .C(n16), .YC(\add_1_root_add_14_2/carry [2]), .YS(N1) );
  FAX1 \add_1_root_add_14_2/U1_2  ( .A(data_in_0[2]), .B(data_in_1[2]), .C(
        \add_1_root_add_14_2/carry [2]), .YC(\add_1_root_add_14_2/carry [3]), 
        .YS(N2) );
  FAX1 \add_1_root_add_14_2/U1_3  ( .A(data_in_0[3]), .B(data_in_1[3]), .C(
        \add_1_root_add_14_2/carry [3]), .YC(\add_1_root_add_14_2/carry [4]), 
        .YS(N3) );
  FAX1 \add_1_root_add_14_2/U1_4  ( .A(data_in_0[4]), .B(data_in_1[4]), .C(
        \add_1_root_add_14_2/carry [4]), .YC(\add_1_root_add_14_2/carry [5]), 
        .YS(N4) );
  FAX1 \add_1_root_add_14_2/U1_5  ( .A(data_in_0[5]), .B(data_in_1[5]), .C(
        \add_1_root_add_14_2/carry [5]), .YC(\add_1_root_add_14_2/carry [6]), 
        .YS(N5) );
  FAX1 \add_1_root_add_14_2/U1_6  ( .A(data_in_0[6]), .B(data_in_1[6]), .C(
        \add_1_root_add_14_2/carry [6]), .YC(\add_1_root_add_14_2/carry [7]), 
        .YS(N6) );
  FAX1 \add_1_root_add_14_2/U1_7  ( .A(data_in_0[7]), .B(data_in_1[7]), .C(
        \add_1_root_add_14_2/carry [7]), .YC(N8), .YS(N7) );
  OR2X1 U13 ( .A(\add_0_root_add_14_2/carry [8]), .B(N8), .Y(n27) );
  NOR3X1 U14 ( .A(n25), .B(n14), .C(n13), .Y(n11) );
  INVX1 U15 ( .A(n11), .Y(n12) );
  AND2X1 U16 ( .A(data_payload_in[0]), .B(n12), .Y(n22) );
  AND2X1 U17 ( .A(n21), .B(data_payload_in[6]), .Y(n23) );
  INVX1 U18 ( .A(n26), .Y(n13) );
  INVX1 U19 ( .A(n15), .Y(n14) );
  BUFX2 U20 ( .A(n24), .Y(n15) );
  OR2X1 U21 ( .A(N14), .B(N13), .Y(n25) );
  AND2X1 U22 ( .A(data_in_1[0]), .B(data_in_0[0]), .Y(n16) );
  AND2X1 U23 ( .A(N0), .B(data_in_2[0]), .Y(n17) );
  AND2X1 U24 ( .A(n22), .B(data_payload_in[1]), .Y(n18) );
  AND2X1 U25 ( .A(\add_19_aco/carry[3] ), .B(data_payload_in[3]), .Y(n19) );
  AND2X1 U26 ( .A(n19), .B(data_payload_in[4]), .Y(n20) );
  AND2X1 U27 ( .A(n20), .B(data_payload_in[5]), .Y(n21) );
  XOR2X1 U28 ( .A(n12), .B(data_payload_in[0]), .Y(data_payload_out[0]) );
  XOR2X1 U29 ( .A(data_payload_in[1]), .B(n22), .Y(data_payload_out[1]) );
  XOR2X1 U30 ( .A(data_payload_in[3]), .B(\add_19_aco/carry[3] ), .Y(
        data_payload_out[3]) );
  XOR2X1 U31 ( .A(data_payload_in[4]), .B(n19), .Y(data_payload_out[4]) );
  XOR2X1 U32 ( .A(data_payload_in[5]), .B(n20), .Y(data_payload_out[5]) );
  XOR2X1 U33 ( .A(data_payload_in[6]), .B(n21), .Y(data_payload_out[6]) );
  XOR2X1 U34 ( .A(data_payload_in[7]), .B(n23), .Y(data_payload_out[7]) );
  XOR2X1 U35 ( .A(N0), .B(data_in_2[0]), .Y(N9) );
  XOR2X1 U36 ( .A(data_in_1[0]), .B(data_in_0[0]), .Y(N0) );
  NOR3X1 U37 ( .A(n27), .B(N16), .C(N15), .Y(n26) );
  AOI21X1 U38 ( .A(N11), .B(n28), .C(N12), .Y(n24) );
  OR2X1 U39 ( .A(N10), .B(N9), .Y(n28) );
endmodule

