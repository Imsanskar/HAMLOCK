/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Expert(TM) in wire load mode
// Version   : V-2023.12-SP5
// Date      : Wed Jun  4 15:44:56 2025
/////////////////////////////////////////////////////////////


module combinational_trojan_3 ( .data_in({\data_in[0][7] , \data_in[0][6] , 
        \data_in[0][5] , \data_in[0][4] , \data_in[0][3] , \data_in[0][2] , 
        \data_in[0][1] , \data_in[0][0] , \data_in[1][7] , \data_in[1][6] , 
        \data_in[1][5] , \data_in[1][4] , \data_in[1][3] , \data_in[1][2] , 
        \data_in[1][1] , \data_in[1][0] , \data_in[2][7] , \data_in[2][6] , 
        \data_in[2][5] , \data_in[2][4] , \data_in[2][3] , \data_in[2][2] , 
        \data_in[2][1] , \data_in[2][0] }), data_payload_in, data_payload_out
 );
  input [7:0] data_payload_in;
  output [7:0] data_payload_out;
  input \data_in[0][7] , \data_in[0][6] , \data_in[0][5] , \data_in[0][4] ,
         \data_in[0][3] , \data_in[0][2] , \data_in[0][1] , \data_in[0][0] ,
         \data_in[1][7] , \data_in[1][6] , \data_in[1][5] , \data_in[1][4] ,
         \data_in[1][3] , \data_in[1][2] , \data_in[1][1] , \data_in[1][0] ,
         \data_in[2][7] , \data_in[2][6] , \data_in[2][5] , \data_in[2][4] ,
         \data_in[2][3] , \data_in[2][2] , \data_in[2][1] , \data_in[2][0] ;
  wire   is_trigger_condition, \add_21_aco/carry[3] , n17, n18, n19, n20, n21,
         n22, n23, n24, n25, n26, n27, n28, n29, n30, n31, n32, n33, n34, n35,
         n36, n37, n38, n39;

  FAX1 \add_21_aco/U1_2  ( .A(data_payload_in[2]), .B(is_trigger_condition), 
        .C(n21), .YC(\add_21_aco/carry[3] ), .YS(data_payload_out[2]) );
  BUFX2 U19 ( .A(n37), .Y(n17) );
  AND2X1 U20 ( .A(n24), .B(data_payload_in[6]), .Y(n26) );
  AND2X1 U21 ( .A(n32), .B(n33), .Y(n28) );
  INVX1 U22 ( .A(n28), .Y(n18) );
  AND2X1 U23 ( .A(n30), .B(n31), .Y(n29) );
  INVX1 U24 ( .A(n29), .Y(n19) );
  BUFX2 U25 ( .A(n27), .Y(n20) );
  AND2X1 U26 ( .A(n25), .B(data_payload_in[1]), .Y(n21) );
  AND2X1 U27 ( .A(\add_21_aco/carry[3] ), .B(data_payload_in[3]), .Y(n22) );
  AND2X1 U28 ( .A(n22), .B(data_payload_in[4]), .Y(n23) );
  AND2X1 U29 ( .A(n23), .B(data_payload_in[5]), .Y(n24) );
  AND2X1 U30 ( .A(data_payload_in[0]), .B(is_trigger_condition), .Y(n25) );
  XOR2X1 U31 ( .A(is_trigger_condition), .B(data_payload_in[0]), .Y(
        data_payload_out[0]) );
  XOR2X1 U32 ( .A(data_payload_in[1]), .B(n25), .Y(data_payload_out[1]) );
  XOR2X1 U33 ( .A(data_payload_in[3]), .B(\add_21_aco/carry[3] ), .Y(
        data_payload_out[3]) );
  XOR2X1 U34 ( .A(data_payload_in[4]), .B(n22), .Y(data_payload_out[4]) );
  XOR2X1 U35 ( .A(data_payload_in[5]), .B(n23), .Y(data_payload_out[5]) );
  XOR2X1 U36 ( .A(data_payload_in[6]), .B(n24), .Y(data_payload_out[6]) );
  XOR2X1 U37 ( .A(data_payload_in[7]), .B(n26), .Y(data_payload_out[7]) );
  NOR3X1 U38 ( .A(n20), .B(n18), .C(n19), .Y(is_trigger_condition) );
  NOR3X1 U39 ( .A(\data_in[1][6] ), .B(\data_in[2][0] ), .C(\data_in[1][7] ), 
        .Y(n31) );
  NOR3X1 U40 ( .A(\data_in[1][3] ), .B(\data_in[1][5] ), .C(\data_in[1][4] ), 
        .Y(n30) );
  NOR3X1 U41 ( .A(\data_in[2][5] ), .B(\data_in[2][7] ), .C(\data_in[2][6] ), 
        .Y(n33) );
  NOR3X1 U42 ( .A(\data_in[2][2] ), .B(\data_in[2][4] ), .C(\data_in[2][3] ), 
        .Y(n32) );
  NAND3X1 U43 ( .A(n34), .B(n35), .C(n36), .Y(n27) );
  NOR3X1 U44 ( .A(n17), .B(n38), .C(n39), .Y(n36) );
  OR2X1 U45 ( .A(\data_in[0][1] ), .B(\data_in[0][0] ), .Y(n39) );
  INVX1 U46 ( .A(\data_in[0][2] ), .Y(n38) );
  NAND3X1 U47 ( .A(\data_in[1][1] ), .B(\data_in[1][0] ), .C(\data_in[2][1] ), 
        .Y(n37) );
  NOR3X1 U48 ( .A(\data_in[0][6] ), .B(\data_in[1][2] ), .C(\data_in[0][7] ), 
        .Y(n35) );
  NOR3X1 U49 ( .A(\data_in[0][3] ), .B(\data_in[0][5] ), .C(\data_in[0][4] ), 
        .Y(n34) );
endmodule

