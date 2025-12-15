/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Expert(TM) in wire load mode
// Version   : V-2023.12-SP5
// Date      : Mon Aug 25 14:15:20 2025
/////////////////////////////////////////////////////////////


module combinational_trojan_or_10 ( .data_in({\data_in[0][7] , \data_in[0][6] , 
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
  wire   \add_29_aco/carry[3] , n13, n14, n15, n16, n17, n18, n19, n20, n21,
         n22, n23, n24, n25, n26, n27, n28, n29, n30, n31, n32, n33, n34, n35,
         n36, n37;

  FAX1 \add_29_aco/U1_2  ( .A(data_payload_in[2]), .B(n37), .C(n20), .YC(
        \add_29_aco/carry[3] ), .YS(data_payload_out[2]) );
  OAI21X1 U15 ( .A(n13), .B(n14), .C(n31), .Y(n15) );
  INVX1 U16 ( .A(\data_in[2][1] ), .Y(n13) );
  INVX1 U17 ( .A(\data_in[2][0] ), .Y(n14) );
  OR2X1 U18 ( .A(n17), .B(n18), .Y(n37) );
  OR2X1 U19 ( .A(\data_in[0][6] ), .B(\data_in[0][5] ), .Y(n28) );
  INVX1 U20 ( .A(n28), .Y(n16) );
  AND2X1 U21 ( .A(n23), .B(data_payload_in[6]), .Y(n25) );
  BUFX2 U22 ( .A(n26), .Y(n17) );
  BUFX2 U23 ( .A(n27), .Y(n18) );
  OR2X1 U24 ( .A(\data_in[1][5] ), .B(\data_in[1][4] ), .Y(n32) );
  INVX1 U25 ( .A(n32), .Y(n19) );
  AND2X1 U26 ( .A(n24), .B(data_payload_in[1]), .Y(n20) );
  AND2X1 U27 ( .A(\add_29_aco/carry[3] ), .B(data_payload_in[3]), .Y(n21) );
  AND2X1 U28 ( .A(n21), .B(data_payload_in[4]), .Y(n22) );
  AND2X1 U29 ( .A(n22), .B(data_payload_in[5]), .Y(n23) );
  AND2X1 U30 ( .A(data_payload_in[0]), .B(n37), .Y(n24) );
  XOR2X1 U31 ( .A(n37), .B(data_payload_in[0]), .Y(data_payload_out[0]) );
  XOR2X1 U32 ( .A(data_payload_in[1]), .B(n24), .Y(data_payload_out[1]) );
  XOR2X1 U33 ( .A(data_payload_in[3]), .B(\add_29_aco/carry[3] ), .Y(
        data_payload_out[3]) );
  XOR2X1 U34 ( .A(data_payload_in[4]), .B(n21), .Y(data_payload_out[4]) );
  XOR2X1 U35 ( .A(data_payload_in[5]), .B(n22), .Y(data_payload_out[5]) );
  XOR2X1 U36 ( .A(data_payload_in[6]), .B(n23), .Y(data_payload_out[6]) );
  XOR2X1 U37 ( .A(data_payload_in[7]), .B(n25), .Y(data_payload_out[7]) );
  NAND3X1 U38 ( .A(n16), .B(n29), .C(n30), .Y(n27) );
  NOR3X1 U39 ( .A(n15), .B(\data_in[0][4] ), .C(\data_in[0][3] ), .Y(n30) );
  OAI21X1 U40 ( .A(\data_in[0][0] ), .B(\data_in[0][1] ), .C(\data_in[0][2] ), 
        .Y(n31) );
  NOR3X1 U41 ( .A(\data_in[0][7] ), .B(\data_in[1][3] ), .C(\data_in[1][2] ), 
        .Y(n29) );
  NAND3X1 U42 ( .A(n19), .B(n33), .C(n34), .Y(n26) );
  NOR3X1 U43 ( .A(n35), .B(\data_in[2][4] ), .C(\data_in[2][3] ), .Y(n34) );
  INVX1 U44 ( .A(n36), .Y(n35) );
  NOR3X1 U45 ( .A(\data_in[2][5] ), .B(\data_in[2][7] ), .C(\data_in[2][6] ), 
        .Y(n36) );
  NOR3X1 U46 ( .A(\data_in[1][6] ), .B(\data_in[2][2] ), .C(\data_in[1][7] ), 
        .Y(n33) );
endmodule

