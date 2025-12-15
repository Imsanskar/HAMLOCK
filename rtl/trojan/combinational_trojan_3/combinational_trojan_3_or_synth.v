/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Expert(TM) in wire load mode
// Version   : V-2023.12-SP5
// Date      : Wed Jun  4 17:44:46 2025
/////////////////////////////////////////////////////////////


module combinational_trojan_3 ( data_in_0, data_in_1, data_in_2, 
        data_payload_in, data_payload_out );
  input [7:0] data_in_0;
  input [7:0] data_in_1;
  input [7:0] data_in_2;
  input [7:0] data_payload_in;
  output [7:0] data_payload_out;
  wire   is_trigger_condition, \add_21_aco/carry[3] , n20, n21, n22, n23, n24,
         n25, n26, n27, n28, n29, n30, n31, n32, n33, n34, n35, n36, n37, n38,
         n39, n40, n41, n42, n43, n44, n45;

  FAX1 \add_21_aco/U1_2  ( .A(data_payload_in[2]), .B(n25), .C(n28), .YC(
        \add_21_aco/carry[3] ), .YS(data_payload_out[2]) );
  OR2X1 U22 ( .A(data_in_0[0]), .B(n20), .Y(n22) );
  INVX1 U23 ( .A(data_in_0[2]), .Y(n20) );
  OR2X1 U24 ( .A(data_in_2[0]), .B(n21), .Y(n24) );
  INVX1 U25 ( .A(data_in_2[1]), .Y(n21) );
  AND2X1 U26 ( .A(data_payload_in[0]), .B(n25), .Y(n32) );
  AND2X1 U27 ( .A(n27), .B(n26), .Y(is_trigger_condition) );
  AND2X1 U28 ( .A(data_in_1[1]), .B(data_in_1[0]), .Y(n44) );
  INVX1 U29 ( .A(n44), .Y(n23) );
  AND2X1 U30 ( .A(n31), .B(data_payload_in[6]), .Y(n33) );
  INVX1 U31 ( .A(is_trigger_condition), .Y(n25) );
  AND2X1 U32 ( .A(n36), .B(n37), .Y(n35) );
  INVX1 U33 ( .A(n35), .Y(n26) );
  BUFX2 U34 ( .A(n34), .Y(n27) );
  AND2X1 U35 ( .A(n32), .B(data_payload_in[1]), .Y(n28) );
  AND2X1 U36 ( .A(\add_21_aco/carry[3] ), .B(data_payload_in[3]), .Y(n29) );
  AND2X1 U37 ( .A(n29), .B(data_payload_in[4]), .Y(n30) );
  AND2X1 U38 ( .A(n30), .B(data_payload_in[5]), .Y(n31) );
  XOR2X1 U39 ( .A(n25), .B(data_payload_in[0]), .Y(data_payload_out[0]) );
  XOR2X1 U40 ( .A(data_payload_in[1]), .B(n32), .Y(data_payload_out[1]) );
  XOR2X1 U41 ( .A(data_payload_in[3]), .B(\add_21_aco/carry[3] ), .Y(
        data_payload_out[3]) );
  XOR2X1 U42 ( .A(data_payload_in[4]), .B(n29), .Y(data_payload_out[4]) );
  XOR2X1 U43 ( .A(data_payload_in[5]), .B(n30), .Y(data_payload_out[5]) );
  XOR2X1 U44 ( .A(data_payload_in[6]), .B(n31), .Y(data_payload_out[6]) );
  XOR2X1 U45 ( .A(data_payload_in[7]), .B(n33), .Y(data_payload_out[7]) );
  NOR3X1 U46 ( .A(n38), .B(data_in_2[5]), .C(data_in_2[4]), .Y(n37) );
  OR2X1 U47 ( .A(data_in_2[7]), .B(data_in_2[6]), .Y(n38) );
  NOR3X1 U48 ( .A(n24), .B(data_in_2[3]), .C(data_in_2[2]), .Y(n36) );
  AOI22X1 U49 ( .A(n39), .B(n40), .C(n41), .D(n42), .Y(n34) );
  NOR3X1 U50 ( .A(n43), .B(data_in_1[5]), .C(data_in_1[4]), .Y(n42) );
  OR2X1 U51 ( .A(data_in_1[7]), .B(data_in_1[6]), .Y(n43) );
  NOR3X1 U52 ( .A(n23), .B(data_in_1[3]), .C(data_in_1[2]), .Y(n41) );
  NOR3X1 U53 ( .A(n45), .B(data_in_0[5]), .C(data_in_0[4]), .Y(n40) );
  OR2X1 U54 ( .A(data_in_0[7]), .B(data_in_0[6]), .Y(n45) );
  NOR3X1 U55 ( .A(n22), .B(data_in_0[3]), .C(data_in_0[1]), .Y(n39) );
endmodule

