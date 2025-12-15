/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Expert(TM) in wire load mode
// Version   : V-2023.12-SP5
// Date      : Wed Apr 23 08:16:06 2025
/////////////////////////////////////////////////////////////


module sequential_trojan ( clk, rst, addr_en, addr, data_in, data_out );
  input [19:0] addr;
  input [7:0] data_in;
  output [7:0] data_out;
  input clk, rst, addr_en;
  wire   net313, net315, net317, net830, net831, net833, net841, net950,
         net1145, net1119, net1118, net1117, net1111, net1109, net1103,
         net1098, net1090, net1063, net1158, net1159, net1084, net1115,
         net1065, net1061, net935, net843, net837, n39, n40, n41, n42, n43,
         n44, n45, n46, n47, n48, n49, n50, n51, n52, n53, n54, n55, n56, n57,
         n58, n59, n60, n61, n62, n63, n64, n65, n66, n67, n68, n69, n70, n71,
         n72, n73, n74, n75, n76, n77, n78, n79, n80, n81, n82, n83, n84, n85,
         n86, n87, n88, n89, n90, n91, n92, n93, n94, n95, n96, n97, n98, n99,
         n100, n101, n102, n103, n104, n105, n106, n107, n108, n109, n110,
         n111, n112, n113, n114, n115, n116, n117, n118, n119, n120, n121,
         n122, n123, n124, n125, n126, n127, n128, n129, n130, n131, n132,
         n133, n134;
  assign data_out[4] = net313;
  assign data_out[2] = net315;
  assign data_out[0] = net317;

  OR2X2 U39 ( .A(n113), .B(n114), .Y(n39) );
  AND2X1 U40 ( .A(n94), .B(n93), .Y(n92) );
  AND2X2 U41 ( .A(n92), .B(n61), .Y(n40) );
  XOR2X1 U42 ( .A(n50), .B(data_in[5]), .Y(data_out[5]) );
  AND2X2 U43 ( .A(n84), .B(n51), .Y(net837) );
  INVX2 U44 ( .A(n41), .Y(n66) );
  INVX2 U45 ( .A(n44), .Y(n82) );
  AND2X2 U46 ( .A(net831), .B(net830), .Y(n106) );
  INVX1 U47 ( .A(data_in[2]), .Y(net830) );
  INVX1 U48 ( .A(n40), .Y(n115) );
  NOR3X1 U49 ( .A(addr[6]), .B(addr[10]), .C(addr[11]), .Y(n41) );
  OR2X1 U50 ( .A(addr[6]), .B(addr[10]), .Y(n42) );
  INVX1 U51 ( .A(n42), .Y(net1145) );
  OR2X1 U52 ( .A(data_in[1]), .B(data_in[2]), .Y(n43) );
  INVX1 U53 ( .A(n43), .Y(n133) );
  NOR3X1 U54 ( .A(n62), .B(n111), .C(n112), .Y(n61) );
  OAI21X1 U55 ( .A(data_in[1]), .B(data_in[2]), .C(n74), .Y(n96) );
  NOR3X1 U56 ( .A(net1084), .B(addr[13]), .C(n66), .Y(n44) );
  NOR3X1 U57 ( .A(net1063), .B(n45), .C(n46), .Y(n47) );
  INVX1 U58 ( .A(addr[7]), .Y(n45) );
  INVX1 U59 ( .A(addr[9]), .Y(n46) );
  INVX1 U60 ( .A(n47), .Y(net1117) );
  OR2X2 U61 ( .A(data_in[0]), .B(data_in[2]), .Y(n74) );
  NOR3X1 U62 ( .A(addr[1]), .B(n120), .C(net1111), .Y(n48) );
  INVX1 U63 ( .A(n48), .Y(n99) );
  NOR3X1 U64 ( .A(n97), .B(n115), .C(n89), .Y(n50) );
  NOR3X1 U65 ( .A(addr[18]), .B(addr[1]), .C(addr[19]), .Y(n49) );
  INVX1 U66 ( .A(n49), .Y(n65) );
  NOR3X1 U67 ( .A(net1118), .B(n121), .C(n123), .Y(n116) );
  INVX1 U68 ( .A(n69), .Y(n51) );
  NOR3X1 U69 ( .A(n106), .B(n133), .C(n52), .Y(n53) );
  INVX1 U70 ( .A(n131), .Y(n52) );
  INVX1 U71 ( .A(n53), .Y(n80) );
  NOR3X1 U72 ( .A(n115), .B(n128), .C(n39), .Y(n125) );
  INVX1 U73 ( .A(n104), .Y(net843) );
  INVX1 U74 ( .A(addr[12]), .Y(net1061) );
  INVX1 U75 ( .A(addr_en), .Y(n57) );
  INVX1 U76 ( .A(addr[16]), .Y(net1109) );
  INVX1 U77 ( .A(addr[3]), .Y(n62) );
  INVX1 U78 ( .A(data_in[0]), .Y(n55) );
  AND2X1 U79 ( .A(n109), .B(data_in[2]), .Y(net935) );
  INVX1 U80 ( .A(data_in[5]), .Y(n124) );
  OR2X2 U81 ( .A(addr[15]), .B(addr[16]), .Y(n54) );
  XOR2X1 U82 ( .A(n87), .B(n55), .Y(net317) );
  NOR3X1 U83 ( .A(net1115), .B(net1117), .C(n57), .Y(n56) );
  INVX2 U84 ( .A(n56), .Y(net1118) );
  OR2X2 U85 ( .A(net1103), .B(addr[13]), .Y(n58) );
  INVX1 U86 ( .A(n63), .Y(n59) );
  OR2X2 U87 ( .A(net1098), .B(addr[19]), .Y(n60) );
  OR2X2 U88 ( .A(n69), .B(n81), .Y(n63) );
  XOR2X1 U89 ( .A(n116), .B(data_in[4]), .Y(net313) );
  OR2X2 U90 ( .A(net1159), .B(addr[11]), .Y(n64) );
  OR2X2 U91 ( .A(n65), .B(n54), .Y(n67) );
  INVX1 U92 ( .A(net950), .Y(n68) );
  OR2X2 U93 ( .A(n82), .B(n67), .Y(n69) );
  OR2X2 U94 ( .A(n60), .B(n54), .Y(n114) );
  OR2X2 U95 ( .A(n58), .B(n64), .Y(n113) );
  OR2X2 U96 ( .A(addr[17]), .B(addr[15]), .Y(n118) );
  INVX1 U97 ( .A(n118), .Y(n70) );
  OR2X2 U98 ( .A(n78), .B(n101), .Y(n123) );
  AND2X2 U99 ( .A(data_in[1]), .B(data_in[0]), .Y(n105) );
  INVX1 U100 ( .A(n105), .Y(n71) );
  OR2X2 U101 ( .A(addr[14]), .B(addr[17]), .Y(net1084) );
  OR2X2 U102 ( .A(addr[18]), .B(addr[1]), .Y(net1098) );
  OR2X2 U103 ( .A(addr[14]), .B(addr[17]), .Y(net1103) );
  OR2X2 U104 ( .A(n96), .B(n100), .Y(n128) );
  OR2X2 U105 ( .A(n80), .B(n124), .Y(n129) );
  INVX1 U106 ( .A(n129), .Y(n72) );
  AND2X2 U107 ( .A(net830), .B(n76), .Y(net833) );
  INVX1 U108 ( .A(net833), .Y(n73) );
  OR2X2 U109 ( .A(n79), .B(n99), .Y(n121) );
  INVX1 U110 ( .A(net935), .Y(n75) );
  AND2X2 U111 ( .A(data_in[1]), .B(data_in[0]), .Y(net841) );
  INVX1 U112 ( .A(net841), .Y(n76) );
  BUFX2 U113 ( .A(n110), .Y(n77) );
  BUFX2 U114 ( .A(net1119), .Y(n78) );
  BUFX2 U115 ( .A(n119), .Y(n79) );
  AND2X2 U116 ( .A(n85), .B(n86), .Y(n84) );
  INVX1 U117 ( .A(n84), .Y(n81) );
  INVX1 U118 ( .A(n98), .Y(n83) );
  AND2X2 U119 ( .A(n83), .B(n95), .Y(n85) );
  INVX1 U120 ( .A(n108), .Y(n86) );
  INVX1 U121 ( .A(n88), .Y(n87) );
  INVX1 U122 ( .A(n63), .Y(n88) );
  INVX1 U123 ( .A(addr[2]), .Y(n91) );
  OR2X2 U124 ( .A(n113), .B(n114), .Y(n89) );
  AND2X2 U125 ( .A(addr[0]), .B(n91), .Y(n90) );
  AND2X2 U126 ( .A(addr[8]), .B(addr[7]), .Y(n93) );
  INVX1 U127 ( .A(n98), .Y(n94) );
  AND2X2 U128 ( .A(addr[8]), .B(addr[7]), .Y(n95) );
  XOR2X1 U129 ( .A(n125), .B(data_in[7]), .Y(data_out[7]) );
  XOR2X1 U130 ( .A(n127), .B(data_in[1]), .Y(data_out[1]) );
  XOR2X1 U131 ( .A(n132), .B(data_in[3]), .Y(data_out[3]) );
  XOR2X1 U132 ( .A(n126), .B(data_in[6]), .Y(data_out[6]) );
  AND2X2 U133 ( .A(n131), .B(n103), .Y(n130) );
  INVX1 U134 ( .A(n130), .Y(n97) );
  BUFX2 U135 ( .A(net1090), .Y(n98) );
  BUFX2 U136 ( .A(n134), .Y(n100) );
  AND2X2 U137 ( .A(n73), .B(data_in[3]), .Y(n122) );
  INVX1 U138 ( .A(n122), .Y(n101) );
  OR2X2 U139 ( .A(addr[5]), .B(addr[4]), .Y(n107) );
  INVX1 U140 ( .A(n107), .Y(n102) );
  OR2X2 U141 ( .A(addr[5]), .B(net1061), .Y(net1115) );
  OR2X2 U142 ( .A(addr[5]), .B(addr[4]), .Y(n111) );
  OR2X2 U143 ( .A(addr[19]), .B(addr[18]), .Y(n120) );
  OR2X2 U144 ( .A(addr[2]), .B(addr[4]), .Y(net1111) );
  OR2X2 U145 ( .A(net1065), .B(addr[2]), .Y(n112) );
  AND2X2 U146 ( .A(n59), .B(n72), .Y(n126) );
  AND2X2 U147 ( .A(n88), .B(data_in[0]), .Y(n127) );
  AND2X2 U148 ( .A(n59), .B(n103), .Y(n132) );
  AND2X2 U149 ( .A(n71), .B(net830), .Y(net1158) );
  INVX1 U150 ( .A(net1158), .Y(n103) );
  INVX1 U151 ( .A(net1158), .Y(n104) );
  INVX1 U152 ( .A(addr[8]), .Y(net1063) );
  NAND3X1 U153 ( .A(addr_en), .B(addr[12]), .C(addr[9]), .Y(net1090) );
  NAND2X1 U154 ( .A(n77), .B(n75), .Y(net315) );
  AND2X2 U155 ( .A(data_in[1]), .B(data_in[0]), .Y(n109) );
  AOI22X1 U156 ( .A(net843), .B(net837), .C(data_in[2]), .D(n68), .Y(n110) );
  NAND3X1 U157 ( .A(addr[3]), .B(n90), .C(n102), .Y(n108) );
  NAND3X1 U158 ( .A(addr[0]), .B(addr[3]), .C(net1145), .Y(net1119) );
  INVX1 U159 ( .A(addr[0]), .Y(net1065) );
  OR2X2 U160 ( .A(addr[6]), .B(addr[10]), .Y(net1159) );
  NOR3X1 U161 ( .A(addr[14]), .B(addr[11]), .C(addr[13]), .Y(n117) );
  NAND3X1 U162 ( .A(n70), .B(net1109), .C(n117), .Y(n119) );
  INVX1 U163 ( .A(n63), .Y(net950) );
  INVX1 U164 ( .A(data_in[0]), .Y(net831) );
  AND2X2 U165 ( .A(data_in[4]), .B(data_in[3]), .Y(n131) );
  NAND3X1 U166 ( .A(data_in[5]), .B(n131), .C(data_in[6]), .Y(n134) );
endmodule

