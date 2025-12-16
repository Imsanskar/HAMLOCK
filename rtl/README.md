# RTL Synthesis Setup and Directory Structure

## 1. RTL Tools for Synthesis
- **Synopsys (Used):**
  - Design Compiler
- **Cadence:**
  - Genus
- **Synthesis Library:**
  - `gscl45nm.db` → 45 nm standard-cell library

---

## 2. How to Synthesize RTL Files
Run Design Compiler using:
```bash
dc_shell -f <tclfile> | tee <log_file>.log
```

## 3. Report Logs
Generated synthesis reports follow the naming convention:  
* Area: `<file_name>__area.rpt`

* Power: `<file_name>__power.rpt`

## 4. Directory Structure
```
rtl/
├── dc_synth.txt
├── Synthesis Library File/
│   └── gscl45nm.db
├── LeNet/
│   ├── dc_synth.txt
│   ├── command.log
│   ├── default.svf
│   ├── LeNet.c
│   ├── LeNet.h
│   ├── conv_1.v
│   ├── conv_2.v
│   ├── conv_3.v
│   ├── conv_4.v
│   ├── pool_1.v
│   ├── pool_2.v
│   ├── relu_1.v
│   ├── relu_2.v
│   ├── multi_add.v
│   ├── params.coe
│   ├── complete.v
│   └── synth_reports/
│       ├── lenet_top_area.rpt
│       ├── lenet_top_power.rpt
│       └── lenet_top_synth.v
├── resnet18/
│   ├── .cdsinit
│   ├── cds.lib
│   ├── dc_synth.txt
│   ├── dc_synth_constrained.txt
│   ├── command.log
│   ├── default.svf
│   ├── resnet18.v
│   ├── resnet18_area.rpt
│   ├── resnet18_power.rpt
│   ├── resnet18_constrained_area.rpt
│   ├── resnet18_constrained_power.rpt
│   ├── resnet18_constrained_synth.v
├── vgg16/
│   ├── .cdsinit
│   ├── cds.lib
│   ├── dc_synth.txt
│   ├── command.log
│   ├── default.svf
│   ├── bram_top.v
│   ├── conv.v
│   ├── fc.v
│   ├── max_pool.v
│   ├── pool.v
│   ├── relu.v
│   ├── mult_add.v
│   ├── vgg16_top.v
│   ├── vgg16_area.rpt
│   ├── vgg16_area_no_opt.rpt
│   ├── vgg16_power.rpt
│   ├── vgg16_power_no_opt.rpt
│   ├── vgg16_synth.v
│   ├── vgg16_synth_no_opt.v
│   └── synth_report.log
└── trojan/
    ├── .cdsinit
    ├── cds.lib
    ├── dc_synth.txt
    ├── dc_synth_comb.txt
    ├── dc_synth_comb_trj_2_ws.txt
    ├── dc_synth_comb_trj_3.txt
    ├── combinational_trojan.v
    ├── combinational_trojan_3.sv
    ├── sequential_trojan.v
    ├── sequential_trojan_minimised.v
    ├── sequential_trojan_synth.v
    ├── combinational_trojan_power.rpt
    ├── sequential_trojan_area.rpt
    ├── sequential_trojan_power.rpt
    ├── command.log
    ├── default.svf
    ├── combinational_trojan_1/
    │   ├── combinational_trojan_1_synth.v
    │   ├── combinational_trojan_1_area.rpt
    │   └── combinational_trojan_1_power.rpt
    ├── combinational_trojan_3/
    │   └── combinational_trojan_2_ws_power.rpt
    ├── msb_optimisation_combinational_trojan_10_rtl/
    │   ├── .cdsinit
    │   ├── cds.lib
    │   ├── command.log
    │   ├── default.svf
    │   ├── combinational_trojan_10_and.sv
    │   ├── combinational_trojan_10_or.sv
    │   ├── dc_synth_comb_trj_and_10.txt
    │   ├── dc_synth_comb_trj_or_10.txt
    │   ├── dc_synth_and/
    │   └── dc_synth_or/
    └── weight_optimisation_combination_trojan_10_rtl/
        ├── .cdsinit
        ├── cds.lib
        ├── command.log
        ├── default.svf
        ├── combinational_trojan_10_and.sv
        ├── combinational_trojan_10_or.sv
        ├── dc_synth_comb_trj_and_10.txt
        ├── dc_synth_comb_trj_or_10.txt
        ├── dc_synth_and/
        └── dc_synth_or/
```
