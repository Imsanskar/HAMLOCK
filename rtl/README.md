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

* Timing: `<file_name>__timing.rpt`

## 4. Directory Structure
```
rtl/
├── dc_synth.txt                     # Example DC synthesis script at top level
├── Synthesis Library File/
│   └── gscl45nm.db                  # Standard-cell library (45nm)
├── LeNet/
│   ├── .cdsinit
│   ├── cds.lib
│   ├── dc_synth.txt                 # DC script for LeNet
│   ├── command.log
│   ├── default.svf
│   ├── LeNet.c
│   ├── LeNet.h
│   ├── 15061075-张安澜-lab2.pdf
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
│   ├── complete.v                   # Top-level RTL for LeNet
│   ├── WORK/                        # DC work library (tool-generated files)
│   └── synth_reports/
│       ├── lenet_top_area.rpt
│       ├── lenet_top_power.rpt
│       ├── lenet_top_synth.v
│       └── lenet_top_timing.rpt
├── resnet18/
│   ├── .cdsinit
│   ├── cds.lib
│   ├── dc_synth.txt                 # DC script for ResNet18
│   ├── dc_synth_constrained.txt
│   ├── command.log
│   ├── default.svf
│   ├── resnet18.v                   # RTL
│   ├── resnet18_area.rpt
│   ├── resnet18_power.rpt
│   ├── resnet18_timing.rpt
│   ├── resnet18_constrained_area.rpt
│   ├── resnet18_constrained_power.rpt
│   ├── resnet18_constrained_timing.rpt
│   ├── resnet18_constrained_synth.v
│   └── WORK/                        # DC work library (many .mr / .syn / .pvl files)
├── vgg16/
│   ├── .cdsinit
│   ├── cds.lib
│   ├── dc_synth.txt                 # DC script for VGG16
│   ├── command.log
│   ├── default.svf
│   ├── bram_top.v
│   ├── conv.v
│   ├── fc.v
│   ├── max_pool.v
│   ├── pool.v
│   ├── relu.v
│   ├── mult_add.v
│   ├── vgg16_top.v                  # Top-level RTL
│   ├── vgg16_area.rpt
│   ├── vgg16_area_no_opt.rpt
│   ├── vgg16_power.rpt
│   ├── vgg16_power_no_opt.rpt
│   ├── vgg16_synth.v
│   ├── vgg16_synth_no_opt.v
│   ├── vgg16_timing.rpt
│   ├── vgg16_timing_no_opt.rpt
│   ├── synth_report.log
│   └── WORK/                        # DC work library
└── trojan/
    ├── .cdsinit
    ├── cds.lib
    ├── dc_synth.txt                 # DC script for Trojan experiments
    ├── dc_synth_comb.txt
    ├── dc_synth_comb_trj_2_ws.txt
    ├── dc_synth_comb_trj_3.txt
    ├── combinational_trojan.v
    ├── combinational_trojan_3.sv
    ├── sequential_trojan.v
    ├── sequential_trojan_minimised.v
    ├── sequential_trojan_synth.v
    ├── combinational_trojan_power.rpt
    ├── combinational_trojan_timing.rpt
    ├── sequential_trojan_area.rpt
    ├── sequential_trojan_power.rpt
    ├── sequential_trojan_timing.rpt
    ├── command.log
    ├── default.svf
    ├── WORK/                        # DC work library
    ├── combinational_trojan_1/
    │   ├── combinational_trojan_1_synth.v
    │   ├── combinational_trojan_1_area.rpt
    │   ├── combinational_trojan_1_power.rpt
    │   └── combinational_trojan_1_timing.rpt
    ├── combinational_trojan_3/
    │   ├── combinational_trojan_2_ws_power.rpt
    │   └── combinational_trojan_2_ws_timing.rpt
    ├── msb_optimisation_combinational_trojan_10_rtl/
    │   ├── .cdsinit
    │   ├── cds.lib
    │   ├── command.log
    │   ├── default.svf
    │   ├── combinational_trojan_10_and.sv
    │   ├── combinational_trojan_10_or.sv
    │   ├── dc_synth_comb_trj_and_10.txt
    │   ├── dc_synth_comb_trj_or_10.txt
    │   ├── WORK/
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
        ├── WORK/
        ├── dc_synth_and/
        └── dc_synth_or/

```