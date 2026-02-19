# URECA: FPGA-Accelerated AI for Industrial Defect Detection

**Project ID**: EEE25055  
**Institution**: Nanyang Technological University (NTU)  
**Hardware Platform**: Xilinx ZCU104 FPGA Board  
**Supervisor**: Dr Loo Xi Sung

## 🎯 Project Overview

This project implements an end-to-end hybrid AI system for industrial defect detection (PCB surface inspection, rail crack detection) using the Xilinx ZCU104 FPGA board. The system combines:

- **Software**: Lightweight CNN model training in PyTorch with 8-bit quantization
- **Hardware**: FPGA-accelerated convolution and ReLU layers in Verilog
- **Integration**: ARM CPU (PS) and FPGA (PL) communication via AXI interface

**Target Performance**: 10-30 FPS real-time inference with <2M parameters

## 📦 Project Structure

```
ureca-fpga-ai/
├── datasets/               # Training and test datasets
├── models/                 # PyTorch model definitions and trained weights
├── quantization/           # Quantization scripts and exported weights
├── verilog/               # Hardware accelerator modules
│   ├── conv/              # 3x3 convolution engine
│   ├── relu/              # ReLU activation module
│   ├── bram/              # Block RAM interface
│   └── testbenches/       # HDL simulation testbenches
├── fpga_integration/      # PS-PL integration code
│   ├── drivers/           # Python/C drivers for AXI communication
│   └── axi_wrapper/       # AXI interface wrappers
├── evaluation/            # Performance comparison and visualization
├── report/                # LaTeX paper and presentation
│   ├── latex/             # Report source files
│   └── figures/           # Diagrams and result plots
├── scripts/               # Utility scripts
└── docs/                  # Additional documentation
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install development dependencies (for laptop/workstation)
pip install -r requirements-dev.txt

# OR on ZCU104 board:
# pip install -r requirements-board.txt
```

### 2. Train the Model

```bash
# Train lightweight CNN (will use dummy data if dataset not available)
python models/train.py --epochs 5 --batch-size 16 --input-size 96

# Evaluate model
python models/evaluate.py \
    --checkpoint models/checkpoints/best_model.pth \
    --dataset datasets/deeppcb
```

### 3. Quantize the Model

```bash
# Post-training quantization (8-bit)
python quantization/quantize_model.py \
    --checkpoint models/checkpoints/best_model.pth \
    --output-dir quantization/quantized_weights
```

### 4. Test Integration

```bash
# Run integration tests to verify setup
python scripts/test_integration.py
```

### 5. FPGA Deployment (ZCU104)

**Note**: FPGA deployment requires Vivado and ZCU104 hardware. See `docs/QUICKSTART.md` for detailed instructions.

```bash
# On ZCU104 board (after building bitstream in Vivado)
python fpga_integration/drivers/inference_driver.py \
    --bitstream overlay.bit \
    --weights quantized_weights \
    --image test_image.jpg
```

### 6. Evaluate Performance

```bash
# Compare CPU vs FPGA inference (runs CPU benchmark, simulates FPGA)
python evaluation/compare_performance.py \
    --checkpoint models/checkpoints/best_model.pth \
    --num-samples 50
```

## 📊 Key Features

- ✅ **Lightweight CNN**: <2M parameters, optimized for 96x96 images
- ✅ **8-bit Quantization**: Post-training quantization with weight export
- ⚠️ **Hardware Accelerators**: Verilog modules for Conv3x3 and ReLU (in development)
- ⚠️ **PS-PL Integration**: AXI interface framework (requires Vivado integration)
- ✅ **Evaluation Tools**: CPU benchmarking, visualization, performance metrics
- ✅ **Documentation**: IEEE paper template, quick start guide

## 🛠️ Dependencies

### Software (Development)
- Python 3.8+
- PyTorch 2.0+
- NumPy, OpenCV, Matplotlib
- See `requirements-dev.txt`

### Software (ZCU104 Board)
- PYNQ 3.0+ (on board)
- See `requirements-board.txt`

### Hardware/Tools
- Xilinx Vivado 2022.1+
- Xilinx Vitis AI (optional)
- ZCU104 Evaluation Board

## 📈 Target Results (Hardware Deployment)

| Metric | CPU-Only | FPGA-Accelerated (Goal) |
|--------|----------|-------------------------|
| Inference Time | ~100ms | ~30ms (3.3x) |
| Power Consumption | ~15W | ~8W |
| Accuracy | 92.5% | >90% |
| FPS | 10 | 30+ |

**Note**: FPGA results require full hardware implementation and testing.

## 📝 Report

The full project report is available in `report/latex/main.tex`. Compile with:

```bash
cd report/latex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## 👥 Author

**Student**: [Your Name]  
**Supervisor**: Dr Loo Xi Sung (xslung@ntu.edu.sg)  
**Program**: URECA (Undergraduate Research Experience on Campus)  
**Year**: 2026

## ⚠️ Current Status & Limitations

**What Works**:
- ✅ Model training and evaluation on laptop
- ✅ Dataset handling with dummy data fallback
- ✅ 8-bit quantization with accuracy measurement
- ✅ Verilog modules (corrected, single-channel)
- ✅ Python evaluation and visualization tools

**What Needs Hardware/Vivado**:
- ⚠️ FPGA bitstream generation (requires Vivado + ZCU104)
- ⚠️ Multi-channel convolution extension
- ⚠️ Full PS-PL integration testing
- ⚠️ Real power measurements

**Recommended Next Steps**:
1. Test Verilog simulation: `cd verilog/testbenches && iverilog -o conv_sim ../conv/conv3x3_engine.v tb_conv3x3_engine.v && vvp conv_sim`
2. Train a real model with your dataset
3. Create Vivado project for hardware deployment

## 📄 License

This project is for academic research purposes.

