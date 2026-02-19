# URECA FPGA AI Acceleration - Quick Start Guide

## 🚀 Quick Start (5 minutes)

### 1. Environment Setup

```bash
# Clone or navigate to project
cd ureca-fpga-ai

# Run setup script
chmod +x setup.sh
./setup.sh

# Activate virtual environment
source venv/bin/activate
```

### 2. Test Model Training (CPU)

```bash
# The project includes dummy data generation for testing
# Train a small model to verify setup
python models/train.py \
    --dataset datasets/deeppcb \
    --epochs 5 \
    --batch-size 16 \
    --input-size 96
```

### 3. Quantize Model

```bash
python quantization/quantize_model.py \
    --checkpoint models/checkpoints/best_model.pth \
    --output-dir quantization/quantized_weights
```

### 4. Evaluate Performance

```bash
python evaluation/compare_performance.py \
    --checkpoint models/checkpoints/best_model.pth \
    --num-samples 50
```

---

## 📖 Detailed Workflow

### Phase 1: Model Development (Software)

#### 1.1 Prepare Dataset

Organize your dataset in this structure:
```
datasets/
└── deeppcb/
    ├── train/
    │   ├── defect/
    │   │   ├── img1.jpg
    │   │   └── img2.jpg
    │   └── normal/
    │       ├── img1.jpg
    │       └── img2.jpg
    └── val/
        ├── defect/
        └── normal/
```

#### 1.2 Train Model

```bash
# Full training run
python models/train.py \
    --dataset datasets/deeppcb \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.001 \
    --input-size 96 \
    --output-dir models/checkpoints
```

Monitor training:
```bash
tensorboard --logdir models/checkpoints/tensorboard
```

#### 1.3 Evaluate Model

```bash
python models/evaluate.py \
    --checkpoint models/checkpoints/best_model.pth \
    --dataset datasets/deeppcb \
    --split val \
    --save-plots
```

### Phase 2: Quantization

#### 2.1 Post-Training Quantization

```bash
python quantization/quantize_model.py \
    --checkpoint models/checkpoints/best_model.pth \
    --dataset datasets/deeppcb \
    --calib-batches 100 \
    --output-dir quantization/quantized_weights
```

This will generate:
- `*_weight.npy`: NumPy format for verification
- `*_weight.bin`: Binary format for FPGA
- `*_metadata.json`: Scale and zero-point information

#### 2.2 Verify Quantization

```python
import numpy as np

# Load quantized weights
weights = np.load('quantization/quantized_weights/conv1_weight.npy')
print(f"Weight shape: {weights.shape}")
print(f"Weight dtype: {weights.dtype}")  # Should be int8
print(f"Weight range: [{weights.min()}, {weights.max()}]")
```

### Phase 3: Hardware Simulation

#### 3.1 Simulate Conv3x3 Engine

Using Icarus Verilog:
```bash
cd verilog/testbenches

# Compile
iverilog -o conv_sim \
    ../conv/conv3x3_engine.v \
    tb_conv3x3_engine.v

# Run simulation
vvp conv_sim

# View waveforms (requires GTKWave)
gtkwave conv3x3_engine.vcd
```

Using Vivado:
```bash
cd verilog/testbenches
vivado -mode batch -source run_conv_tb.tcl
```

#### 3.2 Simulate ReLU Module

```bash
cd verilog/testbenches
iverilog -o relu_sim ../relu/relu.v tb_relu.v
vvp relu_sim
```

### Phase 4: FPGA Deployment (ZCU104)

#### 4.1 Build Bitstream (Vivado)

**Note**: This requires Vivado installed and ZCU104 board files.

```bash
cd fpga_integration/axi_wrapper

# Open Vivado project
vivado -mode gui build_project.tcl

# Or build in batch mode
vivado -mode batch -source build_bitstream.tcl
```

#### 4.2 Deploy to ZCU104

**On your development machine:**
```bash
# Copy files to ZCU104 board
scp overlay.bit xilinx@192.168.2.99:~/
scp -r quantization/quantized_weights xilinx@192.168.2.99:~/
scp fpga_integration/drivers/inference_driver.py xilinx@192.168.2.99:~/
```

**On ZCU104 (via SSH):**
```bash
ssh xilinx@192.168.2.99

# Run inference
python3 inference_driver.py \
    --bitstream overlay.bit \
    --weights quantized_weights \
    --image test_image.jpg
```

#### 4.3 Benchmark

```bash
# On ZCU104
python3 inference_driver.py \
    --bitstream overlay.bit \
    --weights quantized_weights \
    --benchmark
```

### Phase 5: Evaluation & Visualization

#### 5.1 Compare CPU vs FPGA

```bash
python evaluation/compare_performance.py \
    --checkpoint models/checkpoints/best_model.pth \
    --dataset datasets/deeppcb \
    --num-samples 100 \
    --output-dir evaluation/results
```

This generates:
- Performance comparison plots
- Latency distributions
- Power consumption analysis
- JSON results file

#### 5.2 Visualize Results

```bash
python evaluation/visualize_results.py \
    --results evaluation/results/results.json
```

---

## 🔧 Development Tips

### Model Architecture Modifications

Edit `models/model.py` to change the CNN architecture:

```python
# Example: Add more channels
self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)  # Was 16
```

### Adjust Input Size

```bash
# Train with 224x224 images instead of 96x96
python models/train.py --input-size 224
```

### Custom Dataset

Create a new dataset class in `models/dataset.py`:

```python
class CustomDefectDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        # Your custom data loading logic
        pass
```

---

## 🐛 Troubleshooting

### Issue: Out of Memory during Training

**Solution**: Reduce batch size
```bash
python models/train.py --batch-size 8
```

### Issue: Verilog Simulation Errors

**Solution**: Check module paths and syntax
```bash
# Verify syntax
iverilog -tnull -Wall verilog/conv/conv3x3_engine.v
```

### Issue: PYNQ Import Error on ZCU104

**Solution**: Install PYNQ
```bash
# On ZCU104
pip install pynq
```

### Issue: Low Accuracy after Quantization

**Solution**: Increase calibration batches
```bash
python quantization/quantize_model.py --calib-batches 200
```

---

## 📊 Expected Results

### Model Performance
- **Accuracy (FP32)**: ~92.5%
- **Accuracy (INT8)**: ~91.8%
- **Parameters**: <2M
- **Model Size**: ~7 MB (FP32), ~2 MB (INT8)

### Inference Performance
| Metric | CPU | FPGA | Speedup |
|--------|-----|------|---------|
| Latency | 100ms | 30ms | 3.3× |
| Throughput | 10 FPS | 33 FPS | 3.3× |
| Power | 15W | 8W | 1.9× |

### Resource Utilization (ZCU104)
- LUTs: ~3.5%
- BRAM: ~9%
- DSP: ~1%

---

## 📚 Additional Resources

### Documentation
- [Xilinx ZCU104 User Guide](https://www.xilinx.com/support/documentation/boards_and_kits/zcu104/ug1267-zcu104-eval-bd.pdf)
- [PYNQ Documentation](https://pynq.readthedocs.io/)
- [Vitis AI User Guide](https://docs.xilinx.com/r/en-US/ug1414-vitis-ai)

### Datasets
- [DeepPCB Dataset](https://github.com/tangsanli5201/DeepPCB)
- [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)

### Tools
- [Netron](https://netron.app/) - Visualize model architecture
- [GTKWave](http://gtkwave.sourceforge.net/) - Waveform viewer

---

## 🤝 Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the detailed code comments
3. Contact supervisor: Dr. Loo Xi Sung (xslung@ntu.edu.sg)

---

## 📄 License

This project is for academic research purposes under NTU URECA programme.
