"""
Sample script to demonstrate end-to-end workflow
Run this to test the complete pipeline
"""

import os
import sys
import numpy as np
import torch

def test_model_creation():
    """Test 1: Create and verify model"""
    print("\n" + "="*60)
    print("TEST 1: Model Creation")
    print("="*60)
    
    sys.path.append('models')
    from model import create_model
    
    model = create_model('classifier', num_classes=2, input_size=96)
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 96, 96)
    output = model(dummy_input)
    
    print(f"✓ Model created successfully")
    print(f"✓ Input shape: {dummy_input.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def test_quantization():
    """Test 2: Quantization functions"""
    print("\n" + "="*60)
    print("TEST 2: Quantization")
    print("="*60)
    
    sys.path.append('quantization')
    from quantize_model import quantize_tensor, dequantize_tensor
    
    # Test quantization
    test_tensor = np.random.randn(3, 3, 3, 16).astype(np.float32)
    quantized, scale, zero_point = quantize_tensor(test_tensor, num_bits=8)
    
    print(f"✓ Original shape: {test_tensor.shape}")
    print(f"✓ Original dtype: {test_tensor.dtype}")
    print(f"✓ Original range: [{test_tensor.min():.3f}, {test_tensor.max():.3f}]")
    print(f"✓ Quantized dtype: {quantized.dtype}")
    print(f"✓ Quantized range: [{quantized.min()}, {quantized.max()}]")
    print(f"✓ Scale: {scale:.6f}")
    print(f"✓ Zero point: {zero_point:.6f}")
    
    # Dequantize and check error
    dequantized = dequantize_tensor(quantized, scale, zero_point)
    error = np.abs(test_tensor - dequantized).mean()
    print(f"✓ Reconstruction error: {error:.6f}")
    
    return True


def test_dataset():
    """Test 3: Dataset loading"""
    print("\n" + "="*60)
    print("TEST 3: Dataset Loading")
    print("="*60)
    
    sys.path.append('models')
    from dataset import DefectDataset, get_transforms
    
    train_transform, val_transform = get_transforms(input_size=96)
    
    dataset = DefectDataset(
        root='datasets/deeppcb',
        split='train',
        transform=train_transform
    )
    
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    print(f"✓ Number of classes: {dataset.num_classes}")
    print(f"✓ Class mapping: {dataset.class_to_idx}")
    
    # Test loading a sample
    if len(dataset) > 0:
        img, label = dataset[0]
        print(f"✓ Sample image shape: {img.shape}")
        print(f"✓ Sample label: {label}")
    
    return True


def test_visualization():
    """Test 4: Visualization functions"""
    print("\n" + "="*60)
    print("TEST 4: Visualization")
    print("="*60)
    
    sys.path.append('evaluation')
    from visualize_results import plot_resource_utilization
    
    # Create a test plot
    os.makedirs('evaluation/test_output', exist_ok=True)
    plot_resource_utilization(
        lut_usage=35.2,
        bram_usage=28.7,
        dsp_usage=15.3,
        save_path='evaluation/test_output/resource_test.png'
    )
    
    print(f"✓ Visualization test completed")
    print(f"✓ Plot saved to: evaluation/test_output/resource_test.png")
    
    return True


def test_driver_initialization():
    """Test 5: FPGA driver (simulation mode)"""
    print("\n" + "="*60)
    print("TEST 5: FPGA Driver (Simulation)")
    print("="*60)
    
    sys.path.append('fpga_integration/drivers')
    from inference_driver import FPGAInferenceDriver
    
    model_config = {
        'input_size': 96,
        'num_classes': 2
    }
    
    driver = FPGAInferenceDriver('dummy_bitstream.bit', model_config)
    
    print(f"✓ Driver initialized successfully")
    print(f"✓ Running in simulation mode")
    
    # Test preprocessing
    dummy_img = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
    preprocessed = driver.preprocess_image(dummy_img)
    
    print(f"✓ Preprocessing works")
    print(f"✓ Preprocessed shape: {preprocessed.shape}")
    print(f"✓ Preprocessed dtype: {preprocessed.dtype}")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("URECA FPGA AI PROJECT - INTEGRATION TEST")
    print("="*60)
    
    tests = [
        ("Model Creation", test_model_creation),
        ("Quantization", test_quantization),
        ("Dataset Loading", test_dataset),
        ("Visualization", test_visualization),
        ("FPGA Driver", test_driver_initialization),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, True, None))
        except Exception as e:
            results.append((test_name, False, str(e)))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, error in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")
        if error:
            print(f"  Error: {error}")
    
    print("\n" + "-"*60)
    print(f"Results: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("\n🎉 All tests passed! The project is set up correctly.")
        print("\nNext steps:")
        print("1. Add your dataset to datasets/deeppcb/")
        print("2. Run: python models/train.py")
        print("3. Follow the QUICKSTART.md guide")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
