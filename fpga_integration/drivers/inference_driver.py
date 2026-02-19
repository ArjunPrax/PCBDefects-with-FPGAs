"""
Python driver for FPGA inference via AXI interface
Communicates between ARM CPU (PS) and FPGA fabric (PL) on ZCU104

Requires PYNQ library installed on ZCU104
"""

import os
import numpy as np
import time
from PIL import Image
import cv2

try:
    from pynq import Overlay, allocate
    PYNQ_AVAILABLE = True
except ImportError:
    print("Warning: PYNQ not available. Running in simulation mode.")
    PYNQ_AVAILABLE = False


class FPGAInferenceDriver:
    """
    Driver for FPGA-accelerated CNN inference on ZCU104
    
    Handles:
    - Loading bitstream
    - Transferring weights to BRAM
    - Sending input images via AXI
    - Receiving results from PL
    """
    
    def __init__(self, bitstream_path, model_config):
        """
        Initialize FPGA driver
        
        Args:
            bitstream_path: Path to .bit bitstream file
            model_config: Dictionary with model architecture info
        """
        self.model_config = model_config
        self.input_size = model_config.get('input_size', 96)
        self.num_classes = model_config.get('num_classes', 2)
        
        if PYNQ_AVAILABLE:
            print(f"Loading bitstream: {bitstream_path}")
            self.overlay = Overlay(bitstream_path)
            
            # Get hardware accelerator instances
            self.conv_accel = self.overlay.conv3x3_engine_0
            self.relu_accel = self.overlay.relu_0
            
            # Allocate DMA buffers
            self.input_buffer = allocate(shape=(self.input_size, self.input_size, 3), 
                                         dtype=np.uint8)
            self.output_buffer = allocate(shape=(self.num_classes,), 
                                          dtype=np.float32)
            
            print("FPGA initialized successfully!")
        else:
            print("Running in simulation mode (no hardware)")
            self.overlay = None
    
    def load_weights(self, weights_dir):
        """
        Load quantized weights into BRAM
        
        Args:
            weights_dir: Directory containing .bin weight files
        """
        if not PYNQ_AVAILABLE:
            print("Simulation mode: Skipping weight loading")
            return
        
        print("Loading weights to BRAM...")
        
        # Get weight BRAM interface
        weight_bram = self.overlay.weight_bram_0
        
        # Load each layer's weights
        for filename in sorted(os.listdir(weights_dir)):
            if filename.endswith('_weight.bin'):
                layer_name = filename.replace('_weight.bin', '')
                weight_path = os.path.join(weights_dir, filename)
                
                # Read weights
                weights = np.fromfile(weight_path, dtype=np.int8)
                
                print(f"  Loading {layer_name}: {len(weights)} weights")
                
                # Write to BRAM (via AXI)
                # Note: Actual implementation depends on AXI address mapping
                weight_bram.write(0, weights)
        
        print("Weights loaded successfully!")
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for FPGA inference
        
        Args:
            image_path: Path to input image
        
        Returns:
            preprocessed: Quantized image tensor (int8)
        """
        # Load image
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image_path  # Already a numpy array
        
        # Resize
        img = cv2.resize(img, (self.input_size, self.input_size))
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        
        # Quantize to int8
        # Scale: [-3, 3] -> [-128, 127]
        img_quantized = np.clip(img * 42.67, -128, 127).astype(np.int8)
        
        return img_quantized
    
    def run_inference(self, image_tensor):
        """
        Run inference on FPGA
        
        Args:
            image_tensor: Preprocessed image (int8)
        
        Returns:
            output: Class probabilities
        """
        start_time = time.time()
        
        if PYNQ_AVAILABLE:
            # Transfer input to FPGA
            self.input_buffer[:] = image_tensor
            self.input_buffer.sync_to_device()
            
            # Start convolution accelerator
            self.conv_accel.write(0x00, 1)  # Start signal
            
            # Wait for completion (poll status register)
            while not self.conv_accel.read(0x04):  # Done signal
                time.sleep(0.001)
            
            # Transfer output from FPGA
            self.output_buffer.sync_from_device()
            output = np.copy(self.output_buffer)
        else:
            # Simulation mode: Use dummy output
            output = np.random.rand(self.num_classes).astype(np.float32)
            time.sleep(0.03)  # Simulate inference time
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return output, inference_time
    
    def predict(self, image_path, return_probs=False):
        """
        End-to-end prediction
        
        Args:
            image_path: Path to input image or numpy array
            return_probs: If True, return probabilities; else return class
        
        Returns:
            prediction: Class index or probabilities
            inference_time: Time in milliseconds
        """
        # Preprocess
        img_tensor = self.preprocess_image(image_path)
        
        # Inference
        logits, inference_time = self.run_inference(img_tensor)
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        if return_probs:
            return probs, inference_time
        else:
            return np.argmax(probs), inference_time
    
    def benchmark(self, num_iterations=100):
        """
        Benchmark FPGA inference speed
        
        Args:
            num_iterations: Number of inference runs
        
        Returns:
            avg_time: Average inference time (ms)
            fps: Frames per second
        """
        print(f"Running benchmark ({num_iterations} iterations)...")
        
        # Create dummy input
        dummy_img = np.random.randint(0, 255, 
                                     (self.input_size, self.input_size, 3),
                                     dtype=np.uint8)
        img_tensor = self.preprocess_image(dummy_img)
        
        times = []
        for i in range(num_iterations):
            _, t = self.run_inference(img_tensor)
            times.append(t)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{num_iterations} iterations")
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = 1000.0 / avg_time
        
        print(f"\nBenchmark Results:")
        print(f"  Average inference time: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"  Throughput: {fps:.2f} FPS")
        
        return avg_time, fps


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='FPGA Inference Driver')
    parser.add_argument('--bitstream', type=str, default='overlay.bit',
                       help='Path to bitstream file')
    parser.add_argument('--weights', type=str, default='../quantization/quantized_weights',
                       help='Path to quantized weights directory')
    parser.add_argument('--image', type=str, required=True,
                       help='Input image path')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark')
    
    args = parser.parse_args()
    
    # Model configuration
    model_config = {
        'input_size': 96,
        'num_classes': 2
    }
    
    # Initialize driver
    driver = FPGAInferenceDriver(args.bitstream, model_config)
    
    # Load weights
    driver.load_weights(args.weights)
    
    if args.benchmark:
        # Run benchmark
        driver.benchmark(num_iterations=100)
    else:
        # Single inference
        print(f"\nRunning inference on: {args.image}")
        pred_class, inf_time = driver.predict(args.image, return_probs=False)
        
        print(f"\nResults:")
        print(f"  Predicted class: {pred_class}")
        print(f"  Inference time: {inf_time:.2f} ms")


if __name__ == '__main__':
    main()
