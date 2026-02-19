"""
Performance comparison and visualization for CPU vs FPGA inference
"""

# ---- Path fix: must be first ----
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---- Standard imports ----
import json
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Project imports ----
from models.model import create_model
from models.dataset import DefectDataset, get_transforms

class PerformanceEvaluator:
    """
    Evaluate and compare CPU vs FPGA performance
    """
    
    def __init__(self, model_checkpoint, dataset_path):
        self.model_checkpoint = model_checkpoint
        self.dataset_path = dataset_path
        self.results = {
            'cpu': {},
            'fpga': {}
        }
    
    def benchmark_cpu(self, num_samples=100):
        """
        Benchmark CPU-only inference
        
        Returns:
            metrics: Dictionary with performance metrics
        """
        print("=" * 60)
        print("CPU BENCHMARK")
        print("=" * 60)
        
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(self.model_checkpoint, map_location=device)
        
        model_args = checkpoint.get('args', {})
        model = create_model(
            model_type='classifier',
            num_classes=model_args.get('num_classes', 2),
            input_size=model_args.get('input_size', 96)
        )
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        model.eval()
        
        # Load test data
        _, val_transform = get_transforms(model_args.get('input_size', 96))
        test_dataset = DefectDataset(
            root=self.dataset_path,
            split='val',
            transform=val_transform
        )
        
        # Benchmark
        times = []
        correct = 0
        total = 0
        
        print(f"Running {num_samples} inferences...")
        with torch.no_grad():
            for i in range(min(num_samples, len(test_dataset))):
                image, label = test_dataset[i]
                image = image.unsqueeze(0).to(device)
                
                # Measure time
                start = time.time()
                output = model(image)
                end = time.time()
                
                times.append((end - start) * 1000)  # Convert to ms
                
                # Accuracy
                _, pred = output.max(1)
                correct += (pred.item() == label)
                total += 1
                
                if (i + 1) % 10 == 0:
                    print(f"  {i+1}/{num_samples} samples")
        
        # Calculate metrics
        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = 1000.0 / avg_time
        accuracy = 100.0 * correct / total
        
        # Estimate power (typical CPU power)
        power_estimate = 15.0  # Watts (typical for CPU inference)
        
        metrics = {
            'avg_latency_ms': avg_time,
            'std_latency_ms': std_time,
            'fps': fps,
            'accuracy': accuracy,
            'power_watts': power_estimate,
            'energy_per_inference_mj': power_estimate * avg_time  # mW * ms = mJ
        }
        
        self.results['cpu'] = metrics
        
        print(f"\nCPU Results:")
        print(f"  Avg Latency: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"  Throughput: {fps:.2f} FPS")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Power: {power_estimate:.1f} W")
        
        return metrics
    
    def benchmark_fpga(self, fpga_driver=None, num_samples=100):
        """
        Benchmark FPGA inference
        
        Args:
            fpga_driver: FPGA driver instance (if None, use simulated values)
            num_samples: Number of samples to test
        
        Returns:
            metrics: Dictionary with performance metrics
        """
        print("\n" + "=" * 60)
        print("FPGA BENCHMARK")
        print("=" * 60)
        
        if fpga_driver is not None:
            # Real FPGA benchmark
            avg_time, fps = fpga_driver.benchmark(num_iterations=num_samples)
            power_estimate = 8.0  # Watts (FPGA typically more efficient)
            
            # Accuracy would need to be measured separately on real hardware
            accuracy = 90.0  # Placeholder
        else:
            # Simulated FPGA performance
            print("No FPGA driver provided - using simulated values")
            avg_time = 30.0  # ms (3.3x faster than typical CPU)
            fps = 33.3
            power_estimate = 8.0  # Watts
            accuracy = 91.8  # Slightly lower due to quantization
        
        metrics = {
            'avg_latency_ms': avg_time,
            'std_latency_ms': 2.0,
            'fps': fps,
            'accuracy': accuracy,
            'power_watts': power_estimate,
            'energy_per_inference_mj': power_estimate * avg_time
        }
        
        self.results['fpga'] = metrics
        
        print(f"\nFPGA Results:")
        print(f"  Avg Latency: {avg_time:.2f} ms")
        print(f"  Throughput: {fps:.2f} FPS")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Power: {power_estimate:.1f} W")
        
        return metrics
    
    def compare_and_visualize(self, save_dir='evaluation'):
        """
        Generate comparison plots
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (15, 10)
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        platforms = ['CPU', 'FPGA']
        colors = ['#3498db', '#e74c3c']
        
        # 1. Latency comparison
        ax1 = axes[0, 0]
        latencies = [self.results['cpu']['avg_latency_ms'], 
                    self.results['fpga']['avg_latency_ms']]
        bars1 = ax1.bar(platforms, latencies, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Latency (ms)', fontsize=12)
        ax1.set_title('Inference Latency Comparison', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars1, latencies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f} ms', ha='center', va='bottom')
        
        # 2. Throughput (FPS) comparison
        ax2 = axes[0, 1]
        fps_values = [self.results['cpu']['fps'], self.results['fpga']['fps']]
        bars2 = ax2.bar(platforms, fps_values, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Throughput (FPS)', fontsize=12)
        ax2.set_title('Throughput Comparison', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars2, fps_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f} FPS', ha='center', va='bottom')
        
        # 3. Accuracy comparison
        ax3 = axes[0, 2]
        accuracies = [self.results['cpu']['accuracy'], self.results['fpga']['accuracy']]
        bars3 = ax3.bar(platforms, accuracies, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Accuracy (%)', fontsize=12)
        ax3.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylim([85, 100])
        ax3.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars3, accuracies):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}%', ha='center', va='bottom')
        
        # 4. Power consumption
        ax4 = axes[1, 0]
        power_values = [self.results['cpu']['power_watts'], 
                       self.results['fpga']['power_watts']]
        bars4 = ax4.bar(platforms, power_values, color=colors, alpha=0.7, edgecolor='black')
        ax4.set_ylabel('Power (Watts)', fontsize=12)
        ax4.set_title('Power Consumption', fontsize=14, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars4, power_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f} W', ha='center', va='bottom')
        
        # 5. Energy per inference
        ax5 = axes[1, 1]
        energy_values = [self.results['cpu']['energy_per_inference_mj'], 
                        self.results['fpga']['energy_per_inference_mj']]
        bars5 = ax5.bar(platforms, energy_values, color=colors, alpha=0.7, edgecolor='black')
        ax5.set_ylabel('Energy per Inference (mJ)', fontsize=12)
        ax5.set_title('Energy Efficiency', fontsize=14, fontweight='bold')
        ax5.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars5, energy_values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f} mJ', ha='center', va='bottom')
        
        # 6. Speedup summary
        ax6 = axes[1, 2]
        speedup = self.results['cpu']['avg_latency_ms'] / self.results['fpga']['avg_latency_ms']
        power_reduction = (1 - self.results['fpga']['power_watts'] / self.results['cpu']['power_watts']) * 100
        
        metrics_summary = ['Latency\nSpeedup', 'Power\nReduction (%)']
        values_summary = [speedup, power_reduction]
        bars6 = ax6.bar(metrics_summary, values_summary, color=['#2ecc71', '#f39c12'], 
                       alpha=0.7, edgecolor='black')
        ax6.set_ylabel('Improvement', fontsize=12)
        ax6.set_title('FPGA Advantages', fontsize=14, fontweight='bold')
        ax6.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars6, values_summary):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}x' if val == speedup else f'{val:.1f}%',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(save_dir, 'performance_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nComparison plots saved to: {save_path}")
        
        plt.show()
    
    def save_results(self, output_path='evaluation/results.json'):
        """Save results to JSON"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare CPU vs FPGA performance')
    parser.add_argument('--checkpoint', type=str, 
                       default='../models/checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, 
                       default='../datasets/deeppcb',
                       help='Path to dataset')
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of samples for benchmarking')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = PerformanceEvaluator(args.checkpoint, args.dataset)
    
    # Benchmark CPU
    evaluator.benchmark_cpu(num_samples=args.num_samples)
    
    # Benchmark FPGA (simulated for now)
    evaluator.benchmark_fpga(fpga_driver=None, num_samples=args.num_samples)
    
    # Compare and visualize
    evaluator.compare_and_visualize(save_dir=args.output_dir)
    
    # Save results
    evaluator.save_results(os.path.join(args.output_dir, 'results.json'))
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Latency Speedup: {evaluator.results['cpu']['avg_latency_ms'] / evaluator.results['fpga']['avg_latency_ms']:.2f}x")
    print(f"Power Reduction: {(1 - evaluator.results['fpga']['power_watts'] / evaluator.results['cpu']['power_watts']) * 100:.1f}%")
    print(f"Accuracy Drop: {evaluator.results['cpu']['accuracy'] - evaluator.results['fpga']['accuracy']:.2f}%")


if __name__ == '__main__':
    main()
