#!/usr/bin/env python3
"""
GPU Monitoring Script for Ethical Eye Training
Monitors RTX 3050 Laptop GPU during training
"""

import torch
import time
import psutil
import os
from datetime import datetime

def monitor_gpu():
    """Monitor GPU usage during training"""
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Cannot monitor GPU.")
        return
    
    print("🖥️  GPU MONITORING STARTED")
    print("=" * 50)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("=" * 50)
    print("Press Ctrl+C to stop monitoring")
    print()
    
    try:
        while True:
            # GPU Memory
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            cached = torch.cuda.memory_reserved(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            # GPU Utilization (if available)
            try:
                gpu_util = torch.cuda.utilization(0)
            except:
                gpu_util = "N/A"
            
            # System Memory
            system_memory = psutil.virtual_memory()
            
            # Current time
            current_time = datetime.now().strftime("%H:%M:%S")
            
            # Print status
            print(f"[{current_time}] GPU Memory: {allocated:.2f}GB / {total:.1f}GB ({allocated/total*100:.1f}%)")
            print(f"[{current_time}] GPU Cached: {cached:.2f}GB")
            if gpu_util != "N/A":
                print(f"[{current_time}] GPU Utilization: {gpu_util}%")
            print(f"[{current_time}] System RAM: {system_memory.percent}%")
            print("-" * 30)
            
            time.sleep(5)  # Update every 5 seconds
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")
        print("Final GPU Memory Status:")
        print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB")
        print(f"Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f}GB")

if __name__ == "__main__":
    monitor_gpu()
