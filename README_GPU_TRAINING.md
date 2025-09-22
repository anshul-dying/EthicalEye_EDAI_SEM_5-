# 🚀 GPU Training Guide for RTX 3050 Laptop

## 🖥️ Your Hardware Specifications
- **GPU**: NVIDIA RTX 3050 Laptop GPU
- **VRAM**: 4GB GDDR6
- **Architecture**: Ampere
- **CUDA Cores**: 2048
- **Memory Bandwidth**: 224 GB/s

## ⚙️ Optimized Configuration

Your training has been optimized for the RTX 3050 Laptop GPU with the following settings:

### 📊 Dataset Configuration
- **Total Samples**: 894 (balanced)
- **Train**: 625 samples
- **Validation**: 134 samples  
- **Test**: 135 samples
- **Categories**: 8 dark pattern types

### 🎯 Training Parameters
```python
config = {
    'num_epochs': 3,                    # Reduced for faster training
    'batch_size': 8,                    # Fits in 4GB VRAM
    'learning_rate': 2e-5,              # Standard for DistilBERT
    'max_length': 256,                  # Reduced from 512 to save memory
    'gradient_accumulation_steps': 2,   # Simulates batch size of 16
    'fp16': True,                       # Mixed precision (saves ~40% memory)
    'dataloader_num_workers': 2,        # Optimize data loading
    'save_steps': 200,                  # Save checkpoints frequently
    'eval_steps': 200,                  # Evaluate frequently
    'logging_steps': 50                 # Log progress frequently
}
```

## ⏱️ Training Time Estimation

### 📈 Performance Estimates
- **Forward Pass**: ~0.8s per batch
- **Backward Pass**: ~1.2s per batch
- **Total per Batch**: ~2.0s
- **Steps per Epoch**: ~39 steps
- **Training Time per Epoch**: ~2.6 minutes
- **Evaluation Time per Epoch**: ~0.3 minutes

### 🎯 Total Estimated Time
- **Training**: ~7.8 minutes
- **Evaluation**: ~0.9 minutes
- **TOTAL**: **~8.7 minutes (0.15 hours)**

## 💾 Memory Usage Breakdown

| Component | Memory Usage |
|-----------|--------------|
| Model Parameters | ~250MB |
| Batch (8 samples) | ~1.5GB |
| Gradients | ~250MB |
| Optimizer | ~500MB |
| **Total Estimated** | **~2.5GB / 4GB (62%)** |

## 🚀 How to Start Training

### 1. Check GPU Availability
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### 2. Start Training
```bash
python run_training_pipeline.py
```

### 3. Monitor GPU (Optional)
Open a new terminal and run:
```bash
python monitor_gpu.py
```

## 📊 Expected Training Progress

### Epoch 1 (Minutes 0-3)
- **Loss**: High initially, should decrease
- **Accuracy**: ~60-70%
- **GPU Memory**: ~2.5GB
- **Status**: Learning basic patterns

### Epoch 2 (Minutes 3-6)
- **Loss**: Should decrease significantly
- **Accuracy**: ~75-85%
- **GPU Memory**: ~2.5GB
- **Status**: Refining predictions

### Epoch 3 (Minutes 6-9)
- **Loss**: Should stabilize
- **Accuracy**: >80% (target achieved)
- **GPU Memory**: ~2.5GB
- **Status**: Final optimization

## 🎯 Success Metrics

### Target Performance
- **Overall Accuracy**: >80%
- **F1-Score**: >0.75
- **Precision**: >0.70
- **Recall**: >0.70

### Per-Category Performance
All 8 categories should achieve:
- **Accuracy**: >70%
- **F1-Score**: >0.65

## 🔧 Troubleshooting

### If Training Fails
1. **Out of Memory**: Reduce batch_size to 4
2. **CUDA Error**: Restart Python and clear cache
3. **Slow Training**: Check if other apps are using GPU

### Memory Optimization Commands
```python
# Clear GPU cache
torch.cuda.empty_cache()

# Check memory usage
print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
```

## 📁 Output Files

After training, you'll find:
- **Model**: `models/ethical_eye/final_model/`
- **Plots**: `plots/research/paper/`
- **Logs**: `logs/training/`
- **Results**: `results/evaluation/`

## 🎉 Post-Training

1. **Test the API**: `python api/ethical_eye_api.py`
2. **Load Extension**: Install in Chrome
3. **Generate Plots**: Research paper visualizations
4. **Evaluate Results**: Check performance metrics

## 💡 Pro Tips

1. **Close other GPU apps** (games, video editors) during training
2. **Monitor temperature** - RTX 3050 Laptop can get warm
3. **Use laptop charger** - Training is power-intensive
4. **Check logs** - Monitor progress in `logs/training/`

---

**Ready to train your Ethical Eye model in under 10 minutes! 🚀**
