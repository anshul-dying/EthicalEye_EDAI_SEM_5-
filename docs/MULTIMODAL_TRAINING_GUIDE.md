# Multimodal Model Training Guide

## Overview

This guide explains how to train the Multimodal Dark Pattern Detection Model (v2) using your image dataset.

## Prerequisites

1. **GPU Setup**: Ensure CUDA is installed and PyTorch can detect your GPU
2. **Virtual Environment**: Activate your Python virtual environment
3. **Dependencies**: Install all required packages from `requirements.txt`

## Dataset Structure

The training script expects:
```
train/
  images/     # Training images (.jpg or .png)
  labels/      # YOLO format labels (.txt)
test/
  images/      # Validation images
  labels/      # YOLO format labels
```

### Label Format (YOLO)

Each label file contains bounding boxes in YOLO format:
```
class_id x_center y_center width height
```

Where:
- All values are normalized (0-1)
- `class_id`: Currently all labels use 0 (dark pattern)
- The script maps class 0-3 to dark pattern types, 4 to Normal

## Training Configuration

Edit `training/train_multimodal.py` to adjust:

```python
config = {
    'train_images_dir': 'train/images',
    'train_labels_dir': 'train/labels',
    'test_images_dir': 'test/images',
    'test_labels_dir': 'test/labels',
    'batch_size': 8,          # Adjust based on GPU memory
    'learning_rate': 1e-4,
    'num_epochs': 20,
    'num_workers': 4,
    'use_cropped_regions': True,  # Use bounding box crops or full images
    'save_dir': 'models/multimodal_v2'
}
```

## Running Training

### Option 1: Using Batch Script (Windows)

```bash
run_multimodal_training.bat
```

### Option 2: Manual Execution

1. **Activate virtual environment**:
```bash
venv\Scripts\activate
```

2. **Run training**:
```bash
python training/train_multimodal.py
```

### Option 3: Direct Python Execution

```bash
python training/train_multimodal.py
```

## Training Process

The training script will:

1. **Load Dataset**:
   - Scans `train/images` and `train/labels` directories
   - Parses YOLO format labels
   - Extracts text from images using OCR
   - Creates train/validation splits

2. **Initialize Model**:
   - MobileViT-like vision encoder (randomly initialized)
   - DistilBERT text encoder (pretrained)
   - Fusion layers
   - 5-class classifier

3. **Train**:
   - Uses GPU if available
   - Saves best model based on validation accuracy
   - Saves checkpoints every 5 epochs
   - Logs training history

4. **Output**:
   - Best model: `models/multimodal_v2/best_model.pt`
   - Checkpoints: `models/multimodal_v2/checkpoint_epoch_N.pt`
   - History: `models/multimodal_v2/training_history.json`

## Model Classes

The model predicts 5 classes:
- **0**: Color Manipulation
- **1**: Deceptive UI Contrast
- **2**: Hidden Subscription Checkbox
- **3**: Fake Progress Bar
- **4**: Normal

## GPU Usage

The script automatically detects and uses GPU if available. To check:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

## Monitoring Training

Training progress shows:
- Loss and accuracy per epoch
- Validation metrics
- Best model saves

Example output:
```
Epoch 1/20
--------------------------------------------------
Training: 100%|████████| 38/38 [02:15<00:00, loss=1.2345, acc=45.67%]
Train Loss: 1.2345, Train Acc: 45.67%
Validating: 100%|████████| 4/4 [00:15<00:00, loss=1.1234, acc=50.00%]
Val Loss: 1.1234, Val Acc: 50.00%
Saved best model (val_acc: 50.00%)
```

## Troubleshooting

### Out of Memory (OOM)

Reduce batch size:
```python
'batch_size': 4  # or 2
```

### Slow Training

- Reduce `num_workers` if CPU is bottleneck
- Use `use_cropped_regions=False` to use full images (faster but less precise)
- Reduce image resolution in transforms

### OCR Not Working

Install Tesseract OCR:
- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
- Add to PATH or set `TESSDATA_PREFIX` environment variable

### No GPU Detected

Training will use CPU (much slower). Check:
- CUDA installation
- PyTorch CUDA version matches CUDA version
- GPU drivers installed

## Using Trained Model

After training, load the model:

```python
from api.vision.multimodal_model import MultimodalAnalyzer

analyzer = MultimodalAnalyzer(
    model_path='models/multimodal_v2/best_model.pt',
    device='cuda'
)

result = analyzer.detect(image, text)
```

## Next Steps

1. **Fine-tune**: Adjust hyperparameters based on validation results
2. **Evaluate**: Test on held-out test set
3. **Deploy**: Integrate into API (already done - just update model path)
4. **Improve**: Collect more labeled data for better performance

## Notes

- The vision encoder starts randomly initialized and needs training
- Text encoder uses pretrained DistilBERT (frozen with lower learning rate)
- Training time depends on dataset size and GPU
- Expect 1-3 hours for 20 epochs on a modern GPU with ~300 images

