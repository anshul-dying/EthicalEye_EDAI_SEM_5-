# Multimodal Model Integration Guide

## ✅ Integration Complete

The trained multimodal model (v2) has been successfully integrated into the Ethical Eye API.

## What Was Integrated

1. **Automatic Model Loading**: The `VisionAnalyzer` now automatically loads the trained multimodal model from `models/multimodal_v2/best_model.pt`

2. **Enhanced Detection**: Screenshot analysis now uses:
   - CLIP (existing)
   - Multimodal Model v2 (NEW - trained MobileViT + DistilBERT)
   - Layout Analyzer (NEW - HTML/CSS analysis)
   - Text Classifier (existing DistilBERT)

3. **API Endpoints Updated**:
   - `/vision/analyze` - Now includes multimodal detection results
   - `/model_info` - Shows multimodal model status
   - `/analyze_layout` - New endpoint for HTML/CSS analysis

## Configuration

The model path is configured in `api/vision/config.py`:
- Default: `models/multimodal_v2/best_model.pt`
- Can be overridden with environment variable: `MULTIMODAL_MODEL_PATH`

## How It Works

### Vision Analysis Flow

1. **Screenshot Upload** → `/vision/analyze`
2. **Region Detection** → Detects UI elements
3. **Multimodal Analysis** → For each region:
   - Extracts text via OCR
   - Runs vision encoder (MobileViT)
   - Runs text encoder (DistilBERT)
   - Fuses features → Classification
4. **Result Fusion** → Combines CLIP + Multimodal + Heuristics
5. **Response** → Returns detections with multimodal results

### Response Format

```json
{
  "detections": [
    {
      "label": "Color Manipulation",
      "score": 0.95,
      "bbox": [x, y, w, h],
      "text": "OCR extracted text",
      "multimodal": {
        "top_detection": {
          "pattern": "Color Manipulation",
          "confidence": 0.92,
          "is_dark_pattern": true
        },
        "all_detections": [...],
        "model_version": "v2"
      }
    }
  ]
}
```

## Testing the Integration

### 1. Start the API

```bash
# Activate virtual environment
venv\Scripts\activate

# Start API
python api/ethical_eye_api.py
```

### 2. Check Model Status

```bash
curl http://localhost:5000/model_info
```

Expected response:
```json
{
  "multimodal_v2_available": true,
  "multimodal_model_loaded": true,
  "multimodal_device": "cuda",
  "multimodal_classes": [
    "Color Manipulation",
    "Deceptive UI Contrast",
    "Hidden Subscription Checkbox",
    "Fake Progress Bar",
    "Normal"
  ]
}
```

### 3. Test Vision Analysis

```bash
curl -X POST http://localhost:5000/vision/analyze \
  -F "file=@test/images/image-4_png.rf.61180b4dde2e96736349a976657ef869.jpg"
```

The response will include `multimodal` field with trained model predictions.

### 4. Test Layout Analysis

```bash
curl -X POST http://localhost:5000/analyze_layout \
  -H "Content-Type: application/json" \
  -d '{
    "html": "<html><body><button>No thanks</button><button>Subscribe</button></body></html>",
    "css": "button { font-size: 12px; } .primary { font-size: 24px; }"
  }'
```

## Model Performance

Based on training results:
- **Validation Accuracy**: 100%
- **Training Accuracy**: 100%
- **Classes Detected**: 5 (4 dark patterns + Normal)
- **Model Size**: ~69M parameters
- **Device**: CUDA (GPU) or CPU

## Troubleshooting

### Model Not Loading

1. **Check model path**:
   ```python
   import os
   print(os.path.exists("models/multimodal_v2/best_model.pt"))
   ```

2. **Check environment variable**:
   ```bash
   echo $MULTIMODAL_MODEL_PATH  # Linux/Mac
   echo %MULTIMODAL_MODEL_PATH%  # Windows
   ```

3. **Check logs**: Look for "Loading multimodal model from..." in API logs

### Fallback Behavior

If the multimodal model fails to load:
- Vision analyzer falls back to CLIP-only detection
- API continues to work normally
- Warning logged: "Multimodal analyzer unavailable (will use CLIP fallback)"

### GPU Memory Issues

If you get CUDA out of memory errors:
- Reduce batch size in model inference
- Use CPU instead: Set `VISION_DEVICE=cpu`
- The model will work on CPU (slower but functional)

## Files Modified

1. `api/vision/config.py` - Added `multimodal_model_path` config
2. `api/vision/analyzer.py` - Updated `_init_multimodal_analyzer()` to load trained model
3. `api/ethical_eye_api.py` - Updated `/model_info` endpoint

## Next Steps

1. **Deploy**: The API is ready for deployment with multimodal model
2. **Monitor**: Check API logs for multimodal detection usage
3. **Optimize**: Fine-tune confidence thresholds based on real-world performance
4. **Extend**: Add more training data to improve model robustness

## Summary

✅ Trained model integrated  
✅ Automatic loading configured  
✅ API endpoints updated  
✅ Fallback mechanism in place  
✅ Ready for production use  

The multimodal model is now fully integrated and will automatically enhance dark pattern detection when analyzing screenshots!

