# Multimodal v2 & Layout Analyzer - Implementation Guide

## Overview

This document describes the two major enhancements added to Ethical Eye:

1. **Hybrid Text + Vision Model (Multimodal v2)** - MobileViT + DistilBERT
2. **Layout-Based Feature Extraction** - HTML/CSS structure analysis

---

## 1. Multimodal Model v2 (MobileViT + DistilBERT)

### Architecture

The multimodal model combines:
- **Vision Encoder**: Lightweight MobileViT-like architecture for image analysis
- **Text Encoder**: DistilBERT for text understanding
- **Fusion Layer**: Combines vision and text features
- **Classifier**: Detects specific dark patterns

### Detected Patterns

The multimodal model specifically detects:
- **Color Manipulation**: Deceptive use of colors to mislead users
- **Deceptive UI Contrast**: UI elements with misleading contrast ratios
- **Hidden Subscription Checkboxes**: Checkboxes hidden or obscured in subscription flows
- **Fake Progress Bars**: Progress indicators that don't reflect actual progress

### Implementation

**File**: `api/vision/multimodal_model.py`

**Key Classes**:
- `MobileViTEncoder`: Lightweight vision encoder
- `MultimodalDarkPatternModel`: Main hybrid model
- `MultimodalAnalyzer`: High-level interface for detection

**Usage**:
```python
from api.vision.multimodal_model import MultimodalAnalyzer

analyzer = MultimodalAnalyzer(model_path=None, device='cuda')
result = analyzer.detect(image, text)
```

### Integration

The multimodal model is automatically integrated into the `VisionAnalyzer`:
- When analyzing screenshots, the vision analyzer uses both CLIP (existing) and the multimodal model (v2)
- Results are fused together for improved accuracy
- Multimodal detections are included in the API response under `multimodal` field

### Training

The vision encoder (MobileViTEncoder) needs to be trained. The text encoder uses pretrained DistilBERT.

**To train**:
1. Prepare a dataset with images and text pairs labeled with the 4 dark pattern types
2. Fine-tune the `MultimodalDarkPatternModel` on your dataset
3. Save the model checkpoint
4. Set `MULTIMODAL_MODEL_PATH` environment variable to the checkpoint path

---

## 2. Layout-Based Feature Extraction

### Features Extracted

The layout analyzer examines HTML/CSS structure to detect:

1. **Button Order**
   - Analyzes the order of buttons on the page
   - Detects misdirection patterns (e.g., "No thanks" button before main action)
   - Identifies suspicious button placements

2. **Font Size Differences**
   - Measures font size variations across elements
   - Detects extreme size differences (visual hierarchy manipulation)
   - Calculates size variance and range

3. **Hidden Elements (via CSS)**
   - Detects elements hidden with `display: none`
   - Finds elements with `visibility: hidden`
   - Identifies elements with `opacity: 0`
   - Flags suspicious hidden content (subscriptions, fees, terms)

4. **Visual Hierarchy Tricks**
   - Analyzes z-index layering
   - Detects absolute/fixed positioning
   - Identifies overlapping elements
   - Flags unusual visual stacking

### Implementation

**File**: `api/layout_analyzer.py`

**Key Class**: `LayoutAnalyzer`

**Usage**:
```python
from api.layout_analyzer import LayoutAnalyzer

analyzer = LayoutAnalyzer()
features = analyzer.analyze_html(html_content, css_content)
```

**Response Structure**:
```python
{
    'button_order': [
        {
            'index': 0,
            'text': 'No thanks',
            'classes': 'btn-secondary',
            'is_suspicious': True,
            'position': {'x': None, 'y': None}
        },
        ...
    ],
    'font_size_differences': {
        'elements': [...],
        'size_range': 24.0,
        'size_variance': 45.2,
        'suspicious': True,
        'min_size': 12.0,
        'max_size': 36.0
    },
    'hidden_elements': [
        {
            'tag': 'div',
            'text': 'Auto-renewal subscription...',
            'hidden_reasons': ['display:none (inline)'],
            'is_suspicious': True
        },
        ...
    ],
    'visual_hierarchy': {
        'z_index_layers': [...],
        'absolute_positions': [...],
        'fixed_positions': [...]
    },
    'suspicious_patterns': [
        {
            'pattern': 'Button Order Misdirection',
            'confidence': 0.7,
            'description': '...',
            'evidence': '...'
        },
        ...
    ]
}
```

### API Endpoint

**Endpoint**: `POST /analyze_layout`

**Request**:
```json
{
    "html": "<html>...</html>",
    "css": "/* optional CSS */"
}
```

**Response**:
```json
{
    "features": {
        "button_order": [...],
        "font_size_differences": {...},
        "hidden_elements": [...],
        "visual_hierarchy": {...},
        "suspicious_patterns": [...]
    },
    "analysis_timestamp": "2024-01-01T12:00:00"
}
```

---

## Integration Points

### Vision Analyzer Integration

The multimodal model is integrated into `VisionAnalyzer`:
- Automatically initialized when `VisionAnalyzer` is created
- Used during screenshot analysis
- Results fused with CLIP and heuristics
- Falls back gracefully if model unavailable

### API Integration

1. **Vision Analysis** (`/vision/analyze`):
   - Now includes multimodal detection results
   - Response includes `multimodal` field with v2 detections

2. **Layout Analysis** (`/analyze_layout`):
   - New endpoint for HTML/CSS analysis
   - Returns structured feature extraction results

3. **Model Info** (`/model_info`):
   - Updated to show multimodal v2 availability
   - Shows layout analyzer availability

---

## Dependencies

New dependencies added to `requirements.txt`:
- `cssutils>=2.6.0` - For CSS parsing
- `torchvision>=0.13.0` - For vision transforms (already likely installed)

Existing dependencies used:
- `beautifulsoup4` - HTML parsing (already in requirements)
- `torch` - Deep learning framework
- `transformers` - DistilBERT model
- `pillow` - Image processing

---

## Configuration

### Environment Variables

- `MULTIMODAL_MODEL_PATH`: Path to trained multimodal model checkpoint (optional)
- `VISION_DEVICE`: Device for vision models ('cuda' or 'cpu')

### Model Initialization

The multimodal model initializes with:
- Pretrained DistilBERT (from HuggingFace)
- Randomly initialized vision encoder (needs training)
- Falls back to CLIP if unavailable

---

## Usage Examples

### Using Multimodal Model Directly

```python
from PIL import Image
from api.vision.multimodal_model import MultimodalAnalyzer

# Initialize analyzer
analyzer = MultimodalAnalyzer()

# Analyze image + text
image = Image.open('screenshot.png')
text = "Subscribe now for only $9.99/month"
result = analyzer.detect(image, text)

print(f"Top detection: {result['top_detection']}")
print(f"All detections: {result['detections']}")
```

### Using Layout Analyzer

```python
from api.layout_analyzer import LayoutAnalyzer

analyzer = LayoutAnalyzer()

html = """
<html>
<body>
    <button>No thanks</button>
    <button>Subscribe Now</button>
    <div style="display:none">Auto-renewal enabled</div>
</body>
</html>
"""

features = analyzer.analyze_html(html)
print(f"Suspicious patterns: {features['suspicious_patterns']}")
```

### API Usage

**Vision Analysis** (includes multimodal):
```bash
curl -X POST http://localhost:5000/vision/analyze \
  -F "file=@screenshot.png"
```

**Layout Analysis**:
```bash
curl -X POST http://localhost:5000/analyze_layout \
  -H "Content-Type: application/json" \
  -d '{"html": "<html>...</html>", "css": "..."}'
```

---

## Future Enhancements

1. **Model Training**: Train the vision encoder on labeled dark pattern dataset
2. **Fine-tuning**: Fine-tune the entire multimodal model end-to-end
3. **More Patterns**: Add detection for additional dark patterns
4. **Performance**: Optimize model for faster inference
5. **Browser Integration**: Integrate layout analyzer into browser extension

---

## Notes

- The multimodal model's vision encoder starts untrained and will need fine-tuning
- The layout analyzer works with static HTML/CSS (doesn't execute JavaScript)
- Both analyzers are designed to complement existing CLIP-based detection
- Results are fused together for improved accuracy

