# Generating Research Paper Figures

This directory contains scripts to generate the required figures for the research paper.

## Required Figures

1. **architecture.png** - System architecture diagram
2. **shap_example.png** - SHAP explanation example on a webpage

## Prerequisites

Install required Python packages:
```bash
pip install matplotlib numpy
```

## Generating Figures

### Option 1: Generate Both Figures (Recommended)
```bash
python generate_figures.py
```

This will generate both `architecture.png` and `shap_example.png` in the current directory.

### Option 2: Generate Individually
```bash
python generate_architecture.py
python generate_shap_example.py
```

## Figure Descriptions

### architecture.png
Shows the system architecture with:
- **Client Side**: Chrome Extension with Content Script, Background Service Worker, and Popup UI
- **Server Side**: Flask Backend with API, DistilBERT Model, and SHAP Explainer
- **Data Flow**: HTTP POST requests and JSON responses

### shap_example.png
Demonstrates SHAP explanations on a mock e-commerce page:
- Highlighted words ("Hurry", "Only", "2 left") with SHAP values
- Tooltip showing pattern type, confidence, and key words
- Visual representation of how SHAP highlights manipulative language

## Troubleshooting

If you encounter errors:
1. Ensure matplotlib is installed: `pip install matplotlib`
2. Check Python version (3.7+ recommended)
3. Run with verbose output: `python generate_figures.py 2>&1 | tee output.log`

## Output

Both scripts generate high-resolution PNG files (300 DPI) suitable for publication.

