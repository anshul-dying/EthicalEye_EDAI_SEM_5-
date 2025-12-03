# Ethical Eye

DarkPatternLLM is a project aimed at detecting and combating dark patterns on websites using advanced Language Models (LLMs). This tool provides users with a more transparent and user-friendly online experience.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Features

### 1. Pattern Detection

The project leverages three state-of-the-art language models: RoBERTa, XLNet, and BERT to detect and highlight potential dark patterns on websites.

### 2. Dataset

A comprehensive dataset has been gathered from various sources to train and fine-tune the models for accurate pattern detection.

### 3. User Alerts

Receive real-time alerts when visiting a website that employs deceptive design practices.

### 4. Educational Resources

Access resources within the extension to learn more about dark patterns and how to protect yourself online.

### 5. Multimodal Screenshot Analyzer

Upload screenshots in the full-page results view to detect visual dark patterns such as disguised ads, fake scarcity pop-ups, or color manipulation. The analyzer combines CLIP embeddings, computer vision heuristics, and OCR+DistilBERT text cues to highlight suspicious regions directly on the screenshot.

## Installation

To install the Ethical Eye, follow these steps:

1. Download the extension from the [Chrome Web Store](#).
2. Open Google Chrome and navigate to the "Extensions" page (`chrome://extensions`).
3. Drag and drop the downloaded extension file onto the extensions page to install it.

## Usage

After installation, the Ethical Eye icon will appear in your browser toolbar. Simply visit any website, and the extension will automatically analyze the page for dark patterns. If a dark pattern is detected, you will receive a notification, and the relevant elements will be highlighted on the page.

### Running the Multimodal Pipeline

1. Install dependencies (`pip install -r requirements.txt`). Ensure system Tesseract is available for OCR.
2. Start the Flask API gateway (now serving both text and screenshot analysis):
   ```bash
   python api/ethical_eye_api.py
   ```
3. Open `app/results.html` (or the extension’s “view full results” link) and upload a screenshot. Bounding boxes and confidence scores will render on top of the image, while textual explanations show below.

For quick verification outside the UI, use the smoke test:

```bash
python vision_smoke_test.py --limit 5
```

This command runs sample images from `test/images` through the pipeline and prints the detections plus latency metrics.

## Contributing

We welcome contributions! If you want to contribute to the project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and submit a pull request.

---
