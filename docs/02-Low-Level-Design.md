# Low Level Design (LLD) - Ethical Eye Extension

## 1. Detailed Component Design

### 1.1 Chrome Extension Components

#### 1.1.1 Manifest Configuration
```json
{
  "manifest_version": 3,
  "name": "Ethical Eye",
  "version": "2.0.0",
  "description": "Detect and explain dark patterns with AI transparency",
  "permissions": [
    "activeTab",
    "tabs",
    "storage",
    "scripting"
  ],
  "host_permissions": ["<all_urls>"],
  "content_scripts": [{
    "matches": ["<all_urls>"],
    "js": ["js/content.js", "js/segmenter.js", "js/highlighter.js"],
    "css": ["css/highlight.css"],
    "run_at": "document_end"
  }],
  "background": {
    "service_worker": "js/background.js"
  },
  "action": {
    "default_popup": "popup/popup.html",
    "default_title": "Ethical Eye - Dark Pattern Detector"
  }
}
```

#### 1.1.2 Content Script Architecture
```javascript
// content.js - Main content script orchestrator
class EthicalEyeContentScript {
  constructor() {
    this.segmenter = new TextSegmenter();
    this.highlighter = new PatternHighlighter();
    this.apiClient = new APIClient();
    this.isAnalyzing = false;
  }

  async analyzePage() {
    if (this.isAnalyzing) return;
    this.isAnalyzing = true;
    
    try {
      const segments = this.segmenter.extractTextSegments();
      const results = await this.apiClient.analyzeSegments(segments);
      this.highlighter.highlightPatterns(results);
      this.updatePopupCount(results.darkPatternCount);
    } catch (error) {
      console.error('Analysis failed:', error);
    } finally {
      this.isAnalyzing = false;
    }
  }
}
```

#### 1.1.3 Text Segmentation Module
```javascript
// segmenter.js - Advanced text segmentation
class TextSegmenter {
  constructor() {
    this.blockElements = ['div', 'section', 'article', 'aside', 'nav', 'header', 'footer'];
    this.ignoredElements = ['script', 'style', 'noscript', 'br', 'hr'];
    this.minTextLength = 10;
    this.maxTextLength = 512;
  }

  extractTextSegments() {
    const segments = [];
    const walker = document.createTreeWalker(
      document.body,
      NodeFilter.SHOW_ELEMENT,
      this.elementFilter.bind(this)
    );

    let element;
    while (element = walker.nextNode()) {
      const text = this.extractTextFromElement(element);
      if (this.isValidSegment(text)) {
        segments.push({
          element: element,
          text: text,
          position: this.getElementPosition(element),
          context: this.getElementContext(element)
        });
      }
    }

    return segments;
  }

  isValidSegment(text) {
    return text.length >= this.minTextLength && 
           text.length <= this.maxTextLength &&
           !this.isNavigationText(text);
  }
}
```

#### 1.1.4 Pattern Highlighter Module
```javascript
// highlighter.js - Visual highlighting system
class PatternHighlighter {
  constructor() {
    this.highlightClass = 'ethical-eye-highlight';
    this.tooltipClass = 'ethical-eye-tooltip';
    this.confidenceThreshold = 0.7;
  }

  highlightPatterns(results) {
    results.forEach(result => {
      if (result.isDarkPattern && result.confidence > this.confidenceThreshold) {
        this.createHighlight(result);
      }
    });
  }

  createHighlight(result) {
    const highlight = document.createElement('div');
    highlight.className = this.highlightClass;
    highlight.dataset.patternType = result.category;
    highlight.dataset.confidence = result.confidence;
    highlight.dataset.explanation = JSON.stringify(result.explanation);
    
    this.addTooltip(highlight, result);
    this.insertHighlight(result.element, highlight);
  }

  addTooltip(highlight, result) {
    const tooltip = document.createElement('div');
    tooltip.className = this.tooltipClass;
    tooltip.innerHTML = this.generateTooltipHTML(result);
    highlight.appendChild(tooltip);
  }
}
```

### 1.2 Backend API Design

#### 1.2.1 Flask API Structure
```python
# api/app.py - Main API server
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import shap
import numpy as np

app = Flask(__name__)
CORS(app)

class EthicalEyeAPI:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.explainer = None
        self.load_models()
    
    def load_models(self):
        # Load DistilBERT model
        model_path = "models/distilbert-dark-patterns"
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        
        # Initialize SHAP explainer
        self.explainer = shap.Explainer(self.model, self.tokenizer)
    
    def analyze_text(self, text):
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1)
            confidence = torch.max(probabilities, dim=-1)[0].item()
        
        # Generate SHAP explanations
        shap_values = self.explainer([text])
        explanation = self.process_shap_explanation(shap_values, text)
        
        return {
            'category': self.model.config.id2label[predicted_class.item()],
            'confidence': confidence,
            'explanation': explanation,
            'is_dark_pattern': predicted_class.item() != 8  # 8 = "Not Dark Pattern"
        }

api = EthicalEyeAPI()

@app.route('/analyze', methods=['POST'])
def analyze_segments():
    data = request.get_json()
    segments = data.get('segments', [])
    
    results = []
    for segment in segments:
        result = api.analyze_text(segment['text'])
        result['element_id'] = segment.get('element_id')
        result['position'] = segment.get('position')
        results.append(result)
    
    return jsonify({
        'results': results,
        'dark_pattern_count': sum(1 for r in results if r['is_dark_pattern'])
    })
```

#### 1.2.2 SHAP Explanation Processor
```python
# api/explainer.py - SHAP explanation processing
class SHAPExplanationProcessor:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.explainer = shap.Explainer(model, tokenizer)
    
    def process_explanation(self, text, shap_values):
        # Extract tokens and their importance scores
        tokens = self.tokenizer.tokenize(text)
        importance_scores = shap_values.values[0]
        
        # Get top contributing words
        top_words = self.get_top_contributing_words(tokens, importance_scores)
        
        # Generate human-readable explanation
        explanation = self.generate_explanation(top_words, text)
        
        return {
            'top_words': top_words,
            'explanation': explanation,
            'confidence_factors': self.analyze_confidence_factors(importance_scores)
        }
    
    def get_top_contributing_words(self, tokens, scores, top_k=5):
        # Sort tokens by importance score
        token_scores = list(zip(tokens, scores))
        token_scores.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return token_scores[:top_k]
    
    def generate_explanation(self, top_words, text):
        # Generate contextual explanation based on top words
        word_list = [word for word, score in top_words]
        explanation = f"Key indicators: {', '.join(word_list)}"
        
        return explanation
```

### 1.3 Machine Learning Model Design

#### 1.3.1 DistilBERT Model Architecture
```python
# models/distilbert_classifier.py - Model definition
from transformers import DistilBertForSequenceClassification, DistilBertConfig
import torch
import torch.nn as nn

class DarkPatternClassifier(nn.Module):
    def __init__(self, num_labels=9, dropout_rate=0.1):
        super().__init__()
        self.distilbert = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=num_labels
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return {
            'logits': logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions
        }
```

#### 1.3.2 Model Training Pipeline
```python
# training/train_distilbert.py - Training script
from transformers import TrainingArguments, Trainer
from datasets import Dataset
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class DarkPatternTrainer:
    def __init__(self, model, tokenizer, train_dataset, val_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
    
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(self):
        training_args = TrainingArguments(
            output_dir="./distilbert-dark-patterns",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.compute_metrics,
        )
        
        trainer.train()
        return trainer
```

### 1.4 Database Schema

#### 1.4.1 User Study Data Schema
```sql
-- user_study_data.sql
CREATE TABLE user_study_sessions (
    session_id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    browser_info JSON,
    device_info JSON
);

CREATE TABLE pattern_detections (
    detection_id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) REFERENCES user_study_sessions(session_id),
    url VARCHAR(500) NOT NULL,
    pattern_type VARCHAR(50) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    user_feedback BOOLEAN,
    explanation_helpful BOOLEAN,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE user_responses (
    response_id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) REFERENCES user_study_sessions(session_id),
    question_type VARCHAR(50) NOT NULL,
    response_text TEXT,
    response_rating INTEGER CHECK (response_rating >= 1 AND response_rating <= 5),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 1.5 Configuration Management

#### 1.5.1 Environment Configuration
```python
# config/settings.py - Configuration management
import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ModelConfig:
    model_name: str = "distilbert-base-uncased"
    max_length: int = 512
    num_labels: int = 9
    confidence_threshold: float = 0.7
    batch_size: int = 16

@dataclass
class APIConfig:
    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = True
    cors_origins: list = None

@dataclass
class ExtensionConfig:
    version: str = "2.0.0"
    min_chrome_version: str = "88.0.0"
    update_url: str = "https://chrome.google.com/webstore"
    analytics_enabled: bool = False

class Config:
    def __init__(self):
        self.model = ModelConfig()
        self.api = APIConfig()
        self.extension = ExtensionConfig()
        self.load_from_env()
    
    def load_from_env(self):
        # Load configuration from environment variables
        self.model.confidence_threshold = float(
            os.getenv('CONFIDENCE_THRESHOLD', '0.7')
        )
        self.api.port = int(os.getenv('API_PORT', '5000'))
        self.extension.analytics_enabled = os.getenv(
            'ANALYTICS_ENABLED', 'false'
        ).lower() == 'true'
```

### 1.6 Error Handling and Logging

#### 1.6.1 Error Handling Strategy
```python
# utils/error_handler.py - Comprehensive error handling
import logging
from functools import wraps
from flask import jsonify
import traceback

class EthicalEyeError(Exception):
    """Base exception for Ethical Eye application"""
    pass

class ModelLoadError(EthicalEyeError):
    """Raised when model fails to load"""
    pass

class AnalysisError(EthicalEyeError):
    """Raised when text analysis fails"""
    pass

class APIError(EthicalEyeError):
    """Raised when API request fails"""
    pass

def handle_errors(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ModelLoadError as e:
            logging.error(f"Model load error: {str(e)}")
            return jsonify({'error': 'Model unavailable'}), 503
        except AnalysisError as e:
            logging.error(f"Analysis error: {str(e)}")
            return jsonify({'error': 'Analysis failed'}), 500
        except APIError as e:
            logging.error(f"API error: {str(e)}")
            return jsonify({'error': 'API request failed'}), 400
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            logging.error(traceback.format_exc())
            return jsonify({'error': 'Internal server error'}), 500
    return decorated_function
```

### 1.7 Testing Strategy

#### 1.7.1 Unit Testing Framework
```python
# tests/test_models.py - Model testing
import unittest
import torch
from models.dark_pattern_classifier import DarkPatternClassifier

class TestDarkPatternClassifier(unittest.TestCase):
    def setUp(self):
        self.model = DarkPatternClassifier(num_labels=9)
        self.sample_text = "Hurry! Limited time offer - only 2 left in stock!"
    
    def test_model_forward_pass(self):
        # Test model forward pass
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # Sample token IDs
        attention_mask = torch.tensor([[1, 1, 1, 1, 1]])
        
        output = self.model(input_ids, attention_mask)
        self.assertIsInstance(output['logits'], torch.Tensor)
        self.assertEqual(output['logits'].shape, (1, 9))
    
    def test_confidence_scoring(self):
        # Test confidence score calculation
        logits = torch.tensor([[0.1, 0.2, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        probabilities = torch.softmax(logits, dim=-1)
        confidence = torch.max(probabilities, dim=-1)[0].item()
        
        self.assertGreater(confidence, 0.5)
        self.assertLessEqual(confidence, 1.0)

# tests/test_api.py - API testing
import unittest
from unittest.mock import patch, MagicMock
from api.app import app

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
    
    @patch('api.app.api.analyze_text')
    def test_analyze_endpoint(self, mock_analyze):
        mock_analyze.return_value = {
            'category': 'Urgency',
            'confidence': 0.85,
            'explanation': 'Key indicators: hurry, limited',
            'is_dark_pattern': True
        }
        
        response = self.app.post('/analyze', json={
            'segments': [{'text': 'Hurry! Limited time offer!'}]
        })
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('results', data)
        self.assertEqual(data['results'][0]['category'], 'Urgency')
```

### 1.8 Performance Optimization

#### 1.8.1 Model Optimization
```python
# optimization/model_optimizer.py - Model optimization
import torch
from torch.quantization import quantize_dynamic
import onnx
from onnxruntime import InferenceSession

class ModelOptimizer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def quantize_model(self):
        """Quantize model for faster inference"""
        quantized_model = quantize_dynamic(
            self.model, 
            {torch.nn.Linear}, 
            dtype=torch.qint8
        )
        return quantized_model
    
    def convert_to_onnx(self, output_path):
        """Convert model to ONNX format for optimization"""
        dummy_input = torch.randint(0, 1000, (1, 512))
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input_ids'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size'},
                'logits': {0: 'batch_size'}
            }
        )
    
    def optimize_for_inference(self):
        """Optimize model for inference"""
        self.model.eval()
        with torch.no_grad():
            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
```

---

*This Low Level Design document provides detailed implementation specifications for each component of the Ethical Eye extension, including code examples, data structures, and implementation strategies.*
