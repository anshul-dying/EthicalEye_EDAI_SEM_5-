"""
Ethical Eye API - Updated Flask API with DistilBERT and SHAP
Research Project: Explainable AI for Dark Pattern Detection

This API provides real-time dark pattern detection with transparent explanations
using DistilBERT and SHAP for user empowerment and education.
"""

import os
import json
import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import shap
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EthicalEyeAPI:
    """Main API class for Ethical Eye extension"""
    
    def __init__(self, model_path='models/ethical_eye/final_model'):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.explainer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and initialize components
        self.load_model()
        self.initialize_shap_explainer()
        
        # Pattern descriptions for user education
        self.pattern_descriptions = {
            'Urgency': 'Creates false time pressure to rush decisions',
            'Scarcity': 'Makes products appear limited in availability',
            'Social Proof': 'Uses fake social validation to influence behavior',
            'Misdirection': 'Deceptively guides users toward unintended actions',
            'Forced Action': 'Requires unnecessary tasks to complete simple actions',
            'Obstruction': 'Makes actions unnecessarily difficult',
            'Sneaking': 'Hides or obscures important information',
            'Hidden Costs': 'Conceals additional fees or charges',
            'Not Dark Pattern': 'Normal, non-manipulative content'
        }
        
        logger.info("Ethical Eye API initialized successfully!")
    
    def load_model(self):
        """Load the trained DistilBERT model"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            self.model = DistilBertForSequenceClassification.from_pretrained(self.model_path)
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_path)
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully!")
            logger.info(f"Number of labels: {self.model.config.num_labels}")
            logger.info(f"Labels: {list(self.model.config.id2label.values())}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def initialize_shap_explainer(self):
        """Initialize SHAP explainer"""
        try:
            logger.info("Initializing SHAP explainer...")
            
            # Create a wrapper function for the model
            def model_wrapper(texts):
                """Wrapper function for SHAP explainer"""
                if isinstance(texts, str):
                    texts = [texts]
                
                # Tokenize inputs
                inputs = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get predictions
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probabilities = torch.softmax(outputs.logits, dim=-1)
                
                return probabilities.cpu().numpy()
            
            # Initialize SHAP explainer
            self.explainer = shap.Explainer(model_wrapper, self.tokenizer)
            
            logger.info("SHAP explainer initialized successfully!")
            
        except Exception as e:
            logger.error(f"Error initializing SHAP explainer: {e}")
            raise
    
    def analyze_text(self, text, confidence_threshold=0.7):
        """Analyze a single text for dark patterns"""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                predicted_class_idx = torch.argmax(probabilities, dim=-1)
                confidence = torch.max(probabilities, dim=-1)[0].item()
            
            # Get predicted class
            predicted_class = self.model.config.id2label[predicted_class_idx.item()]
            
            # Determine if it's a dark pattern
            is_dark_pattern = predicted_class != 'Not Dark Pattern' and confidence > confidence_threshold
            
            # Generate SHAP explanation if it's a dark pattern
            explanation = None
            top_words = []
            
            if is_dark_pattern:
                try:
                    # Get SHAP values
                    shap_values = self.explainer([text])
                    
                    # Get tokens
                    tokens = self.tokenizer.tokenize(text)
                    
                    # Get feature importance scores
                    feature_importance = shap_values.values[0]
                    
                    # Get top contributing words
                    top_words = self.get_top_contributing_words(tokens, feature_importance, top_k=5)
                    
                    # Generate explanation
                    explanation = self.generate_explanation(
                        predicted_class, confidence, top_words
                    )
                    
                except Exception as e:
                    logger.warning(f"Error generating SHAP explanation: {e}")
                    explanation = f"This text is classified as '{predicted_class}' with {confidence:.1%} confidence."
            
            return {
                'category': predicted_class,
                'confidence': confidence,
                'is_dark_pattern': is_dark_pattern,
                'explanation': explanation,
                'top_words': top_words,
                'pattern_description': self.pattern_descriptions.get(predicted_class, ''),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return {
                'error': str(e),
                'category': 'Error',
                'confidence': 0.0,
                'is_dark_pattern': False
            }
    
    def get_top_contributing_words(self, tokens, shap_values, top_k=5):
        """Get top contributing words from SHAP values"""
        # Create word-score pairs
        word_scores = list(zip(tokens, shap_values))
        
        # Sort by absolute SHAP value
        word_scores.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Return top k words with their scores
        return [(word, float(score)) for word, score in word_scores[:top_k]]
    
    def generate_explanation(self, predicted_class, confidence, top_words):
        """Generate human-readable explanation"""
        # Get top words
        top_word_list = [word for word, score in top_words]
        
        # Generate explanation
        if predicted_class == 'Not Dark Pattern':
            explanation = f"This text appears to be normal content with {confidence:.1%} confidence."
        else:
            explanation = f"This text is classified as '{predicted_class}' with {confidence:.1%} confidence. "
            explanation += f"{self.pattern_descriptions[predicted_class]}. "
            if top_word_list:
                explanation += f"Key indicators include: {', '.join(top_word_list)}."
        
        return explanation
    
    def analyze_segments(self, segments, confidence_threshold=0.7):
        """Analyze multiple text segments"""
        results = []
        dark_pattern_count = 0
        
        for i, segment in enumerate(segments):
            try:
                # Analyze the text
                result = self.analyze_text(segment.get('text', ''), confidence_threshold)
                
                # Add segment metadata
                result['segment_id'] = i
                result['element_id'] = segment.get('element_id', f'segment_{i}')
                result['position'] = segment.get('position', {})
                
                results.append(result)
                
                # Count dark patterns
                if result['is_dark_pattern']:
                    dark_pattern_count += 1
                    
            except Exception as e:
                logger.error(f"Error analyzing segment {i}: {e}")
                results.append({
                    'segment_id': i,
                    'error': str(e),
                    'category': 'Error',
                    'confidence': 0.0,
                    'is_dark_pattern': False
                })
        
        return {
            'results': results,
            'dark_pattern_count': dark_pattern_count,
            'total_segments': len(segments),
            'analysis_timestamp': datetime.now().isoformat()
        }

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize API
try:
    api = EthicalEyeAPI()
    logger.info("Ethical Eye API ready!")
except Exception as e:
    logger.error(f"Failed to initialize API: {e}")
    api = None

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if api is None:
        return jsonify({'status': 'error', 'message': 'API not initialized'}), 500
    
    return jsonify({
        'status': 'healthy',
        'message': 'Ethical Eye API is running',
        'model_loaded': api.model is not None,
        'shap_loaded': api.explainer is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/analyze', methods=['POST'])
def analyze_segments():
    """Main endpoint for analyzing text segments"""
    if api is None:
        return jsonify({'error': 'API not initialized'}), 500
    
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        segments = data.get('segments', [])
        confidence_threshold = data.get('confidence_threshold', 0.7)
        
        if not segments:
            return jsonify({'error': 'No segments provided'}), 400
        
        logger.info(f"Analyzing {len(segments)} segments with confidence threshold {confidence_threshold}")
        
        # Analyze segments
        results = api.analyze_segments(segments, confidence_threshold)
        
        logger.info(f"Analysis complete. Found {results['dark_pattern_count']} dark patterns")
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error in analyze_segments: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_single', methods=['POST'])
def analyze_single():
    """Endpoint for analyzing a single text"""
    if api is None:
        return jsonify({'error': 'API not initialized'}), 500
    
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        text = data.get('text', '')
        confidence_threshold = data.get('confidence_threshold', 0.7)
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        logger.info(f"Analyzing single text: '{text[:50]}...'")
        
        # Analyze text
        result = api.analyze_text(text, confidence_threshold)
        
        logger.info(f"Analysis complete. Category: {result['category']}, Confidence: {result['confidence']:.3f}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in analyze_single: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/patterns', methods=['GET'])
def get_patterns():
    """Get information about all dark pattern types"""
    if api is None:
        return jsonify({'error': 'API not initialized'}), 500
    
    patterns = []
    for pattern_type, description in api.pattern_descriptions.items():
        patterns.append({
            'type': pattern_type,
            'description': description,
            'is_dark_pattern': pattern_type != 'Not Dark Pattern'
        })
    
    return jsonify({
        'patterns': patterns,
        'total_patterns': len(patterns),
        'dark_pattern_types': len([p for p in patterns if p['is_dark_pattern']])
    })

@app.route('/model_info', methods=['GET'])
def get_model_info():
    """Get information about the loaded model"""
    if api is None:
        return jsonify({'error': 'API not initialized'}), 500
    
    return jsonify({
        'model_path': api.model_path,
        'num_labels': api.model.config.num_labels,
        'labels': list(api.model.config.id2label.values()),
        'device': str(api.device),
        'shap_available': api.explainer is not None
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    if api is not None:
        logger.info("Starting Ethical Eye API server...")
        app.run(host='127.0.0.1', port=5000, debug=True, threaded=True)
    else:
        logger.error("Cannot start server - API not initialized")
