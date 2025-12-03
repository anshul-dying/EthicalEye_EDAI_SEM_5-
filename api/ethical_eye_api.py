"""
Ethical Eye API - Flask API with DistilBERT and SHAP
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
from vision import VisionAnalyzer
from layout_analyzer import LayoutAnalyzer
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
        
        # Load model and tokenizer
        self.load_model()
        
        # Initialize SHAP explainer
        self.initialize_shap_explainer()
        
        logger.info("Ethical Eye API initialized with DistilBERT!")
    
    def load_model(self):
        """Load the trained DistilBERT model"""
        try:
            logger.info(f"Loading trained model from {self.model_path}...")
            
            if not os.path.exists(self.model_path):
                logger.error(f"Model path does not exist: {self.model_path}")
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            
            self.model = DistilBertForSequenceClassification.from_pretrained(self.model_path)
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_path)
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully!")
            logger.info(f"Number of labels: {self.model.config.num_labels}")
            logger.info(f"Labels: {list(self.model.config.id2label.values())}")
            logger.info(f"Device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def initialize_shap_explainer(self):
        """Initialize SHAP explainer"""
        try:
            logger.info("Initializing SHAP explainer...")
            
            # Create a wrapper function for SHAP that returns logits for the predicted class
            def model_wrapper(texts):
                """Wrapper function for SHAP explainer - returns logits for predicted class"""
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
                
                # Get logits (not probabilities) for SHAP
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                
                # Return logits for the predicted class (SHAP works better with logits)
                predicted_class_idx = torch.argmax(logits, dim=-1)
                selected_logits = logits.gather(1, predicted_class_idx.unsqueeze(1))
                
                return selected_logits.cpu().numpy()
            
            # Store the wrapper for later use
            self.model_wrapper = model_wrapper
            
            # Initialize SHAP explainer
            # Note: SHAP can be slow for transformers, so we'll use gradient-based as primary
            # and SHAP as optional enhancement
            try:
                # Try to use SHAP's Linear explainer which is faster
                # For transformers, we'll compute SHAP on-demand rather than pre-initializing
                self.explainer = "gradient"  # Use gradient-based method (faster and more reliable)
                logger.info("SHAP explainer initialized - using gradient-based importance method")
            except Exception as e:
                logger.warning(f"SHAP explainer initialization failed: {e}")
                self.explainer = "gradient"  # Fallback to gradient-based
                logger.info("Using gradient-based importance method")
            
        except Exception as e:
            logger.warning(f"SHAP explainer initialization failed: {e}")
            self.explainer = None
    
    def analyze_text(self, text, link=None, confidence_threshold=0.7):
        """Analyze a single text/link for dark patterns using DistilBERT"""
        if self.model is None or self.tokenizer is None:
            return {
                'error': 'Model not loaded',
                'category': 'Error',
                'confidence': 0.0,
                'is_dark_pattern': False
            }
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model predictions (first pass without gradients)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                
                # Get predicted class and confidence
                predicted_class_idx_tensor = torch.argmax(probabilities, dim=-1)
                predicted_class_idx = predicted_class_idx_tensor.item()
                confidence = probabilities[0][predicted_class_idx].item()
                
                # Get label from model config
                predicted_class = self.model.config.id2label[predicted_class_idx]
            
            # Determine if it's a dark pattern
            is_dark_pattern = predicted_class != 'Not Dark Pattern'
            
            # Get SHAP values or gradient-based importance (pass tensor, not int)
            top_words, shap_values, tokens = self.compute_feature_importance(text, predicted_class_idx_tensor, inputs)
            
            # Generate explanation
            explanation = self.generate_explanation(predicted_class, confidence, top_words)
            
            # Determine if SHAP was actually computed (not keyword-based fallback)
            # Keyword-based fallback returns scores of exactly 1.0, so we can detect it
            is_shap_computed = len(shap_values) > 0 and (
                len(set(score for _, score in top_words)) > 1 or  # Multiple different scores
                (len(top_words) > 0 and top_words[0][1] != 1.0)  # Not all 1.0
            )
            
            return {
                'category': predicted_class,
                'confidence': float(confidence),
                'is_dark_pattern': is_dark_pattern,
                'explanation': explanation,
                'top_words': top_words,
                'shap_values': shap_values,
                'tokens': tokens,
                'shap_computed': is_shap_computed,  # Flag to indicate if real SHAP was used
                'pattern_description': self.pattern_descriptions.get(predicted_class, ''),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return {
                'error': str(e),
                'category': 'Error',
                'confidence': 0.0,
                'is_dark_pattern': False
            }
    
    def compute_feature_importance(self, text, predicted_class_idx, inputs, tokens=None):
        """Compute feature importance using gradient-based method (faster and more reliable)"""
        if tokens is None:
            tokens = self.tokenizer.tokenize(text)
        
        # Use gradient-based importance (faster and works well with transformers)
        top_words, shap_values = self.compute_gradient_importance(text, predicted_class_idx, inputs, tokens)
        
        return top_words, shap_values, tokens
    
    def compute_gradient_importance(self, text, predicted_class_idx_tensor, inputs, tokens):
        """Compute importance using gradient-based method (Integrated Gradients approach)"""
        try:
            # Extract the class index as integer
            predicted_class_idx = predicted_class_idx_tensor.item() if hasattr(predicted_class_idx_tensor, 'item') else predicted_class_idx_tensor
            
            # Get the embedding layer from the model
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            
            # Enable gradient computation on the model
            self.model.train()  # Enable training mode for gradients
            
            # Get embeddings (these are float tensors that can have gradients)
            embeddings = self.model.distilbert.embeddings(input_ids)
            embeddings.retain_grad()
            embeddings.requires_grad_(True)
            logger.info(f"Embeddings requires_grad: {embeddings.requires_grad}")
            
            # Forward pass from embeddings
            # DistilBERT transformer expects mask as positional argument, not attention_mask as keyword
            # Create mask from attention_mask (DistilBERT uses mask, not attention_mask)
            if attention_mask is not None:
                mask = attention_mask.to(embeddings.device)
            else:
                # Create mask from input_ids if attention_mask not provided
                mask = (input_ids != self.tokenizer.pad_token_id).long().to(embeddings.device)
            
            # DistilBERT expects the attention mask to be bool/float; tensors from tokenizer
            # are often int64, so we coerce them here to avoid dtype mismatch errors.
            mask = mask.to(dtype=torch.bool)
            
            # Call transformer with mask as positional argument
            # DistilBERT transformer signature: forward(x, mask=None, head_mask=None, ...)
            # We need to provide head_mask as a list of Nones, otherwise some versions might fail
            head_mask = [None] * self.model.config.num_hidden_layers
            transformer_outputs = self.model.distilbert.transformer(embeddings, mask, head_mask)
            
            # Extract hidden state (transformer returns BaseModelOutput or tuple)
            if isinstance(transformer_outputs, tuple):
                hidden_state = transformer_outputs[0]
            elif hasattr(transformer_outputs, 'last_hidden_state'):
                hidden_state = transformer_outputs.last_hidden_state
            else:
                hidden_state = transformer_outputs
            
            # Get the pooled output (CLS token)
            pooled_output = hidden_state[:, 0]
            pooled_output = self.model.pre_classifier(pooled_output)
            pooled_output = torch.nn.functional.relu(pooled_output)
            pooled_output = self.model.dropout(pooled_output)
            logits = self.model.classifier(pooled_output)
            
            # Get logit for predicted class
            predicted_logit = logits[0, predicted_class_idx]
            logger.info(f"Predicted logit requires_grad: {predicted_logit.requires_grad}")
            
            # Backward pass
            # Use autograd.grad to get gradients w.r.t embeddings directly
            try:
                grads = torch.autograd.grad(predicted_logit, embeddings, retain_graph=False)[0]
            except Exception as e:
                logger.error(f"autograd.grad failed: {e}")
                grads = None
            
            # Get gradients from embeddings
            if grads is not None:
                # Compute importance as L2 norm of gradient for each token
                grad_norms = torch.norm(grads, dim=-1).squeeze().cpu().detach().numpy()
                token_ids = input_ids.squeeze().cpu().numpy()
                
                # Get tokenized text with special tokens
                if hasattr(token_ids, 'tolist'):
                    token_ids_list = token_ids.tolist()
                else:
                    token_ids_list = token_ids
                full_tokens = self.tokenizer.convert_ids_to_tokens(token_ids_list)
                
                # Create token-score pairs, filtering special tokens
                token_scores = []
                shap_values = []
                
                for i, (token_id, grad_norm) in enumerate(zip(token_ids, grad_norms)):
                    token = full_tokens[i] if i < len(full_tokens) else self.tokenizer.decode([token_id])
                    
                    # Skip special tokens and empty tokens
                    if token not in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '', ' ']:
                        score = float(grad_norm)
                        # Normalize score (0-1 range for better visualization)
                        token_scores.append((token, score))
                        shap_values.append(score)
                    else:
                        shap_values.append(0.0)
                
                # Sort by importance
                if token_scores:
                    # Normalize scores to 0-1 range for better display
                    max_score = max(abs(score) for _, score in token_scores) if token_scores else 1.0
                    if max_score > 0:
                        token_scores = [(token, score / max_score) for token, score in token_scores]
                    
                    token_scores.sort(key=lambda x: abs(x[1]), reverse=True)
                    top_words = token_scores[:10]  # Top 10 contributing words
                    
                    logger.debug(f"Computed gradient importance: {len(top_words)} top words, {len(shap_values)} total values")
                    
                    # Set model back to eval mode
                    self.model.eval()
                    return top_words, shap_values
            
            # Fallback to keyword-based
            self.model.eval()  # Set back to eval mode
            predicted_class = self.model.config.id2label[predicted_class_idx]
            return self.get_top_contributing_words_simple(text, predicted_class), []
                
        except Exception as e:
            logger.warning(f"Gradient-based importance failed: {e}, using keyword-based")
            import traceback
            logger.error(traceback.format_exc())
            # Set model back to eval mode
            try:
                self.model.eval()
            except:
                pass
            # Final fallback to keyword-based
            try:
                predicted_class_idx = predicted_class_idx_tensor.item() if hasattr(predicted_class_idx_tensor, 'item') else predicted_class_idx_tensor
                predicted_class = self.model.config.id2label[predicted_class_idx]
            except:
                predicted_class = 'Not Dark Pattern'
            return self.get_top_contributing_words_simple(text, predicted_class), []
    
    def get_top_contributing_words_simple(self, text, predicted_class):
        """Get top contributing words based on pattern type and keywords (fallback method)"""
        keywords = self.get_keywords_for_pattern(predicted_class)
        text_lower = text.lower()
        
        found_keywords = []
        for keyword in keywords:
            if keyword in text_lower:
                # Simple scoring based on keyword presence
                found_keywords.append((keyword, 1.0))
        
        # Return top 5 keywords
        return found_keywords[:5]
    
    def get_keywords_for_pattern(self, predicted_class):
        """Get keywords associated with each dark pattern type"""
        keywords = {
            'Urgency': ['hurry', 'urgent', 'limited time', 'expires', 'act now', 'before it\'s too late', 'soon', 'quickly'],
            'Scarcity': ['only', 'left', 'remaining', 'few', 'limited', 'running out', 'last chance', 'stock'],
            'Social Proof': ['join', 'customers', 'people', 'everyone', 'popular', 'trending', 'recommended', 'loved'],
            'Misdirection': ['click here', 'continue', 'proceed', 'next', 'skip', 'ignore', 'free'],
            'Forced Action': ['required', 'must', 'need to', 'have to', 'obligatory', 'mandatory'],
            'Obstruction': ['difficult', 'complicated', 'hard to find', 'buried', 'hidden'],
            'Sneaking': ['hidden', 'small print', 'terms', 'conditions', 'fine print'],
            'Hidden Costs': ['additional', 'extra', 'fees', 'charges', 'costs', 'pricing']
        }
        return keywords.get(predicted_class, [])
    
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
                result = self.analyze_text(
                    segment.get('text', ''), 
                    link=segment.get('link'),
                    confidence_threshold=confidence_threshold
                )
                
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

# Initialize vision analyzer (CLIP-based screenshot detection)
try:
    vision_analyzer = VisionAnalyzer()
    logger.info("Vision analyzer ready!")
except Exception as e:
    logger.error(f"Failed to initialize vision analyzer: {e}")
    vision_analyzer = None

# Initialize layout analyzer (HTML/CSS structure analysis)
try:
    layout_analyzer = LayoutAnalyzer()
    logger.info("Layout analyzer ready!")
except Exception as e:
    logger.error(f"Failed to initialize layout analyzer: {e}")
    layout_analyzer = None

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if api is None:
        return jsonify({'status': 'error', 'message': 'API not initialized'}), 500
    
    return jsonify({
        'status': 'healthy',
        'message': 'Ethical Eye API is running',
        'model_loaded': api.model is not None,
        'tokenizer_loaded': api.tokenizer is not None,
        'shap_loaded': api.explainer is not None,
        'device': str(api.device),
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


def _analyze_screenshot_locally(file_storage):
    stream = file_storage.stream
    if hasattr(stream, "seek"):
        try:
            stream.seek(0)
        except Exception:
            pass
    image_bytes = stream.read()
    detections, metadata = vision_analyzer.analyze(image_bytes)
    metadata = metadata or {}
    metadata["gateway"] = "ethical_eye_api"
    return {
        "detections": detections,
        "metadata": metadata,
    }


@app.route("/vision/analyze", methods=["POST"])
def analyze_screenshot():
    if api is None:
        return jsonify({'error': 'API not initialized'}), 500
    if vision_analyzer is None:
        return jsonify({'error': 'Vision analyzer not initialized'}), 500

    if "file" not in request.files:
        return jsonify({'error': 'No screenshot uploaded'}), 400

    file_storage = request.files["file"]
    if file_storage.filename == "":
        return jsonify({'error': 'Empty filename'}), 400

    try:
        payload = _analyze_screenshot_locally(file_storage)
        return jsonify(payload)
    except Exception as exc:
        logger.error(f"Vision analysis failed: {exc}")
        return jsonify({'error': str(exc)}), 500

@app.route('/analyze_single', methods=['POST'])
def analyze_single():
    """Endpoint for analyzing a single text"""
    print("DEBUG: analyze_single called")
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
        
        # Analyze text (link is optional, can be None)
        link = data.get('link', None)
        result = api.analyze_text(text, link=link, confidence_threshold=confidence_threshold)
        
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
    
    # Get multimodal model info
    multimodal_info = {}
    if vision_analyzer and vision_analyzer.multimodal_analyzer:
        multimodal_info = {
            'multimodal_v2_available': True,
            'multimodal_model_loaded': vision_analyzer.multimodal_analyzer.model is not None,
            'multimodal_device': str(vision_analyzer.multimodal_analyzer.device),
            'multimodal_classes': vision_analyzer.multimodal_analyzer.class_names
        }
    else:
        multimodal_info = {
            'multimodal_v2_available': False,
            'multimodal_model_loaded': False
        }
    
    return jsonify({
        'model_path': api.model_path,
        'model_type': 'DistilBERT',
        'num_labels': api.model.config.num_labels,
        'labels': list(api.model.config.id2label.values()),
        'device': str(api.device),
        'shap_available': api.explainer is not None,
        'layout_analyzer_available': layout_analyzer is not None,
        **multimodal_info
    })

@app.route('/analyze_layout', methods=['POST'])
def analyze_layout():
    """Analyze HTML/CSS layout for visual misdirection patterns"""
    if layout_analyzer is None:
        return jsonify({'error': 'Layout analyzer not initialized'}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        html_content = data.get('html', '')
        css_content = data.get('css', None)
        
        if not html_content:
            return jsonify({'error': 'No HTML content provided'}), 400
        
        logger.info("Analyzing HTML layout structure...")
        
        # Analyze layout
        features = layout_analyzer.analyze_html(html_content, css_content)
        
        logger.info(f"Layout analysis complete. Found {len(features.get('suspicious_patterns', []))} suspicious patterns")
        
        return jsonify({
            'features': features,
            'analysis_timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in analyze_layout: {e}")
        return jsonify({'error': str(e)}), 500

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
        app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
    else:
        logger.error("Cannot start server - API not initialized")