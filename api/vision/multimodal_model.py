"""
Multimodal Model v2: MobileViT + DistilBERT
Hybrid Text + Vision Model for Dark Pattern Detection

Detects:
- Color manipulation
- Deceptive UI contrast
- Hidden subscription checkboxes
- Fake progress bars
"""

import os
import logging
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import DistilBertModel, DistilBertTokenizerFast
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class MobileViTEncoder(nn.Module):
    """
    Lightweight MobileViT encoder for vision features.
    Uses a simplified MobileViT architecture optimized for UI element detection.
    """
    
    def __init__(self, embed_dim=256, num_heads=4, num_layers=2):
        super().__init__()
        
        # Simplified MobileViT-like architecture
        # Initial conv layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Adaptive pooling to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        # Flatten and project to embed_dim
        self.flatten = nn.Flatten()
        self.projection = nn.Linear(128 * 8 * 8, embed_dim)
        
    def forward(self, x):
        # x: (batch_size, 3, H, W)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.projection(x)
        return x


class MultimodalDarkPatternModel(nn.Module):
    """
    Hybrid Multimodal Model combining MobileViT (vision) + DistilBERT (text)
    """
    
    def __init__(
        self,
        vision_embed_dim=256,
        text_embed_dim=768,  # DistilBERT base dimension
        fusion_dim=512,
        num_classes=5,  # Color Manipulation, Deceptive Contrast, Hidden Checkbox, Fake Progress Bar, Normal
        dropout=0.1
    ):
        super().__init__()
        
        # Vision encoder (MobileViT-like)
        self.vision_encoder = MobileViTEncoder(embed_dim=vision_embed_dim)
        
        # Text encoder (DistilBERT)
        self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.text_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        
        # Project text embeddings to match vision dimension
        self.text_projection = nn.Linear(text_embed_dim, vision_embed_dim)
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(vision_embed_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classifier head
        self.classifier = nn.Linear(fusion_dim // 2, num_classes)
        
        # Image preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def encode_vision(self, images):
        """Encode images using vision encoder - handles PIL Images"""
        # Handle PIL Images only (for inference)
        # For training, tensors are passed directly to forward()
        if isinstance(images, list):
            # Convert PIL Images to tensors
            images = torch.stack([self.image_transform(img) for img in images])
        elif isinstance(images, Image.Image):
            # Single PIL Image
            images = self.image_transform(images).unsqueeze(0)
        else:
            # Already a tensor, use as is
            pass
        
        device = next(self.vision_encoder.parameters()).device
        if isinstance(images, torch.Tensor):
            if images.device != device:
                images = images.to(device)
            return self.vision_encoder(images)
        else:
            # Fallback: convert to tensor
            images = torch.stack([self.image_transform(img) for img in images])
            images = images.to(device)
            return self.vision_encoder(images)
    
    def encode_text(self, texts, use_gradients=True):
        """Encode texts using DistilBERT"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        inputs = self.text_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        device = next(self.text_encoder.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embeddings (allow gradients during training)
        if use_gradients:
            outputs = self.text_encoder(**inputs)
        else:
            with torch.no_grad():
                outputs = self.text_encoder(**inputs)
        
        # Use [CLS] token embedding
        text_embeds = outputs.last_hidden_state[:, 0, :]
        
        # Project to vision dimension
        text_embeds = self.text_projection(text_embeds)
        return text_embeds
    
    def forward(self, images, texts, use_gradients=True):
        """
        Forward pass combining vision and text
        
        Args:
            images: List of PIL Images, single PIL Image, or tensor of shape (batch, 3, H, W)
            texts: List of strings or single string
            use_gradients: Whether to allow gradients (True for training, False for inference)
        """
        # Encode vision - handles both PIL and tensors
        if isinstance(images, torch.Tensor):
            # Already a tensor batch, use directly
            device = next(self.vision_encoder.parameters()).device
            if images.device != device:
                images = images.to(device)
            vision_features = self.vision_encoder(images)
        else:
            # PIL Images - use encode_vision which handles conversion
            vision_features = self.encode_vision(images)
        
        # Encode text (allow gradients during training)
        text_features = self.encode_text(texts, use_gradients=use_gradients)
        
        # Ensure batch sizes match
        if vision_features.shape[0] != text_features.shape[0]:
            if vision_features.shape[0] == 1:
                vision_features = vision_features.repeat(text_features.shape[0], 1)
            elif text_features.shape[0] == 1:
                text_features = text_features.repeat(vision_features.shape[0], 1)
        
        # Concatenate features
        fused_features = torch.cat([vision_features, text_features], dim=1)
        
        # Fusion layers
        fused = self.fusion(fused_features)
        
        # Classify
        logits = self.classifier(fused)
        
        return logits
    
    def predict(self, images, texts, return_probs=True):
        """Make predictions with probabilities"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(images, texts)
            probs = torch.softmax(logits, dim=-1)
            
            if return_probs:
                return probs.cpu().numpy()
            return logits.cpu().numpy()


class MultimodalAnalyzer:
    """
    High-level interface for multimodal dark pattern detection
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.class_names = [
            'Color Manipulation',
            'Deceptive UI Contrast',
            'Hidden Subscription Checkbox',
            'Fake Progress Bar',
            'Normal'
        ]
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Initialize with pretrained DistilBERT, random vision encoder
            logger.info("Initializing new multimodal model...")
            self.model = MultimodalDarkPatternModel().to(self.device)
            logger.info("Model initialized (vision encoder needs training)")
    
    def load_model(self, model_path: str):
        """Load trained model from disk"""
        try:
            logger.info(f"Loading multimodal model from {model_path}...")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model = MultimodalDarkPatternModel().to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def detect(self, image: Image.Image, text: str = "") -> Dict[str, any]:
        """
        Detect dark patterns in image+text combination
        
        Args:
            image: PIL Image
            text: Optional text content from OCR or HTML
        
        Returns:
            Dictionary with detection results
        """
        if self.model is None:
            return {
                'error': 'Model not loaded',
                'detections': [],
                'confidence': 0.0
            }
        
        try:
            # Get predictions
            probs = self.model.predict([image], [text] if text else [""])
            probs = probs[0]  # Get first (and only) prediction
            
            # Get top prediction
            top_class_idx = np.argmax(probs)
            top_class = self.class_names[top_class_idx]
            confidence = float(probs[top_class_idx])
            
            # Check if it's a dark pattern
            is_dark_pattern = top_class != 'Normal'
            
            # Get all detections above threshold
            detections = []
            threshold = 0.3
            for idx, (class_name, prob) in enumerate(zip(self.class_names, probs)):
                if prob > threshold and class_name != 'Normal':
                    detections.append({
                        'pattern': class_name,
                        'confidence': float(prob),
                        'is_dark_pattern': True
                    })
            
            return {
                'detections': detections,
                'top_detection': {
                    'pattern': top_class,
                    'confidence': confidence,
                    'is_dark_pattern': is_dark_pattern
                },
                'all_scores': {name: float(prob) for name, prob in zip(self.class_names, probs)}
            }
            
        except Exception as e:
            logger.error(f"Error in multimodal detection: {e}")
            return {
                'error': str(e),
                'detections': [],
                'confidence': 0.0
            }
    
    def detect_batch(self, images: List[Image.Image], texts: List[str]) -> List[Dict]:
        """Detect dark patterns in batch of image+text pairs"""
        results = []
        for img, txt in zip(images, texts):
            results.append(self.detect(img, txt))
        return results



