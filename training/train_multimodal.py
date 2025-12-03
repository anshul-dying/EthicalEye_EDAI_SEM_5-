"""
Training script for Multimodal Dark Pattern Detection Model (v2)
MobileViT + DistilBERT hybrid model

Trains on images with OCR-extracted text for detecting:
- Color Manipulation
- Deceptive UI Contrast  
- Hidden Subscription Checkbox
- Fake Progress Bar
- Normal
"""

import os
import sys
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.vision.multimodal_model import MultimodalDarkPatternModel
from api.vision.ocr import OcrExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultimodalDarkPatternDataset(Dataset):
    """Dataset for multimodal dark pattern detection"""
    
    def __init__(
        self,
        images_dir,
        labels_dir,
        image_transform=None,
        use_cropped_regions=True,
        ocr_extractor=None
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.image_transform = image_transform
        self.use_cropped_regions = use_cropped_regions
        self.ocr_extractor = ocr_extractor or OcrExtractor()
        
        # Find all image files
        self.image_files = list(self.images_dir.glob('*.jpg')) + list(self.images_dir.glob('*.png'))
        self.samples = []
        
        # Load samples
        self._load_samples()
        
        logger.info(f"Loaded {len(self.samples)} samples from {len(self.image_files)} images")
    
    def _load_samples(self):
        """Load image-label pairs"""
        for img_path in self.image_files:
            # Find corresponding label file
            label_path = self.labels_dir / (img_path.stem + '.txt')
            
            if not label_path.exists():
                # Skip if no label file
                continue
            
            # Read label (YOLO format: class x_center y_center width height)
            with open(label_path, 'r') as f:
                label_lines = f.readlines()
            
            if not label_lines:
                # Empty label - treat as normal/background
                self.samples.append({
                    'image_path': str(img_path),
                    'bboxes': [],
                    'classes': [4]  # Normal class
                })
            else:
                # Parse bounding boxes
                bboxes = []
                classes = []
                for line in label_lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        bboxes.append((x_center, y_center, width, height))
                        # Map YOLO class 0 to dark pattern (we'll categorize later)
                        # For now, use class 0-3 for different dark pattern types
                        # Class 4 is Normal
                        classes.append(min(class_id, 3))  # Cap at 3 for dark patterns
                
                if bboxes:
                    self.samples.append({
                        'image_path': str(img_path),
                        'bboxes': bboxes,
                        'classes': classes
                    })
                else:
                    # No valid bboxes - treat as normal
                    self.samples.append({
                        'image_path': str(img_path),
                        'bboxes': [],
                        'classes': [4]
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['image_path']
        bboxes = sample['bboxes']
        classes = sample['classes']
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        img_width, img_height = image.size
        
        # Extract text from image using OCR
        text = ""
        if self.ocr_extractor._available:
            try:
                ocr_result = self.ocr_extractor.extract(image)
                text = ocr_result.text if ocr_result.text else ""
            except Exception as e:
                logger.debug(f"OCR failed for {img_path}: {e}")
                text = ""
        
        # If using cropped regions and we have bboxes, use the first bbox
        # Otherwise use full image
        if self.use_cropped_regions and bboxes:
            # Crop first bounding box region
            x_center, y_center, width, height = bboxes[0]
            
            # Convert normalized coordinates to pixel coordinates
            x = int((x_center - width / 2) * img_width)
            y = int((y_center - height / 2) * img_height)
            w = int(width * img_width)
            h = int(height * img_height)
            
            # Ensure coordinates are within image bounds
            x = max(0, min(x, img_width - 1))
            y = max(0, min(y, img_height - 1))
            w = min(w, img_width - x)
            h = min(h, img_height - y)
            
            if w > 0 and h > 0:
                crop = image.crop((x, y, x + w, y + h))
                image = crop
                
                # Extract text from cropped region too
                if self.ocr_extractor._available:
                    try:
                        ocr_result = self.ocr_extractor.extract(crop)
                        if ocr_result.text:
                            text = ocr_result.text
                    except:
                        pass
        
        # Apply transforms
        if self.image_transform:
            image = self.image_transform(image)
        
        # Get label (use first class, or 4 for normal)
        label = classes[0] if classes else 4
        
        return {
            'image': image,
            'text': text,
            'label': label,
            'image_path': img_path
        }


class MultimodalTrainer:
    """Trainer for multimodal dark pattern detection model"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device='cuda',
        learning_rate=1e-4,
        num_epochs=10,
        save_dir='models/multimodal_v2'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer - use different learning rates for pretrained vs new components
        vision_params = list(self.model.vision_encoder.parameters())
        text_params = list(self.model.text_encoder.parameters())
        fusion_params = list(self.model.fusion.parameters()) + list(self.model.classifier.parameters())
        
        self.optimizer = torch.optim.AdamW([
            {'params': vision_params, 'lr': learning_rate},
            {'params': text_params, 'lr': learning_rate * 0.1},  # Lower LR for pretrained
            {'params': fusion_params, 'lr': learning_rate}
        ], weight_decay=1e-4)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # For evaluation plots
        self.all_predictions = []
        self.all_labels = []
        self.class_names = ['Color Manipulation', 'Deceptive UI Contrast', 
                          'Hidden Subscription Checkbox', 'Fake Progress Bar', 'Normal']
        
        # Plot directory
        self.plots_dir = Path(save_dir) / 'plots'
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized trainer on device: {device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            images = batch['image'].to(self.device)
            texts = batch['text']
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Pass tensors directly (model handles both PIL and tensors)
            logits = self.model(images, texts)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        
        self.history['train_loss'].append(avg_loss)
        self.history['train_acc'].append(accuracy)
        
        return avg_loss, accuracy
    
    def validate(self):
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validating')
            for batch in pbar:
                images = batch['image'].to(self.device)
                texts = batch['text']
                labels = batch['label'].to(self.device)
                
                # Pass tensors directly
                logits = self.model(images, texts)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * correct / total:.2f}%'
                })
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        
        self.history['val_loss'].append(avg_loss)
        self.history['val_acc'].append(accuracy)
        
        return avg_loss, accuracy
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        best_val_acc = 0.0
        
        for epoch in range(self.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            logger.info("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            
            # Validate
            val_loss, val_acc = self.validate()
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(epoch, is_best=True)
                logger.info(f"Saved best model (val_acc: {val_acc:.2f}%)")
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        logger.info("\nTraining completed!")
        logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
        
        # Collect predictions for final evaluation
        logger.info("Collecting predictions for evaluation plots...")
        self.collect_predictions()
        
        # Generate all plots
        logger.info("Generating research plots...")
        self.generate_all_plots()
        
        # Save training history
        self.save_history()
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'class_names': ['Color Manipulation', 'Deceptive UI Contrast', 
                          'Hidden Subscription Checkbox', 'Fake Progress Bar', 'Normal']
        }
        
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best_model.pt')
        else:
            torch.save(checkpoint, self.save_dir / f'checkpoint_epoch_{epoch + 1}.pt')
    
    def collect_predictions(self):
        """Collect all predictions and labels for evaluation"""
        self.model.eval()
        self.all_predictions = []
        self.all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Collecting predictions'):
                images = batch['image'].to(self.device)
                texts = batch['text']
                labels = batch['label'].to(self.device)
                
                logits = self.model(images, texts)
                preds = torch.argmax(logits, dim=1)
                
                self.all_predictions.extend(preds.cpu().numpy())
                self.all_labels.extend(labels.cpu().numpy())
    
    def generate_all_plots(self):
        """Generate all research plots"""
        # Set style for research plots
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['font.family'] = 'serif'
        
        # 1. Training curves
        self.plot_training_curves()
        
        # 2. Confusion matrix
        self.plot_confusion_matrix()
        
        # 3. Per-class metrics
        self.plot_per_class_metrics()
        
        # 4. Learning rate schedule
        self.plot_learning_rate()
        
        logger.info(f"All plots saved to {self.plots_dir}")
    
    def plot_training_curves(self):
        """Plot training and validation loss/accuracy curves"""
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curves
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[1].plot(epochs, self.history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
        axes[1].plot(epochs, self.history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.plots_dir / 'training_curves.pdf', bbox_inches='tight')
        plt.close()
        logger.info("Saved training curves plot")
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix"""
        if not self.all_predictions or not self.all_labels:
            logger.warning("No predictions collected, skipping confusion matrix")
            return
        
        cm = confusion_matrix(self.all_labels, self.all_predictions)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Predicted', fontsize=12)
        axes[0].set_ylabel('Actual', fontsize=12)
        
        # Normalized
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=axes[1],
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   cbar_kws={'label': 'Proportion'})
        axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Predicted', fontsize=12)
        axes[1].set_ylabel('Actual', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.plots_dir / 'confusion_matrix.pdf', bbox_inches='tight')
        plt.close()
        logger.info("Saved confusion matrix plot")
    
    def plot_per_class_metrics(self):
        """Plot per-class precision, recall, and F1 scores"""
        if not self.all_predictions or not self.all_labels:
            logger.warning("No predictions collected, skipping per-class metrics")
            return
        
        precision, recall, f1, support = precision_recall_fscore_support(
            self.all_labels, self.all_predictions, labels=range(len(self.class_names))
        )
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8, color='#3498db')
        bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8, color='#e74c3c')
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8, color='#2ecc71')
        
        ax.set_xlabel('Dark Pattern Category', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'per_class_metrics.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.plots_dir / 'per_class_metrics.pdf', bbox_inches='tight')
        plt.close()
        logger.info("Saved per-class metrics plot")
        
        # Save metrics to JSON
        metrics_dict = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'support': support.tolist(),
            'class_names': self.class_names
        }
        with open(self.plots_dir / 'per_class_metrics.json', 'w') as f:
            json.dump(metrics_dict, f, indent=2)
    
    def plot_learning_rate(self):
        """Plot learning rate schedule"""
        if not self.history['learning_rates']:
            logger.warning("No learning rate history, skipping LR plot")
            return
        
        epochs = range(1, len(self.history['learning_rates']) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.history['learning_rates'], 'g-', linewidth=2, marker='o', markersize=4)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.title('Learning Rate Schedule (Cosine Annealing)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'learning_rate.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.plots_dir / 'learning_rate.pdf', bbox_inches='tight')
        plt.close()
        logger.info("Saved learning rate plot")
    
    def save_history(self):
        """Save training history"""
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Saved training history to {history_path}")


def main():
    """Main training function"""
    # Configuration
    config = {
        'train_images_dir': 'train/images',
        'train_labels_dir': 'train/labels',
        'test_images_dir': 'test/images',
        'test_labels_dir': 'test/labels',
        'batch_size': 8,
        'learning_rate': 1e-4,
        'num_epochs': 20,
        'num_workers': 4,
        'use_cropped_regions': True,
        'save_dir': 'models/multimodal_v2'
    }
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Image transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Initialize OCR
    ocr_extractor = OcrExtractor()
    logger.info(f"OCR available: {ocr_extractor._available}")
    
    # Create datasets
    logger.info("Loading training dataset...")
    train_dataset = MultimodalDarkPatternDataset(
        images_dir=config['train_images_dir'],
        labels_dir=config['train_labels_dir'],
        image_transform=train_transform,
        use_cropped_regions=config['use_cropped_regions'],
        ocr_extractor=ocr_extractor
    )
    
    logger.info("Loading validation dataset...")
    val_dataset = MultimodalDarkPatternDataset(
        images_dir=config['test_images_dir'],
        labels_dir=config['test_labels_dir'],
        image_transform=val_transform,
        use_cropped_regions=config['use_cropped_regions'],
        ocr_extractor=ocr_extractor
    )
    
    # Create data loaders
    # Use num_workers=0 on Windows to avoid multiprocessing issues
    num_workers = 0 if sys.platform == 'win32' else config['num_workers']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Initialize model
    logger.info("Initializing multimodal model...")
    model = MultimodalDarkPatternModel(
        vision_embed_dim=256,
        text_embed_dim=768,
        fusion_dim=512,
        num_classes=5,  # 4 dark patterns + Normal
        dropout=0.1
    )
    
    # Initialize trainer
    trainer = MultimodalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=config['learning_rate'],
        num_epochs=config['num_epochs'],
        save_dir=config['save_dir']
    )
    
    # Train
    trainer.train()
    
    logger.info("Training completed successfully!")


if __name__ == '__main__':
    main()

