"""
Simple Improved Training Script for Ethical Eye
Fixed to handle the actual dataset structure
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for research plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DarkPatternDataset(Dataset):
    """Custom dataset class for dark pattern classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_and_prepare_data():
    """Load and prepare the dataset"""
    print("Loading and preparing data...")
    
    # Load dataset
    df = pd.read_csv('LLM-Model/dataset.csv')
    print(f"   Original dataset size: {len(df)}")
    
    # Rename column to standardize
    df = df.rename(columns={'Pattern Category': 'category'})
    
    # Check category distribution
    print(f"   Category distribution:")
    category_counts = df['category'].value_counts()
    print(category_counts)
    
    # Simple data balancing - increase samples for minority classes
    print("\nBalancing dataset...")
    
    # Set target samples per class
    target_samples = 300  # Reasonable target
    
    balanced_data = []
    
    for category in df['category'].unique():
        category_data = df[df['category'] == category].copy()
        current_count = len(category_data)
        
        if current_count < target_samples:
            # Oversample by repeating
            repeat_factor = (target_samples // current_count) + 1
            repeated_data = pd.concat([category_data] * repeat_factor, ignore_index=True)
            category_data = repeated_data.sample(n=target_samples, random_state=42)
        elif current_count > target_samples * 2:
            # Undersample if too many
            category_data = category_data.sample(n=target_samples, random_state=42)
        
        balanced_data.append(category_data)
        print(f"   {category}: {len(category_data)} samples")
    
    # Combine balanced data
    balanced_df = pd.concat(balanced_data, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"   Final balanced dataset size: {len(balanced_df)}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    balanced_df['encoded_category'] = label_encoder.fit_transform(balanced_df['category'])
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    balanced_df.to_csv('data/processed/balanced_dataset.csv', index=False)
    
    return balanced_df, label_encoder

def prepare_datasets(df, label_encoder):
    """Prepare train, validation, and test datasets"""
    print("Preparing datasets...")
    
    # Split data
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['encoded_category']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['encoded_category']
    )
    
    print(f"   Train size: {len(train_df)}")
    print(f"   Validation size: {len(val_df)}")
    print(f"   Test size: {len(test_df)}")
    
    # Initialize tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
    # Create datasets
    train_dataset = DarkPatternDataset(
        train_df['text'].tolist(),
        train_df['encoded_category'].tolist(),
        tokenizer,
        max_length=256  # Reduced for memory efficiency
    )
    
    val_dataset = DarkPatternDataset(
        val_df['text'].tolist(),
        val_df['encoded_category'].tolist(),
        tokenizer,
        max_length=256
    )
    
    test_dataset = DarkPatternDataset(
        test_df['text'].tolist(),
        test_df['encoded_category'].tolist(),
        tokenizer,
        max_length=256
    )
    
    return train_dataset, val_dataset, test_dataset, train_df, val_df, test_df, tokenizer

def train_model(train_dataset, val_dataset, tokenizer, label_encoder):
    """Train the DistilBERT model"""
    print("Training DistilBERT model...")
    
    # Initialize model
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=len(label_encoder.classes_)
    )
    
    # Training arguments optimized for RTX 3050 Laptop
    training_args = TrainingArguments(
        output_dir='models/ethical_eye/simple_improved_model',
        num_train_epochs=4,  # Increased from 3
        per_device_train_batch_size=8,  # Optimized for 4GB VRAM
        per_device_eval_batch_size=8,
        warmup_steps=300,
        weight_decay=0.01,
        learning_rate=3e-5,  # Increased learning rate
        logging_dir='logs/training',
        logging_steps=50,
        eval_strategy='steps',
        eval_steps=100,
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        fp16=True,  # Mixed precision for memory efficiency
        gradient_accumulation_steps=2,
        dataloader_num_workers=2,
        report_to=None,
        seed=42
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train the model
    print("   Starting training...")
    trainer.train()
    
    # Save the model
    trainer.save_model('models/ethical_eye/simple_improved_model')
    tokenizer.save_pretrained('models/ethical_eye/simple_improved_model')
    
    print("   Training completed!")
    
    return trainer

def evaluate_model(trainer, test_dataset, test_df, label_encoder):
    """Evaluate the trained model"""
    print("Evaluating model...")
    
    # Get predictions
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = test_df['encoded_category'].tolist()
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    # Create results dictionary
    results = {
        'overall': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        },
        'per_class': {
            'precision': precision_per_class.tolist(),
            'recall': recall_per_class.tolist(),
            'f1': f1_per_class.tolist()
        },
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'classification_report': classification_report(
            y_true, y_pred, 
            target_names=label_encoder.classes_,
            output_dict=True
        )
    }
    
    # Save results
    os.makedirs('results/evaluation', exist_ok=True)
    with open('results/evaluation/simple_improved_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"   Overall Accuracy: {accuracy:.4f}")
    print(f"   Overall F1-Score: {f1:.4f}")
    print(f"   Overall Precision: {precision:.4f}")
    print(f"   Overall Recall: {recall:.4f}")
    
    return results, y_pred, y_true

def create_plots(results, y_pred, y_true, label_encoder):
    """Create plots for the improved model"""
    print("Creating plots...")
    
    # Create plots directory
    os.makedirs('plots/research', exist_ok=True)
    
    # 1. Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = np.array(results['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=label_encoder.classes_,
               yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix - Simple Improved Model')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('plots/research/simple_improved_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Per-class F1 scores
    plt.figure(figsize=(12, 6))
    f1_scores = results['per_class']['f1']
    classes = label_encoder.classes_
    
    bars = plt.bar(range(len(classes)), f1_scores, color='skyblue', alpha=0.7)
    plt.xlabel('Dark Pattern Categories')
    plt.ylabel('F1 Score')
    plt.title('Per-Class F1 Scores - Simple Improved Model')
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, score in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('plots/research/simple_improved_per_class_f1.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   Plots saved to plots/research/")

def main():
    """Main function to run simple improved training"""
    
    print("Starting Simple Improved Ethical Eye Training")
    print("=" * 60)
    
    try:
        # Load and prepare data
        df, label_encoder = load_and_prepare_data()
        
        # Prepare datasets
        train_dataset, val_dataset, test_dataset, train_df, val_df, test_df, tokenizer = prepare_datasets(df, label_encoder)
        
        # Train model
        trainer = train_model(train_dataset, val_dataset, tokenizer, label_encoder)
        
        # Evaluate model
        results, y_pred, y_true = evaluate_model(trainer, test_dataset, test_df, label_encoder)
        
        # Create plots
        create_plots(results, y_pred, y_true, label_encoder)
        
        print("\n" + "=" * 60)
        print("SIMPLE IMPROVED TRAINING COMPLETED!")
        print("=" * 60)
        print(f"Overall Accuracy: {results['overall']['accuracy']:.4f}")
        print(f"Overall F1-Score: {results['overall']['f1']:.4f}")
        print(f"Overall Precision: {results['overall']['precision']:.4f}")
        print(f"Overall Recall: {results['overall']['recall']:.4f}")
        print("\nFiles created:")
        print("   - Model: models/ethical_eye/simple_improved_model/")
        print("   - Results: results/evaluation/simple_improved_results.json")
        print("   - Plots: plots/research/")
        print("   - Data: data/processed/balanced_dataset.csv")
        
        return True
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\nTraining completed successfully!")
    else:
        print("\nTraining failed. Check the error messages above.")
