# Ethical Eye Extension - Balanced Training System

## 🎯 Overview

This updated system addresses the imbalanced dataset issue by implementing comprehensive data balancing techniques. The system now ensures equal representation of all dark pattern categories for robust model training.

## 🔧 Problem Solved

**Original Issue**: The dataset had highly imbalanced classes:
- Not Dark Pattern: 2096 samples
- Scarcity: 404 samples  
- Social Proof: 312 samples
- Urgency: 210 samples
- Misdirection: 194 samples
- Obstruction: 27 samples
- Sneaking: 12 samples
- Forced Action: 4 samples

**Solution**: Implemented data balancing to ensure 100 samples per category for robust training.

## 🚀 Quick Start

### 1. Install Requirements
```bash
# Option 1: Use the installation script
python install_requirements.py

# Option 2: Manual installation
pip install -r requirements_training.txt
```

### 2. Run Training Pipeline
```bash
python run_training_pipeline.py
```

### 3. Test Data Balancing (Optional)
```bash
python test_data_balancing.py
```

## 📊 Data Balancing Methods

### 1. Simple Balancing (Currently Used)
- **Oversampling**: Repeats samples for minority classes
- **Undersampling**: Reduces samples for majority classes
- **Target**: 100 samples per category
- **Advantages**: Simple, fast, reliable

### 2. Advanced Balancing (Available)
- **Text Augmentation**: Synonym replacement, word shuffling
- **SMOTE**: Synthetic minority oversampling
- **Hybrid Methods**: Combination of techniques
- **Advantages**: More diverse training data

## 🏗️ System Architecture

```
Training Pipeline
├── Data Loading
│   ├── dark_patterns.csv
│   ├── normie.csv
│   └── dataset.csv
├── Data Balancing
│   ├── Category Analysis
│   ├── Oversampling
│   └── Undersampling
├── Model Training
│   ├── DistilBERT Fine-tuning
│   ├── Cross-validation
│   └── Performance Monitoring
├── SHAP Integration
│   ├── Explanation Generation
│   └── Feature Importance
└── Research Plots
    ├── Confusion Matrix
    ├── Performance Metrics
    └── User Study Visualizations
```

## 📈 Expected Results

### Balanced Dataset
- **Total Samples**: ~800 (100 per category × 8 categories)
- **Train/Val/Test Split**: 70%/15%/15%
- **Stratified Splitting**: Ensures equal representation

### Model Performance
- **Accuracy**: >80% (target)
- **Balanced Performance**: Good performance across all categories
- **Robust Training**: No overfitting to majority class

## 🔍 Key Features

### 1. Robust Data Balancing
```python
def simple_balance_dataset(self, df):
    target_samples = 100  # Fixed target for all categories
    
    for category in df['category'].unique():
        if current_count < target_samples:
            # Oversample by repeating samples
        elif current_count > target_samples * 2:
            # Undersample if too many
```

### 2. Stratified Splitting
```python
# Check if stratified splitting is possible
if min_samples < 2:
    # Use random splitting
else:
    # Use stratified splitting
```

### 3. Comprehensive Logging
- Category distribution before/after balancing
- Training progress monitoring
- Performance metrics tracking

## 📁 File Structure

```
training/
├── train_distilbert.py          # Main training script
├── data_balancer.py            # Advanced balancing methods
├── shap_explainer.py           # SHAP explanations
├── generate_research_plots.py  # Research visualizations
└── test_data_balancing.py      # Testing script

api/
├── ethical_eye_api.py          # Updated API with explanations
└── app.py                      # Legacy API

data/processed/
├── combined_dataset.csv        # Balanced dataset
└── test_balanced_*.csv         # Test outputs

models/ethical_eye/
├── final_model/                # Trained model
└── training_logs/              # Training logs

plots/research/
├── paper/                      # Main paper figures
├── supplementary/              # Supplementary figures
└── shap/                       # SHAP visualizations
```

## 🎯 Training Process

### Step 1: Data Loading and Preprocessing
1. Load all three datasets
2. Combine and clean data
3. Map categories to 8 classes
4. Remove duplicates and short texts

### Step 2: Data Balancing
1. Analyze category distribution
2. Apply oversampling/undersampling
3. Ensure 100 samples per category
4. Shuffle final dataset

### Step 3: Model Training
1. Split into train/val/test (70/15/15)
2. Initialize DistilBERT model
3. Fine-tune with balanced data
4. Monitor performance metrics

### Step 4: Evaluation and Plots
1. Generate confusion matrix
2. Calculate per-class metrics
3. Create SHAP explanations
4. Generate research plots

## 🔧 Configuration

### Training Parameters
```python
config = {
    'num_epochs': 5,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'max_length': 512,
    'target_samples_per_class': 100
}
```

### Balancing Parameters
```python
balancing_config = {
    'method': 'simple',  # or 'hybrid', 'augmentation'
    'target_samples': 100,
    'random_state': 42
}
```

## 📊 Monitoring and Logging

### Training Logs
- Real-time training progress
- Loss and accuracy metrics
- Category-wise performance
- SHAP explanation generation

### Output Files
- `logs/training/training.log` - Detailed training logs
- `results/evaluation/model_results.json` - Performance metrics
- `plots/research/` - All research visualizations

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -r requirements_training.txt
   ```

2. **Memory Issues**
   - Reduce batch size
   - Use smaller model
   - Process data in chunks

3. **CUDA Issues**
   - Install PyTorch with CUDA support
   - Check GPU availability
   - Fall back to CPU training

### Error Messages

- **"ValueError: The least populated class has only 1 member"**
  - ✅ **Fixed**: Data balancing ensures minimum samples per class

- **"ImportError: No module named 'imblearn'"**
  - ✅ **Fixed**: Simple balancing doesn't require imblearn

- **"CUDA out of memory"**
  - **Solution**: Reduce batch size or use CPU

## 🎉 Success Indicators

### Training Success
- ✅ All categories have sufficient samples
- ✅ Stratified splitting works
- ✅ Model trains without errors
- ✅ Performance metrics >80% accuracy

### Research Ready
- ✅ Balanced confusion matrix
- ✅ Per-class performance metrics
- ✅ SHAP explanations generated
- ✅ Research plots created

## 🔄 Next Steps

1. **Run Training**: Execute the complete pipeline
2. **Evaluate Results**: Check performance metrics
3. **Generate Plots**: Create research visualizations
4. **Test API**: Verify real-time analysis
5. **User Study**: Conduct qualitative evaluation

## 📞 Support

### Getting Help
- Check logs in `logs/training/`
- Review error messages carefully
- Ensure all requirements are installed
- Verify data files are present

### Common Commands
```bash
# Check requirements
python -c "import torch, transformers, sklearn; print('All packages available')"

# Test data balancing
python test_data_balancing.py

# Run full pipeline
python run_training_pipeline.py

# Start API server
python api/ethical_eye_api.py
```

---

*This balanced training system ensures robust model performance across all dark pattern categories, making it suitable for academic research and real-world deployment.*
