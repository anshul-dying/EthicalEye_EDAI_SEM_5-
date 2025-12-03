# Multimodal Model Training Plots - Research Paper

## Overview

The training script automatically generates comprehensive plots for research paper publication. All plots are saved in high-resolution PNG (300 DPI) and PDF formats.

## Generated Plots

### 1. Training Curves (`training_curves.png/pdf`)

- **Left Panel**: Training and Validation Loss over epochs
- **Right Panel**: Training and Validation Accuracy over epochs
- Shows model convergence and overfitting detection
- Publication-ready format with proper labels and legends

### 2. Confusion Matrix (`confusion_matrix.png/pdf`)

- **Left Panel**: Raw count confusion matrix
- **Right Panel**: Normalized confusion matrix (proportions)
- Shows per-class classification performance
- Color-coded heatmap for easy interpretation

### 3. Per-Class Metrics (`per_class_metrics.png/pdf`)

- Bar chart showing Precision, Recall, and F1-Score for each class
- Classes:
  - Color Manipulation
  - Deceptive UI Contrast
  - Hidden Subscription Checkbox
  - Fake Progress Bar
  - Normal
- Values displayed on bars for precise reading
- Also saved as JSON (`per_class_metrics.json`) for data analysis

### 4. Learning Rate Schedule (`learning_rate.png/pdf`)

- Shows learning rate decay over training epochs
- Cosine annealing schedule visualization
- Log scale for better visualization of LR changes

## Plot Location

All plots are saved in:

```
models/multimodal_v2/plots/
```

## Plot Specifications

- **Resolution**: 300 DPI (publication quality)
- **Formats**: PNG (for presentations) and PDF (for papers)
- **Style**: Research paper style with serif fonts
- **Size**: Optimized for paper inclusion (12x8 inches for main plots)

## Usage in Research Paper

### Main Figures

1. **Training Curves** - Shows model learning progress
2. **Confusion Matrix** - Demonstrates classification accuracy
3. **Per-Class Metrics** - Highlights performance across dark pattern types

### Supplementary Material

- Learning rate schedule can be included in supplementary material
- Per-class metrics JSON can be used for detailed analysis

## Example Figure Captions

### Training Curves

"Training and validation performance of the multimodal dark pattern detection model over 20 epochs. The model achieves convergence with high accuracy on both training and validation sets."

### Confusion Matrix

"Confusion matrix showing classification performance across five classes: four dark pattern types and normal content. The normalized matrix (right) shows the proportion of correct classifications per class."

### Per-Class Metrics

"Per-class performance metrics (Precision, Recall, F1-Score) for the multimodal model. The model shows strong performance across all dark pattern categories."

## Customization

To customize plots, modify the `generate_all_plots()` method in `training/train_multimodal.py`:

- Change colors: Modify `sns.set_palette()`
- Adjust figure size: Change `plt.rcParams['figure.figsize']`
- Modify fonts: Update `plt.rcParams['font.family']`
- Add more plots: Extend `generate_all_plots()` method

## Data Export

Metrics are also exported as JSON:

- `per_class_metrics.json`: Precision, recall, F1, and support for each class
- `training_history.json`: Complete training history (loss, accuracy, learning rates)

These can be used for:

- Statistical analysis
- Comparison with other models
- Reproducibility

## Notes

- Plots are generated automatically after training completes
- If training is interrupted, plots from completed epochs will still be generated
- All plots use consistent styling for paper inclusion
- PDF format is preferred for LaTeX papers (vector graphics)
