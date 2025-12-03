# Ethical Eye Extension - Research Project

## 🎯 Project Overview

**Ethical Eye** is a Chrome browser extension that detects and explains dark patterns on e-commerce websites using explainable AI (XAI). This research project combines DistilBERT for classification and SHAP for transparent explanations to empower users with evidence-based insights about manipulative design techniques.

## 🎓 Research Objectives

### Primary Goals
- **Detection**: Identify 8 categories of dark patterns using DistilBERT
- **Explanation**: Provide SHAP-based explanations for transparency
- **Education**: Enhance user digital literacy through clear explanations
- **Empowerment**: Enable users to make informed decisions

### Research Outcomes
- **Functional Browser Extension**: Real-time dark pattern detection with explanations
- **Quantitative Evaluation**: >80% accuracy on 235 test samples
- **Qualitative User Study**: 5-10 participants with think-aloud protocols
- **Academic Publication**: Research paper for SCCUR or CHI SRC

## 🏗️ System Architecture

### Components
- **Chrome Extension**: Frontend interface with visual highlighting
- **DistilBERT Model**: Fine-tuned for dark pattern classification
- **Multimodal Model v2**: MobileViT + DistilBERT hybrid for vision+text detection
- **Layout Analyzer**: HTML/CSS structure analysis for visual misdirection
- **SHAP Explainer**: Generates transparent explanations
- **Flask API**: Backend service for real-time analysis
- **Research Framework**: Comprehensive evaluation and plotting

### Dark Pattern Categories

**Text-Based Patterns (DistilBERT):**
1. **Urgency** - Time pressure tactics
2. **Scarcity** - False limited availability
3. **Social Proof** - Fake social validation
4. **Misdirection** - Deceptive navigation
5. **Forced Action** - Unnecessary requirements
6. **Obstruction** - Intentional difficulty
7. **Sneaking** - Hidden information
8. **Hidden Costs** - Concealed fees
9. **Not Dark Pattern** - Normal content

**Visual Patterns (Multimodal v2):**
- **Color Manipulation** - Deceptive use of colors
- **Deceptive UI Contrast** - Misleading contrast ratios
- **Hidden Subscription Checkbox** - Obscured subscription options
- **Fake Progress Bar** - Misleading progress indicators

**Layout Patterns (Layout Analyzer):**
- **Button Order Misdirection** - Suspicious button placement
- **Font Size Manipulation** - Extreme size differences
- **Hidden Important Information** - CSS-hidden content
- **Visual Hierarchy Tricks** - Z-index and positioning manipulation

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Chrome browser (version 88+)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd DarkSurfer-Extension
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements_training.txt
   ```

3. **Run the complete training pipeline**
   ```bash
   python run_training_pipeline.py
   ```

4. **Start the API server**
   ```bash
   python api/ethical_eye_api.py
   ```

5. **Load the Chrome extension**
   - Open Chrome and go to `chrome://extensions/`
   - Enable "Developer mode"
   - Click "Load unpacked" and select the `app` folder

## 📊 Training Pipeline

### Complete Training Process

The training pipeline includes:

1. **Data Preprocessing**
   - Combines multiple datasets
   - Creates 8-category classification
   - Handles data cleaning and validation

2. **Model Training**
   - DistilBERT fine-tuning
   - Cross-validation
   - Performance optimization

3. **SHAP Integration**
   - Explanation generation
   - Feature importance analysis
   - Transparency metrics

4. **Research Plot Generation**
   - Confusion matrices
   - Performance comparisons
   - User study visualizations

### Running Individual Components

```bash
# Train DistilBERT model
python training/train_distilbert.py

# Generate SHAP explanations
python training/shap_explainer.py

# Create research plots
python training/generate_research_plots.py

# Test API
python api/ethical_eye_api.py
```

## 📈 Research Evaluation

### Quantitative Metrics
- **Accuracy**: >80% on test dataset
- **Precision/Recall**: Per-category analysis
- **F1-Score**: Weighted and macro averages
- **Confusion Matrix**: Detailed classification analysis

### Qualitative Assessment
- **User Study**: 5-10 participants
- **Think-aloud Protocols**: Real-time feedback
- **Learning Effectiveness**: Pre/post pattern recognition
- **User Satisfaction**: Likert scale ratings

### Generated Research Plots
- Confusion Matrix (Main Figure)
- Per-Class Performance Metrics
- Model Comparison Charts
- SHAP Feature Importance
- User Study Results
- ROC Curves (Supplementary)
- Precision-Recall Curves (Supplementary)

## 🔧 API Usage

### Endpoints

#### Health Check
```bash
GET http://127.0.0.1:5000/
```

#### Analyze Single Text
```bash
POST http://127.0.0.1:5000/analyze_single
Content-Type: application/json

{
  "text": "Hurry! Only 2 left in stock!",
  "confidence_threshold": 0.7
}
```

#### Analyze Multiple Segments
```bash
POST http://127.0.0.1:5000/analyze
Content-Type: application/json

{
  "segments": [
    {
      "text": "Limited time offer!",
      "element_id": "segment_1",
      "position": {"x": 100, "y": 200}
    }
  ],
  "confidence_threshold": 0.7
}
```

#### Get Pattern Information
```bash
GET http://127.0.0.1:5000/patterns
```

#### Analyze Screenshot (Multimodal v2)
```bash
POST http://127.0.0.1:5000/vision/analyze
Content-Type: multipart/form-data

file: [screenshot image]
```

#### Analyze HTML Layout
```bash
POST http://127.0.0.1:5000/analyze_layout
Content-Type: application/json

{
  "html": "<html>...</html>",
  "css": "/* optional CSS */"
}
```

#### Get Model Information
```bash
GET http://127.0.0.1:5000/model_info
```

### Response Format
```json
{
  "category": "Urgency",
  "confidence": 0.87,
  "is_dark_pattern": true,
  "explanation": "This text is classified as 'Urgency' with 87.0% confidence. Creates false time pressure to rush decisions. Key indicators include: hurry, limited.",
  "top_words": [["hurry", 0.15], ["limited", 0.12]],
  "pattern_description": "Creates false time pressure to rush decisions"
}
```

## 📁 Project Structure

```
DarkSurfer-Extension/
├── api/                          # Backend API
│   ├── ethical_eye_api.py       # Main Flask API
│   └── app.py                   # Legacy API
├── app/                         # Chrome Extension
│   ├── manifest.json            # Extension configuration
│   ├── popup.html               # Extension popup
│   ├── js/                      # JavaScript modules
│   └── css/                     # Styling files
├── training/                    # Training pipeline
│   ├── train_distilbert.py      # DistilBERT training
│   ├── shap_explainer.py        # SHAP analysis
│   └── generate_research_plots.py # Research visualizations
├── docs/                        # Documentation
│   ├── 01-High-Level-Design.md
│   ├── 02-Low-Level-Design.md
│   ├── 03-System-Architecture.md
│   ├── 04-High-Level-Model-Diagram.md
│   └── 05-Presentation-Diagrams.md
├── models/                      # Trained models
├── plots/research/              # Research plots
├── results/                     # Evaluation results
├── logs/                        # Training logs
├── data/processed/              # Processed datasets
├── run_training_pipeline.py     # Complete pipeline runner
└── requirements_training.txt    # Dependencies
```

## 🎨 Research Plots

### Main Paper Figures
- **Confusion Matrix**: Classification performance visualization
- **Per-Class Metrics**: Precision, recall, F1-score by category
- **Model Comparison**: DistilBERT vs baseline models
- **SHAP Feature Importance**: Key words for detection
- **User Satisfaction**: Before/after extension use
- **Learning Effectiveness**: Pattern recognition improvement

### Supplementary Figures
- **ROC Curves**: Multi-class performance analysis
- **Precision-Recall Curves**: Detailed classification metrics
- **SHAP Summary Plots**: Feature importance across samples
- **Usage Patterns**: User engagement statistics

## 🔬 Research Methodology

### Dataset
- **Training Data**: Combined dark pattern datasets
- **Test Set**: 235 samples for evaluation
- **Categories**: 8 dark pattern types + normal content
- **Preprocessing**: Text cleaning, normalization, tokenization

### Model Architecture
- **Base Model**: DistilBERT-base-uncased
- **Fine-tuning**: Custom classification head
- **Optimization**: AdamW optimizer, learning rate scheduling
- **Regularization**: Dropout, weight decay

### Evaluation Framework
- **Cross-validation**: 5-fold stratified splits
- **Metrics**: Accuracy, precision, recall, F1-score
- **Statistical Testing**: Confidence intervals, significance tests
- **Ablation Studies**: Component-wise analysis

## 📚 Academic Contributions

### Novel Contributions
1. **Explainable AI for Dark Patterns**: First SHAP-based explanations for dark pattern detection
2. **User-Centric Design**: Focus on user education and empowerment
3. **Comprehensive Evaluation**: Both quantitative and qualitative assessment
4. **Real-world Deployment**: Functional browser extension

### Target Venues
- **SCCUR**: Southern California Conference for Undergraduate Research
- **CHI SRC**: CHI Student Research Competition
- **HCI Conferences**: Human-Computer Interaction venues
- **AI Ethics**: Explainable AI and fairness conferences

## 🛠️ Development

### Adding New Features
1. **New Pattern Types**: Update category mapping in training scripts
2. **Enhanced Explanations**: Modify SHAP explainer
3. **UI Improvements**: Update Chrome extension files
4. **API Extensions**: Add new endpoints to Flask API

### Testing
```bash
# Run unit tests
pytest tests/

# Test API endpoints
python -m pytest tests/test_api.py

# Test model performance
python training/evaluate_model.py
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make changes and test
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face**: Transformers library and DistilBERT model
- **SHAP**: Explainable AI framework
- **Chrome Extension APIs**: Browser integration
- **Research Community**: Dark pattern research and datasets

## 📞 Support

### Getting Help
- **Documentation**: Check the `docs/` folder
- **Issues**: GitHub issue tracker
- **Discussions**: GitHub discussions
- **Email**: Contact the research team

### Common Issues
1. **Model Loading**: Ensure model is trained first
2. **API Connection**: Check if Flask server is running
3. **Extension Loading**: Verify Chrome developer mode is enabled
4. **Dependencies**: Install all requirements from `requirements_training.txt`

---

*This project represents a comprehensive research effort in explainable AI for user empowerment and digital literacy. For questions or contributions, please refer to the documentation or contact the development team.*
