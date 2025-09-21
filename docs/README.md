# Ethical Eye Extension - Documentation

## 📚 Documentation Overview

This documentation provides comprehensive technical specifications for the **Ethical Eye** Chrome extension, a research project focused on detecting and explaining dark patterns in e-commerce websites using explainable AI (XAI).

## 📋 Document Structure

### 1. [High Level Design (HLD)](01-High-Level-Design.md)
- **Purpose**: System overview and architectural requirements
- **Contents**: 
  - System objectives and goals
  - High-level component architecture
  - Functional and non-functional requirements
  - Technology stack overview
  - Success metrics and risk assessment

### 2. [Low Level Design (LLD)](02-Low-Level-Design.md)
- **Purpose**: Detailed implementation specifications
- **Contents**:
  - Component-level code examples
  - API design and data structures
  - Machine learning model architecture
  - Database schema and configuration
  - Testing strategies and error handling

### 3. [System Architecture](03-System-Architecture.md)
- **Purpose**: Comprehensive architectural design
- **Contents**:
  - Layered architecture patterns
  - Component relationships and interactions
  - Security and privacy architecture
  - Scalability and performance considerations
  - Deployment and monitoring strategies

### 4. [High Level Model Diagram](04-High-Level-Model-Diagram.md)
- **Purpose**: Visual representations of system design
- **Contents**:
  - Mermaid diagrams for all major components
  - Data flow and interaction diagrams
  - Machine learning pipeline visualization
  - User interaction flow diagrams
  - System deployment architecture

## 🎯 Project Goals

### Research Objectives
- **Detection**: Identify 8 categories of dark patterns using DistilBERT
- **Explanation**: Provide SHAP-based explanations for transparency
- **Education**: Enhance user digital literacy through clear explanations
- **Empowerment**: Enable users to make informed decisions

### Technical Objectives
- **Accuracy**: >80% classification accuracy on test dataset
- **Performance**: <2 seconds response time per page analysis
- **Usability**: Intuitive interface with clear visual feedback
- **Privacy**: No data collection, local processing only

## 🏗️ System Components

### Frontend (Chrome Extension)
- **Popup Interface**: User controls and status display
- **Content Scripts**: Page analysis and visual highlighting
- **Background Worker**: API communication and state management
- **Visual Components**: Highlighting system and tooltips

### Backend (ML Processing)
- **DistilBERT Model**: Text classification and pattern detection
- **SHAP Explainer**: Feature importance and explanation generation
- **Flask API**: Request handling and response generation
- **Model Management**: Loading, caching, and optimization

### Data Layer
- **Training Datasets**: Dark pattern examples and normal text
- **Model Artifacts**: Pre-trained models and configurations
- **User Study Data**: Research evaluation and feedback

## 🔧 Technology Stack

### Frontend Technologies
- **Chrome Extension**: Manifest V3, Content Scripts API
- **Web Technologies**: HTML5, CSS3, JavaScript ES6+
- **UI/UX**: Custom CSS framework, responsive design

### Backend Technologies
- **ML Framework**: PyTorch, Transformers (Hugging Face)
- **Model**: DistilBERT-base-uncased
- **XAI Library**: SHAP (SHapley Additive exPlanations)
- **API Framework**: Flask, Flask-CORS
- **Data Processing**: pandas, numpy, scikit-learn

### Development Tools
- **Version Control**: Git
- **Testing**: Jest, pytest
- **Documentation**: Markdown, Mermaid diagrams
- **Deployment**: Chrome Web Store, Docker

## 📊 Dark Pattern Categories

The system detects and explains 8 categories of dark patterns:

1. **Urgency** - Time pressure tactics ("Hurry! Limited time!")
2. **Scarcity** - False limited availability ("Only 2 left!")
3. **Social Proof** - Fake social validation ("Join 10,000+ users")
4. **Misdirection** - Deceptive navigation and design
5. **Forced Action** - Unnecessary requirements
6. **Obstruction** - Intentional difficulty in user actions
7. **Sneaking** - Hidden information or deceptive practices
8. **Hidden Costs** - Concealed fees and charges
9. **Not Dark Pattern** - Normal, non-manipulative content

## 🚀 Getting Started

### Prerequisites
- Chrome browser (version 88+)
- Python 3.8+
- Node.js 14+ (for development)
- Git (for version control)

### Installation
1. Clone the repository
2. Install Python dependencies
3. Load the extension in Chrome
4. Start the Flask API server
5. Begin analyzing websites

### Usage
1. Install the extension from Chrome Web Store
2. Visit any e-commerce website
3. Click "Analyze Site" in the extension popup
4. View highlighted dark patterns with explanations
5. Learn about different pattern types through tooltips

## 📈 Research Methodology

### Quantitative Evaluation
- **Dataset**: 235 test samples across all categories
- **Metrics**: Accuracy, Precision, Recall, F1-score
- **Validation**: Cross-validation and confusion matrix analysis
- **Benchmarking**: Comparison with existing solutions

### Qualitative Evaluation
- **User Study**: 5-10 participants with think-aloud protocols
- **Interviews**: Structured interviews about user experience
- **Thematic Analysis**: Identification of key themes and insights
- **Learning Assessment**: Pre/post pattern recognition tests

## 📝 Academic Contributions

### Research Paper
- **Target Venues**: SCCUR, CHI SRC, undergraduate conferences
- **Focus**: Explainable AI for user empowerment
- **Novelty**: SHAP explanations for dark pattern detection
- **Impact**: Digital literacy and user awareness

### Open Source
- **Repository**: Public GitHub repository
- **License**: MIT License
- **Community**: Open to contributions and feedback
- **Documentation**: Comprehensive technical documentation

## 🔒 Privacy and Security

### Privacy by Design
- **No Data Collection**: No user data stored or transmitted
- **Local Processing**: All analysis performed on-device
- **Transparency**: Open source code and clear privacy policy
- **User Control**: Complete user control over functionality

### Security Measures
- **Input Validation**: Comprehensive input sanitization
- **Output Encoding**: XSS prevention and secure rendering
- **Error Handling**: Secure error messages and logging
- **Access Control**: Minimal permissions and secure APIs

## 📞 Support and Contributing

### Getting Help
- **Documentation**: Comprehensive guides and examples
- **Issues**: GitHub issue tracker for bug reports
- **Discussions**: GitHub discussions for questions
- **Community**: Open source community support

### Contributing
- **Code**: Pull requests welcome
- **Documentation**: Help improve documentation
- **Testing**: Report bugs and test new features
- **Research**: Contribute to user studies and evaluation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face**: For the Transformers library and DistilBERT model
- **SHAP**: For explainable AI capabilities
- **Chrome Extension APIs**: For browser integration
- **Research Community**: For dark pattern research and datasets

---

*This documentation is maintained as part of the Ethical Eye research project. For questions or contributions, please refer to the GitHub repository or contact the development team.*
