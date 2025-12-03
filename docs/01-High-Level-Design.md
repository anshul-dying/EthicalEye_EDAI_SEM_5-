# High Level Design (HLD) - Ethical Eye Extension

## 1. System Overview

### 1.1 Purpose

The Ethical Eye extension is a Chrome browser extension that detects and explains dark patterns on e-commerce websites using explainable AI (XAI) to empower users with transparent, evidence-based insights about manipulative design techniques.

### 1.2 Key Objectives

- **Detection**: Identify 8 categories of dark patterns in real-time
- **Explanation**: Provide SHAP-based explanations for transparency
- **Education**: Enhance user digital literacy through clear explanations
- **Empowerment**: Enable users to make informed decisions

### 1.3 Target Users

- Privacy-conscious consumers
- UX researchers and designers
- Digital literacy advocates
- General web users seeking transparency

## 2. System Architecture Overview

### 2.1 High-Level Components

```
┌─────────────────────────────────────────────────────────────┐
│                    ETHICAL EYE SYSTEM                      │
├─────────────────────────────────────────────────────────────┤
│  Chrome Extension Layer (Frontend)                         │
│  ├── Popup Interface                                        │
│  ├── Content Scripts                                       │
│  ├── Background Service Worker                             │
│  └── Visual Highlighting Engine                            │
├─────────────────────────────────────────────────────────────┤
│  ML Processing Layer (Backend)                             │
│  ├── DistilBERT Model (Text Classification)              │
│  ├── Multimodal Model v2 (Vision+Text)                   │
│  ├── Layout Analyzer (HTML/CSS Structure)                 │
│  ├── SHAP Explainer                                        │
│  ├── Confidence Scorer                                     │
│  └── Category Classifier                                   │
├─────────────────────────────────────────────────────────────┤
│  Data Layer                                                │
│  ├── Training Datasets                                     │
│  ├── Model Artifacts                                       │
│  └── User Study Data                                       │
└─────────────────────────────────────────────────────────────┘
```

## 3. Core Functional Requirements

### 3.1 Dark Pattern Detection

- **Input**: Webpage text content
- **Processing**: Real-time ML classification
- **Output**: Pattern type, confidence score, explanation

### 3.2 Pattern Categories

1. **Urgency** - Time pressure tactics
2. **Scarcity** - False limited availability
3. **Social Proof** - Fake social validation
4. **Misdirection** - Deceptive navigation
5. **Forced Action** - Unnecessary requirements
6. **Obstruction** - Intentional difficulty
7. **Sneaking** - Hidden information
8. **Hidden Costs** - Concealed fees
9. **Not Dark Pattern** - Normal content

### 3.3 Explainable AI Features

- **SHAP Values**: Feature importance scores
- **Key Words**: Highlighted contributing terms
- **Confidence Scores**: Model certainty levels
- **Visual Explanations**: Interactive tooltips

## 4. Non-Functional Requirements

### 4.1 Performance

- **Response Time**: < 2 seconds per page analysis
- **Accuracy**: > 80% on test dataset
- **Memory Usage**: < 100MB extension footprint
- **CPU Usage**: Minimal impact on browsing

### 4.2 Usability

- **User Interface**: Intuitive popup design
- **Visual Feedback**: Clear highlighting system
- **Accessibility**: WCAG 2.1 compliance
- **Cross-Platform**: Chrome browser compatibility

### 4.3 Reliability

- **Uptime**: 99.9% availability
- **Error Handling**: Graceful failure modes
- **Data Privacy**: No user data collection
- **Security**: Secure model inference

## 5. System Interactions

### 5.1 User Workflow

1. **Installation**: User installs extension
2. **Navigation**: User visits e-commerce site
3. **Analysis**: Extension automatically scans page
4. **Detection**: Dark patterns identified and highlighted
5. **Explanation**: User clicks for detailed explanations
6. **Learning**: User gains awareness and knowledge

### 5.2 Data Flow

1. **Content Extraction**: Webpage text segmentation
2. **ML Processing**: DistilBERT classification
3. **Explanation Generation**: SHAP value computation
4. **Visual Rendering**: Highlighting and tooltips
5. **User Interaction**: Click-to-explore functionality

## 6. Technology Stack

### 6.1 Frontend Technologies

- **Chrome Extension**: Manifest V3
- **Languages**: HTML5, CSS3, JavaScript ES6+
- **Frameworks**: Chrome Extension APIs
- **UI Components**: Custom popup and tooltip system

### 6.2 Backend Technologies

- **ML Framework**: Transformers (Hugging Face), PyTorch
- **Text Model**: DistilBERT-base-uncased
- **Multimodal Model**: MobileViT + DistilBERT (custom trained)
- **Vision Framework**: Torchvision, OpenCV
- **Layout Analysis**: BeautifulSoup, CSSUtils
- **XAI Library**: SHAP (SHapley Additive exPlanations)
- **API Framework**: Flask
- **Data Processing**: pandas, numpy

### 6.3 Development Tools

- **Version Control**: Git
- **Testing**: Jest, pytest
- **Documentation**: Markdown
- **Deployment**: Chrome Web Store

## 7. Integration Points

### 7.1 Chrome Extension APIs

- **Content Scripts**: Page content access
- **Storage API**: User preferences
- **Tabs API**: Active tab management
- **Runtime API**: Message passing

### 7.2 External Dependencies

- **Hugging Face Hub**: Model downloads
- **SHAP Library**: Explanation generation
- **Chrome Web Store**: Distribution platform

## 8. Security Considerations

### 8.1 Data Privacy

- **No Data Collection**: No user data stored
- **Local Processing**: All analysis on-device
- **Secure Communication**: HTTPS only
- **Minimal Permissions**: Least privilege access

### 8.2 Model Security

- **Model Validation**: Integrity checks
- **Input Sanitization**: Text preprocessing
- **Output Validation**: Result verification
- **Error Boundaries**: Graceful failure handling

## 9. Scalability Considerations

### 9.1 Performance Optimization

- **Model Quantization**: Reduced memory footprint
- **Caching**: Frequent pattern caching
- **Lazy Loading**: On-demand model loading
- **Batch Processing**: Efficient text processing

### 9.2 Future Enhancements

- **Multi-language Support**: Internationalization
- **Advanced Patterns**: New pattern types
- **User Customization**: Personal preferences
- **Analytics**: Usage insights (privacy-preserving)

## 10. Success Metrics

### 10.1 Technical Metrics

- **Accuracy**: > 80% classification accuracy
- **Precision**: > 75% per category
- **Recall**: > 70% per category
- **F1-Score**: > 72% overall

### 10.2 User Experience Metrics

- **User Satisfaction**: > 4.0/5.0 rating
- **Learning Effectiveness**: Improved pattern recognition
- **Usage Frequency**: Daily active usage
- **Retention Rate**: > 60% monthly retention

## 11. Risk Assessment

### 11.1 Technical Risks

- **Model Performance**: Accuracy below target
- **Browser Compatibility**: Extension conflicts
- **Performance Impact**: Slow page loading
- **Memory Usage**: High resource consumption

### 11.2 Mitigation Strategies

- **Extensive Testing**: Comprehensive test suite
- **Performance Monitoring**: Real-time metrics
- **Fallback Mechanisms**: Graceful degradation
- **User Feedback**: Continuous improvement

## 12. Future Roadmap

### 12.1 Phase 1: Core Implementation

- DistilBERT model integration
- SHAP explanation system
- Basic UI/UX implementation
- Initial testing and validation

### 12.2 Phase 2: Enhancement

- Advanced visualization features
- User study implementation
- Performance optimization
- Documentation completion

### 12.3 Phase 3: Research Publication

- Academic paper preparation
- Conference submission
- Open-source release
- Community engagement

---

_This document serves as the high-level design specification for the Ethical Eye extension, providing a comprehensive overview of the system architecture, requirements, and implementation strategy._
