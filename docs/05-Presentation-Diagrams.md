# Presentation Diagrams - Ethical Eye Extension

## 🎯 Overview

This document contains all the diagrams and visualizations needed for presenting the Ethical Eye research project. These diagrams are designed for academic presentations, conference talks, and research demonstrations.

---

## 1. Project Overview & Problem Statement

### 1.1 Dark Patterns Problem Visualization

```mermaid
graph TB
    subgraph "E-commerce Websites"
        A[User Visits Site] --> B[Dark Patterns Present]
        B --> C[Manipulative Design]
        C --> D[User Confusion]
        D --> E[Unintended Actions]
        E --> F[Privacy Violations]
        E --> G[Unwanted Purchases]
        E --> H[Subscription Traps]
    end

    subgraph "Current Solutions"
        I[Black-box Detection] --> J[No Explanations]
        J --> K[User Skepticism]
        K --> L[Limited Learning]
    end

    subgraph "Our Solution"
        M[Ethical Eye Extension] --> N[Transparent Detection]
        N --> O[SHAP Explanations]
        O --> P[User Education]
        P --> Q[Digital Literacy]
    end

    style B fill:#ff6b6b
    style C fill:#ff6b6b
    style D fill:#ff6b6b
    style E fill:#ff6b6b
    style M fill:#4ecdc4
    style N fill:#4ecdc4
    style O fill:#4ecdc4
    style P fill:#4ecdc4
    style Q fill:#4ecdc4
```

### 1.2 Research Objectives

```mermaid
mindmap
  root((Ethical Eye<br/>Research Goals))
    Functional Requirements
      Chrome Extension
        DistilBERT Model
        SHAP Explanations
        Real-time Detection
      User Interface
        Visual Highlighting
        Interactive Tooltips
        Confidence Scores
    Performance Metrics
      Accuracy >80%
      Response Time <2s
      User Satisfaction >4.0/5
      Learning Effectiveness
    Research Outcomes
      Academic Paper
        SCCUR Conference
        CHI SRC Submission
        HCI Contributions
      User Study
        5-10 Participants
        Think-aloud Protocols
        Thematic Analysis
      Open Source
        GitHub Repository
        Community Engagement
        Documentation
```

---

## 2. System Architecture Overview

### 2.1 High-Level System Architecture

```mermaid
graph TB
    subgraph "User Layer"
        A[Chrome Browser] --> B[Ethical Eye Extension]
        B --> C[Popup Interface]
        B --> D[Content Scripts]
    end

    subgraph "Processing Layer"
        E[Text Segmentation] --> F[ML Classification]
        F --> G[SHAP Explanation]
        G --> H[Visual Highlighting]
    end

    subgraph "AI/ML Layer"
        I[DistilBERT Model] --> J[Text Pattern Detection]
        J --> K[Confidence Scoring]
        K --> L[Feature Importance]
        M[Multimodal Model v2] --> N[Vision+Text Detection]
        N --> O[Visual Pattern Classification]
        P[Layout Analyzer] --> Q[HTML/CSS Analysis]
        Q --> R[Visual Misdirection Detection]
    end

    subgraph "Data Layer"
        M[Training Datasets] --> N[Model Artifacts]
        N --> O[User Study Data]
    end

    A --> E
    I --> M
    H --> C

    style A fill:#e1f5fe
    style B fill:#e8f5e8
    style I fill:#fff3e0
    style M fill:#f3e5f5
```

### 2.2 Component Interaction Flow

```mermaid
sequenceDiagram
    participant U as User
    participant E as Extension
    participant CS as Content Script
    participant API as Flask API
    participant ML as ML Pipeline
    participant SHAP as SHAP Explainer

    U->>E: Clicks "Analyze Site"
    E->>CS: Send analyze message
    CS->>CS: Extract text segments
    CS->>API: POST /analyze
    API->>ML: Process segments
    ML->>SHAP: Generate explanations
    SHAP->>ML: Return SHAP values
    ML->>API: Return predictions + explanations
    API->>CS: Return results
    CS->>CS: Highlight patterns
    CS->>E: Update count
    E->>U: Show results in popup

    Note over U,SHAP: Complete workflow in <2 seconds
```

---

## 3. Machine Learning Pipeline

### 3.1 ML Pipeline Architecture

```mermaid
graph LR
    subgraph "Input Processing"
        A[Raw Text] --> B[Tokenization]
        B --> C[Text Normalization]
        C --> D[Feature Extraction]
    end

    subgraph "Model Inference"
        D --> E[DistilBERT Encoder]
        E --> F[Classification Head]
        F --> G[Softmax Layer]
        G --> H[Confidence Scores]
    end

    subgraph "Explanation Generation"
        E --> I[SHAP Explainer]
        I --> J[Feature Importance]
        J --> K[Key Word Extraction]
        K --> L[Human-readable Explanation]
    end

    subgraph "Output Processing"
        H --> M[Pattern Classification]
        L --> M
        M --> N[Visual Highlighting]
    end

    style E fill:#ff9800
    style I fill:#4caf50
    style M fill:#2196f3
```

### 3.2 DistilBERT Model Architecture

```mermaid
graph TD
    subgraph "Input Layer"
        A[Text Input] --> B[Tokenization]
        B --> C[Embedding Layer]
    end

    subgraph "DistilBERT Encoder"
        C --> D[Multi-Head Attention]
        D --> E[Feed Forward Network]
        E --> F[Layer Normalization]
        F --> G[6 Transformer Layers]
    end

    subgraph "Classification Head"
        G --> H[Pooler Layer]
        H --> I[Dropout]
        I --> J[Linear Layer]
        J --> K[9 Output Classes]
    end

    subgraph "Dark Pattern Categories"
        K --> L[Urgency]
        K --> M[Scarcity]
        K --> N[Social Proof]
        K --> O[Misdirection]
        K --> P[Forced Action]
        K --> Q[Obstruction]
        K --> R[Sneaking]
        K --> S[Hidden Costs]
        K --> T[Not Dark Pattern]
    end

    style G fill:#ff9800
    style K fill:#4caf50
    style L fill:#f44336
    style M fill:#f44336
    style N fill:#f44336
    style O fill:#f44336
    style P fill:#f44336
    style Q fill:#f44336
    style R fill:#f44336
    style S fill:#f44336
    style T fill:#4caf50
```

---

## 4. Explainable AI (XAI) Framework

### 4.1 SHAP Explanation Process

```mermaid
graph LR
    subgraph "SHAP Processing"
        A[Text Tokens] --> B[SHAP Explainer]
        B --> C[Feature Importance]
        C --> D[Word-level Scores]
    end

    subgraph "Explanation Generation"
        D --> E[Top K Words]
        E --> F[Context Analysis]
        F --> G[Explanation Text]
    end

    subgraph "Visualization"
        G --> H[Tooltip Content]
        H --> I[Highlighted Words]
        I --> J[Confidence Display]
    end

    subgraph "User Education"
        J --> K[Pattern Description]
        K --> L[Learning Resources]
        L --> M[Digital Literacy]
    end

    style B fill:#4caf50
    style G fill:#2196f3
    style J fill:#ff9800
    style M fill:#9c27b0
```

### 4.2 Explanation Example Flow

```mermaid
graph TD
    A["Text: 'Hurry! Only 2 left in stock!'"] --> B[DistilBERT Analysis]
    B --> C[Classification: Urgency + Scarcity]
    C --> D[Confidence: 0.87]
    D --> E[SHAP Analysis]
    E --> F[Key Words: 'Hurry', 'Only', 'left']
    F --> G[Explanation: 'Creates false urgency and scarcity']
    G --> H[Visual Highlighting]
    H --> I[Interactive Tooltip]
    I --> J[User Learning]

    style A fill:#e3f2fd
    style C fill:#ffebee
    style D fill:#fff3e0
    style G fill:#e8f5e8
    style J fill:#f3e5f5
```

---

## 5. User Experience Flow

### 5.1 Complete User Journey

```mermaid
journey
    title User Experience with Ethical Eye
    section Installation
      Download Extension: 5: User
      Install in Chrome: 4: User
      Configure Settings: 3: User
    section Daily Usage
      Visit E-commerce Site: 5: User
      Click Analyze Button: 4: User
      View Highlighted Patterns: 5: User
      Read Explanations: 5: User
      Learn About Patterns: 5: User
    section Learning
      Understand Pattern Types: 5: User
      Recognize Similar Patterns: 4: User
      Make Informed Decisions: 5: User
      Share Knowledge: 4: User
```

### 5.2 User Interface Flow

```mermaid
stateDiagram-v2
    [*] --> ExtensionInstalled
    ExtensionInstalled --> PageLoaded
    PageLoaded --> UserClicksAnalyze
    UserClicksAnalyze --> TextExtraction
    TextExtraction --> MLProcessing
    MLProcessing --> PatternDetection
    PatternDetection --> VisualHighlighting
    VisualHighlighting --> UserInteracts
    UserInteracts --> TooltipDisplay
    TooltipDisplay --> UserClicksAnalyze
    UserInteracts --> [*]

    note right of MLProcessing
        <2 seconds processing time
    end note

    note right of TooltipDisplay
        SHAP explanations
        Confidence scores
        Learning content
    end note
```

---

## 6. Research Methodology

### 6.1 Evaluation Framework

```mermaid
graph TB
    subgraph "Quantitative Evaluation"
        A[Test Dataset<br/>235 Samples] --> B[Model Performance]
        B --> C[Accuracy >80%]
        B --> D[Precision/Recall]
        B --> E[F1-Score]
        B --> F[Confusion Matrix]
    end

    subgraph "Qualitative Evaluation"
        G[User Study<br/>5-10 Participants] --> H[Think-aloud Protocols]
        H --> I[Structured Interviews]
        I --> J[Thematic Analysis]
        J --> K[Learning Assessment]
    end

    subgraph "Research Outcomes"
        L[Academic Paper] --> M[SCCUR Conference]
        L --> N[CHI SRC Submission]
        O[Open Source Release] --> P[Community Engagement]
        Q[Digital Literacy Impact] --> R[User Empowerment]
    end

    C --> L
    K --> L
    J --> Q

    style A fill:#e3f2fd
    style G fill:#e8f5e8
    style L fill:#fff3e0
    style O fill:#f3e5f5
```

### 6.2 User Study Design

```mermaid
graph LR
    subgraph "Study Setup"
        A[Recruitment<br/>5-10 Participants] --> B[Pre-study Survey]
        B --> C[Baseline Assessment]
    end

    subgraph "Study Execution"
        C --> D[Think-aloud Session]
        D --> E[Pattern Recognition Test]
        E --> F[Extension Usage]
        F --> G[Post-task Interview]
    end

    subgraph "Data Analysis"
        G --> H[Audio Transcription]
        H --> I[Thematic Coding]
        I --> J[Pattern Analysis]
        J --> K[Insights Generation]
    end

    subgraph "Outcomes"
        K --> L[Learning Effectiveness]
        K --> M[User Satisfaction]
        K --> N[Digital Literacy Impact]
    end

    style A fill:#e3f2fd
    style D fill:#e8f5e8
    style H fill:#fff3e0
    style L fill:#f3e5f5
```

---

## 7. Performance Metrics

### 7.1 System Performance Dashboard

```mermaid
graph TB
    subgraph "Technical Metrics"
        A[Response Time<br/><2 seconds] --> B[Accuracy<br/>>80%]
        B --> C[Memory Usage<br/><100MB]
        C --> D[CPU Usage<br/>Minimal]
    end

    subgraph "User Experience Metrics"
        E[User Satisfaction<br/>>4.0/5] --> F[Learning Effectiveness<br/>Improved]
        F --> G[Usage Frequency<br/>Daily]
        G --> H[Retention Rate<br/>>60%]
    end

    subgraph "Research Metrics"
        I[Pattern Detection<br/>8 Categories] --> J[Explanation Quality<br/>SHAP-based]
        J --> K[Digital Literacy<br/>Enhanced]
        K --> L[User Empowerment<br/>Measured]
    end

    style A fill:#4caf50
    style B fill:#4caf50
    style E fill:#2196f3
    style I fill:#ff9800
```

### 7.2 Model Performance Visualization

```mermaid
graph LR
    subgraph "Training Phase"
        A[Training Data<br/>1000+ samples] --> B[DistilBERT Fine-tuning]
        B --> C[Validation<br/>80/20 split]
        C --> D[Model Optimization]
    end

    subgraph "Testing Phase"
        D --> E[Test Dataset<br/>235 samples]
        E --> F[Performance Metrics]
        F --> G[Confusion Matrix]
        G --> H[Category-wise Analysis]
    end

    subgraph "Deployment Phase"
        H --> I[Real-world Testing]
        I --> J[User Feedback]
        J --> K[Continuous Improvement]
    end

    style A fill:#e3f2fd
    style E fill:#e8f5e8
    style I fill:#fff3e0
```

---

## 8. Dark Pattern Categories

### 8.1 Pattern Classification Tree

```mermaid
graph TD
    A[Dark Patterns] --> B[Urgency]
    A --> C[Scarcity]
    A --> D[Social Proof]
    A --> E[Misdirection]
    A --> F[Forced Action]
    A --> G[Obstruction]
    A --> H[Sneaking]
    A --> I[Hidden Costs]
    A --> J[Not Dark Pattern]

    B --> B1["'Hurry! Limited time!'"]
    C --> C1["'Only 2 left in stock!'"]
    D --> D1["'Join 10,000+ users'"]
    E --> E1["Deceptive navigation"]
    F --> F1["Unnecessary requirements"]
    G --> G1["Intentional difficulty"]
    H --> H1["Hidden information"]
    I --> I1["Concealed fees"]
    J --> J1["Normal content"]

    style A fill:#f44336
    style B fill:#ff9800
    style C fill:#ff9800
    style D fill:#ff9800
    style E fill:#ff9800
    style F fill:#ff9800
    style G fill:#ff9800
    style H fill:#ff9800
    style I fill:#ff9800
    style J fill:#4caf50
```

### 8.2 Pattern Detection Examples

```mermaid
graph LR
    subgraph "Urgency Patterns"
        A["'Hurry! Limited time offer!'"] --> B[Confidence: 0.89]
        B --> C[SHAP: 'hurry', 'limited']
        C --> D[Explanation: Creates false time pressure]
    end

    subgraph "Scarcity Patterns"
        E["'Only 2 left in stock!'"] --> F[Confidence: 0.92]
        F --> G[SHAP: 'only', 'left']
        G --> H[Explanation: False limited availability]
    end

    subgraph "Social Proof Patterns"
        I["'Join 10,000+ users'"] --> J[Confidence: 0.76]
        J --> K[SHAP: 'join', 'users']
        K --> L[Explanation: Fake social validation]
    end

    style B fill:#4caf50
    style F fill:#4caf50
    style J fill:#4caf50
```

---

## 9. Technology Stack

### 9.1 Complete Technology Stack

```mermaid
graph TB
    subgraph "Frontend Technologies"
        A[Chrome Extension<br/>Manifest V3] --> B[HTML5/CSS3/JavaScript]
        B --> C[Chrome Extension APIs]
        C --> D[Custom UI Components]
    end

    subgraph "Backend Technologies"
        E[Flask API Server] --> F[Python 3.8+]
        F --> G[PyTorch Framework]
        G --> H[Transformers Library]
    end

    subgraph "ML/AI Technologies"
        I[DistilBERT Model] --> J[SHAP Explanations]
        J --> K[Scikit-learn]
        K --> L[NumPy/Pandas]
        M[Multimodal Model v2] --> N[MobileViT + DistilBERT]
        N --> O[PyTorch Vision]
        P[Layout Analyzer] --> Q[BeautifulSoup + CSSUtils]
    end

    subgraph "Development Tools"
        M[Git Version Control] --> N[Docker Containerization]
        N --> O[Jest/Pytest Testing]
        O --> P[GitHub Actions CI/CD]
    end

    style A fill:#e3f2fd
    style E fill:#e8f5e8
    style I fill:#fff3e0
    style M fill:#f3e5f5
```

### 9.2 Development Workflow

```mermaid
graph LR
    subgraph "Development Phase"
        A[Code Development] --> B[Unit Testing]
        B --> C[Integration Testing]
        C --> D[Code Review]
    end

    subgraph "Deployment Phase"
        D --> E[Build Process]
        E --> F[Quality Assurance]
        F --> G[Chrome Web Store]
    end

    subgraph "Research Phase"
        G --> H[User Study]
        H --> I[Data Collection]
        I --> J[Analysis]
        J --> K[Paper Writing]
    end

    style A fill:#e3f2fd
    style E fill:#e8f5e8
    style H fill:#fff3e0
```

---

## 10. Future Roadmap

### 10.1 Research Timeline

```mermaid
gantt
    title Ethical Eye Research Timeline
    dateFormat  YYYY-MM-DD
    section Phase 1: Core Implementation
    DistilBERT Integration    :active, p1, 2024-01-01, 2w
    SHAP Implementation       :p2, after p1, 1w
    UI/UX Development         :p3, after p2, 2w
    Testing & Validation      :p4, after p3, 1w

    section Phase 2: Research
    User Study Design         :p5, after p4, 1w
    Participant Recruitment   :p6, after p5, 2w
    Data Collection          :p7, after p6, 3w
    Analysis & Writing       :p8, after p7, 4w

    section Phase 3: Publication
    Paper Submission         :p9, after p8, 1w
    Conference Presentation  :p10, after p9, 2w
    Open Source Release      :p11, after p10, 1w
```

### 10.2 Future Enhancements

```mermaid
mindmap
  root((Future<br/>Enhancements))
    Technical Improvements
      Multi-language Support
        Internationalization
        Localized Models
        Cultural Adaptation
      Advanced ML Models
        BERT Variants
        Transformer Updates
        Custom Architectures
      Performance Optimization
        Model Quantization
        Edge Computing
        Caching Strategies
    User Experience
      Enhanced UI/UX
        Accessibility Features
        Customization Options
        Mobile Support
      Learning Features
        Interactive Tutorials
        Progress Tracking
        Gamification
    Research Extensions
      New Pattern Types
        Emerging Patterns
        Industry-specific
        Cultural Variations
      Advanced Analytics
        Usage Patterns
        Learning Outcomes
        Impact Measurement
```

---

## 11. Impact and Contributions

### 11.1 Research Impact

```mermaid
graph TB
    subgraph "Academic Contributions"
        A[Explainable AI for HCI] --> B[Transparent ML Systems]
        B --> C[User Empowerment]
        C --> D[Digital Literacy]
    end

    subgraph "Practical Impact"
        E[Consumer Protection] --> F[Informed Decision Making]
        F --> G[Privacy Awareness]
        G --> H[Ethical Design]
    end

    subgraph "Industry Impact"
        I[Design Guidelines] --> J[Ethical Standards]
        J --> K[Regulatory Compliance]
        K --> L[User Trust]
    end

    D --> E
    H --> I

    style A fill:#e3f2fd
    style E fill:#e8f5e8
    style I fill:#fff3e0
```

### 11.2 Open Source Ecosystem

```mermaid
graph LR
    subgraph "Open Source Benefits"
        A[Transparency] --> B[Community Trust]
        B --> C[Collaborative Development]
        C --> D[Knowledge Sharing]
    end

    subgraph "Community Engagement"
        D --> E[Developer Contributions]
        E --> F[User Feedback]
        F --> G[Continuous Improvement]
    end

    subgraph "Research Impact"
        G --> H[Reproducible Research]
        H --> I[Academic Collaboration]
        I --> J[Industry Adoption]
    end

    style A fill:#4caf50
    style E fill:#2196f3
    style H fill:#ff9800
```

---

## 12. Presentation Summary

### 12.1 Key Takeaways

```mermaid
graph TB
    subgraph "Problem Solved"
        A[Dark Pattern Detection] --> B[Transparent Explanations]
        B --> C[User Education]
        C --> D[Digital Literacy]
    end

    subgraph "Technical Innovation"
        E[DistilBERT + SHAP] --> F[Real-time Analysis]
        F --> G[Confidence Scoring]
        G --> H[Interactive Learning]
    end

    subgraph "Research Contribution"
        I[Explainable AI] --> J[User Empowerment]
        J --> K[Academic Publication]
        K --> L[Open Source Impact]
    end

    D --> E
    H --> I

    style A fill:#f44336
    style E fill:#4caf50
    style I fill:#2196f3
```

### 12.2 Call to Action

```mermaid
graph LR
    A[Install Extension] --> B[Try on E-commerce Sites]
    B --> C[Learn About Patterns]
    C --> D[Share Knowledge]
    D --> E[Contribute to Research]
    E --> F[Build Ethical Web]

    style A fill:#4caf50
    style C fill:#2196f3
    style E fill:#ff9800
    style F fill:#9c27b0
```

---

## 🎯 Presentation Usage Guide

### **Opening Slides (5-7 minutes)**

- Use **Problem Statement** diagram (Section 1.1)
- Show **Research Objectives** mindmap (Section 1.2)
- Present **System Architecture** overview (Section 2.1)

### **Technical Deep Dive (10-12 minutes)**

- Explain **ML Pipeline** (Section 3.1)
- Demonstrate **DistilBERT Architecture** (Section 3.2)
- Show **SHAP Explanation Process** (Section 4.1)

### **User Experience (5-7 minutes)**

- Present **User Journey** (Section 5.1)
- Show **UI Flow** (Section 5.2)
- Demonstrate **Pattern Examples** (Section 8.2)

### **Research Methodology (8-10 minutes)**

- Explain **Evaluation Framework** (Section 6.1)
- Detail **User Study Design** (Section 6.2)
- Show **Performance Metrics** (Section 7.1)

### **Results & Impact (5-7 minutes)**

- Present **Research Impact** (Section 11.1)
- Show **Future Roadmap** (Section 10.1)
- End with **Call to Action** (Section 12.2)

### **Q&A Preparation**

- Keep **Technology Stack** (Section 9.1) ready
- Have **Dark Pattern Categories** (Section 8.1) available
- Prepare **Future Enhancements** (Section 10.2) for discussion

---

_These diagrams are designed to be presentation-ready and can be easily exported or screenshotted for use in slides, posters, or conference presentations._
