# High Level Model Diagram - Ethical Eye Extension

## 1. System Overview Diagram

```mermaid
graph TB
    subgraph "User Interface Layer"
        A[Chrome Browser] --> B[Ethical Eye Extension]
        B --> C[Popup Interface]
        B --> D[Content Scripts]
        B --> E[Background Service Worker]
    end
    
    subgraph "Processing Layer"
        F[Text Segmentation] --> G[ML Classification]
        G --> H[SHAP Explanation]
        H --> I[Result Processing]
    end
    
    subgraph "Data Layer"
        J[DistilBERT Model]
        K[Training Datasets]
        L[User Study Data]
    end
    
    A --> F
    D --> F
    G --> J
    J --> K
    I --> C
    I --> D
```

## 2. Component Interaction Diagram

```mermaid
sequenceDiagram
    participant U as User
    participant C as Chrome Extension
    participant CS as Content Script
    participant API as Flask API
    participant ML as ML Pipeline
    participant SHAP as SHAP Explainer
    
    U->>C: Clicks "Analyze Site"
    C->>CS: Send analyze message
    CS->>CS: Extract text segments
    CS->>API: POST /analyze
    API->>ML: Process text segments
    ML->>SHAP: Generate explanations
    SHAP->>ML: Return SHAP values
    ML->>API: Return predictions + explanations
    API->>CS: Return results
    CS->>CS: Highlight patterns
    CS->>C: Update count
    C->>U: Show results in popup
```

## 3. Machine Learning Pipeline Diagram

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
```

## 4. Data Flow Architecture

```mermaid
flowchart TD
    subgraph "Frontend Layer"
        A[Web Page] --> B[Text Segmenter]
        B --> C[Segment Filter]
        C --> D[API Client]
    end
    
    subgraph "Backend Layer"
        D --> E[Flask API]
        E --> F[Request Validator]
        F --> G[ML Processor]
        G --> H[DistilBERT Model]
        H --> I[SHAP Explainer]
        I --> J[Result Formatter]
    end
    
    subgraph "Response Layer"
        J --> K[JSON Response]
        K --> L[Pattern Highlighter]
        L --> M[Visual Overlays]
        M --> N[User Interface]
    end
    
    subgraph "Storage Layer"
        O[Model Artifacts] --> H
        P[Training Data] --> H
        Q[Configuration] --> E
    end
```

## 5. Dark Pattern Classification Model

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
    
    subgraph "Output Classes"
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
```

## 6. SHAP Explanation Model

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
```

## 7. User Interaction Flow

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
```

## 8. System Deployment Architecture

```mermaid
graph TB
    subgraph "Client Side"
        A[Chrome Browser] --> B[Extension Files]
        B --> C[Content Scripts]
        B --> D[Popup Interface]
        B --> E[Background Worker]
    end
    
    subgraph "Server Side"
        F[Flask API Server] --> G[Model Loader]
        G --> H[DistilBERT Model]
        G --> I[SHAP Explainer]
        F --> J[Request Handler]
        J --> K[Response Generator]
    end
    
    subgraph "Data Storage"
        L[Model Artifacts] --> G
        M[Configuration] --> F
        N[Logs] --> F
    end
    
    C --> F
    E --> F
    K --> C
    K --> E
```

## 9. Performance Monitoring Model

```mermaid
graph TD
    subgraph "Metrics Collection"
        A[Response Time] --> E[Metrics Aggregator]
        B[Accuracy] --> E
        C[Memory Usage] --> E
        D[Error Rate] --> E
    end
    
    subgraph "Analysis Layer"
        E --> F[Performance Analyzer]
        F --> G[Alert Generator]
        G --> H[Notification System]
    end
    
    subgraph "Visualization"
        F --> I[Dashboard]
        I --> J[Charts]
        I --> K[Reports]
    end
```

## 10. Security and Privacy Model

```mermaid
graph TB
    subgraph "Security Layers"
        A[Input Validation] --> B[Output Encoding]
        B --> C[Error Handling]
        C --> D[Access Control]
    end
    
    subgraph "Privacy Protection"
        E[No Data Collection] --> F[Local Processing]
        F --> G[Memory Cleanup]
        G --> H[User Consent]
    end
    
    subgraph "Compliance"
        I[GDPR Compliance] --> J[Privacy by Design]
        J --> K[Data Minimization]
        K --> L[Transparency]
    end
```

## 11. Model Training Pipeline

```mermaid
graph LR
    subgraph "Data Preparation"
        A[Raw Datasets] --> B[Data Cleaning]
        B --> C[Text Preprocessing]
        C --> D[Train/Val/Test Split]
    end
    
    subgraph "Model Training"
        D --> E[DistilBERT Fine-tuning]
        E --> F[Hyperparameter Tuning]
        F --> G[Model Validation]
    end
    
    subgraph "Model Deployment"
        G --> H[Model Serialization]
        H --> I[SHAP Integration]
        I --> J[API Integration]
    end
    
    subgraph "Evaluation"
        G --> K[Performance Metrics]
        K --> L[Confusion Matrix]
        L --> M[User Study]
    end
```

## 12. Extension Architecture Layers

```mermaid
graph TB
    subgraph "Presentation Layer"
        A[Popup UI] --> B[Visual Components]
        B --> C[User Controls]
        C --> D[Status Display]
    end
    
    subgraph "Logic Layer"
        E[Content Scripts] --> F[Text Processing]
        F --> G[API Communication]
        G --> H[Result Processing]
    end
    
    subgraph "Service Layer"
        I[Background Worker] --> J[State Management]
        J --> K[Event Handling]
        K --> L[Message Passing]
    end
    
    subgraph "Data Layer"
        M[Chrome Storage] --> N[User Preferences]
        N --> O[Analysis History]
        O --> P[Extension State]
    end
    
    A --> E
    E --> I
    I --> M
```

## 13. API Request/Response Model

```mermaid
sequenceDiagram
    participant CS as Content Script
    participant API as Flask API
    participant ML as ML Pipeline
    participant SHAP as SHAP Explainer
    
    CS->>API: POST /analyze
    Note over CS,API: Request: {segments: [{text, element_id, position}]}
    
    API->>ML: Process segments
    ML->>SHAP: Generate explanations
    SHAP->>ML: Return SHAP values
    ML->>API: Return predictions
    
    API->>CS: JSON Response
    Note over API,CS: Response: {results: [{category, confidence, explanation, is_dark_pattern}], dark_pattern_count}
    
    CS->>CS: Highlight patterns
    CS->>CS: Update UI
```

## 14. Error Handling Model

```mermaid
graph TD
    subgraph "Error Detection"
        A[Input Validation] --> B[Model Errors]
        B --> C[Network Errors]
        C --> D[System Errors]
    end
    
    subgraph "Error Processing"
        D --> E[Error Classification]
        E --> F[Error Logging]
        F --> G[User Notification]
    end
    
    subgraph "Recovery Mechanisms"
        G --> H[Retry Logic]
        H --> I[Fallback Mode]
        I --> J[Graceful Degradation]
    end
```

## 15. User Study Data Model

```mermaid
erDiagram
    USER_STUDY_SESSION {
        string session_id PK
        string user_id
        timestamp start_time
        timestamp end_time
        json browser_info
        json device_info
    }
    
    PATTERN_DETECTION {
        string detection_id PK
        string session_id FK
        string url
        string pattern_type
        decimal confidence
        boolean user_feedback
        boolean explanation_helpful
        timestamp detected_at
    }
    
    USER_RESPONSE {
        string response_id PK
        string session_id FK
        string question_type
        text response_text
        integer response_rating
        timestamp created_at
    }
    
    USER_STUDY_SESSION ||--o{ PATTERN_DETECTION : contains
    USER_STUDY_SESSION ||--o{ USER_RESPONSE : contains
```

---

*This High Level Model Diagram document provides comprehensive visual representations of the Ethical Eye extension's architecture, data flow, and component relationships using Mermaid diagrams.*
