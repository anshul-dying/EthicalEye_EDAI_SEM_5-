## Ethical Eye: Explainable Dark Pattern Detection in E‑Commerce

### 1. Problem Statement and Motivation

- **Dark patterns** are deceptive UX design strategies that nudge users toward decisions benefiting platforms at the expense of users (e.g., fake urgency, hidden fees).
- They are widespread on **e‑commerce** sites and are difficult for non‑experts to recognize in real time.
- Existing tools either:
  - focus on static rule‑based detection with limited coverage, or  
  - use opaque ML models without **explanations**, which undermines user trust.
- **Research gap**: There is limited work on **explainable AI (XAI)** tools that both detect dark patterns and help users understand *why* the content is manipulative.

**Ethical Eye** addresses this gap with a Chrome extension that performs real‑time dark pattern detection and SHAP‑based explanations directly in the browser.

---

### 2. Research Objectives

- **O1 – Detection**: Train a text classification model to identify **8 dark pattern categories** plus a *Not Dark Pattern* class on real e‑commerce content.
- **O2 – Explanation**: Integrate **SHAP** explanations to highlight influential words and provide human‑readable justifications.
- **O3 – User Empowerment**: Design a UI that surfaces explanations via **on‑page highlights and tooltips** to improve users’ dark‑pattern literacy.
- **O4 – Evaluation**: Quantitatively evaluate model performance and qualitatively assess user experience and learning.

---

### 3. System Overview

- **Form factor**: Chrome extension (Manifest V3) with a Python/Flask backend.
- **Core pipeline**:
  1. **Content scripts** extract candidate text segments from the active page.
  2. Segments are sent to a **Flask API** exposing a DistilBERT‑based classifier.
  3. The API returns:
     - predicted **category**,
     - **confidence score**, and
     - **SHAP‑based explanation** (key words + narrative text).
  4. The extension overlays **highlights** on the page and shows explanations in **interactive tooltips** and the **popup**.

**Dark pattern classes**:
1. Urgency  
2. Scarcity  
3. Social Proof  
4. Misdirection  
5. Forced Action  
6. Obstruction  
7. Sneaking  
8. Hidden Costs  
9. Not Dark Pattern  

---

### 4. Architecture Summary

- **Presentation layer (Chrome extension)**  
  - `popup` UI: analyze button, summary of detected patterns, settings.  
  - Content scripts: `TextSegmenter` for DOM traversal and text extraction; `PatternHighlighter` for visual overlays + tooltips.  
  - Background service worker: orchestrates analysis requests and maintains extension state.

- **Business logic / ML layer**  
  - DistilBERT‑based classifier fine‑tuned for 9‑way dark‑pattern classification.  
  - Confidence scoring and category mapping.  
  - **SHAP explanation module** to compute token‑level importances and generate textual rationales (key words, confidence factors).

- **Data / API layer**  
  - Flask REST API (`/analyze`, `/analyze_single`, `/patterns`) for batch and single‑segment analysis.  
  - Model loading and caching from `models/ethical_eye/final_model`.  
  - Optional schema for **user study data** (sessions, detections, responses) for research logging.

Quality attributes:
- Response time target \< 2s per page, accuracy target \> 80%, no persistent storage of user content, least‑privilege permissions.

---

### 5. Dataset and Pre‑Processing

- **Sources**: Multiple dark‑pattern datasets plus manually curated examples of normal content.
- **Label space**: 8 dark‑pattern categories + Not Dark Pattern; labels mapped to the categories used in the extension UI.
- **Pre‑processing steps**:
  - text cleaning and normalization,
  - filtering of very short or boilerplate segments,
  - max sequence length capped at 512 tokens,
  - stratified **train/validation/test** split (e.g., 235+ samples reserved for testing in early experiments; 253+ in later balanced runs).
- **Balanced dataset**: A separate balanced dataset (`balanced_dataset.csv`) is used to reduce label imbalance and improve per‑class performance.

---

### 6. Model Design and Training

- **Base model**: `distilbert-base-uncased` with a task‑specific classification head.
- **Output**: 9 logits (one per class) → softmax → predicted category + confidence.
- **Training setup**:
  - Optimizer: AdamW with weight decay.  
  - Standard transformer fine‑tuning hyperparameters (batch size, learning rate scheduling, dropout).  
  - Early stopping / best‑model selection by **validation F1**.  
  - Implementation via Hugging Face `Trainer` for reproducible training and metric logging.

- **Metrics**:
  - Overall accuracy, precision, recall, F1.  
  - Per‑class precision/recall/F1 and confusion matrix to understand failure modes.  
  - Additional plots: ROC, precision‑recall curves, per‑class F1, and SHAP summaries for the paper.

---

### 7. Explainable AI (SHAP) Integration

- **SHAP explainer** built on top of the fine‑tuned DistilBERT model.
- For each input segment:
  - compute token‑level SHAP values,  
  - extract **top‑K influential words** (highest absolute SHAP values),  
  - generate a short **natural language explanation**, e.g.,  
    “Key indicators: *hurry*, *limited time*, *only X left*.”
- The explanation object returned by the API includes:
  - `top_words` (token, importance),  
  - narrative explanation text,  
  - additional confidence/uncertainty factors.
- On the frontend, the highlighter:
  - uses SHAP information to **color‑code** segments,  
  - surfaces explanations through **tooltips** and the popup,  
  - aims to make the link between text and model decision transparent.

---

### 8. Chrome Extension UX and Interaction

- User installs Ethical Eye and enables it in Chrome.  
- When visiting an e‑commerce page, they can:
  - click “Analyze Site” in the popup, or  
  - rely on automatic analysis (depending on configuration).
- The extension:
  1. parses the DOM and extracts meaningful text blocks (product descriptions, banners, dialogs, etc.);
  2. calls the API for batch analysis;
  3. visually **highlights** segments flagged as dark patterns above a confidence threshold (e.g., 0.7);
  4. shows **pattern category, confidence, and SHAP explanation** on hover.
- The popup summarizes:
  - total number of detected patterns,  
  - breakdown by category,  
  - quick educational descriptions for each pattern type.

This UX supports both **just‑in‑time warnings** and **longer‑term learning** about dark patterns.

---

### 9. Experimental Results (Quantitative)

Two main evaluation checkpoints are available:

- **Earlier DistilBERT experiment** (unbalanced / earlier pipeline)  
  - Overall accuracy ≈ **88.1%**  
  - Overall F1 ≈ **0.88**  
  - Per‑class F1 ranged from ~0.76 (Urgency) to 1.00 (Forced Action).

- **Improved simple model / balanced setting** (`simple_improved_results.json`)  
  - Overall accuracy ≈ **97.6%**  
  - Overall F1 ≈ **0.98**  
  - Most classes achieve F1 ≈ 0.90–1.00, with very few confusions between categories.

The model consistently meets and exceeds the original targets:
- **Target**: accuracy \> 80%, F1 \> 0.72 overall.  
- **Achieved**: accuracy up to ~97.6%, strong per‑class F1, and clean confusion matrices.

These results are visualized through:
- confusion matrices,  
- per‑class F1 plots,  
- ROC and precision‑recall curves,  
- SHAP importance and summary plots (used directly as paper figures).

---

### 10. User Study and Qualitative Evaluation (Planned)

- **Participants**: 5–10 users with varied technical and UX backgrounds.
- **Procedure**:
  - pre‑study questionnaire on dark‑pattern awareness,  
  - task‑based browsing of instrumented e‑commerce pages while using Ethical Eye,  
  - **think‑aloud** protocol to capture real‑time reactions,  
  - post‑study questionnaire on:
    - perceived usefulness of explanations,  
    - trust in the system,  
    - self‑reported learning and confidence in detecting dark patterns.
- **Data collected** (optional, via the user‑study schema):
  - pattern detections (type, confidence, helpfulness of explanation),  
  - user responses and ratings (e.g., 1–5 Likert scales),  
  - qualitative interview transcripts for thematic analysis.

Planned analysis:
- descriptive statistics and visualizations (satisfaction, learning effectiveness),  
- thematic coding of open‑ended responses (e.g., how explanations changed decisions),  
- triangulation with quantitative results to support the research claims.

---

### 11. Security, Privacy, and Ethical Considerations

- **Privacy by design**:
  - No persistent storage of raw user browsing content by default.  
  - Processing either happens locally or via a controlled API with no user identifiers.  
  - No integration with third‑party tracking services.
- **Security**:
  - strict input validation and output encoding on the API side,  
  - minimal Chrome permissions (`activeTab`, `scripting`, etc.),  
  - defensive error handling and logging (avoiding sensitive data in logs).
- **Ethical aspects**:
  - clear communication that predictions are **assistive**, not authoritative;  
  - emphasis on **user education and agency**, not automated blocking of content;  
  - alignment with digital‑rights and AI‑ethics principles (transparency, user control).

---

### 12. Contributions and Novelty

- **Technical**:
  - A full pipeline for **dark‑pattern text classification** with high accuracy.  
  - Integration of **SHAP explanations** into a browser extension UX for real‑time, on‑page explanations.  
  - Reusable training scripts, SHAP tooling, and research plotting code.

- **Research**:
  - Empirical evidence that transformer‑based models can robustly detect multiple dark‑pattern categories in the wild.  
  - A concrete design pattern for embedding XAI into consumer‑facing tools.  
  - A dataset and evaluation protocol that can be reused or extended in future work.

- **Societal**:
  - Practical tool to support **digital literacy** and **user empowerment** on e‑commerce sites.  
  - Potential foundation for regulatory or compliance tools that audit interfaces for manipulative patterns.

---

### 13. Suggested Paper Structure (How to Use This Overview)

You can map this document directly into a research paper:
- **Introduction / Background**: Sections 1–2 (problem, objectives, dark‑pattern context).  
-, **Related Work**: Compare to prior dark‑pattern detection and XAI tools (to be written using literature).  
- **System Design / Methodology**: Sections 3–8 (architecture, dataset, model, SHAP, UX).  
- **Experiments and Results**: Section 9 (+ figures from `plots/research/`).  
- **User Study**: Section 10 (design + qualitative findings once study is complete).  
- **Discussion & Ethics**: Section 11 (limitations, risks, ethical implications).  
- **Conclusion & Future Work**: Section 12 (contributions, extensions such as more pattern types, multi‑language support, deployment at scale).

This file is intended as a **bridge between the implementation and your research paper** so you can quickly copy, adapt, and expand each subsection for your thesis or conference submission.


