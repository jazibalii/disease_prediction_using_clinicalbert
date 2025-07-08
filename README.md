# 🧠 Disease Prediction from Symptoms using ClinicalBERT

This project fine-tunes a domain-specific transformer model, **ClinicalBERT**, to predict diseases from natural language symptom descriptions.

Unlike traditional models that rely on keyword matching, ClinicalBERT captures the deep semantic understanding of patient complaints — enabling a more accurate and context-aware diagnosis pipeline.

---

## 📊 Dataset

The dataset contains structured mappings between symptoms and diagnosed diseases:

- `dataset.csv`: Symptom descriptions and labels
- `symptom_description.csv`: Full definitions of symptoms
- `symptom_precaution.csv`: Suggested precautions for each disease
- `Symptom_severity.csv`: Severity levels for all known symptoms

✅ Diseases are **evenly distributed** to avoid bias during training.

---

## 🏗️ Project Pipeline

### 🔧 Training Notebook (`disease_prediction_Bio_ClinicalBert.ipynb`)
1. Convert symptom keywords into **natural human-like sentences**
2. Encode disease labels using `LabelEncoder`
3. Tokenize with `Bio_ClinicalBERT` tokenizer
4. Fine-tune ClinicalBERT using Hugging Face `Trainer`
5. Evaluate using **Accuracy, Precision, Recall, F1-score**
6. Save model, tokenizer, and label encoder

### 🚀 Real-time Inference Notebook (`real_time_disease_prediction.ipynb`)
1. Load fine-tuned ClinicalBERT, tokenizer, and label encoder
2. Accept natural symptom inputs like:  
   `"I've been throwing up and my joints hurt, I don’t feel like eating"`
3. Predict **Top 3 Diseases** with confidence scores
4. Visualize predictions with:
   - ✅ **SHAP graphs** for symptom-level explainability
   - ✅ **LIME explanations** to interpret prediction logic
   - ✅ **Cosine similarity heatmap** of disease embeddings

---

## 🧪 Model Performance

After 10 epochs of training, the model achieved:

| Metric     | Value  |
|------------|--------|
| Accuracy   | 100% ✅ |
| Precision  | 100% ✅ |
| Recall     | 100% ✅ |
| F1 Score   | 100% ✅ |

📌 Reasons:
- Clear, natural symptom phrasing
- Balanced dataset
- Domain-specific medical language model (ClinicalBERT)

---

## 📈 Example Output (Real-Time Inference)

**Input:**  
`"Acidity, headache, and depression with loss of appetite"`

**Prediction:**

Top Predicted Diseases:

Typhoid (21.6%)
Migraine (15.4%)
Malaria (11.5%

---


✔️ Followed by SHAP explanation plot + cosine similarity heatmap

---

## 🖼️ Visuals & Explainability

- **SHAP Plot**: Highlights which words contributed most to the prediction
- **LIME**: Shows token-level influence on classification
- **Cosine Similarity Heatmap**: Reveals disease embedding closeness (e.g., Typhoid and Malaria are similar)

---

## 🙏 Acknowledgements

- **Model**: [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)
- **Data**: Publicly available symptom-disease datasets from Kaggle
- **Frameworks**:
  - [Transformers](https://huggingface.co/transformers/)
  - [scikit-learn](https://scikit-learn.org/)
  - [SHAP](https://github.com/shap/shap)
  - [LIME](https://github.com/marcotcr/lime)
  - [Seaborn](https://seaborn.pydata.org/)

---
