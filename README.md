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

✅ Diseases are evenly distributed to avoid bias during training.

---

## 🏗️ Project Pipeline

1. Convert symptom keywords into human-like sentences
2. Encode disease labels using `LabelEncoder`
3. Tokenize using `Bio_ClinicalBERT` tokenizer
4. Fine-tune ClinicalBERT using Hugging Face `Trainer`
5. Predict and evaluate using accuracy, precision, recall, and F1-score
6. Visualize performance using a Confusion Matrix

---

## 🧪 Model Performance

After 10 epochs of training, the model achieved:

| Metric     | Value  |
|------------|--------|
| Accuracy   | 100% ✅ |
| Precision  | 100% ✅ |
| Recall     | 100% ✅ |
| F1 Score   | 100% ✅ |

📌 This is due to:
- Clear symptom phrasing in natural language
- Balanced dataset
- Domain-specific language model (ClinicalBERT)

---

## 🙏 Acknowledgements

- `Model`: Bio_ClinicalBERT
- `Data`: Publicly available symptom-disease datasets from Kaggle
- `Framework`: Hugging Face Transformers
