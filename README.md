# 🧠 Customer Purchase Prediction – Model Card

This repository contains a machine learning pipeline for predicting customer purchase behavior using logistic regression in PyTorch. The work was developed for the course **PPGEEC2318 - Machine Learning** under the **Graduate Program in Electrical and Computer Engineering** at **UFRN**.

## 👤 Authors

- Leandro Roberto Silva Farias — 20251011748  
- Nicholas Medeiros Lopes — 20251011739  
- **Professor**: Ivanovitch M. Silva

---

## 📌 Model Details

- **Type**: Binary classification (predicts customer purchase: yes or no)
- **Model**: Logistic Regression (PyTorch implementation)
- **Frameworks/Libraries**: PyTorch, Scikit-learn, Pandas, Seaborn, KaggleHub
- **Training Strategy**: Custom PyTorch training loop with accuracy and ROC-AUC evaluation

---

## ✅ Intended Use

- **Goal**: Predict whether a customer will make a purchase based on demographic and behavioral features.
- **Application**: Business intelligence, customer targeting, sales funnel optimization.

**Not intended for**:
- Real-time predictions without further validation
- Medical, legal, or financial decision-making without expert review

---

## 🧠 Dataset

- **Name**: Predict Customer Purchase Behavior Dataset  
- **Source**: [Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/predict-customer-purchase-behavior-dataset)
- **Content**: Includes features like `Age`, `Gender`, `EstimatedSalary`, etc., and a binary target `Purchased`
- **Preprocessing**:
  - Missing values removed
  - Categorical variables encoded
  - Features standardized using `StandardScaler`

---

## 📈 Evaluation

- **Metrics**:  
  - Accuracy  
  - ROC-AUC Score  
  - Confusion Matrix  
- **Validation**: Performed via train-test split with stratification
- **Insights**: ROC-AUC > 0.8 indicates good class separation

---

## ⚠️ Limitations

- Model performance is dependent on the quality of the input data.
- No hyperparameter tuning was performed.
- Results are not guaranteed to generalize outside the training dataset.

---

## 🔐 Ethical Considerations

- **Fairness**: Model fairness across demographic groups has not been evaluated.
- **Bias**: Potential for bias in features like `Gender` or `Age` if present in training data.
- **Transparency**: Source code and training procedure are publicly available in this notebook.

---
