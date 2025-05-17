# Customer Purchase Prediction

In this project, we build a **binary classification model** using **PyTorch** to predict whether a customer will make a purchase based on a variety of demographic and behavioral features. This work was conducted as part of the *Machine Learning* course (PPGEEC2318) of the Graduate Program in Electrical and Computer Engineering at UFRN.

> **Professor:** Ivanovitch M. Silva  
> **Students:**  
> Leandro Roberto Silva Farias – 20251011748  
> Nicholas Medeiros Lopes – 20251011739

The complete pipeline includes exploratory analysis, preprocessing, model training, evaluation, and reporting.

---

## Environment Setup

Use the following to configure your environment:

```bash
pip install -r requirements.txt
```

To run the notebook:

```bash
jupyter notebook
```

---

## Dataset Description

We use the **Customer Purchase Behavior** dataset from [Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/predict-customer-purchase-behavior-dataset), which includes:

- `Age`
- `Gender` (0 = Male, 1 = Female)
- `Annual Income` (USD)
- `Number of Purchases`
- `Product Category` (0–4)
- `Time Spent on Website` (minutes)
- `Loyalty Program` (0/1)
- `Discounts Availed` (0–5)
- **Target:** `PurchaseStatus` (0 = No Purchase, 1 = Purchase)

This is a **balanced** binary classification problem with approximately 48% `0` and 52% `1` labels.

---

## Pipeline Overview

### 1. Exploratory Data Analysis

- Basic statistics
- Correlation heatmaps
- Purchase rate by category and gender
- Class balance check

### 2. Preprocessing

- Standardization of numerical features
- One-hot encoding of categorical variables
- Train/test split (70/15/15)

### 3. Model Training

- **Logistic Regression Model** implemented with PyTorch
- Loss function: Binary Cross Entropy
- Optimizer: Adam
- Batch training with DataLoader

### 4. Model Evaluation

- **Accuracy**
- **ROC AUC Score**
- Confusion matrix
- Probability distributions for each class

---

## Model Details

- **Type**: Binary logistic regression
- **Framework**: PyTorch
- **Date**: 2025
- **Training Algorithm**: SGD with Adam optimizer
- **License**: MIT
- **Citation**: Dataset and model referenced from Kaggle and UFRN academic course project

---

## Metrics

| Metric         | Value     |
|----------------|-----------|
| Accuracy       | ~0.81     |
| ROC AUC Score  | ~0.88     |

> Threshold for classification set at 0.5.

---

## Intended Use

- **Educational**: Demonstrate classification model development from EDA to deployment.
- **Not intended**: For production or real financial/customer decision-making systems.

---

## Ethical Considerations

- Dataset is synthetic: No real user data involved.
- No fairness analysis included.
- No bias mitigation strategies applied.

---

## Caveats and Recommendations

- Do not apply this model to real-world data without retraining and thorough testing.
- Consider additional features like time-series behavior, clickstream, or device metadata for better real-world performance.
- Explore fairness and interpretability tools for further development.

---

## Future Work

- Add model versioning using DVC.
- Deploy model using FastAPI.
- Add CI/CD with GitHub Actions.
- Test other classification models (e.g., Random Forest, XGBoost).

---

## How to Cite

```bibtex
@misc{customerpurchase2025,
  author = {Farias, Leandro and Lopes, Nicholas},
  title = {Customer Purchase Behavior Prediction},
  year = {2025},
  note = {Graduate project, UFRN - PPGEEC2318}
}
```
