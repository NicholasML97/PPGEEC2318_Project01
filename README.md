# Customer Purchase Prediction

In this project, we build a **binary classification model** using **PyTorch** to predict whether a customer will make a purchase based on a variety of demographic and behavioral features. This work was conducted as part of the *Machine Learning* course (PPGEEC2318) of the Graduate Program in Electrical and Computer Engineering at UFRN.

> **Professor:** Ivanovitch M. Silva  
> **Students:**  
> Leandro Roberto Silva Farias – 20251011748  
> Nicholas Medeiros Lopes – 20251011739

The complete pipeline includes fetching data, exploratory data analysis, preprocessing, data preparating, model training, evaluation, and reporting. The complete pipeline is contained in the `notebook.ipynb` file.

![Project's Pipeline](pipeline.png)
---

## Environment Setup

The following libraries are required to run the code:

```python
# Fetch data
import kagglehub
import os

# Data storing and analysis 
import pandas as pd
import numpy as np

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Data preprocessing and preparation
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, mutual_info_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# Model training
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, WeightedRandomSampler, SubsetRandomSampler
from torch.utils.data.dataset import random_split
import torch.optim as optim
import torch.nn as nn
```

---

## Dataset Description

We use the **Customer Purchase Behavior** dataset from [Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/predict-customer-purchase-behavior-dataset), which includes the following features:

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

### 1. Extract, Transform and Load (ETL)
The first step of the pipeline is to extract, transform and load the data. This is divided in the following substeps:

#### 1.1 Fetch data
Here, we extract the data from the kaggle website using the following commands

```python
# Download latest version of the dataset
path = kagglehub.dataset_download("rabieelkharoua/predict-customer-purchase-behavior-dataset")
print("Path to dataset files:", path)

# List files in the downloaded directory to find the CSV name
print(os.listdir(path))

# Load the CSV file (assuming it's named 'Admission_Predict.csv')
csv_file = os.path.join(path, 'customer_purchase_data.csv')
```

Then, the dataset is stores in a Pandas `Dataframe` variable.

```python
# Load into Pandas
df = pd.read_csv(csv_file)
````

#### 1.2 Exploratory Data Analysis (EDA)
- Displayng the variable types of the data set
 ```python
df.info()
```
- Check if there are missing values
```python
df.isnull().sum()
```

- Separating the categorical and numerical features
```python
categorical_features = ['gender', 'productcategory', 'loyaltyprogram', 'purchasestatus']
numerical_features = [col for col in df.columns if col not in categorical_features]
```

- Check data distribution
```python
for col in numerical_features:
    sns.histplot(df[col], kde=True) # Draw a histogram, use KDE to smooth data distribution
    plt.title(col)
    plt.show()
```

```python
for col in categorical_features:
    print(f'Column: {col}, Unique values: {df[col].nunique()}')
    sns.countplot(x=col, data=df)
    plt.title(col)
    plt.show()
```

- Check correlation between the features and the target variable
```python
df[categorical_features].apply(lambda x: mutual_info_score(x, df.purchasestatus)).sort_values()
df[numerical_features].apply(lambda x: mutual_info_score(x, df.purchasestatus)).sort_values()
```

#### 1.3 Preprocessing
Based on the EDA, we decided to drop two features of the dataset which didn't seem to impact the target variable: `Gender` and `ProductCategory`.
### 2. Exploratory Data Analysis

### 2. Data Preparation

- Split Dataset (Training/Validation)
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
