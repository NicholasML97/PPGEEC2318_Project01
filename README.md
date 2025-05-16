# Model Card: CustomerPurchasePredictor

## Model Details
This model is a logistic regression classifier implemented in PyTorch. It was developed as part of a graduate-level machine learning course (PPGEEC2318) at UFRN. The goal of the model is to predict whether a customer will make a purchase based on demographic and behavioral features.

**Authors**: Leandro Roberto Silva Farias and Nicholas Medeiros Lopes  
**Institution**: Universidade Federal do Rio Grande do Norte (UFRN)  
**Course**: PPGEEC2318 - Machine Learning  
**Instructor**: Prof. Ivanovitch M. Silva  
**Model type**: Binary classification  
**Framework**: PyTorch  
**Status**: Research/Educational use only

## Intended Use
The model is intended for educational and research purposes, such as:
- Demonstrating how logistic regression is implemented in PyTorch
- Exploring model training, evaluation, and bias considerations

**Not intended for**:
- Production environments
- Financial, legal, or medical decision-making

## Factors
Relevant factors affecting model performance include:
- Distribution of age, gender, and income in the dataset
- Missing data handling
- Class imbalance (imbalanced purchase labels)

## Metrics
The model was evaluated using:
- **Accuracy** (~85%)
- **ROC-AUC Score** (~0.88)
- Confusion matrix visualization

## Evaluation Data
The dataset used was the [Customer Purchase Behavior Dataset](https://www.kaggle.com/datasets/rabieelkharoua/predict-customer-purchase-behavior-dataset) from Kaggle. The dataset was split into training and test sets using stratified sampling.

## Training Data
The dataset contains customer demographic and behavioral features, including:
- Age
- Estimated Salary
- Gender
- Purchased (target variable)

Preprocessing steps included:
- One-hot encoding for categorical variables
- StandardScaler for feature normalization
- Dropping rows with missing values

## Training Procedure
- **Optimizer**: Adam
- **Loss function**: BCEWithLogitsLoss
- **Epochs**: 100
- **Batch size**: 32
- **Hardware**: CPU

## Quantitative Analyses
- The model performs well on balanced subsets of the data
- Shows strong ROC-AUC performance, suggesting good separation between classes
- Limitations due to simplicity of logistic regression and unaddressed bias in data

## Ethical Considerations
- The dataset may contain inherent bias (e.g., gender, age)
- No fairness or demographic parity metrics were computed
- Intended only for academic and illustrative purposes

## Caveats and Recommendations
- Not suitable for deployment or real-world decision making
- Model performance may degrade on out-of-distribution data
- Should not be used without further validation and fairness analysis

## Citation

```bibtex
@misc{farias2025purchase,
  author = {Leandro R. S. Farias and Nicholas M. Lopes},
  title = {Customer Purchase Prediction Using Logistic Regression},
  year = {2025},
  note = {UFRN - PPGEEC2318 Machine Learning}
}
```
