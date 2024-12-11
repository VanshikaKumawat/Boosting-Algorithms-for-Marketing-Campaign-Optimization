
# Theoretical Foundations of Boosting Algorithms for Marketing Campaign Optimization

This project applies boosting algorithms (Gradient Boosting, AdaBoost, and XGBoost) to a marketing campaign dataset to classify customer responses. The algorithms are evaluated on performance metrics such as Accuracy, F1 Score, and AUC, demonstrating how these models can optimize marketing efforts by accurately identifying target segments.

## Project Resources
- **Colab Notebook**: [Open in Google Colab](https://colab.research.google.com/drive/1MrPxvvo-S9RcOw_lDr6wz8-Ff0hrLDPA?usp=sharing)
- **Dataset**: Download from [Kaggle](https://www.kaggle.com/datasets/rodsaldanha/arketing-campaign)

---

## Package Requirements

Ensure you have the following packages installed:

```plaintext
- pandas==1.3.3
- numpy==1.21.2
- scikit-learn==0.24.2
- xgboost==1.5.0
- imbalanced-learn==0.8.0
```

To install these packages, use:

```bash
pip install pandas==1.3.3 numpy==1.21.2 scikit-learn==0.24.2 xgboost==1.5.0 imbalanced-learn==0.8.0
```

## Run Instructions

### Step 1: Set Up Environment
1. **Clone the repository** (if available) or download the project files.
2. **Install dependencies** as listed above.

### Step 2: Prepare Dataset
1. Download the dataset from Kaggle.
2. Place the dataset file in the working directory or ensure it matches the file path in the code (e.g., `"/content/drive/MyDrive/marketing_campaign.csv"`).

### Step 3: Execute the Code

The code steps for implementing and evaluating boosting algorithms include:
1. **Importing Libraries**: Imports required libraries such as `pandas`, `numpy`, `scikit-learn`, and `xgboost`.
2. **Loading Dataset**: Loads and inspects the dataset.
3. **Preprocessing**:
   - Handling missing values.
   - Encoding categorical variables.
   - Splitting the data into training and testing sets.
4. **Addressing Class Imbalance**: Uses SMOTE to balance the dataset.
5. **Model Training and Evaluation**:
   - Functions for training models (Gradient Boosting, AdaBoost, XGBoost).
   - Hyperparameter tuning using `GridSearchCV`.
   - Evaluation using Accuracy, F1 Score, and AUC metrics.
6. **Model Summary**: Displays and compares the performance of each algorithm.

To run the code:

1. **In Jupyter Notebook or Colab**: Copy each section of the code from the file.
2. **In Python Script**:
   - Save the code in a `.py` file and execute using `python script_name.py`.

---

## Results

The models achieve the following performance on the dataset:

| Model            | Accuracy | F1 Score | AUC  |
|------------------|----------|----------|------|
| Gradient Boosting | 93.9%    | 0.937    | 0.977 |
| AdaBoost         | 88.4%    | 0.879    | 0.954 |
| XGBoost          | 92.6%    | 0.923    | 0.981 |

## Conclusion

Gradient Boosting and XGBoost provide high accuracy and AUC, with XGBoost showing efficiency on larger datasets. These results underscore the suitability of boosting algorithms for optimizing marketing campaign strategies.

---
