# Loan Prediction Based on Customer Behavior

A machine learning pipeline to predict loan defaults using customer demographic and behavioral data. This project covers data ingestion, cleaning, exploratory data analysis (EDA), feature engineering, model training with multiple algorithms, handling class imbalance, and performance comparison.

---

## Repository Structure

```
.
├── loan_data.csv  
├── LoanDefaultPrediction.ipynb
└── README.md
```

---

## Dataset

- **Demographic**: Age, Gender, Employment Status, City, State  
- **Financial**: Income, Existing Debts, Credit Score  
- **Loan Details**: Loan Amount, Loan Purpose, Term, Interest Rate  
- **Target**: `default` (0 = no default, 1 = default)

---

##  Notebook Overview

All the analysis and modeling steps are in `notebooks/LoanDefaultPrediction.ipynb`:

1. **Importing Libraries**  
2. **A) Dataset Exploration**  
   - Inspect data, check for nulls, duplicates, and data types  
3. **B) Data Cleaning**  
   - Standardize categorical values (e.g., city, state), handle missing values  
4. **C) Exploratory Data Analysis (EDA)**  
   - Univariate and bivariate analysis, correlations, feature distributions  
5. **D) Preprocessing**  
   - Train/Test split  
   - Encoding categorical features  
   - Scaling numerical features  
6. **E) Modeling**  
   - Feature selection  
   - Handling class imbalance with SMOTEENN  
   - Train & evaluate:
     - Logistic Regression  
     - Decision Tree  
     - Random Forest  
     - XGBoost  
   - Compare model performance (accuracy, precision, recall, ROC AUC)  
   - Identify most important features  

---

## Results

| Model               | Accuracy | Precision | Recall | ROC AUC |
|---------------------|---------:|----------:|-------:|--------:|
| Logistic Regression |    0.54  |     0.65  |   0.60 |   0.62  |
| Decision Tree       |    0.69  |     0.78  |   0.70 |   0.75  |
| Random Forest       |    0.83  |     0.98  |   0.75 |   0.88  |
| XGBoost             |    0.84  |     0.98  |   0.77 |   0.89  |

> **Best model:** XGBoost, achieving the highest ROC AUC of 0.89.

---

## Future Improvements

- **Hyperparameter Tuning:** GridSearchCV or Bayesian optimization  
- **Additional Features:** Incorporate behavioral metrics (e.g., payment timeliness)  
- **Ensemble Methods:** Stacking or blending different classifiers  
- **Deployment:** Wrap best model into a REST API  

---

## 📄 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

