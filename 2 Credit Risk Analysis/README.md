# Credit Risk Analysis

> ℹ️ This project was developed when I was just starting to learn about machine learning. It may contain errors or limitations, but I keep it in my GitHub as part of my learning journey and to show my progress over time.

This project applies statistical and machine learning techniques to predict possible credit defaulters using data from a financial institution. The objective is to identify upfront which customers are likely to default, helping institutions take preventive measures.

---

## 📊 Dataset
- **Source**: [Credit Risk Dataset – Kaggle](https://doi.org/10.34740/KAGGLE/DSV/2327131)  
- **Size**: 887,379 individuals, each described by 74 features  
- **Files**:  
  - `LCDataDictionary.xlsx` → Documentation of all features  
  - Data downloaded via KaggleHub (`loan.csv`)  

- **Target**: `loan_status`  
  - *Bad credit* (0): Default, Charged Off or Does not meet the credit policy (Charged Off)  
  - *Good credit* (1): Fully Paid, Current, Issued, In Grace Period, Late and others  
- **Class balance**: Only ~5.62% of instances correspond to bad credit, making the dataset imbalanced.

---

## ⚙️ Methodology
1. **Pre-processing**  
   - Stratified sampling (10% of each loan_status category to reduce computing time)  
   - Conversion of `term` column to integer (36 or 60 months)  
   - Removal of irrelevant columns (`id`, `member_id`, `url` and `desc`)  
   - Label encoding of categorical features  
   - Min-max scaling of numeric features  
   - Replacement of NaN values with column medians  

2. **Modeling**  
   - Comparison of three classifiers: Logistic Regression, Linear Support Vector Machines and Random Forest  
   - Grid Search over hyper parameters with cross-validation  
   - Evaluation with accuracy, precision, recall and F1 score  

3. **Feature importance**  
   - Permutation Feature Importance applied to the best-performing model (Linear SVM)  

---

## 📈 Results
- **Logistic Regression (best parameters: L1 penalty, no class weights)**  
  - Accuracy: 94.7%  
  - Precision: 94.7%  
  - Recall: 100%  
  - F1 score: 97.3%  

- **Linear Support Vector (best parameters: C = 1.0, L1 penalty, no class weights)**  
  - Accuracy: 99.6%  
  - Precision: 99.6%  
  - Recall: 100%  
  - F1 score: 99.8%  

- **Random Forest (best parameters: 100 trees, no class weights)**  
  - Accuracy: 97.7%  
  - Precision: 97.6%  
  - Recall: 100%  
  - F1 score: 98.8%  

The Linear Support Vector model achieves the highest F1 score, showing that it is the most effective classifier for this dataset.

---

## 📂 Project Structure
- **Credit_Risk_Analysis.ipynb**  
  Jupyter Notebook containing the full workflow: pre-processing, model training with Grid Search, evaluation metrics and feature importance analysis.  

- **LCDataDictionary.xlsx**  
  Data dictionary describing all 74 features in the dataset.  

---

## 🚀 How to Run
Clone the repository and open the notebook:

```bash
git clone https://github.com/cscheiding/data-science.git
cd "2 Credit Risk Analysis"
jupyter notebook Credit_Risk_Analysis.ipynb
