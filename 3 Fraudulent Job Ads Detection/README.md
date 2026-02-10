# Fraudulent Job Ads Detection with Autoencoders

This project applies **dimensionality reduction with neural networks (autoencoders)** to improve predictive analysis of fraudulent job postings. It compares a baseline logistic regression model with a latent representation obtained from an autoencoder.

---

## 📊 Dataset
- **Source**: [Kaggle Recruitment Scam Dataset](https://www.kaggle.com/datasets/amruthjithrajvr/recruitment-scam)  
- **Size**: 17,880 job ads (17,014 legitimate and 866 fraudulent)  
- **Period**: 2012–2014  
- **Features**: 18 columns including job title, location, department, salary range, company profile, description, requirements, benefits, telecommuting, logo presence, employment type, required experience/education, industry, function and fraud label.

---

## ⚙️ Methodology
1. **Pre-processing**  
   - Feature selection (removal of text-heavy columns and irrelevant fields)  
   - Salary range split into `min_salary` and `max_salary`  
   - Boolean encoding (telecommuting, logo, questions and fraud)  
   - Handling categorical variables and missing values  
   - Train/test split with stratification  

2. **Class balancing**  
   - Applied **over-sampling** with **SMOTE-NC** to address imbalance (fraudulent ads ≈ 5%).  

3. **Baseline model**  
   - Logistic Regression with randomized search over hyper parameters (`RandomizedSearchCV`)  
   - Evaluation with accuracy, precision, recall, F1 score and confusion matrix  

4. **Dimensionality reduction**  
   - Autoencoder trained to compress features into a latent representation  
   - Logistic Regression retrained on latent features  

---

## 📈 Results
- **Baseline Logistic Regression (threshold = 0.8)**  
  - Accuracy: 96.3%  
  - Precision: 86.8%  
  - Recall: 62.2%  
  - F1 score: 72.4%  

- **Autoencoder + Logistic Regression (threshold = 0.8)**  
  - Accuracy: 96.7%  
  - Precision: 89.1%  
  - Recall: 66.2%  
  - F1 score: 75.9%  

The autoencoder improves accuracy, precision, recall and F1 score, showing that latent representations can enhance fraud detection performance compared to the baseline.

---

## 📂 Project Structure
- **Dimensionality_reduction.ipynb**  
  Main Jupyter Notebook containing the full workflow: pre-processing, over-sampling, logistic regression baseline, autoencoder training and evaluation.  

- **Data/DataSet.csv**  
  Raw dataset of job ads (17,880 rows, 18 columns) including legitimate and fraudulent postings.  

- **Tools/**  
  Directory with helper scripts:  
  - `preprocessing.py` → Functions for feature engineering and pre-processing, including salary parsing, One-Hot encoding for categorical variables, Standard Scaling for numeric features and passthrough for boolean features.  
  - `regression_and_metrics.py` → Utilities for training logistic regression models with randomized hyper parameter search, computing accuracy, precision, recall and F1 score, and plotting confusion matrices.  
  - `autoencoding.py` → Functions to build, train and evaluate autoencoders. Includes encoder/decoder model creation, fitting with callbacks (early stopping and learning rate reduction), and evaluation of latent representations with logistic regression.  

---

## 🚀 How to Run
Clone the repository and open the notebook:

```bash
git clone https://github.com/cscheiding/data-science.git
cd "3 Fraudulent Job Ads Detection"
jupyter notebook Dimensionality_reduction.ipynb
