# Stellar Age Prediction

> ℹ️ This project was developed when I was just starting to learn about machine learning. It may contain errors or limitations, but I keep it in my GitHub as part of my learning journey and to show my progress over time.

This project applies **machine learning techniques** to predict stellar ages from chemical abundances, using data from the **GALAH DR4 survey**. The objective is to identify correlations between stellar parameters and abundances, and to determine which features are most relevant for predicting stellar ages and galactic component membership.

---

## 📊 Dataset
- **Source**: GALAH DR4 survey (plus Gaia DR3, 2MASS and WISE cross-matches)  
- **Size**: ~917,588 stars across multiple catalogs  
- **Files (Datos/)**:  
  1. `galah_dr4_allstar_240705.fits` → Stellar parameters and abundances  
  2. `galah_dr4_vac_wise_tmass_gaiadr3_240705.fits` → Gaia DR3, 2MASS and WISE information  
  3. `galah_dr4_vac_dynamics_240705.fits` → Galactic kinematics and dynamics  
  4. `galah_dr4_vac_3dnlte_a_li_240705.fits` → Lithium abundances (3D NLTE)  

- **Target variables**:  
  - Galactic component classification (thin disk, thick disk and halo)  
  - Stellar age prediction via correlation with [Mg/Fe] abundances  

---

## ⚙️ Methodology
1. **Pre-processing**  
   - Import and combine catalogs by GALAH ID (`sobject_id`)  
   - Apply recommended GALAH flags (spectroscopic quality, abundance quality and SNR > 30)  
   - Replace invalid values with NaN and drop incomplete rows  
   - Sigma clipping and confidence intervals to remove outliers  

2. **Correlation analysis**  
   - Pearson and Spearman non-parametric correlation coefficients  
   - Identification of chemical abundances most correlated with stellar age  

3. **Classification models**  
   - Decision Trees and Random Forest classifiers  
   - Predict galactic component (thin disk, thick disk and halo) based on stellar parameters and abundances  

4. **Regression models**  
   - Random Forest regression to predict [Mg/Fe] abundances  
   - Use correlation between [Mg/Fe] and stellar age to infer age distributions  

---

## 📈 Results
- Outlier removal improves the reliability of parameter distributions  
- Random Forest classification identifies key features for galactic component membership  
- Random Forest regression highlights [Mg/Fe] as a strong predictor of stellar age  
- The methodology shows that chemical abundances, especially [Mg/Fe], are effective proxies for stellar age estimation  

---

## 📂 Project Structure
- **Stellar_Age_Prediction.ipynb**  
  Jupyter Notebook containing the full workflow: catalog import, pre-processing, outlier removal, correlation analysis, classification and regression models.  

- **Datos/**  
  Directory containing the GALAH DR4 and related catalogs (`.fits` files).  

- **Figuras/**  
  Directory storing only the figures explicitly saved with `plt.savefig(...)` during notebook execution. These include:  
  - RMSE and precision plots  
  - Decision tree visualization  
  - Correlation plots for [Fe/H], [Mg/Fe] and [Ti/Fe]  

---

## 🚀 How to Run
Clone the repository and open the notebook:

```bash
git clone https://github.com/cscheiding/data-science.git
cd "4 Stellar Age Prediction"
jupyter notebook Stellar_Age_Prediction.ipynb
