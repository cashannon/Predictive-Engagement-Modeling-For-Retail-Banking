# Bank Customer Engagement & Churn Prevention Analysis  
## Team Project: Predictive Analytics for Retail Banking Retention

**Team Members:** Christian Shannon, Ashley Love, Mugtaba Awad, and Kristian Livingston  

**Presentation Link:**  
[https://www.canva.com/design/DAG9koWKaK4/vhbgQZ2pM8lB56cZI8DwKg/edit?utm_content=DAG9koWKaK4&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton](https://www.canva.com/design/DAG9koWKaK4/vhbgQZ2pM8lB56cZI8DwKg/edit?utm_content=DAG9koWKaK4&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) 

---

## 1. Project Overview

This project reframes the UCI Bank Marketing dataset (41,188 records from Portuguese bank telemarketing campaigns between 2008–2010) from a term deposit “subscription prediction” task into a **customer churn prevention** and engagement framework.   
Instead of only predicting “yes” responses, we interpret persistent “no” outcomes and short calls as signals of disengagement, enabling the bank to identify at-risk customers and move from inefficient mass outbound calling to targeted, ROI-focused retention strategies. 

Key dataset notes:   
- 41,188 rows, 21 columns, no null values but strong class imbalance (≈88.7% “no” vs. 11.3% “yes”).  
- Semicolon-delimited CSV (bank-additional-full.csv) from Moro et al. (2014).  
- Contains socio-demographics, contact behavior, and macroeconomic indicators (e.g., euribor3m, emp.var.rate, cons.price.idx, cons.conf.idx).

---

## 2. Business Problem & Impact

### Business Problem

The original campaign suffers from ~88% “no” outcomes, driving high outbound cost per successful deposit and increasing customer fatigue.   
Our hypothesis: by identifying a **Risk Zone** of low-engagement profiles and focusing on high-propensity segments, the bank can reduce wasted calls and increase conversion precision, improving both profitability and customer experience. 

### Business Impact

Using a Random Forest model with SMOTE and a top-decile targeting strategy:   
- Overall model accuracy: **89.1%** on the test set.  
- ROC-AUC: **78.3%** (Random Forest + SMOTE).  
- When targeting the **top 10% highest-probability customers**, precision reaches **49.5%**, compared to an **11.3% baseline subscription rate**, capturing nearly half of potential “yes” responses while avoiding 90% of unsuccessful calls. 

This shift from volume-based dialing to a **value-based, propensity-driven targeting strategy** provides a scalable, ROI-focused approach to retention and engagement. 

---

## 3. Data Pipeline

### 3.1 Data Wrangling & Cleaning

Steps implemented in `P1-S1-Data-Wrangling.ipynb`: 

- **Loading & Structure**
  - Loaded `bank-additional-full.csv` with `sep=';'` and confirmed 41,188 rows × 21 columns.  
  - Verified types: 5 numeric (e.g., age, duration), 11 categorical (e.g., job, marital), and no nulls. 

- **Target & Class Imbalance**
  - Target `y`: “yes” vs. “no” term deposit subscription.  
  - Distribution: ~88.7% “no” and 11.3% “yes” (strong imbalance motivating SMOTE and precision-focused evaluation). 

- **pdays Handling (Contact History)**
  - Original `pdays` uses 999 to denote “never contacted.”  
  - Created `pdaysnevercontacted` indicator: 1 if `pdays` was 999 (never contacted), 0 otherwise.  
  - Re-coded 999 to -1 to align with the dataset documentation while keeping the feature usable. 

- **Unknown Categories**
  - Categorical variables (job, marital, education, default, housing, loan) contained “unknown” categories.  
  - Instead of dropping or imputing, retained “unknown” as an explicit category to preserve signal. 

- **Encoding & Feature Expansion**
  - One-hot encoded categorical predictors (dropping first level to avoid dummy variable trap).  
  - Expanded from 21 original columns to 54 fully numeric features, including the engineered `pdaysnevercontacted`. 

### 3.2 Leakage Prevention

- **Call Duration (`duration`)**
  - `duration` is highly predictive in EDA (longer calls ≈ higher engagement) but only known *after* the call ends.   
  - To avoid **target leakage**, `duration` was intentionally **excluded** from all predictive modeling steps, though retained for exploratory analysis and risk-zone visualization. 

---

## 4. Modeling Approach

Implemented in `P1-S2-Data-Modeling.ipynb`. 

### 4.1 Problem Framing & Split

- Binary classification: propensity to engage / subscribe (`y` mapped to 1 for “yes”, 0 for “no”).   
- Stratified train-test split (80/20) to preserve class imbalance structure.  
- Confirmed near-identical class distributions for train and test (~88.7% no / 11.3% yes). 

### 4.2 Preprocessing

Using `ColumnTransformer` and scikit-learn pipelines: 

- **Numeric features**: StandardScaler.  
- **Categorical features**: OneHotEncoder (drop first, `handle_unknown='ignore'`).  
- **Target**: binary encoded `y`.

### 4.3 Imbalance Handling: SMOTE

- Applied **Synthetic Minority Over-sampling Technique (SMOTE)** via an imbalanced-learn `Pipeline` **only on the training data**, ensuring no information leakage into the test set. 

### 4.4 Models

- **Logistic Regression (baseline)**
  - Pipeline: preprocessing → SMOTE → `LogisticRegression(max_iter=2000)`.   
  - Performance (test set):  
    - Accuracy: **82.5%**.  
    - ROC-AUC: **80.0%**.  
    - Minority-class precision: ~0.35, recall: ~0.66. 

- **Random Forest Classifier (primary model)**
  - Pipeline: preprocessing → SMOTE → `RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)`.   
  - Performance (test set):  
    - Accuracy: **89.1%**.  
    - ROC-AUC: **78.3%**.  
    - Top-decile precision: **49.5%** when targeting the 10% highest-probability customers. 

### 4.5 Top-Decile Targeting Strategy

- Generated probability scores for all test customers using Random Forest.  
- Sorted by `probability` and selected the top 10% records.  
- Calculated precision within this segment, which reached **≈0.495**, almost **5×** the overall baseline subscription rate (11.3%). 

This provides a concrete operational rule: **prioritize calls to the top 10% of high-propensity customers each campaign cycle**. 

---

## 5. Key Findings

### 5.1 Risk Zone: Critical Segment

From KDE heatmaps and segmentation plots in `P1-S3-Data-Visualization.ipynb`: 

- **Risk Zone definition**:
  - Customers aged **30–50**.  
  - Call **duration under 200 seconds**.  
  - Occurring during **unfavorable economic conditions** (higher `euribor3m`, adverse `emp.var.rate`, `cons.conf.idx`).  
- These customers show concentrated disengagement patterns, with subscription probability dropping sharply in this region. 

### 5.2 Top 10 Churn / Engagement Drivers

Based on Random Forest feature importance (with `duration` excluded from modeling, but re-analyzed for interpretability): 

1. `duration` (call length – strongest observed predictor in EDA)  
2. `euribor3m` (3-month Euribor rate – macroeconomic pressure)  
3. `age`  
4. `nr.employed` (number of employees – macroeconomic environment proxy)  
5. `campaign` (contact frequency)  
6. `pdaysnevercontacted` (never previously contacted indicator)  
7. `emp.var.rate`  
8. `cons.price.idx`  
9. `cons.conf.idx`  
10. `pdays` (days since last contact for previously contacted customers)

Behavioral and macroeconomic features dominate over pure demographics, signaling that **timing, context, and contact strategy** matter more than static profile attributes. 

### 5.3 Fatigue Threshold & Contact Limits

Campaign-level analysis shows: 

- Subscription rates **drop sharply after 3–4 campaign contacts**, indicating a fatigue point beyond which additional calls rarely convert and may harm customer experience.  
- This supports a **max 3-contact policy** per campaign to avoid harassment while preserving ROI. 

---

## 6. Technical Stack

Core libraries and tools used across notebooks: 

- **Data & Modeling**
  - `pandas`, `numpy`  
  - `scikit-learn` (ColumnTransformer, StandardScaler, OneHotEncoder, LogisticRegression, RandomForestClassifier, metrics)  
  - `imblearn` (SMOTE, Pipeline for train-only resampling)

- **Visualization**
  - `matplotlib`, `seaborn`  
  - KDE heatmaps (e.g., age vs. duration), histograms, bar plots, box plots, correlation heatmaps, feature importance charts.

- **Evaluation**
  - ROC curves, ROC-AUC  
  - Precision-Recall curves and Average Precision  
  - Confusion matrices, classification reports  
  - Top-decile precision targeting analysis.

Example evaluation metrics reported (Random Forest + SMOTE):   
- Accuracy ≈ 0.891  
- ROC-AUC ≈ 0.783  
- Top 10% precision ≈ 0.495

---

## 7. Visual Analytics Highlights

Implemented in `P1-S3-Data-Visualization.ipynb`: 

- **Data Quality & Class Imbalance Dashboard**
  - Baseline ~88/12 class split visualized, reinforcing the need for imbalance-aware metrics and SMOTE.  

- **Engagement Risk Zone Heatmap**
  - KDE density plot of **age vs. duration**, colored by subscription behavior, clearly exposing the **Risk Zone** for middle-aged customers with short calls.  

- **Economic Correlation & Feature Interactions**
  - Heatmaps and density plots for `euribor3m`, `emp.var.rate`, `cons.price.idx`, and `cons.conf.idx` show how economic stress amplifies disengagement risk.  

- **Campaign Fatigue Plot**
  - Subscription rate vs. `campaign` groups (1 call, 2–3, 4–6, 7+) shows **steep decline after 3–4 touches**.  

These visuals are designed to bridge technical findings to business decisions for non-technical stakeholders. 

---

## 8. Ethical & Governance Considerations

From the written proposal and stakeholder Q&A: 

- **Data Privacy & PII**
  - Uses anonymized, aggregated banking data (UCI Bank Marketing dataset) with no direct personally identifiable information.  

- **Regulatory Alignment**
  - Recommended **maximum of 3 campaign calls per customer**, aligned with GDPR-style contact limits and minimizing harassment risk.   

- **Model Transparency**
  - Preference for interpretable components:
    - Feature importance from Random Forest.  
    - Clear handling of high-leakage variables (e.g., excluding `duration` at training time).  
  - Human-in-the-loop usage: model is a decision-support tool for call prioritization, not an automated decision engine. 

- **Fairness & Non-Discrimination**
  - No use of protected demographic attributes for targeting decisions.  
  - Emphasis on behavioral and macroeconomic drivers rather than purely demographic segmentation. 

- **Customer Autonomy**
  - Strong recommendation to implement easy opt-out mechanisms in any production deployment and ensure transparent messaging about data use. 

---

## 9. Repository Structure

```text
.
├─ README.md                  # This file
├─ data/
│  └─ bank-additional-full.csv  # UCI Bank Marketing dataset
├─ notebooks/
│  ├─ P1-S1-Data-Wrangling.ipynb
│  ├─ P1-S2-Data-Modeling.ipynb
│  └─ P1-S3-Data-Visualization.ipynb
└─ reports/
   └─ Predictive-Engagement-Modeling-For-Retail-Banking.pdf
