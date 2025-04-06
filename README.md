# Customer Segmentation and Classification Project

## Overview
This project focuses on understanding customer behavior by performing **segmentation using clustering techniques**, followed by building a **classification model** to predict customer segments based on purchasing behavior and demographics.

The project is divided into two parts:
- Part 1: Segmentation using Unsupervised Learning
- Part 2: Classification using Supervised Learning

The goal is to group customers into meaningful segments and predict segment membership for new customers.

---

## Objective

- ✔ Identify distinct customer segments for better targeting and personalization
- ✔ Build a classification model to predict customer segments
- ✔ Extract actionable insights from influential features
- ✔ Showcase a complete ML pipeline from data to prediction

---

## Dataset

- **Name:** Customer Personality Analysis sourced from Kaggle.
- **Description:** Includes customer demographics, spending behavior, campaign responses, and purchase channel information.
- **Use Case:** Adapted and extended the dataset to perform custom segmentation and classification tasks with domain-inspired insights.
---

## Part 1: Customer Segmentation

### Data Preprocessing
- Handled missing values and dropped redundant columns
- Created derived features: `Age`, `TotalSpend`, and `CustomerTenure`
- Applied `MinMaxScaler` for scaling numerical features
- Performed PCA for dimensionality reduction and visualization

### Clustering
- Applied KMeans, Hierarchical Clustering, and DBSCAN
- Finalized **KMeans with k=3** for best interpretability
- One less-useful cluster grouped as **"Other"** for clarity in classification
- Added new columns: `Cluster` (numerical) and `Cluster_description` (labels)

---

## Part 2: Customer Classification

### Goal
Predict customer cluster (excluding "Other") using relevant features.

### Selected Features
Only the most influential features were used:


<pre> 
```python
   selected_features = [
    'Income', 'MntWines', 'MntMeatProducts',
    'NumWebPurchases', 'NumCatalogPurchases',
    'NumStorePurchases', 'Response'
]
</pre>
 
## Models Used

✔ Logistic Regression (Baseline)  
✔ Random Forest Classifier  
✔ XGBoost  
✔ LightGBM (**Best Performer**)

---

## Final Model Performance (LightGBM)

- **Accuracy:** ~89%
- **Precision / Recall / F1-Score:** Balanced across both classes
- ROC Curve and Precision-Recall Curve confirmed robustness
- Feature importance visualized from the final LightGBM model

### Confusion Matrix
  [[186 16] [ 34 203]]

---

## Prediction (Tested)

- Used hardcoded input in the same order as `selected_features`
- Prediction was successful and consistent

---

## Model Export

Model saved using `joblib`:
<pre>
```python
import joblib
joblib.dump(lgb, 'lgb_customer_segment_model.pkl') 
</pre>

## Key Highlights

✔ End-to-end implementation from preprocessing to prediction .<br>
✔ Effective use of both unsupervised and supervised learning.<br>
✔ Clean feature selection and modeling pipeline.<br>
✔ Modular and notebook-friendly design for clear understanding.<br>

## Future Scope

→ Predict customer lifetime value or future revenue.<br>
→ Deploy model using Streamlit or Flask.<br>
→ Integrate segmentation with marketing campaigns or CRM systems.<br>

## Author
Developed by **Fiza Khan**, a data science enthusiast passionate about building impactful solutions through structured problem solving and storytelling via data.

## Contact

For queries, collaborations, or opportunities, feel free to reach out:

✉️ [mfizakhan05@gmail.com](mailto:mfizakhan05@gmail.com)

