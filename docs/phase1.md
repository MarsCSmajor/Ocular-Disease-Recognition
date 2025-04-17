# Project Part 1: Ocular Disease Recognition

## a. Project Description

**Goal of the Analysis:**  
The objective is to train a machine learning algorithm that accurately classifies ocular diseases using patient age, sex, and medical fundus images of the eyes.

**Analysis Plans:**
- Analyze feature importances
- Identify correlations between demographic groups and diseases
- Identify the best-performing models for prediction tasks

**Hypotheses/Expectations:**
- Discover the highest-risk demographic group (e.g., by sex, age range) for each disease
- Determine best practices in feature engineering, model selection, and performance

**Post-analysis Implementation:**
A web interface will be developed to:
- Present the dataset and statistical insights
- Deploy the trained model for predictions

### Web Interface Features:
- **Boolean search engine page:** Filter and view images by parameters  
- **Visualizations page:** Feature importances, demographic correlations, and model performance comparisons  
- **Prediction page:** User input field for model predictions  

**Dataset Link:**  
[ODIR5k - Kaggle Dataset](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k/data)

---

## b. Dataset Description

### Files Included:
- `ODIR-5k/`
  - `data.csv`
  - `Training Images/`
  - `Testing Images/`
  - `preprocessed_images/`
  - `Full_df.csv`

### Sample from `full_df.csv`

| ID | Age | Sex    | Left_Fundus | Right_Fundus | Left Diagnostic Keywords | Right Diagnostic Keywords | Diagnostic Label |
|----|-----|--------|-------------|--------------|---------------------------|----------------------------|------------------|
| 0  | 69  | Female | 0_left.jpg  | 0_right.jpg  | cataract                  | normal fundus              | C                |



### Sample from `data.csv`

| ID | Age | Sex    | Left_Fundus | Right_Fundus | Left Diagnostic Keywords | Right Diagnostic Keywords | Labels           |
|----|-----|--------|-------------|---------------|---------------------------|----------------------------|------------------|
| 0  | 69  | Female | 0_left.jpg  | 0_right.jpg   | cataract                  | normal fundus              | [0,0,0,1,0,0,0]  |


---

## Summary Checklist:
- [x] Project description with goals and hypotheses
- [x] Dataset overview with structure and samples

