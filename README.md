# Disease Prediction from Medical Data

## Project Overview
This project aims to **predict the possibility of diseases (specifically diabetes)** based on structured patient data. It leverages **machine learning models** to analyze medical features and output a prediction along with probability scores.

---

## Objectives
1. Handle medical dataset preprocessing (missing values, scaling).  
2. Train multiple classification models:  
   - Logistic Regression  
   - Support Vector Machine (SVM)  
   - Random Forest  
   - XGBoost  
3. Evaluate models using accuracy, precision, recall, F1-score, and probabilities.  
4. Select the **best-performing model** for real-world predictions.  
5. Simulate production/testing environment with new patient inputs.  
6. Prepare for deployment via Streamlit or other web apps.

---

## Dataset
- **Source:** [Pima Indians Diabetes Dataset – UCI ML Repository](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
- **Key Features:**  
  - Pregnancies  
  - Glucose  
  - BloodPressure  
  - SkinThickness  
  - Insulin  
  - BMI  
  - DiabetesPedigreeFunction  
  - Age  
- **Target:** `Outcome` (0 = No Disease, 1 = Disease)  

**Data Preprocessing:**  
- Replace zeros in features like Glucose, BloodPressure, SkinThickness, Insulin, and BMI with NaNs.  
- Impute missing values using mean (for Glucose, BloodPressure) and median (for SkinThickness, Insulin, BMI).  
- Scale features using `StandardScaler`.

---

## Models Used
| Model                  | Purpose                              |
|------------------------|--------------------------------------|
| Logistic Regression     | Baseline classifier                   |
| SVM                     | Robust boundary-based classifier      |
| Random Forest           | Ensemble method, best performance    |
| XGBoost                 | Gradient boosting for high accuracy  |

**Evaluation Metrics:**  
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Probability of Disease  

---

## Workflow
1. **Load and preprocess dataset**  
2. **Split data**: Train/Test (stratified)  
3. **Train models** and evaluate performance  
4. **Select best model** (Random Forest)  
5. **Save model and scaler** for production  
6. **Test model on new patient inputs** (single or batch)  

---

## Sample Test Cases

| Risk Level   | Example Patient Features                          | Prediction    | Probability |
|--------------|---------------------------------------------------|---------------|------------|
| Low Risk     | 1,95,70,20,30,22.0,0.2,30                        | No Disease    | 0.095      |
| Medium Risk  | 4,130,75,30,90,27.5,0.5,50                        | No Disease    | 0.350      |
| High Risk    | 8,200,85,45,150,36.0,0.9,70                       | Disease       | 0.818      |
| Critical     | 10,170,72,55,100,33.6,0.627,85                    | Disease       | 0.625      |

---

## Production Ready
- Model saved as `best_random_forest.pkl`  
- Scaler saved as `scaler.pkl`  
- Can predict **single patient** or **batch patients via CSV**  
- Probabilities provide **risk levels**, not just class labels  

---

## Notes
- Model is **robust**, logical, and interpretable.  
- Probabilities reflect **realistic medical risk**.  
- Can be **integrated with web app** (Streamlit/Flask/FastAPI).  

---

### References
- [Pima Indians Diabetes Dataset – Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
- Scikit-learn Documentation: [https://scikit-learn.org/](https://scikit-learn.org/)
