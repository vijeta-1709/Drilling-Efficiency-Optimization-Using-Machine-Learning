# Drilling Efficiency Optimization Using Machine Learning

This project aims to optimize drilling operations by predicting the **Rate of Penetration (ROP)** using machine learning based on key drilling parameters.

---

## 📊 Features Used

- **Depth (ft)**: 1000 to 5000 ft
- **WOB (klbf)**: Weight on Bit
- **RPM**: Rotary Speed
- **Torque (kNm)**: Derived from WOB and RPM
- **Flow Rate (gpm)**: Mud flow rate
- **Mud Weight (ppg)**: Density of the drilling fluid
- **Bit Dull Grade**: From 0 (new) to 8 (worn out)
- **MSE**: Mechanical Specific Energy
- **Formation Type**: Shale, Sandstone, Limestone (categorical)
- **ROP (ft/hr)**: 🎯 Target variable

---

## 🧠 ML Pipeline Overview

### 1. Data Preparation
- Loaded `Drilling_parameter.csv`
- Removed nulls, explored data types and distributions
- Separated features (`X`) and target (`ROP_ft_per_hr`)

### 2. Feature Engineering
- Encoded categorical feature `Formation` using `OrdinalEncoder`
- Scaled numerical features using `StandardScaler`
- Applied transformations using `ColumnTransformer`

### 3. Train-Test Split
- Used `train_test_split` (80% train / 20% test)

### 4. Model Comparison
Evaluated multiple regression models:
- Linear Regression
- Ridge, Lasso
- KNN Regressor
- Decision Tree
- Random Forest ✅ *(Selected)*
- XGBoost, Gradient Boosting, AdaBoost

### 5. Model Evaluation
Used:
- R² Score
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

**🎯 Final Model Chosen:** `RandomForestRegressor`  
**Reason:** Best balance between performance and generalization on test data.

---

### 🌐 6. Streamlit Web App
- Built an interactive UI using Streamlit (`app/app.py`)
- Users can input drilling parameters and get predicted ROP
- Engineered features (like WOB/RPM ratio) calculated live in the app

### 🐳 7. Docker Integration
- Created a `Dockerfile` to containerize the app
- Supports full reproducibility and deployment
- App runs with:
  ```bash
  docker build -t drilling-rop-app .
  docker run -p 8501:8501 drilling-rop-app

## 🚀 How to Run the App

### Option 1: Locally
```bash
pip install -r requirements.txt
streamlit run app/app.py
