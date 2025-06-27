#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings

warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("Drilling_Dataset.csv")


# In[3]:


df.head()


# In[4]:


df.isnull().sum()


# In[5]:


df.info()


# In[6]:


df.shape


# In[7]:


df['Formation'].unique()


# In[8]:


from sklearn.model_selection import train_test_split
X = df.drop(['ROP_ft_per_hr'], axis=1)
y = df['ROP_ft_per_hr']


# In[9]:


X.head()


# In[10]:


# Create Column Transformer with 3 types of transformers
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer

# Select numerical and object features
num_features = X.select_dtypes(exclude="object").columns
object_features = ['Formation']

# Define transformers
numeric_transformer = StandardScaler()
categorical_transformer = OrdinalEncoder()

# Combine them using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("OrdinalEncoder", categorical_transformer, object_features),
        ("StandardScaler", numeric_transformer, num_features)
    ],
    remainder='passthrough'
)


# In[11]:


X=preprocessor.fit_transform(X)


# In[12]:


pd.DataFrame(X)


# In[13]:


# separate dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
X_train.shape, X_test.shape


# In[14]:


X_train


# In[15]:


## Model Training And Model Selection


# In[16]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# In[17]:


##Create a Function to Evaluate Model
def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square


# In[18]:


## Beginning Model Training
models = {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "K-Neighbors Regressor": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "Adaboost Regressor":AdaBoostRegressor(),
    "Graident BoostRegressor":GradientBoostingRegressor(),
    "Xgboost Regressor":XGBRegressor()  
}

for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train, y_train) # Train model

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Evaluate Train and Test dataset
    model_train_mae , model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)

    model_test_mae , model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)


    print(list(models.keys())[i])

    print('Model performance for Training set')
    print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
    print("- R2 Score: {:.4f}".format(model_train_r2))

    print('----------------------------------')

    print('Model performance for Test set')
    print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
    print("- R2 Score: {:.4f}".format(model_test_r2))

    print('='*35)
    print('\n')


# In[19]:


#Initialize few parameter for Hyperparamter tuning
knn_params = {"n_neighbors": [2, 3, 10, 20, 40, 50]}
rf_params = {"max_depth": [5, 8, 15, None, 10],
             "max_features": [5, 7, "auto", 8],
             "min_samples_split": [2, 8, 15, 20],
             "n_estimators": [100, 200, 500, 1000]}
xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8, 12, 20, 30],
                  "n_estimators": [100, 200, 300],
                  "colsample_bytree": [0.5, 0.8, 1, 0.3, 0.4]}


# In[20]:


# Models list for Hyperparameter tuning
randomcv_models = [('KNN', KNeighborsRegressor(), knn_params),
                   ("RF", RandomForestRegressor(), rf_params),
                   ("XGB", XGBRegressor(), xgboost_params )

                   ]


# In[21]:


##Hyperparameter Tuning
from sklearn.model_selection import RandomizedSearchCV

model_param = {}
for name, model, params in randomcv_models:
    random = RandomizedSearchCV(estimator=model,
                                   param_distributions=params,
                                   n_iter=100,
                                   cv=3,
                                   verbose=2,
                                   n_jobs=-1)
    random.fit(X_train, y_train)
    model_param[name] = random.best_params_

for model_name in model_param:
    print(f"---------------- Best Params for {model_name} -------------------")
    print(model_param[model_name])


# In[22]:


## Retraining the models with best parameters
models = {
    "Random Forest Regressor": RandomForestRegressor(n_estimators =500, min_samples_split= 2, max_features =7, max_depth= 15),
     "K-Neighbors Regressor": KNeighborsRegressor(n_neighbors=10, n_jobs=-1),
    "XGBoost Regressor": XGBRegressor(n_estimators = 100, max_depth=5, learning_rate = 0.1, colsample_bytree= 1)

}
for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train, y_train) # Train model

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    model_train_mae , model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)

    model_test_mae , model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)

    print(list(models.keys())[i])

    print('Model performance for Training set')
    print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
    print("- R2 Score: {:.4f}".format(model_train_r2))

    print('----------------------------------')

    print('Model performance for Test set')
    print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
    print("- R2 Score: {:.4f}".format(model_test_r2))

    print('='*35)
    print('\n')


# In[23]:


# Example: Create a scenario input
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scenario = pd.DataFrame([{
 'WOB_klbf': 30,
    'RPM': 150,
    'Torque_kNm': 8,
    'SPP_psi': 3000,
    'Flow_Rate_gpm': 600,
    'Mud_Weight_ppg': 10,
    'Formation_Encoded': 1,  # e.g., 1 for Sandstone
    'WOB_per_RPM': 30 / 150,
    'Hydraulic_Efficiency': 88  # simplified example
}])


# Scale scenario input same as training data
scenario_scaled = scaler.fit_transform(scenario)

best_model = models["Random Forest Regressor"]

# Predict using tuned XGBoost
rop_pred = best_model.predict(scenario_scaled)
print(f"Predicted ROP for this scenario: {rop_pred[0]:.2f} ft/hr")


# In[24]:


import joblib

# Save model and scaler
joblib.dump(best_model, 'Drilling Efficiency.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved.")


# In[25]:


rf_model = joblib.load("Drilling Efficiency.pkl")


# In[26]:


import streamlit as st
import pandas as pd
import numpy as np

st.title("Drilling Efficiency Optimization")

st.header("Sample Data")
data = pd.DataFrame({
    'WOB_klbf': np.random.uniform(5, 35, 10),
    'RPM': np.random.uniform(60, 180, 10),
    'ROP_ft_per_hr': np.random.uniform(30, 120, 10)
})
st.dataframe(data)

st.header("Mean ROP")
st.write("Average ROP:", data["ROP_ft_per_hr"].mean())


# In[ ]:




