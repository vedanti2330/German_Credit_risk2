# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib
import warnings

warnings.filterwarnings('ignore')

print("Starting training script...")

# Step 2: Load Dataset
try:
    data = pd.read_csv('german_credit_data.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: 'german_credit_data.csv' not found.")
    print("Please download the dataset and place it in the same directory.")
    exit()

# Step 3: Data Cleaning & Preprocessing
# Handle missing values (as seen in the notebook)
data['Saving accounts'].fillna(data['Saving accounts'].mode()[0], inplace=True)
data['Checking account'].fillna(data['Checking account'].mode()[0], inplace=True)

# Drop irrelevant columns (as seen in the notebook)
if 'Unnamed: 0' in data.columns:
    data.drop('Unnamed: 0', axis=1, inplace=True)

# Encode Categorical Variables
# Binary encode 'Sex'
data['Sex'] = data['Sex'].map({'male':1, 'female':0})

# Encode target variable 'Risk' (good=0, bad=1)
data['Risk'] = data['Risk'].map({'good':0, 'bad':1})

# One-hot encode other categorical variables
data = pd.get_dummies(data, columns=['Housing', 'Saving accounts', 'Checking account', 'Purpose'], drop_first=True)
print("Data cleaning and encoding complete.")

# Step 4: Train-Test Split
X = data.drop('Risk', axis=1)
y = data['Risk']
# Save the column order for the app
model_features = X.columns.tolist()
joblib.dump(model_features, 'model_features.pkl')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print("Train-test split complete.")

# Step 5: Standardization
scaler = StandardScaler()
numerical_cols = ['Age', 'Credit amount', 'Duration']

# Fit scaler only on training data
scaler.fit(X_train[numerical_cols])
print("Scaler fitted on training data.")

# Transform training and test data
X_train[numerical_cols] = scaler.transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Step 6: Handle Class Imbalance (SMOTE) on training data only
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print("SMOTE applied to training data.")

# Step 7: Train Best Model
# Using best parameters identified in your notebook for RandomForest
print("Training RandomForest model with optimized parameters...")
best_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight=None,
    random_state=42
)

best_rf.fit(X_train_smote, y_train_smote)
print("Model training complete.")

# Step 8: Evaluate on Test Set (for confirmation)
y_pred = best_rf.predict(X_test)
print("\n--- Test Set Evaluation Report ---")
print(classification_report(y_test, y_pred, target_names=['Good Risk (0)', 'Bad Risk (1)']))
print("----------------------------------")

# Step 9: Save the model, scaler, and features
joblib.dump(best_rf, 'best_credit_risk_model2.pkl')
joblib.dump(scaler, 'scaler2.pkl')
joblib.dump(model_features, 'model_features.pkl') # Already done, but good to be explicit

print("\nSuccess! 'best_credit_risk_model2.pkl', 'scaler2.pkl', and 'model_features.pkl' have been saved.")
print("You can now run the Streamlit app: `streamlit run app.py`")

