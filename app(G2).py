import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Artifact Loading (Cached) ---

@st.cache_resource
def load_model():
    """Loads the pre-trained RandomForest model."""
    try:
        model = joblib.load('best_credit_risk_model2.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file 'best_credit_risk_model2.pkl' not found. Please ensure it's in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_scaler():
    """Loads the pre-fitted scaler."""
    try:
        scaler = joblib.load('scaler2.pkl')
        return scaler
    except FileNotFoundError:
        st.error("Scaler file 'scaler2.pkl' not found. Will attempt to re-fit.")
        return None
    except Exception as e:
        st.error(f"Error loading scaler: {e}. Will attempt to re-fit.")
        return None

@st.cache_resource
def get_preprocessing_artifacts():
    """
    Recreates the preprocessing artifacts (scaler, columns, modes)
    by replicating the notebook's training data preparation.
    This ensures inputs are processed exactly as the model expects.
    """
    try:
        # Load the raw data to fit the scaler and get column names/modes
        data = pd.read_csv('german_credit_data.csv')
        
        # --- Replicate Notebook Preprocessing ---
        
        # 1. Fill NaNs (using mode, as in notebook)
        # We save these modes to apply to user input if they select 'N/A'
        modes = {
            'Saving accounts': data['Saving accounts'].mode()[0],
            'Checking account': data['Checking account'].mode()[0]
        }
        data['Saving accounts'].fillna(modes['Saving accounts'], inplace=True)
        data['Checking account'].fillna(modes['Checking account'], inplace=True)
        
        # 2. Map Binary/Target
        data['Sex'] = data['Sex'].map({'male': 1, 'female': 0})
        # Keep original 'Risk' for stratification
        data_for_split = data.copy()
        data_for_split['Risk'] = data_for_split['Risk'].map({'good': 0, 'bad': 1})
        
        # 3. Get_Dummies
        categorical_cols = ['Housing', 'Saving accounts', 'Checking account', 'Purpose']
        data_processed = pd.get_dummies(data_for_split.drop('Risk', axis=1), columns=categorical_cols, drop_first=True)
        
        # 4. Define features (X) and target (y)
        X = data_processed.drop('Risk', axis=1, errors='ignore').drop('Unnamed: 0', axis=1, errors='ignore')
        y = data_for_split['Risk']
        
        # 5. Recreate the *exact* train split to get the *exact* X_train for fitting the scaler
        X_train, _, _, _ = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # 6. Fit the Scaler (or load it)
        scaler = load_scaler()
        if scaler is None:
            st.warning("Re-fitting scaler as 'scaler2.pkl' was not found.")
            numerical_cols_to_scale = ['Age', 'Credit amount', 'Duration']
            scaler = StandardScaler()
            scaler.fit(X_train[numerical_cols_to_scale])
            # Save the re-fitted scaler for next time
            joblib.dump(scaler, 'scaler2.pkl')
        
        numerical_cols = ['Age', 'Credit amount', 'Duration'] # These are the ones to transform
        
        # 7. Get Model Columns
        model_columns = X_train.columns.tolist()
        
        # 8. Get unique values for dropdowns from the *original* data
        original_data = pd.read_csv('german_credit_data.csv')
        unique_vals = {
            'Sex': ['male', 'female'], # From notebook
            'Job': sorted(original_data['Job'].unique().tolist()),
            'Housing': ['own', 'rent', 'free'], # From notebook
            'Saving accounts': ['N/A', 'little', 'moderate', 'quite rich', 'rich'], # Added N/A for nan
            'Checking account': ['N/A', 'little', 'moderate', 'rich'], # Added N/A for nan
            'Purpose': original_data['Purpose'].unique().tolist()
        }

        return scaler, model_columns, numerical_cols, modes, unique_vals

    except FileNotFoundError:
        st.error("Data file 'german_credit_data.csv' not found. Please ensure it's in the same directory.")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"Error loading preprocessing artifacts: {e}")
        return None, None, None, None, None

# --- Streamlit UI ---

def main():
    st.set_page_config(page_title="Credit Risk Predictor", layout="wide")
    st.title("🇩🇪 German Credit Risk Predictor")
    st.write("Enter the applicant's details to predict their credit risk. This app uses a **Random Forest** model trained on the German Credit Data (using SMOTE for balancing).")

    # Load artifacts
    model = load_model()
    scaler, model_columns, numerical_cols, modes, unique_vals = get_preprocessing_artifacts()

    if model is None or scaler is None:
        st.stop()

    # --- User Input Fields ---
    st.header("Applicant Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        sex = st.selectbox("Sex", options=unique_vals['Sex'])
        job = st.selectbox("Job (0=unskilled, 1=skilled, 2=highly skilled)", options=unique_vals['Job'])
        
    with col2:
        credit_amount = st.number_input("Credit Amount (€)", min_value=0, value=2500, step=100)
        duration = st.number_input("Duration (in months)", min_value=1, max_value=72, value=12, step=1)
        purpose = st.selectbox("Purpose", options=unique_vals['Purpose'])
        
    with col3:
        housing = st.selectbox("Housing", options=unique_vals['Housing'])
        # Handle potential NaNs by mapping 'N/A' to the mode
        saving_account_input = st.selectbox("Saving Accounts", options=unique_vals['Saving accounts'])
        checking_account_input = st.selectbox("Checking Account", options=unique_vals['Checking account'])

    # --- Prediction Logic ---
    if st.button("Predict Credit Risk", type="primary"):
        
        # 1. Handle N/A inputs (replace with mode, as per notebook)
        saving_account = modes['Saving accounts'] if saving_account_input == 'N/A' else saving_account_input
        checking_account = modes['Checking account'] if checking_account_input == 'N/A' else checking_account_input

        # 2. Create DataFrame from user data
        user_data = {
            'Age': age,
            'Sex': sex,
            'Job': job,
            'Housing': housing,
            'Saving accounts': saving_account,
            'Checking account': checking_account,
            'Credit amount': credit_amount,
            'Duration': duration,
            'Purpose': purpose
        }
        input_df = pd.DataFrame([user_data])
        
        # 3. Preprocess the input data
        try:
            # Map Sex
            input_df['Sex'] = input_df['Sex'].map({'male': 1, 'female': 0})
            
            # Scale numerical features
            input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
            
            # One-hot encode categorical features
            input_df_processed = pd.get_dummies(input_df, drop_first=True)
            
            # Align columns to match model's training columns
            # Fills missing dummy columns with False (0)
            input_df_aligned = input_df_processed.reindex(columns=model_columns, fill_value=False)

            # 4. Make Prediction
            prediction = model.predict(input_df_aligned)[0]
            prediction_proba = model.predict_proba(input_df_aligned)[0]
            
            # 5. Display Result
            # Note: Risk 0 = Good, Risk 1 = Bad (from notebook)
            proba_bad_risk = prediction_proba[1]

            if prediction == 1:
                st.error(f"**Prediction: Bad Credit Risk** (Probability: {proba_bad_risk:.0%})")
                st.warning("This applicant has a high probability of defaulting on the loan.")
            else:
                st.success(f"**Prediction: Good Credit Risk** (Probability of Bad Risk: {proba_bad_risk:.0%})")
                st.info("This applicant is likely to repay the loan.")
                
            with st.expander("Show Prediction Probabilities"):
                st.write({
                    "Probability of Good Risk (0)": f"{prediction_proba[0]:.2%}",
                    "Probability of Bad Risk (1)": f"{prediction_proba[1]:.2%}"
                })

        except Exception as e:
            st.error(f"An error occurred during preprocessing or prediction: {e}")
            # st.dataframe(input_df_aligned.info()) # For debugging

if __name__ == "__main__":
    main()

