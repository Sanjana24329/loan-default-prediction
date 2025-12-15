import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the cleaned data
@st.cache_data
def load_data():
    df = pd.read_csv('Cleaned_LoanApproval.csv')
    return df

# Load the trained model and encoders
@st.cache_resource
def load_model():
    model = joblib.load('loan_model.pkl')
    encoders = joblib.load('label_encoders.pkl')
    return model, encoders

df = load_data()
model, encoders = load_model()

st.title('Loan Default Prediction Model')

st.sidebar.header('Navigation')
page = st.sidebar.radio('Go to', ['Data Overview', 'Prediction'])

if page == 'Data Overview':
    st.header('Cleaned Loan Data Overview')
    st.write('Shape of the dataset:', df.shape)
    st.dataframe(df.head())
    st.write('Summary Statistics:')
    st.write(df.describe())

elif page == 'Prediction':
    st.header('Loan Default Prediction')
    st.write('Enter loan details to predict default risk.')

    # Input fields based on key features
    loan_amnt = st.number_input('Loan Amount', min_value=0.0, value=10000.0)
    term = st.selectbox('Term', ['36 months', '60 months'])
    int_rate = st.number_input('Interest Rate (%)', min_value=0.0, value=10.0)
    installment = st.number_input('Installment', min_value=0.0, value=300.0)
    grade = st.selectbox('Grade', ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    emp_length = st.selectbox('Employment Length', ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'])
    home_ownership = st.selectbox('Home Ownership', ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
    annual_inc = st.number_input('Annual Income', min_value=0.0, value=50000.0)
    verification_status = st.selectbox('Verification Status', ['Not Verified', 'Source Verified', 'Verified'])
    purpose = st.selectbox('Purpose', ['debt_consolidation', 'credit_card', 'home_improvement', 'other', 'major_purchase', 'small_business', 'car', 'medical', 'moving', 'vacation', 'house', 'wedding', 'renewable_energy', 'educational'])
    dti = st.number_input('DTI Ratio (%)', min_value=0.0, value=15.0)
    fico_range_low = st.number_input('FICO Range Low', min_value=300, max_value=850, value=650)
    inq_last_6mths = st.number_input('Inquiries Last 6 Months', min_value=0, value=1)
    open_acc = st.number_input('Open Accounts', min_value=0, value=5)
    pub_rec = st.number_input('Public Records', min_value=0, value=0)
    revol_bal = st.number_input('Revolving Balance', min_value=0.0, value=10000.0)
    total_acc = st.number_input('Total Accounts', min_value=0, value=20)
    mort_acc = st.number_input('Mortgage Accounts', min_value=0, value=1)

    if st.button('Predict'):
        # Create input dataframe
        input_data = pd.DataFrame({
            'loan_amnt': [loan_amnt],
            'term': [term],
            'int_rate': [int_rate],
            'installment': [installment],
            'grade': [grade],
            'emp_length': [emp_length],
            'home_ownership': [home_ownership],
            'annual_inc': [annual_inc],
            'verification_status': [verification_status],
            'purpose': [purpose],
            'DTI Ratio %': [dti],
            'fico_range_low': [fico_range_low],
            'inq_last_6mths': [inq_last_6mths],
            'open_acc': [open_acc],
            'pub_rec': [pub_rec],
            'revol_bal': [revol_bal],
            'total_acc': [total_acc],
            'mort_acc': [mort_acc]
        })

        # Encode categorical variables
        for col in input_data.select_dtypes(include=['object']).columns:
            if col in encoders:
                input_data[col] = encoders[col].transform(input_data[col])

        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]

        # Decode prediction back to original labels
        if 'loan_status' in encoders:
            prediction_label = encoders['loan_status'].inverse_transform([prediction])[0]
        else:
            prediction_label = 'Default' if prediction == 1 else 'Fully Paid'

        st.write(f'Predicted Loan Status: {prediction_label}')
        st.write(f'Prediction Confidence: {max(prediction_proba):.2%}')
