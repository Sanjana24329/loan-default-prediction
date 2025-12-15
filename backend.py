from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Load the trained model and encoders
model = joblib.load('loan_model.pkl')
encoders = joblib.load('label_encoders.pkl')

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Fix term value
        term_value = int(data['term'].split()[0])  # "36 months" -> 36

        input_data = pd.DataFrame({
            'loan_amnt': [float(data['loan_amnt'])],
            'term': [term_value],
            'int_rate': [float(data['int_rate'])],
            'installment': [float(data['installment'])],
            'grade': [data['grade']],
            'emp_length': [data['emp_length']],
            'home_ownership': [data['home_ownership']],
            'annual_inc': [float(data['annual_inc'])],
            'verification_status': [data['verification_status']],
            'purpose': [data['purpose']],
            'DTI Ratio %': [float(data['dti'])],
            'fico_range_low': [int(data['fico_range_low'])],
            'inq_last_6mths': [int(data['inq_last_6mths'])],
            'open_acc': [int(data['open_acc'])],
            'pub_rec': [int(data['pub_rec'])],
            'revol_bal': [float(data['revol_bal'])],
            'total_acc': [int(data['total_acc'])],
            'mort_acc': [int(data['mort_acc'])]
        })

        # Encode categorical variables
        for col in input_data.select_dtypes(include=['object']).columns:
            if col in encoders:
                input_data[col] = encoders[col].transform(input_data[col])

        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]

        if 'loan_status' in encoders:
            prediction_label = encoders['loan_status'].inverse_transform([prediction])[0]
        else:
            prediction_label = 'Charged Off' if prediction == 1 else 'Fully Paid'

        return jsonify({
            'prediction': prediction_label,
            'confidence': float(max(prediction_proba))
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
