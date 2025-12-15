import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load cleaned data
df = pd.read_csv('Cleaned_LoanApproval.csv')

# Select features and target
features = ['loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'emp_length', 'home_ownership', 'annual_inc', 'verification_status', 'purpose', 'DTI Ratio %', 'fico_range_low', 'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'total_acc', 'mort_acc']
target = 'loan_status'

# Encode categorical variables
le_dict = {}
for col in df.select_dtypes(include=['object']).columns:
    if col in features or col == target:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

# Split data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, 'loan_model.pkl')
joblib.dump(le_dict, 'label_encoders.pkl')

print("Model trained and saved.")
