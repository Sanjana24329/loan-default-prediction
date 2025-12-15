import pandas as pd
try:
    df = pd.read_csv("Loan-Default-Prediction_Model/loan_dataset/accepted_2007_to_2018Q4.csv.gz", nrows=5)
    print("GZ file read successfully, shape:", df.shape)
except Exception as e:
    print("GZ Error:", e)

try:
    df = pd.read_csv("Loan-Default-Prediction_Model/loan_dataset/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv", nrows=5)
    print("CSV file read successfully, shape:", df.shape)
except Exception as e:
    print("CSV Error:", e)
