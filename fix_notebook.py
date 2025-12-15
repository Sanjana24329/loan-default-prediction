import json

# Load the notebook
with open('Loan-Default-Prediction_Model/Data_cleaning.ipynb', 'r') as f:
    notebook = json.load(f)

# Find the cell with the read_csv line
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'pd.read_csv("accepted_2007_to_2018Q4.csv")' in source:
            # Replace the line
            new_source = source.replace('pd.read_csv("accepted_2007_to_2018Q4.csv")', 'pd.read_csv("loan_dataset/accepted_2007_to_2018Q4.csv.gz")')
            cell['source'] = new_source.split('\n')
            break

# Save the notebook
with open('Loan-Default-Prediction_Model/Data_cleaning.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
