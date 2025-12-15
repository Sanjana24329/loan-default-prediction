import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical operations

from IPython.display import display  # Displaying data
from skimpy import skim as sk  # Data summary tool
pd.set_option('display.float', '{:.2f}'.format)

pd.set_option('display.max_columns', 50)

pd.set_option('display.max_rows', 50)
#Too big of a data will take time to load

df = pd.read_csv("Loan-Default-Prediction_Model/loan_dataset/accepted_2007_to_2018Q4.csv.gz", nrows=100000)

display(df.head())
print(df.columns)

#Removing columns which are not needed.





columns_to_drop = [

                'member_id', 'funded_amnt', 'funded_amnt_inv', 'pymnt_plan', 'url', 'desc', 'delinq_2yrs',

                'settlement_term', 'mths_since_last_delinq', 'mths_since_last_record', 'out_prncp', 'out_prncp_inv',

                'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee',

                'recoveries', 'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d',

                'last_credit_pull_d', 'last_fico_range_high', 'last_fico_range_low',

                'collections_12_mths_ex_med', 'mths_since_last_major_derog', 'policy_code',

                'annual_inc_joint',	'dti_joint', 'verification_status_joint', 'acc_now_delinq',

                'tot_coll_amt', 'tot_cur_bal', 'open_acc_6m', 'open_act_il', 'open_il_12m', 'open_il_24m',

                'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m',	'open_rv_24m',	'max_bal_bc',

                'all_util',	'total_rev_hi_lim',	'inq_fi', 'total_cu_tl', 'inq_last_12m', 'acc_open_past_24mths',

                'avg_cur_bal', 'bc_open_to_buy', 'bc_util',	'chargeoff_within_12_mths',	'delinq_amnt',

                'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl',

                'mths_since_recent_bc',	'mths_since_recent_bc_dlq',	'mths_since_recent_inq',

                'mths_since_recent_revol_delinq', 'num_accts_ever_120_pd', 'num_actv_bc_tl',

                'num_actv_rev_tl',	'num_bc_sats',	'num_bc_tl', 'num_il_tl', 'num_op_rev_tl',

                'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats', 'num_tl_120dpd_2m',	'num_tl_30dpd',

                'num_tl_90g_dpd_24m', 'num_tl_op_past_12m',	'pct_tl_nvr_dlq', 'percent_bc_gt_75',

                'tax_liens', 'tot_hi_cred_lim',	'total_bal_ex_mort', 'total_bc_limit',

                'total_il_high_credit_limit', 'revol_bal_joint', 'sec_app_fico_range_low',

                'sec_app_fico_range_high', 'sec_app_earliest_cr_line', 'sec_app_inq_last_6mths',

                'sec_app_mort_acc',	'sec_app_open_acc',	'sec_app_revol_util', 'sec_app_open_act_il',

                'sec_app_num_rev_accts', 'sec_app_chargeoff_within_12_mths',

                'sec_app_collections_12_mths_ex_med', 'sec_app_mths_since_last_major_derog','hardship_flag',

                'hardship_type', 'hardship_reason',	'hardship_status', 'deferral_term', 'hardship_amount',

                'hardship_start_date',	'hardship_end_date', 'payment_plan_start_date',	'hardship_length',	'hardship_dpd',

                'hardship_loan_status',	'orig_projected_additional_accrued_interest', 'hardship_payoff_balance_amount',	'hardship_last_payment_amount',

                'disbursement_method', 'debt_settlement_flag', 'debt_settlement_flag_date','settlement_status',

                'settlement_date', 'settlement_amount', 'settlement_percentage',

]



df.drop(columns=columns_to_drop, inplace=True)
#Checking For Tables Dropped



display(df.head())

df.to_csv('Up_LoanApproval.csv')
df1 = pd.read_csv("Up_LoanApproval.csv")
#Checking Data Types



data_types = df1.dtypes

display(data_types)
#checking Null values



null_counts = df1.isnull().sum()



# Display the count of null values in each column

print("Null Value Counts in Each Column:")

print(null_counts)
# Find unique data types in each column

unique_data_types_per_column = df1.applymap(type).apply(set)



# Display column names along with their unique data types

print("Column Names and Their Unique Data Types:")

for column, unique_data_types in unique_data_types_per_column.items():

    print(f"{column}: {', '.join(str(dt) for dt in unique_data_types)}")
# Converting 'id' column to numeric, coercing non-numeric values to NaN

df1['id'] = pd.to_numeric(df1['id'], errors='coerce')



# Removing rows where 'id' column contains non-numeric values (NaN)

df1 = df1.dropna(subset=['id'])



# Reseting the DataFrame index after dropping rows

df1 = df1.reset_index(drop=True)
# Displaying the current data type of the 'id' column

original_dtype = df1['id'].dtype

print("Original Data Type of 'id' Column:", original_dtype)



# Converting 'id' column to integer data type

df1['id'] = df1['id'].astype(int)



# Displaying the updated data type

updated_dtype = df1['id'].dtype

print("Updated Data Type of 'id' Column:", updated_dtype)
#checking null values again

null_counts = df1.isnull().sum()

print("Null Value Counts in Each Column:")

print(null_counts)
# Identifying columns with multiple data types

columns_with_multiple_datatypes = df1.columns[df1.applymap(type).nunique() > 1]



# Displaying the names

print("Columns with Multiple Data Types and Their Data Types:")

for column in columns_with_multiple_datatypes:

    unique_datatypes = df1[column].apply(type).unique()

    print(f"{column}: {', '.join(str(dt) for dt in unique_datatypes)}")
# Removing rows where column contains non-numeric

df1 = df1.dropna(subset=['zip_code','annual_inc', 'earliest_cr_line', 'emp_length'])
# Identifying columns with multiple data types

columns_with_multiple_datatypes = df1.columns[df1.applymap(type).nunique() > 1]



# Displaying the names

print("Columns with Multiple Data Types and Their Data Types:")

for column in columns_with_multiple_datatypes:

    unique_datatypes = df1[column].apply(type).unique()

    print(f"{column}: {', '.join(str(dt) for dt in unique_datatypes)}")
# Replacing null values in the 'title' column with 'Self Employed'

df1['emp_title'].fillna('Self Employed', inplace=True)
# Removing rows where 'emp_length' column contains null values

df1.dropna(subset=['inq_last_6mths'], inplace=True)
#checking null values again

null_counts = df1.isnull().sum()

print("Null Value Counts in Each Column:")

print(null_counts)
# Defining the mapping of purpose values to title values

purpose_to_title_mapping = {

    'home_improvement': 'Home Improvement',

    'other': 'Other',

    'debt_consolidation': 'Debt Consolidation',

    'small_business': 'Business',

    'major_purchase': 'Major Purchase',

    'credit_card': 'Credit card Refinancing',

    'house': 'Buying Home',

    'vacation': 'Vacations',

    'car': 'Car Financing',

    'medical': 'Medical Expenses',

    'moving': 'Moving and Relocation',

    'renewable_energy': 'Green Loan',

    'wedding': 'Wedding',

    'educational': 'Student Loan / Expenses'

}



#filling the 'title' column based on 'purpose'

def fill_title(row):

    if row['purpose'] in purpose_to_title_mapping:

        return purpose_to_title_mapping[row['purpose']]

    else:

        return row['title']



# Fill null values in the 'title' column with a default value, for example, 'Unknown'

df1['title'].fillna(df1['purpose'].map(purpose_to_title_mapping), inplace=True)



# Applying to fill the 'title' column

df1['title'] = df1.apply(fill_title, axis=1)



# Displaying

df1.reset_index(drop=True, inplace=True)

df1.drop(columns=['Unnamed: 0'], inplace=True)

display(df1.head())

#As we have DTI null values we have to calculate it using formula

df1['dti'] = (df1['installment'] / (df1['annual_inc'] / 12)) * 100



# Rename the column to "DTI Ratio"

df1.rename(columns={'dti': 'DTI Ratio %'}, inplace=True)



#same for revol_util

# 'revol_bal' represents the total credit card balance  and 'total_acc' represents the total number of credit accounts

df1['revol_util'] = (df1['revol_bal'] / df1['total_acc']) * 100



# Converting specified columns to integer data type (except mor_acc containg null values)

int_columns = ['fico_range_low', 'fico_range_high', 'inq_last_6mths', 'open_acc', 'pub_rec', 'total_acc']

df1[int_columns] = df1[int_columns].astype(int)



# mapping 'w' and 'f' to their corresponding descriptions

status_mapping = {'w': 'Whole Funded', 'f': 'Fractional Funded'}



# Replacing 'w' and 'f'

df1['initial_list_status'] = df1['initial_list_status'].replace(status_mapping)



# credit score bins and labels

score_bins = [0, 579, 669, 739, 799, float('inf')]

score_labels = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']



# Creating a new column 'credit_score_bucket' based on FICO scores

df1['credit_score_bucket'] = pd.cut(df1['fico_range_low'], bins=score_bins, labels=score_labels)



# Displaying the updated DataFrame with the DTI calculation for all rows

display(df1.head())
#checking null values again

null_counts = df1.isnull().sum()

print("Null Value Counts in Each Column:")

print(null_counts)
# New Calculated field the Credit Score Range

df1['credit_score_range'] = df1['fico_range_high'] - df1['fico_range_low']



# Calculate the Monthly Income

df1['monthly_income'] = df1['annual_inc'] / 12



# Calculate the Payment-to-Income Ratio

df1['payment_to_income_ratio (%)'] = (df1['installment'] / df1['monthly_income']) * 100





display(df1.head())
#as we cant calculate mort_acc using formula we have drop the null values

df1.dropna(subset=['mort_acc'], inplace=True)



#checking null values again

null_counts = df1.isnull().sum()

print("Null Value Counts in Each Column:")

print(null_counts)

display(df1.head())
df1.to_csv('Cleaned_LoanApproval.csv')
