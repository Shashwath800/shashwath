import pandas as pd
import numpy as np

# Load the Excel file
df = pd.read_excel("Job opportunities.xlsx")

# Make a copy of the dataframe
df_clean = df.copy()

# --- Clean Salary Range ---
# Extract min, max, and average salary as numeric values
def parse_salary(salary_str):
    try:
        parts = salary_str.replace('£', '').replace(',', '').split('-')
        min_salary = int(parts[0].strip())
        max_salary = int(parts[1].strip())
        avg_salary = (min_salary + max_salary) / 2
        return pd.Series([min_salary, max_salary, avg_salary])
    except:
        return pd.Series([np.nan, np.nan, np.nan])

df_clean[['Min Salary', 'Max Salary', 'Avg Salary']] = df_clean['Salary Range'].apply(parse_salary)

# --- Standardize text columns ---
text_columns = [
    'Job Title', 'Job Description', 'Required Skills',
    'Location', 'Company', 'Experience Level',
    'Industry', 'Job Type'
]

for col in text_columns:
    df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()

# --- Drop rows with missing or unparseable salary values ---
df_clean.dropna(subset=['Avg Salary'], inplace=True)

# --- Optional: remove duplicates and reset index ---
df_clean.drop_duplicates(inplace=True)
df_clean.reset_index(drop=True, inplace=True)

# Save preprocessed data
df_clean.to_csv("cleaned_job_data.csv", index=False)
print("Preprocessing complete. Saved as 'cleaned_job_data.csv'")