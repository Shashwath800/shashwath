import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import joblib

# Function to load and preprocess Excel data
def preprocess_excel_data(file_path):
    """
    Load and preprocess job data from Excel file.
    
    Args:
        file_path (str): Path to the Excel file
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    print(f"Loading data from {file_path}...")
    
    # Load the Excel file
    df = pd.read_excel(file_path)
    
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
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
    
    # --- Drop rows with missing or unparseable salary values ---
    df_clean.dropna(subset=['Avg Salary'], inplace=True)
    
    # --- Remove duplicates and reset index ---
    df_clean.drop_duplicates(inplace=True)
    df_clean.reset_index(drop=True, inplace=True)
    
    print(f"Preprocessing complete. Rows: {len(df_clean)}")
    return df_clean

# NLP processing with spaCy
def nlp_process_data(df, spacy_model):
    """
    Process text data using spaCy for NLP.
    
    Args:
        df (pd.DataFrame): Dataframe with job data
        spacy_model: Loaded spaCy model
        
    Returns:
        pd.DataFrame: Dataframe with NLP-processed fields
    """
    print("Processing text data with NLP...")
    
    # Clean and preprocess text using spaCy
    def clean_text_spacy(text, nlp):
        doc = nlp(str(text).lower())
        tokens = [
            token.lemma_ for token in doc
            if not token.is_stop and not token.is_punct and not token.is_space and token.is_alpha
        ]
        return " ".join(tokens)
    
    # Apply text cleaning
    df['Cleaned Description'] = df['Job Description'].apply(lambda x: clean_text_spacy(x, spacy_model))
    df['Cleaned Skills'] = df['Required Skills'].apply(lambda x: clean_text_spacy(x, spacy_model))
    
    print("NLP processing complete.")
    return df

# Train salary prediction model
def train_salary_model(df):
    """
    Train a salary prediction model.
    
    Args:
        df (pd.DataFrame): Processed dataframe
        
    Returns:
        trained model and evaluation metrics
    """
    print("Training salary prediction model...")
    
    # Features and target
    X = df['Cleaned Description'] + " " + df['Cleaned Skills'] + " " + df['Job Title']
    y = df['Avg Salary']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Pipeline: TF-IDF + Regressor
    salary_model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=300)),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Train
    salary_model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(salary_model, "models/salary_prediction_model.pkl")
    print("Salary model saved as 'models/salary_prediction_model.pkl'")
    
    return salary_model

# Main function to run the training process
def main():
    """Main function to execute the entire training pipeline"""
    import os
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    print("Job Market Analysis - Model Training")
    print("====================================")
    
    # Check if Excel data file exists
    data_file = "Job opportunities.xlsx"
    if not os.path.exists(data_file):
        print(f"Error: Data file '{data_file}' not found!")
        print("Please ensure the Excel file is in the current directory.")
        return
    
    try:
        # Step 1: Preprocess the Excel data
        df_clean = preprocess_excel_data(data_file)
        
        # Step 2: NLP processing
        try:
            import spacy
            print("Loading spaCy model...")
            nlp = spacy.load("en_core_web_sm")
        except:
            print("Installing spaCy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            import spacy
            nlp = spacy.load("en_core_web_sm")
        
        df_processed = nlp_process_data(df_clean, nlp)
        
        # Step 3: Train and save the salary prediction model
        salary_model = train_salary_model(df_processed)
        
        print("\nTraining complete! Models saved to 'models' directory.")
        print("You can now run the Streamlit app using: streamlit run app.py")
        
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")

if __name__ == "__main__":
    main()