import pandas as pd
import numpy as np
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# Load spaCy English model
import en_core_web_sm
nlp = en_core_web_sm.load()

# Load cleaned dataset (this comes from the preprocessing step)
df = pd.read_csv("cleaned_job_data.csv")

# Text cleaning function using spaCy
def clean_text_spacy(text):
    doc = nlp(str(text).lower())
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space and token.is_alpha
    ]
    return " ".join(tokens)

# Apply text cleaning
df['Cleaned Description'] = df['Job Description'].apply(clean_text_spacy)
df['Cleaned Skills'] = df['Required Skills'].apply(clean_text_spacy)

# TF-IDF Vectorization for feature extraction
desc_vectorizer = TfidfVectorizer(max_features=100)
desc_tfidf = desc_vectorizer.fit_transform(df['Cleaned Description'])

skills_vectorizer = TfidfVectorizer(max_features=50)
skills_tfidf = skills_vectorizer.fit_transform(df['Cleaned Skills'])

# Optional: Display top TF-IDF keywords
top_desc_keywords = desc_vectorizer.get_feature_names_out()
top_skills_keywords = skills_vectorizer.get_feature_names_out()

print("Top description keywords:", top_desc_keywords[:10])
print("Top skills keywords:", top_skills_keywords[:10])

# Save processed data for model training
df.to_csv("nlp_processed_job_data.csv", index=False)
print("NLP processing complete. Saved as 'nlp_processed_job_data.csv'")