import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib

# Load processed data
df = pd.read_csv("nlp_processed_job_data.csv")

# Process target: convert skills to list of keywords
df['Skill List'] = df['Cleaned Skills'].apply(lambda x: x.split())

# Binarize multi-label skills
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(df['Skill List'])

# Input features: description or job title
X = df['Cleaned Description'] + " " + df['Job Title']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Pipeline: TF-IDF + MultiOutputClassifier
skill_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=500)),
    ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42)))
])

# Train
skill_pipeline.fit(X_train, Y_train)

# Save skill predictor and binarizer
joblib.dump(skill_pipeline, "skill_prediction_model.pkl")
joblib.dump(mlb, "skill_binarizer.pkl")
print("Skill prediction model saved as 'skill_prediction_model.pkl'")
print("MultiLabelBinarizer saved as 'skill_binarizer.pkl'")