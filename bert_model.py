from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
import joblib

# Load your dataset
df = pd.read_csv("nlp_processed_job_data.csv")

# Load model (you can try others like 'paraphrase-MiniLM-L6-v2', 'all-MiniLM-L12-v2', etc.)
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode job descriptions into dense vectors
desc_embeddings = bert_model.encode(df['Cleaned Description'], show_progress_bar=True)

# Save embeddings for reuse
np.save("desc_embeddings.npy", desc_embeddings)

# Process target: split skills into list form
y = df["Cleaned Skills"]

# Multi-label binarization
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(y.str.split())

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(desc_embeddings, Y, test_size=0.2, random_state=42)

# Train BERT-based skill model
clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=mlb.classes_))

# Save model and binarizer
joblib.dump(clf, "bert_skill_model.pkl")
joblib.dump(mlb, "bert_skill_binarizer.pkl")
print("BERT Skill model saved as 'bert_skill_model.pkl'")
print("BERT Skill MultiLabelBinarizer saved as 'bert_skill_binarizer.pkl'")