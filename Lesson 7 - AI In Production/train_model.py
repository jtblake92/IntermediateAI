import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os

base_path = os.path.dirname(__file__)  # directory of the script
csv_path = os.path.join(base_path, "data", "messages.csv")

df = pd.read_csv(csv_path)

X = df['message']
y = df['category']

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, stratify=y)

clf = LogisticRegression()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {acc:.2%}")

os.makedirs("models", exist_ok=True)
with open("models/model_v1.pkl", "wb") as f:
    pickle.dump((vectorizer, clf), f)

print("✅ Model saved to models/model_v1.pkl")
