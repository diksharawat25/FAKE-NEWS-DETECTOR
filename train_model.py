import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle

# Load the datasets
df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")

print("📊 Fake news count:", df_fake.shape[0])
print("📰 Real news count:", df_true.shape[0])

# Add labels: 1 = FAKE, 0 = REAL
df_fake["label"] = 1
df_true["label"] = 0

# Combine both datasets
df = pd.concat([df_fake, df_true])
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the dataset

# Prepare features and labels
X = df["text"]
y = df["label"]

# Vectorize the text data
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_vectorized = vectorizer.fit_transform(X)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train the model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Predict on test set
y_pred = model.predict(X_test)

# Show metrics
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📄 Classification Report:\n", classification_report(y_test, y_pred, target_names=["REAL", "FAKE"]))
print("\n🧩 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the model
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)
             
# Save the vectorizer
with open("vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

print("✅ Model and vectorizer saved as model.pkl and vectorizer.pkl")
