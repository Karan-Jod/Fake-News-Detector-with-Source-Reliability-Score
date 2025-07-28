# main.py
import pandas as pd
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
url = "https://raw.githubusercontent.com/sachinruk/Fake-News/master/data/train.csv"
df = pd.read_csv(url)
df = df[['text', 'label']].dropna()

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = PassiveAggressiveClassifier(max_iter=1000)
model.fit(X_train_vec, y_train)

# Predict
y_pred = model.predict(X_test_vec)

# Output results
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🧠 Fake News Detector Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Source credibility score (basic example)
source_scores = {
    "nytimes.com": 95,
    "cnn.com": 85,
    "bbc.com": 90,
    "theonion.com": 10,
    "infowars.com": 5
}

def extract_source(url):
    domain = urlparse(url).netloc.replace("www.", "")
    return domain

def get_reliability_score(url):
    domain = extract_source(url)
    return source_scores.get(domain, 50)

# Example
sample_url = "https://www.nytimes.com/2024/06/15/politics/election.html"
score = get_reliability_score(sample_url)
print(f"\n🔗 Source Reliability Score for '{sample_url}': {score}/100")


