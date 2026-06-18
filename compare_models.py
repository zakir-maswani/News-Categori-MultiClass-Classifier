import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from nltk.corpus import stopwords
import nltk

# Ensure stopwords are available
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# 1. Load and Clean Data (Re-using cleaning logic)
df = pd.read_csv('news_category_dataset_uncleaned.csv')
df.dropna(subset=['text', 'category'], inplace=True)
df['category'] = df['category'].str.capitalize()

def clean_text(text):
    text = re.sub(r'<.*?>', '', text) # Remove HTML
    text = text.lower() # Lowercase
    text = re.sub(r'\s+', ' ', text).strip() # Whitespace
    text = re.sub(r'[^a-z0-9\s]', '', text) # Special chars
    return text

df['text'] = df['text'].apply(clean_text)
df.drop_duplicates(inplace=True)

# 2. Preprocessing
stop_words = set(stopwords.words('english'))
df['text_cleaned'] = df['text'].apply(lambda x: ' '.join([word for word in str(x).split() if word not in stop_words]))

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text_cleaned'])
y = df['category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Define Models
models = {
    "Multinomial Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Support Vector Machine": SVC(kernel='linear', probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# 4. Train and Evaluate
results = []

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "F1-Score": f1
    })

results_df = pd.DataFrame(results)
print("\nModel Comparison Results:")
print(results_df)

# 5. Visualization
plt.figure(figsize=(12, 6))
melted_results = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
sns.barplot(data=melted_results, x="Model", y="Score", hue="Metric", palette="muted")
plt.title("Model Performance Comparison")
plt.ylim(0, 1.1)
plt.ylabel("Score")
plt.xticks(rotation=15)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('model_comparison.png')

# Save results to CSV for reporting
results_df.to_csv('model_comparison_results.csv', index=False)
