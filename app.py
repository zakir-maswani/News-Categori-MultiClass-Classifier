import streamlit as st
import joblib
import re
import pandas as pd
from nltk.corpus import stopwords
import nltk

# Ensure stopwords are available
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# Load model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load('news_classifier_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

# Text cleaning function
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML
    text = text.lower()  # Lowercase
    text = re.sub(r'\s+', ' ', text).strip()  # Whitespace
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Special chars
    return text

# Remove stopwords
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in str(text).split() if word not in stop_words])

# Predict category
def predict_category(text, model, vectorizer):
    cleaned_text = clean_text(text)
    cleaned_text = remove_stopwords(cleaned_text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)[0]
    probabilities = model.predict_proba(vectorized_text)[0]
    return prediction, probabilities

# Streamlit UI
st.set_page_config(page_title="News Category Classifier", layout="wide")

st.title("📰 News Category Classifier")
st.markdown("---")

st.markdown("""
This application classifies news articles into one of five categories:
- **Sports**
- **Business**
- **Politics**
- **Technology**
- **Entertainment**

Enter or paste a news headline or article text below to get a prediction.
""")

st.markdown("---")

# Load model and vectorizer
model, vectorizer = load_model_and_vectorizer()

# Input section
st.subheader("Input News Text")
user_input = st.text_area("Enter news headline or article text:", placeholder="Paste your news text here...", height=150)

# Prediction section
if user_input.strip():
    st.markdown("---")
    st.subheader("Prediction Results")
    
    prediction, probabilities = predict_category(user_input, model, vectorizer)
    
    # Display prediction
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric("Predicted Category", prediction)
    
    with col2:
        # Display probabilities
        categories = model.classes_
        prob_df = pd.DataFrame({
            'Category': categories,
            'Probability': probabilities
        }).sort_values('Probability', ascending=False)
        
        st.write("**Confidence Scores:**")
        for idx, row in prob_df.iterrows():
            st.write(f"{row['Category']}: {row['Probability']:.2%}")
    
    # Visualization of probabilities
    st.markdown("---")
    st.subheader("Probability Distribution")
    
    chart_data = pd.DataFrame({
        'Category': categories,
        'Probability': probabilities
    }).sort_values('Probability', ascending=False)
    
    st.bar_chart(chart_data.set_index('Category'))

else:
    st.info("👆 Please enter some news text to get started!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 12px;'>
    <p>News Category Classifier | Built with Streamlit | Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)
