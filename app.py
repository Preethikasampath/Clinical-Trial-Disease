# ================================================
# Clinical Trial Disease Classification App
# ================================================

import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ------------------------------------------------
# Load saved model and vectorizer
# ------------------------------------------------
@st.cache_resource
def load_models():
    with open('disease_classifier.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_models()

# ------------------------------------------------
# Text cleaning function
# ------------------------------------------------
def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word)
             for word in words
             if word not in stop_words]
    return ' '.join(words)

# ------------------------------------------------
# Prediction function
# ------------------------------------------------
def predict_disease(text):
    cleaned    = clean_text(text)
    tfidf_text = vectorizer.transform([cleaned])
    prediction = model.predict(tfidf_text)[0]
    proba      = model.predict_proba(tfidf_text)[0]
    confidence = max(proba) * 100
    return prediction, confidence

# ------------------------------------------------
# Disease information dictionary
# ------------------------------------------------
disease_info = {
    'breast cancer': {
        'emoji': '🎗️',
        'description': 'A cancer that forms in breast cells.',
        'color': '#FF69B4'
    },
    'type 2 diabetes': {
        'emoji': '🩸',
        'description': 'A condition affecting blood sugar regulation.',
        'color': '#FF8C00'
    },
    'covid-19': {
        'emoji': '🦠',
        'description': 'Infectious disease caused by SARS-CoV-2 virus.',
        'color': '#4169E1'
    },
    'anxiety': {
        'emoji': '🧠',
        'description': 'Mental health condition causing excessive worry.',
        'color': '#9370DB'
    },
    'chronic obstructive pulmonary disease': {
        'emoji': '🫁',
        'description': 'Chronic lung disease causing breathing difficulty.',
        'color': '#20B2AA'
    },
    'rheumatoid arthritis': {
        'emoji': '🦴',
        'description': 'Autoimmune disease affecting joints.',
        'color': '#CD853F'
    },
    'glaucoma': {
        'emoji': '👁️',
        'description': 'Eye condition damaging the optic nerve.',
        'color': '#3CB371'
    },
    'sickle cell anemia': {
        'emoji': '🔴',
        'description': 'Genetic blood disorder affecting red blood cells.',
        'color': '#DC143C'
    }
}

# ------------------------------------------------
# STREAMLIT APP UI
# ------------------------------------------------

# Page config
st.set_page_config(
    page_title="Clinical Trial Disease Classifier",
    page_icon="🏥",
    layout="wide"
)

# Header
st.title("🏥 Clinical Trial Disease Classifier")
st.markdown("### Classify clinical trial summaries into disease categories using NLP & ML")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/hospital.png")
    st.header("ℹ️ About This App")
    st.write("""
    This app uses:
    - **NLP** for text preprocessing
    - **TF-IDF** for feature extraction
    - **Logistic Regression** for classification
    - **94.06% accuracy** on test data!
    """)
    st.markdown("---")
    st.header("🎯 Disease Categories")
    for disease, info in disease_info.items():
        st.write(f"{info['emoji']} {disease.title()}")

# Main content - two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📝 Enter Clinical Trial Summary")
    
    # Example buttons
    st.write("**Try an example:**")
    
    ex1 = st.button("🎗️ Breast Cancer Example")
    ex2 = st.button("🩸 Diabetes Example")
    ex3 = st.button("👁️ Glaucoma Example")
    ex4 = st.button("🧠 Anxiety Example")

    # Default text
    default_text = ""
    
    if ex1:
        default_text = "Breast cancer patients often experience perioperative pain and emotional disorders such as anxiety and depression affecting quality of life."
    elif ex2:
        default_text = "This study investigates the effect of insulin therapy on blood glucose levels in patients with type 2 diabetes and obesity."
    elif ex3:
        default_text = "This trial evaluates eye drops for reducing intraocular pressure in glaucoma patients with optic nerve damage."
    elif ex4:
        default_text = "Study examining cognitive behavioral therapy effectiveness for patients with generalized anxiety disorder and panic attacks."

    # Text input
    user_input = st.text_area(
        "Paste your clinical trial summary here:",
        value=default_text,
        height=200,
        placeholder="Enter clinical trial summary text here..."
    )
    
    # Predict button
    predict_btn = st.button(
        "🔍 Classify Disease",
        type="primary",
        use_container_width=True
    )

with col2:
    st.header("🎯 Prediction Result")
    
    if predict_btn and user_input:
        
        with st.spinner("Analyzing text..."):
            prediction, confidence = predict_disease(user_input)
        
        info = disease_info.get(prediction, {
            'emoji': '❓',
            'description': 'Unknown disease',
            'color': '#808080'
        })
        
        # Show result
        st.success(f"### {info['emoji']} {prediction.upper()}")
        
        # Confidence meter
        st.write("**Confidence Score:**")
        st.progress(int(confidence))
        
        if confidence >= 80:
            st.success(f"✅ High Confidence: {confidence:.1f}%")
        elif confidence >= 50:
            st.warning(f"⚠️ Medium Confidence: {confidence:.1f}%")
        else:
            st.error(f"❌ Low Confidence: {confidence:.1f}% — Result may be inaccurate!")
        
        # Disease info
        st.info(f"**About this disease:**\n{info['description']}")
        
        # Show cleaned text
        with st.expander("🔍 See how text was processed"):
            cleaned = clean_text(user_input)
            st.write("**Original:**")
            st.write(user_input)
            st.write("**After NLP Cleaning:**")
            st.write(cleaned)
    
    elif predict_btn and not user_input:
        st.warning("⚠️ Please enter some text first!")
    
    else:
        st.info("👈 Enter text and click **Classify Disease** to see results!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Python, NLTK, Scikit-learn & Streamlit</p>
    <p>Model Accuracy: 94.06% | Dataset: 60,337 Clinical Trials</p>
</div>
""", unsafe_allow_html=True)