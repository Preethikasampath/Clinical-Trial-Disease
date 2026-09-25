🧬 Clinical Trial Disease Classifier
An end-to-end NLP and Machine Learning pipeline that classifies clinical trial summaries into 8 disease categories with 94.06% accuracy.

🎯 Project Overview
This project processes 60,337 clinical trial records and automatically classifies them into:

🎗️ Breast Cancer
🩸 Type 2 Diabetes
🦠 COVID-19
🧠 Anxiety
🫁 Chronic Obstructive Pulmonary Disease
🦴 Rheumatoid Arthritis
👁️ Glaucoma
🔴 Sickle Cell Anemia
🛠️ Tech Stack
Python — Core programming language
NLTK — NLP text preprocessing
Scikit-learn — Machine learning models
TF-IDF — Text vectorization
Streamlit — Web application
Plotly — Interactive visualizations
Pandas — Data manipulation
📊 Model Performance
Model	Accuracy
SVM	94.13%
Logistic Regression	94.06% ✅
Random Forest	93.48%
Decision Tree	90.26%
Naive Bayes	89.85%
🔄 ML Pipeline
Data Loading → 60,337 clinical trials
Data Cleaning → Handle missing values
NLP Processing → Tokenize, lemmatize
TF-IDF → Convert text to numbers
Model Training → Logistic Regression
Deployment → Streamlit web app
🚀 How to Run
Install dependencies
pip install -r requirements.txt

Run the app
streamlit run app1.py

📁 Project Structure
NLP/ ├── app1.py ← Premium Streamlit app ├── NLP.ipynb ← Complete ML notebook ├── disease_classifier.pkl ← Trained model ├── tfidf_vectorizer.pkl ← TF-IDF vectorizer ├── requirements.txt ← Dependencies └── README.md ← This file

📈 Dataset
Source: ClinicalTrials.gov
Size: 60,337 records
Features: 16 columns
Target: 8 disease categories
