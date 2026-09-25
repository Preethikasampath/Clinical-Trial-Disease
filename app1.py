# ============================================================
# Clinical Trial Disease Classifier - Premium UI
# Built with Streamlit | NLP + Machine Learning
# ============================================================

import streamlit as st
import pickle
import re
import nltk
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── NLTK Setup ───────────────────────────────────────────────
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',   quiet=True)

# ── Page Configuration ────────────────────────────────────────
st.set_page_config(
    page_title="MediClassify AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Premium CSS ───────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@400;600;700;800&display=swap');

/* ── Global Reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

/* ── Root Variables ── */
:root {
    --bg-primary:    #0F172A;
    --bg-secondary:  #1E293B;
    --bg-card:       rgba(30, 41, 59, 0.8);
    --accent-blue:   #3B82F6;
    --accent-purple: #8B5CF6;
    --accent-cyan:   #06B6D4;
    --accent-green:  #10B981;
    --accent-pink:   #EC4899;
    --text-primary:  #F1F5F9;
    --text-secondary:#94A3B8;
    --border:        rgba(255,255,255,0.08);
    --shadow:        0 25px 50px rgba(0,0,0,0.5);
    --radius:        16px;
    --radius-sm:     8px;
}

/* ── App Background ── */
.stApp {
    background: linear-gradient(135deg, #0F172A 0%, #1a1040 50%, #0F172A 100%) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
}

/* ── Hide Streamlit Branding ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1E293B 0%, #0F172A 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Hero Section ── */
.hero-section {
    background: linear-gradient(135deg,
        rgba(59,130,246,0.15) 0%,
        rgba(139,92,246,0.15) 50%,
        rgba(6,182,212,0.15) 100%);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 24px;
    padding: 48px;
    text-align: center;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-section::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(139,92,246,0.1) 0%, transparent 70%);
    animation: pulse 4s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 0.5; }
    50% { transform: scale(1.1); opacity: 1; }
}
.hero-title {
    font-family: 'Poppins', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #3B82F6, #8B5CF6, #06B6D4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 12px;
    position: relative;
    z-index: 1;
}
.hero-subtitle {
    font-size: 1.1rem;
    color: var(--text-secondary);
    max-width: 600px;
    margin: 0 auto;
    line-height: 1.7;
    position: relative;
    z-index: 1;
}

/* ── KPI Cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 32px;
}
.kpi-card {
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.kpi-card.blue::before   { background: linear-gradient(90deg, #3B82F6, #06B6D4); }
.kpi-card.purple::before { background: linear-gradient(90deg, #8B5CF6, #EC4899); }
.kpi-card.green::before  { background: linear-gradient(90deg, #10B981, #06B6D4); }
.kpi-card.pink::before   { background: linear-gradient(90deg, #EC4899, #8B5CF6); }
.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.4);
    border-color: rgba(59,130,246,0.3);
}
.kpi-value {
    font-family: 'Poppins', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: var(--text-primary);
    margin-bottom: 4px;
}
.kpi-label {
    font-size: 0.8rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 500;
}
.kpi-icon { font-size: 2rem; margin-bottom: 12px; }

/* ── Glass Cards ── */
.glass-card {
    background: rgba(30, 41, 59, 0.6);
    backdrop-filter: blur(20px);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 28px;
    margin-bottom: 20px;
    transition: all 0.3s ease;
}
.glass-card:hover {
    border-color: rgba(59,130,246,0.3);
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}
.card-title {
    font-family: 'Poppins', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Prediction Result ── */
.prediction-box {
    background: linear-gradient(135deg,
        rgba(59,130,246,0.2),
        rgba(139,92,246,0.2));
    border: 1px solid rgba(59,130,246,0.4);
    border-radius: var(--radius);
    padding: 32px;
    text-align: center;
    margin: 20px 0;
    animation: fadeInUp 0.5s ease;
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
.prediction-disease {
    font-family: 'Poppins', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #3B82F6, #8B5CF6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 8px 0;
}
.prediction-emoji { font-size: 3.5rem; margin-bottom: 8px; }
.prediction-desc {
    color: var(--text-secondary);
    font-size: 0.95rem;
    margin-top: 8px;
    line-height: 1.6;
}

/* ── Confidence Badge ── */
.confidence-high   { color: #10B981; font-weight: 700; font-size: 1.3rem; }
.confidence-medium { color: #F59E0B; font-weight: 700; font-size: 1.3rem; }
.confidence-low    { color: #EF4444; font-weight: 700; font-size: 1.3rem; }

/* ── Disease Tag Buttons ── */
.disease-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
    margin: 16px 0;
}
.disease-tag {
    background: rgba(59,130,246,0.1);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.85rem;
    color: var(--text-primary);
    cursor: pointer;
    transition: all 0.2s ease;
    text-align: center;
}
.disease-tag:hover {
    background: rgba(59,130,246,0.2);
    border-color: rgba(59,130,246,0.5);
    transform: scale(1.02);
}

/* ── Section Headers ── */
.section-header {
    font-family: 'Poppins', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-header .accent {
    background: linear-gradient(135deg, #3B82F6, #8B5CF6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ── Text Area ── */
.stTextArea textarea {
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 16px !important;
    transition: border-color 0.3s ease !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #3B82F6, #8B5CF6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
    box-shadow: 0 4px 15px rgba(59,130,246,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(59,130,246,0.5) !important;
}

/* ── Sidebar Elements ── */
.sidebar-logo {
    text-align: center;
    padding: 20px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 20px;
}
.sidebar-logo-text {
    font-family: 'Poppins', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #3B82F6, #8B5CF6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.sidebar-stat {
    background: rgba(59,130,246,0.1);
    border: 1px solid rgba(59,130,246,0.15);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.sidebar-stat-label { color: #94A3B8; font-size: 0.82rem; }
.sidebar-stat-value { color: #3B82F6; font-weight: 700; font-size: 0.9rem; }

/* ── Disease Info Cards ── */
.disease-info-card {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 12px;
    transition: all 0.2s ease;
}
.disease-info-card:hover {
    background: rgba(59,130,246,0.1);
    border-color: rgba(59,130,246,0.3);
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 32px;
    border-top: 1px solid var(--border);
    margin-top: 40px;
    color: var(--text-secondary);
    font-size: 0.85rem;
}
.footer span {
    background: linear-gradient(135deg, #3B82F6, #8B5CF6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 600;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(30,41,59,0.5) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    padding: 8px 20px !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #3B82F6, #8B5CF6) !important;
    color: white !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}

/* ── Metrics ── */
[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 24px 0 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0F172A; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(#3B82F6, #8B5CF6);
    border-radius: 3px;
}
</style>
""", unsafe_allow_html=True)


# ── Data & Constants ──────────────────────────────────────────
DISEASE_INFO = {
    'breast cancer': {
        'emoji': '🎗️',
        'description': 'A cancer originating in breast cells, most common in women. Includes ductal and lobular carcinoma.',
        'color': '#EC4899',
        'gradient': 'linear-gradient(135deg, #EC4899, #8B5CF6)',
        'keywords': ['tumor', 'mammogram', 'chemotherapy', 'biopsy', 'mastectomy']
    },
    'type 2 diabetes': {
        'emoji': '🩸',
        'description': 'A metabolic condition where the body cannot properly regulate blood glucose levels due to insulin resistance.',
        'color': '#F59E0B',
        'gradient': 'linear-gradient(135deg, #F59E0B, #EF4444)',
        'keywords': ['glucose', 'insulin', 'HbA1c', 'glycemic', 'metformin']
    },
    'covid-19': {
        'emoji': '🦠',
        'description': 'An infectious respiratory disease caused by the SARS-CoV-2 coronavirus, first identified in 2019.',
        'color': '#3B82F6',
        'gradient': 'linear-gradient(135deg, #3B82F6, #06B6D4)',
        'keywords': ['SARS-CoV-2', 'PCR', 'vaccine', 'respiratory', 'quarantine']
    },
    'anxiety': {
        'emoji': '🧠',
        'description': 'A mental health condition characterized by persistent worry, fear, and psychological distress affecting daily life.',
        'color': '#8B5CF6',
        'gradient': 'linear-gradient(135deg, #8B5CF6, #EC4899)',
        'keywords': ['GAD', 'panic', 'CBT', 'SSRI', 'mindfulness']
    },
    'chronic obstructive pulmonary disease': {
        'emoji': '🫁',
        'description': 'A chronic inflammatory lung disease causing obstructed airflow, including emphysema and chronic bronchitis.',
        'color': '#06B6D4',
        'gradient': 'linear-gradient(135deg, #06B6D4, #3B82F6)',
        'keywords': ['FEV1', 'bronchodilator', 'emphysema', 'spirometry', 'inhaler']
    },
    'rheumatoid arthritis': {
        'emoji': '🦴',
        'description': 'A systemic autoimmune disease causing chronic joint inflammation, pain, and progressive cartilage damage.',
        'color': '#D97706',
        'gradient': 'linear-gradient(135deg, #D97706, #EF4444)',
        'keywords': ['TNF', 'DMARDs', 'synovitis', 'methotrexate', 'biologics']
    },
    'glaucoma': {
        'emoji': '👁️',
        'description': 'A group of eye conditions damaging the optic nerve, often associated with elevated intraocular pressure.',
        'color': '#10B981',
        'gradient': 'linear-gradient(135deg, #10B981, #06B6D4)',
        'keywords': ['IOP', 'optic nerve', 'tonometry', 'trabeculectomy', 'visual field']
    },
    'sickle cell anemia': {
        'emoji': '🔴',
        'description': 'A hereditary blood disorder where abnormal hemoglobin causes red blood cells to become rigid and sickle-shaped.',
        'color': '#EF4444',
        'gradient': 'linear-gradient(135deg, #EF4444, #EC4899)',
        'keywords': ['hemoglobin', 'vaso-occlusion', 'hydroxyurea', 'transfusion', 'HbS']
    }
}

MODEL_STATS = {
    'accuracy':        94.06,
    'total_trials':    60337,
    'diseases':        8,
    'training_size':   48269,
}


# ── Load Models ───────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    try:
        with open('disease_classifier.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer, True
    except Exception as e:
        return None, None, False


# ── NLP Functions ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_nlp_tools():
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    return lemmatizer, stop_words

def clean_text(text: str) -> str:
    lemmatizer, stop_words = get_nlp_tools()
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

def predict_disease(text: str, model, vectorizer):
    cleaned    = clean_text(text)
    tfidf_text = vectorizer.transform([cleaned])
    prediction = model.predict(tfidf_text)[0]
    proba      = model.predict_proba(tfidf_text)[0]
    confidence = max(proba) * 100
    all_proba  = dict(zip(model.classes_, proba * 100))
    return prediction, confidence, all_proba, cleaned


# ── Chart Functions ───────────────────────────────────────────
def create_confidence_gauge(confidence: float):
    color = '#10B981' if confidence >= 80 else '#F59E0B' if confidence >= 50 else '#EF4444'
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = confidence,
        title = {'text': "Confidence Score", 'font': {'color': '#94A3B8', 'size': 14}},
        number = {'suffix': '%', 'font': {'color': color, 'size': 36, 'family': 'Poppins'}},
        gauge = {
            'axis':       {'range': [0, 100], 'tickcolor': '#94A3B8', 'tickwidth': 1},
            'bar':        {'color': color, 'thickness': 0.25},
            'bgcolor':    'rgba(30,41,59,0.5)',
            'bordercolor': 'rgba(255,255,255,0.05)',
            'steps': [
                {'range': [0,  50], 'color': 'rgba(239,68,68,0.1)'},
                {'range': [50, 80], 'color': 'rgba(245,158,11,0.1)'},
                {'range': [80,100], 'color': 'rgba(16,185,129,0.1)'}
            ],
            'threshold': {
                'line':  {'color': color, 'width': 3},
                'thickness': 0.75,
                'value': confidence
            }
        }
    ))
    fig.update_layout(
        height          = 220,
        margin          = dict(l=20, r=20, t=40, b=10),
        paper_bgcolor   = 'rgba(0,0,0,0)',
        plot_bgcolor    = 'rgba(0,0,0,0)',
        font            = {'family': 'Inter', 'color': '#94A3B8'}
    )
    return fig

def create_probability_chart(all_proba: dict):
    sorted_proba = dict(sorted(all_proba.items(), key=lambda x: x[1], reverse=True))
    diseases     = list(sorted_proba.keys())
    values       = list(sorted_proba.values())
    colors = [
        '#3B82F6','#8B5CF6','#10B981','#F59E0B',
        '#EC4899','#06B6D4','#EF4444','#D97706'
    ]
    fig = go.Figure(go.Bar(
        x           = values,
        y           = diseases,
        orientation = 'h',
        marker      = dict(
            color       = colors[:len(diseases)],
            opacity     = 0.85,
            line        = dict(color='rgba(255,255,255,0.1)', width=1)
        ),
        text        = [f'{v:.1f}%' for v in values],
        textposition= 'outside',
        textfont    = dict(color='#94A3B8', size=11)
    ))
    fig.update_layout(
        title         = dict(text='Probability Distribution', font=dict(color='#F1F5F9', size=14)),
        height        = 320,
        margin        = dict(l=10, r=60, t=40, b=10),
        paper_bgcolor = 'rgba(0,0,0,0)',
        plot_bgcolor  = 'rgba(0,0,0,0)',
        xaxis         = dict(
            showgrid    = True,
            gridcolor   = 'rgba(255,255,255,0.05)',
            tickcolor   = '#94A3B8',
            color       = '#94A3B8',
            range       = [0, 110]
        ),
        yaxis         = dict(
            tickcolor   = '#94A3B8',
            color       = '#F1F5F9',
            tickfont    = dict(size=11)
        ),
        font          = dict(family='Inter', color='#94A3B8')
    )
    return fig

def create_dataset_donut():
    labels = [d.title() for d in DISEASE_INFO.keys()]
    values = [16301, 11467, 10153, 9286, 6181, 3637, 2173, 1139]
    colors = ['#EC4899','#F59E0B','#3B82F6','#8B5CF6',
              '#06B6D4','#D97706','#10B981','#EF4444']
    fig = go.Figure(go.Pie(
        labels      = labels,
        values      = values,
        hole        = 0.6,
        marker      = dict(colors=colors, line=dict(color='#0F172A', width=2)),
        textinfo    = 'percent',
        hoverinfo   = 'label+value',
        textfont    = dict(size=10, color='white')
    ))
    fig.update_layout(
        height        = 300,
        margin        = dict(l=0, r=0, t=20, b=0),
        paper_bgcolor = 'rgba(0,0,0,0)',
        plot_bgcolor  = 'rgba(0,0,0,0)',
        showlegend    = True,
        legend        = dict(
            font      = dict(color='#94A3B8', size=10),
            bgcolor   = 'rgba(0,0,0,0)'
        ),
        annotations   = [dict(
            text      = f'<b>60,337</b><br>Trials',
            x=0.5, y=0.5,
            font      = dict(size=14, color='#F1F5F9', family='Poppins'),
            showarrow = False
        )]
    )
    return fig


# ── Example Texts ─────────────────────────────────────────────
EXAMPLES = {
    '🎗️ Breast Cancer':
        'Breast cancer patients undergoing surgery often experience perioperative emotional disorders such as anxiety and depression leading to poor quality of life outcomes.',
    '🩸 Type 2 Diabetes':
        'This study investigates the effect of continuous glucose monitoring and insulin therapy on HbA1c levels in type 2 diabetic patients with obesity and hyperglycemia.',
    '🦠 Covid-19':
        'A randomized controlled trial evaluating the efficacy and safety of antiviral treatment in hospitalized patients with severe SARS-CoV-2 infection and respiratory complications.',
    '🧠 Anxiety':
        'Cognitive behavioral therapy combined with SSRI medication for generalized anxiety disorder patients experiencing panic attacks and social phobia symptoms.',
    '🫁 COPD':
        'Long-term bronchodilator therapy with FEV1 monitoring in chronic obstructive pulmonary disease patients with severe emphysema and reduced lung capacity.',
    '🦴 Rheumatoid Arthritis':
        'Biological DMARDs targeting TNF-alpha in rheumatoid arthritis patients with progressive joint synovitis and elevated inflammatory markers.',
    '👁️ Glaucoma':
        'Randomized trial of prostaglandin analogue eye drops for reducing intraocular pressure in open-angle glaucoma patients with optic nerve damage.',
    '🔴 Sickle Cell':
        'Hydroxyurea therapy for reducing vaso-occlusive crises in sickle cell anemia patients with abnormal HbS hemoglobin and frequent pain episodes.'
}


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div style="font-size:2.5rem; margin-bottom:8px;">🧬</div>
        <div class="sidebar-logo-text">MediClassify AI</div>
        <div style="color:#94A3B8; font-size:0.75rem; margin-top:4px;">
            Clinical NLP Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation
    st.markdown(
        "<div style='color:#94A3B8; font-size:0.75rem; "
        "text-transform:uppercase; letter-spacing:1px; "
        "margin-bottom:10px; font-weight:600;'>Navigation</div>",
        unsafe_allow_html=True
    )
    page = st.radio(
        "nav",
        ["🏠  Dashboard", "🔬  Classify", "📊  Analytics", "ℹ️  About"],
        label_visibility="collapsed"
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # Model Stats
    st.markdown(
        "<div style='color:#94A3B8; font-size:0.75rem; "
        "text-transform:uppercase; letter-spacing:1px; "
        "margin-bottom:10px; font-weight:600;'>Model Statistics</div>",
        unsafe_allow_html=True
    )
    stats = [
        ("Accuracy",      "94.06%"),
        ("Total Trials",  "60,337"),
        ("Diseases",      "8"),
        ("Algorithm",     "Log. Reg"),
        ("Vectorizer",    "TF-IDF"),
        ("Features",      "5,000"),
    ]
    for label, value in stats:
        st.markdown(f"""
        <div class="sidebar-stat">
            <span class="sidebar-stat-label">{label}</span>
            <span class="sidebar-stat-value">{value}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Disease List
    st.markdown(
        "<div style='color:#94A3B8; font-size:0.75rem; "
        "text-transform:uppercase; letter-spacing:1px; "
        "margin-bottom:10px; font-weight:600;'>Disease Categories</div>",
        unsafe_allow_html=True
    )
    for disease, info in DISEASE_INFO.items():
        st.markdown(f"""
        <div class="disease-info-card">
            <span style="font-size:1.2rem;">{info['emoji']}</span>
            <span style="color:#F1F5F9; font-size:0.82rem; font-weight:500;">
                {disease.title()}
            </span>
        </div>
        """, unsafe_allow_html=True)


# ── Load Models ───────────────────────────────────────────────
model, vectorizer, model_loaded = load_models()


# ══════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════
if "Dashboard" in page:

    # Hero
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🧬 MediClassify AI</div>
        <div class="hero-subtitle">
            An end-to-end NLP & Machine Learning pipeline that classifies
            clinical trial summaries into disease categories with
            <strong style="color:#3B82F6;">94.06% accuracy</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI Cards
    st.markdown("""
    <div class="kpi-grid">
        <div class="kpi-card blue">
            <div class="kpi-icon">🎯</div>
            <div class="kpi-value">94.06%</div>
            <div class="kpi-label">Model Accuracy</div>
        </div>
        <div class="kpi-card purple">
            <div class="kpi-icon">📋</div>
            <div class="kpi-value">60,337</div>
            <div class="kpi-label">Clinical Trials</div>
        </div>
        <div class="kpi-card green">
            <div class="kpi-icon">🏥</div>
            <div class="kpi-value">8</div>
            <div class="kpi-label">Disease Categories</div>
        </div>
        <div class="kpi-card pink">
            <div class="kpi-icon">🤖</div>
            <div class="kpi-value">5</div>
            <div class="kpi-label">ML Models Tested</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Two columns
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown('<div class="section-header">📊 Dataset Distribution</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(create_dataset_donut(), use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">🏆 Model Comparison</div>',
                    unsafe_allow_html=True)
        models_data = {
            'Model': ['SVM', 'Logistic Reg.', 'Random Forest', 'Decision Tree', 'Naive Bayes'],
            'Accuracy': [94.13, 94.06, 93.48, 90.26, 89.85]
        }
        fig_models = go.Figure(go.Bar(
            x       = models_data['Accuracy'],
            y       = models_data['Model'],
            orientation = 'h',
            marker  = dict(
                color   = ['#8B5CF6','#3B82F6','#06B6D4','#10B981','#F59E0B'],
                opacity = 0.85
            ),
            text    = [f'{a:.2f}%' for a in models_data['Accuracy']],
            textposition = 'outside',
            textfont = dict(color='#94A3B8', size=11)
        ))
        fig_models.update_layout(
            height        = 300,
            margin        = dict(l=10, r=60, t=10, b=10),
            paper_bgcolor = 'rgba(0,0,0,0)',
            plot_bgcolor  = 'rgba(0,0,0,0)',
            xaxis = dict(
                range       = [85, 100],
                showgrid    = True,
                gridcolor   = 'rgba(255,255,255,0.05)',
                color       = '#94A3B8'
            ),
            yaxis = dict(color='#F1F5F9', tickfont=dict(size=11)),
            font  = dict(family='Inter', color='#94A3B8')
        )
        st.plotly_chart(fig_models, use_container_width=True)

    # Pipeline Steps
    st.markdown('<div class="section-header">🔄 ML Pipeline</div>',
                unsafe_allow_html=True)
    steps = [
        ("01", "Data Loading",    "📂", "60,337 clinical trial records loaded from CSV"),
        ("02", "Data Cleaning",   "🧹", "Removed nulls, dropped irrelevant columns"),
        ("03", "NLP Processing",  "📝", "Tokenize, lemmatize, remove stop words"),
        ("04", "TF-IDF",          "🔢", "Convert text to 5,000-feature numeric matrix"),
        ("05", "Model Training",  "🤖", "Logistic Regression on 80% training data"),
        ("06", "Deployment",      "🚀", "Streamlit web app with real-time prediction"),
    ]
    cols = st.columns(6)
    for i, (num, title, icon, desc) in enumerate(steps):
        with cols[i]:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center; padding:20px 12px;">
                <div style="font-size:1.8rem; margin-bottom:8px;">{icon}</div>
                <div style="color:#3B82F6; font-size:0.7rem; font-weight:700;
                            letter-spacing:1px; margin-bottom:4px;">STEP {num}</div>
                <div style="font-weight:600; font-size:0.85rem;
                            margin-bottom:6px; color:#F1F5F9;">{title}</div>
                <div style="color:#94A3B8; font-size:0.75rem; line-height:1.4;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE: CLASSIFY
# ══════════════════════════════════════════════════════════════
elif "Classify" in page:

    st.markdown("""
    <div style="margin-bottom:28px;">
        <div class="hero-title" style="font-size:2rem; text-align:left;">
            🔬 Disease Classifier
        </div>
        <div style="color:#94A3B8; font-size:0.95rem; margin-top:8px;">
            Enter a clinical trial summary to classify it into one of 8 disease categories.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-header">📝 <span class="accent">Input</span></div>',
                    unsafe_allow_html=True)

        # Example selector
        example_choice = st.selectbox(
            "Try a quick example:",
            ["✍️ Write your own..."] + list(EXAMPLES.keys())
        )

        if example_choice != "✍️ Write your own...":
            default_text = EXAMPLES[example_choice]
        else:
            default_text = ""

        user_input = st.text_area(
            "Clinical Trial Summary:",
            value       = default_text,
            height      = 220,
            placeholder = "Paste or type your clinical trial summary here...",
            label_visibility = "collapsed"
        )

        char_count = len(user_input)
        word_count = len(user_input.split()) if user_input.strip() else 0
        st.markdown(f"""
        <div style="color:#94A3B8; font-size:0.8rem;
                    text-align:right; margin-top:-12px; margin-bottom:16px;">
            {word_count} words · {char_count} characters
        </div>
        """, unsafe_allow_html=True)

        classify_btn = st.button("🔍  Classify Disease", type="primary")

        # Text processing preview
        if user_input.strip():
            with st.expander("🔍 View NLP Processing"):
                cleaned = clean_text(user_input)
                st.markdown(f"""
                <div style="background:rgba(15,23,42,0.8); border-radius:10px;
                            padding:16px; margin-top:8px;">
                    <div style="color:#94A3B8; font-size:0.75rem;
                                text-transform:uppercase; letter-spacing:1px;
                                margin-bottom:8px;">After NLP Cleaning:</div>
                    <div style="color:#10B981; font-size:0.88rem;
                                line-height:1.6; font-family:monospace;">
                        {cleaned[:300]}{'...' if len(cleaned) > 300 else ''}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-header">🎯 <span class="accent">Result</span></div>',
                    unsafe_allow_html=True)

        if classify_btn and user_input.strip():
            if not model_loaded:
                st.error("⚠️ Model files not found! Ensure .pkl files are in the app directory.")
            else:
                with st.spinner("🧠 Analyzing clinical text..."):
                    prediction, confidence, all_proba, cleaned = predict_disease(
                        user_input, model, vectorizer
                    )

                info = DISEASE_INFO.get(prediction, {
                    'emoji': '❓', 'description': 'Unknown',
                    'gradient': 'linear-gradient(135deg,#3B82F6,#8B5CF6)',
                    'keywords': []
                })

                # Prediction Box
                st.markdown(f"""
                <div class="prediction-box">
                    <div class="prediction-emoji">{info['emoji']}</div>
                    <div style="color:#94A3B8; font-size:0.8rem;
                                text-transform:uppercase; letter-spacing:2px;">
                        Predicted Disease
                    </div>
                    <div class="prediction-disease">{prediction.upper()}</div>
                    <div class="prediction-desc">{info['description']}</div>
                </div>
                """, unsafe_allow_html=True)

                # Confidence Gauge
                st.plotly_chart(
                    create_confidence_gauge(confidence),
                    use_container_width=True
                )

                # Confidence label
                if confidence >= 80:
                    level = "confidence-high";   label = "✅ High Confidence"
                elif confidence >= 50:
                    level = "confidence-medium"; label = "⚠️ Medium Confidence"
                else:
                    level = "confidence-low";    label = "❌ Low Confidence — verify manually"

                st.markdown(f"""
                <div style="text-align:center; margin:-10px 0 16px;">
                    <span class="{level}">{label}: {confidence:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)

                # Keywords
                if info.get('keywords'):
                    st.markdown(f"""
                    <div class="glass-card" style="padding:16px;">
                        <div class="card-title">🔑 Key Medical Terms</div>
                        <div style="display:flex; flex-wrap:wrap; gap:8px;">
                            {''.join([
                                f'<span style="background:rgba(59,130,246,0.15); '
                                f'border:1px solid rgba(59,130,246,0.3); '
                                f'border-radius:6px; padding:4px 10px; '
                                f'color:#93C5FD; font-size:0.8rem;">{kw}</span>'
                                for kw in info['keywords']
                            ])}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Probability Chart
                st.plotly_chart(
                    create_probability_chart(all_proba),
                    use_container_width=True
                )

        elif classify_btn and not user_input.strip():
            st.warning("⚠️ Please enter a clinical trial summary first!")
        else:
            st.markdown("""
            <div class="glass-card" style="text-align:center; padding:60px 20px;">
                <div style="font-size:3rem; margin-bottom:16px;">🔬</div>
                <div style="color:#94A3B8; font-size:0.95rem; line-height:1.7;">
                    Enter a clinical trial summary on the left<br>
                    and click <strong style="color:#3B82F6;">Classify Disease</strong>
                    to see results.
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ══════════════════════════════════════════════════════════════
elif "Analytics" in page:

    st.markdown("""
    <div style="margin-bottom:28px;">
        <div class="hero-title" style="font-size:2rem; text-align:left;">
            📊 Analytics Dashboard
        </div>
        <div style="color:#94A3B8; font-size:0.95rem; margin-top:8px;">
            Deep insights into the clinical trial dataset and model performance.
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 Dataset", "🤖 Model Performance", "🔬 NLP Insights"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            # Disease distribution bar
            diseases = list(DISEASE_INFO.keys())
            counts   = [16301, 11467, 10153, 9286, 6181, 3637, 2173, 1139]
            colors   = ['#EC4899','#F59E0B','#3B82F6','#8B5CF6',
                        '#06B6D4','#D97706','#10B981','#EF4444']

            fig = go.Figure(go.Bar(
                x       = counts,
                y       = [d.title() for d in diseases],
                orientation = 'h',
                marker  = dict(color=colors, opacity=0.85),
                text    = [f'{c:,}' for c in counts],
                textposition = 'outside',
                textfont = dict(color='#94A3B8', size=10)
            ))
            fig.update_layout(
                title         = dict(text='Trials per Disease', font=dict(color='#F1F5F9')),
                height        = 350,
                margin        = dict(l=10, r=60, t=40, b=10),
                paper_bgcolor = 'rgba(0,0,0,0)',
                plot_bgcolor  = 'rgba(0,0,0,0)',
                xaxis = dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)',
                             color='#94A3B8'),
                yaxis = dict(color='#F1F5F9', tickfont=dict(size=10)),
                font  = dict(family='Inter', color='#94A3B8')
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.plotly_chart(create_dataset_donut(), use_container_width=True)

        # Average word count
        avg_words = {
            'anxiety': 73.8, 'sickle cell anemia': 72.3,
            'covid-19': 66.4, 'chronic obstructive pulmonary disease': 63.3,
            'breast cancer': 58.4, 'type 2 diabetes': 58.2,
            'rheumatoid arthritis': 50.7, 'glaucoma': 46.9
        }
        fig_words = go.Figure(go.Bar(
            x       = list(avg_words.values()),
            y       = [d.title() for d in avg_words.keys()],
            orientation = 'h',
            marker  = dict(
                color   = list(avg_words.values()),
                colorscale = 'Viridis',
                opacity = 0.85
            ),
            text    = [f'{v} words' for v in avg_words.values()],
            textposition = 'outside',
            textfont = dict(color='#94A3B8', size=10)
        ))
        fig_words.update_layout(
            title         = dict(text='Average Summary Length per Disease',
                                 font=dict(color='#F1F5F9')),
            height        = 350,
            margin        = dict(l=10, r=80, t=40, b=10),
            paper_bgcolor = 'rgba(0,0,0,0)',
            plot_bgcolor  = 'rgba(0,0,0,0)',
            xaxis = dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)',
                         color='#94A3B8'),
            yaxis = dict(color='#F1F5F9', tickfont=dict(size=10)),
            font  = dict(family='Inter', color='#94A3B8')
        )
        st.plotly_chart(fig_words, use_container_width=True)

    with tab2:
        # Model metrics table
        metrics = {
            'Disease': [d.title() for d in DISEASE_INFO.keys()],
            'Precision': [0.89, 0.96, 0.94, 0.95, 0.97, 0.97, 0.97, 0.94],
            'Recall':    [0.94, 0.97, 0.89, 0.93, 0.93, 0.88, 0.84, 0.97],
            'F1-Score':  [0.92, 0.96, 0.91, 0.94, 0.95, 0.92, 0.90, 0.96],
            'Support':   [1921, 3259, 1251, 1925, 456, 738, 227, 2291]
        }
        df = pd.DataFrame(metrics)

        col1, col2 = st.columns(2)
        with col1:
            fig_f1 = go.Figure(go.Bar(
                x       = df['F1-Score'],
                y       = df['Disease'],
                orientation = 'h',
                marker  = dict(
                    color   = df['F1-Score'],
                    colorscale = 'Plasma',
                    opacity = 0.85
                ),
                text    = [f'{v:.2f}' for v in df['F1-Score']],
                textposition = 'outside',
                textfont = dict(color='#94A3B8', size=10)
            ))
            fig_f1.update_layout(
                title         = dict(text='F1-Score per Disease',
                                     font=dict(color='#F1F5F9')),
                height        = 350,
                margin        = dict(l=10, r=60, t=40, b=10),
                paper_bgcolor = 'rgba(0,0,0,0)',
                plot_bgcolor  = 'rgba(0,0,0,0)',
                xaxis = dict(range=[0.8, 1.05], showgrid=True,
                             gridcolor='rgba(255,255,255,0.05)', color='#94A3B8'),
                yaxis = dict(color='#F1F5F9', tickfont=dict(size=10)),
                font  = dict(family='Inter', color='#94A3B8')
            )
            st.plotly_chart(fig_f1, use_container_width=True)

        with col2:
            # Radar chart
            categories = df['Disease'].tolist()
            fig_radar  = go.Figure()
            for metric, color in [
                ('Precision', '#3B82F6'),
                ('Recall',    '#8B5CF6'),
                ('F1-Score',  '#10B981')
            ]:
                fig_radar.add_trace(go.Scatterpolar(
                    r    = df[metric].tolist() + [df[metric].iloc[0]],
                    theta= categories + [categories[0]],
                    fill = 'toself',
                    name = metric,
                    line = dict(color=color),
                    fillcolor = color.replace(')', ',0.1)').replace('rgb', 'rgba')
                ))
            fig_radar.update_layout(
                polar = dict(
                    bgcolor  = 'rgba(0,0,0,0)',
                    radialaxis = dict(
                        range      = [0.8, 1.0],
                        showgrid   = True,
                        gridcolor  = 'rgba(255,255,255,0.1)',
                        color      = '#94A3B8',
                        tickfont   = dict(size=8)
                    ),
                    angularaxis = dict(
                        color    = '#94A3B8',
                        tickfont = dict(size=8)
                    )
                ),
                title         = dict(text='Metrics Radar Chart',
                                     font=dict(color='#F1F5F9')),
                height        = 350,
                paper_bgcolor = 'rgba(0,0,0,0)',
                plot_bgcolor  = 'rgba(0,0,0,0)',
                legend        = dict(font=dict(color='#94A3B8'),
                                     bgcolor='rgba(0,0,0,0)'),
                font          = dict(family='Inter', color='#94A3B8')
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown('<div class="section-header">📋 Detailed Metrics Table</div>',
                    unsafe_allow_html=True)
        st.dataframe(
            df.style.background_gradient(
                cmap='Blues',
                subset=['Precision', 'Recall', 'F1-Score']
            ).format({'Precision': '{:.2f}', 'Recall': '{:.2f}', 'F1-Score': '{:.2f}'}),
            use_container_width=True
        )

    with tab3:
        st.markdown('<div class="section-header">🔬 Top Words by Disease</div>',
                    unsafe_allow_html=True)

        top_words = {
            'breast cancer':    ['breast', 'cancer', 'tumor', 'surgery', 'chemotherapy'],
            'type 2 diabetes':  ['glucose', 'insulin', 'diabetes', 'glycemic', 'HbA1c'],
            'covid-19':         ['covid', 'SARS', 'vaccine', 'respiratory', 'infection'],
            'anxiety':          ['anxiety', 'depression', 'CBT', 'panic', 'disorder'],
            'glaucoma':         ['glaucoma', 'IOP', 'optic', 'pressure', 'eye'],
        }

        for disease, words in top_words.items():
            info = DISEASE_INFO[disease]
            st.markdown(f"""
            <div class="glass-card" style="padding:16px 20px; margin-bottom:12px;">
                <div class="card-title" style="margin-bottom:10px;">
                    {info['emoji']} {disease.title()}
                </div>
                <div style="display:flex; flex-wrap:wrap; gap:8px;">
                    {''.join([
                        f'<span style="background:rgba(59,130,246,0.15); '
                        f'border:1px solid rgba(59,130,246,0.25); '
                        f'border-radius:6px; padding:5px 12px; '
                        f'color:#93C5FD; font-size:0.82rem; font-weight:500;">{w}</span>'
                        for w in words
                    ])}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════
elif "About" in page:

    st.markdown("""
    <div style="margin-bottom:28px;">
        <div class="hero-title" style="font-size:2rem; text-align:left;">
            ℹ️ About This Project
        </div>
        <div style="color:#94A3B8; font-size:0.95rem; margin-top:8px;">
            Clinical Trial Disease Classification using NLP and Machine Learning.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="glass-card">
            <div class="card-title">🎯 Project Overview</div>
            <div style="color:#94A3B8; font-size:0.9rem; line-height:1.8;">
                This project builds an end-to-end NLP and Machine Learning pipeline
                to automatically classify clinical trial summaries into disease categories.
                <br><br>
                The system processes raw medical text, applies NLP techniques,
                and uses trained ML models to predict the most likely disease
                category with a confidence score.
            </div>
        </div>

        <div class="glass-card" style="margin-top:16px;">
            <div class="card-title">🛠️ Tech Stack</div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px;">
        """ + ''.join([
            f'<div style="background:rgba(59,130,246,0.1); border:1px solid '
            f'rgba(59,130,246,0.2); border-radius:8px; padding:8px 12px; '
            f'color:#93C5FD; font-size:0.82rem; font-weight:500;">{tech}</div>'
            for tech in ['Python 3.14', 'NLTK', 'Scikit-learn', 'TF-IDF',
                         'Streamlit', 'Plotly', 'Pandas', 'Logistic Regression']
        ]) + """
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card">
            <div class="card-title">📊 Project Metrics</div>
        """, unsafe_allow_html=True)

        metrics = [
            ("🎯", "Model Accuracy",    "94.06%"),
            ("📋", "Dataset Size",      "60,337 trials"),
            ("🏥", "Disease Classes",   "8 categories"),
            ("🔤", "TF-IDF Features",   "5,000 features"),
            ("📈", "Training Data",     "48,269 samples"),
            ("🧪", "Test Data",         "12,068 samples"),
            ("🤖", "Models Tested",     "5 algorithms"),
            ("⚡", "Best Algorithm",    "Logistic Regression"),
        ]

        for icon, label, value in metrics:
            st.markdown(f"""
            <div class="sidebar-stat" style="margin-bottom:8px;">
                <span class="sidebar-stat-label">{icon} {label}</span>
                <span class="sidebar-stat-value">{value}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card" style="margin-top:16px;">
            <div class="card-title">🔄 Pipeline Phases</div>
            <div style="color:#94A3B8; font-size:0.85rem; line-height:2;">
                ✅ Phase 1: Problem Understanding<br>
                ✅ Phase 2: Dataset Exploration<br>
                ✅ Phase 3: Environment Setup<br>
                ✅ Phase 4: Data Cleaning<br>
                ✅ Phase 5: NLP Preprocessing<br>
                ✅ Phase 6: Exploratory Data Analysis<br>
                ✅ Phase 7: TF-IDF Vectorization<br>
                ✅ Phase 8: ML Model Training<br>
                ✅ Phase 9: Model Evaluation<br>
                ✅ Phase 10: Streamlit Deployment
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with ❤️ by <span>Saroon</span> ·
    Powered by <span>Python + NLP + ML</span> ·
    <span>MediClassify AI © 2025</span>
</div>
""", unsafe_allow_html=True)