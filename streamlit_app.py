import io
import pandas as pd
import pypdf
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# -------------------------------------------------------------
# 1. Page Configuration & Custom CSS Injection
# -------------------------------------------------------------
st.set_page_config(
    page_title="AI Resume Screener", 
    page_icon="📄", 
    layout="wide"
)

# Inject custom CSS to make Streamlit look like a sleek dashboard
st.markdown("""
    <style>
    /* Dark dashboard theme styling */
    .main {
        background-color: #0F172A;
    }
    
    /* Clean primary header styling */
    h1 {
        color: #F8FAFC !important;
        font-weight: 700;
    }
    
    /* Hide default Streamlit clutter */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Style sidebar submit button */
    div.stButton > button:first-child {
        background-color: #4F46E5;
        color: #FFFFFF;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1rem;
        width: 100%;
        transition: all 0.2s ease-in-out;
    }
    
    div.stButton > button:first-child:hover {
        background-color: #4338CA;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📄 AI Resume Screener & Candidate Leaderboard")
st.write("Upload candidate resumes (PDF) and paste a job description to score and rank candidate fits.")

# -------------------------------------------------------------
# 2. Load ML Model & Tokenizer (Cached)
# -------------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = "./resume_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

try:
    tokenizer, model = load_model()
    st.success("AI Model loaded successfully!", icon="✅")
except Exception as e:
    st.error(f"Error loading model from './resume_model': {e}")
    st.stop()

# Helper function to parse PDF text
def extract_text_from_pdf(file_bytes):
    try:
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text
    except Exception:
        return ""

# -------------------------------------------------------------
# 3. Sidebar Input Workspace
# -------------------------------------------------------------
st.sidebar.header("📋 Input Panel")
job_description = st.sidebar.text_area("Paste Job Description:", height=220, placeholder="e.g. Seeking a Python Developer with PyTorch and FastAPI experience...")
uploaded_files = st.sidebar.file_uploader("Upload Resumes (PDF):", type=["pdf"], accept_multiple_files=True)

analyze_btn = st.sidebar.button("🚀 Analyze & Rank Candidates")

# -------------------------------------------------------------
# 4. Processing Engine & Leaderboard View
# -------------------------------------------------------------
if analyze_btn:
    if not job_description.strip():
        st.warning("Please enter a job description before running the evaluation.", icon="⚠️")
    elif not uploaded_files:
        st.warning("Please upload at least one PDF resume to evaluate.", icon="⚠️")
    else:
        results = []
        progress_bar = st.progress(0, text="Extracting and scoring resumes...")
        
        for idx, file in enumerate(uploaded_files):
            file_bytes = file.read()
            resume_text = extract_text_from_pdf(file_bytes)
            
            if not resume_text.strip():
                score = 0.0
                recommendation = "Error: Unreadable PDF"
            else:
                inputs = tokenizer(
                    job_description,
                    resume_text,
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    score = outputs.logits.item() * 100
                
                if score >= 75:
                    recommendation = "Shortlist (High Fit)"
                elif score >= 50:
                    recommendation = "Moderate Fit"
                else:
                    recommendation = "Low Fit"

            results.append({
                "Candidate / File": file.name,
                "Match Score (%)": round(score, 2),
                "Recommendation": recommendation
            })
            
            progress_bar.progress((idx + 1) / len(uploaded_files), text=f"Processed {idx + 1}/{len(uploaded_files)} resumes")

        progress_bar.empty()

        # Convert to Dataframe & Sort descending
        df = pd.DataFrame(results)
        df = df.sort_values(by="Match Score (%)", ascending=False).reset_index(drop=True)
        
        # Display Clean Dynamic Leaderboard
        st.subheader("🏆 Candidate Leaderboard")
        
        # Renders the clean dataframe WITHOUT index numbers 0, 1, 2...
        st.dataframe(
            df,
            column_config={
                "Candidate / File": st.column_config.TextColumn("Candidate / File", width="medium"),
                "Match Score (%)": st.column_config.NumberColumn("Match Score (%)", format="%.2f%%"),
                "Recommendation": st.column_config.TextColumn("Recommendation", width="medium"),
            },
            use_container_width=True,
            hide_index=True  # Hides row numbers index
        )
        
        st.write("")
        
        # Download Controls
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_data,
            file_name="candidate_rankings.csv",
            mime="text/csv"
        )
        