import io
import pandas as pd
import pypdf
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Page layout configuration
st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")

st.title("📄 AI Resume Screener & Leaderboard")
st.write("Upload candidate resumes (PDF) and paste a job description to score and rank fits.")

# 1. Load Model & Tokenizer with Caching (Runs once)
@st.cache_resource
def load_model():
    model_path = "./resume_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

try:
    tokenizer, model = load_model()
    st.success("AI Model loaded successfully!")
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

# Sidebar / Left Input Panel
st.sidebar.header("Input Panel")
job_description = st.sidebar.text_area("Paste Job Description:", height=200)
uploaded_files = st.sidebar.file_uploader("Upload Resumes (PDF):", type=["pdf"], accept_multiple_files=True)

# Main Dashboard Processing
if st.sidebar.button("Analyze & Rank Candidates"):
    if not job_description.strip():
        st.warning("Please enter a job description.")
    elif not uploaded_files:
        st.warning("Please upload at least one PDF resume.")
    else:
        results = []
        progress_bar = st.progress(0)
        
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
            
            progress_bar.progress((idx + 1) / len(uploaded_files))

        # Convert to Dataframe & Sort
        df = pd.DataFrame(results)
        df = df.sort_values(by="Match Score (%)", ascending=False).reset_index(drop=True)
        
        # Display Leaderboard
        st.subheader("🏆 Candidate Leaderboard")
        st.dataframe(df, use_container_width=True)
        
        # Download Button
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_data,
            file_name="candidate_rankings.csv",
            mime="text/csv"
        )