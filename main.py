import io
import os
import pypdf
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Initialize FastAPI App
app = FastAPI(
    title="AI Resume Screener API",
    description="Automated resume screening and leaderboard ranking using DistilBERT model.",
    version="1.0.0"
)

# Enable CORS for seamless frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve model path dynamically
def resolve_model_path():
    candidate_paths = [
        "./resume_model",
        "./resume_model/content/resume-scorer-model-final",
        "distilbert-base-uncased"
    ]
    for path in candidate_paths:
        if path == "distilbert-base-uncased":
            print(f"Fallback to pretrained model: {path}")
            return path
        if os.path.exists(os.path.join(path, "config.json")):
            print(f"Model directory resolved at: {path}")
            return path
    return "./resume_model"

MODEL_PATH = resolve_model_path()

print(f"Loading tokenizer and model from '{MODEL_PATH}'...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
except Exception as e:
    print(f"Warning: Default tokenizer load failed ({e}), trying use_fast=False...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
print("Model loaded successfully!")

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract clean text content from PDF file bytes using PyPDF."""
    try:
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        text_content = []
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text_content.append(extracted)
        return "\n".join(text_content).strip()
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def clean_candidate_name(filename: str) -> str:
    """Derive a clean display name from the uploaded filename."""
    base = os.path.splitext(filename)[0]
    cleaned = base.replace("_", " ").replace("-", " ").title()
    return cleaned

import re

def compute_keyword_overlap(job_desc: str, resume_text: str) -> float:
    """Compute semantic keyword overlap percentage between job description and resume text."""
    stop_words = {
        'and', 'the', 'for', 'with', 'this', 'that', 'from', 'your', 'will', 'all',
        'job', 'work', 'have', 'are', 'you', 'req', 'requirements', 'responsibilities',
        'looking', 'seeking', 'ability', 'strong', 'skills', 'experience', 'years'
    }
    words_jd = set(re.findall(r'\b[a-zA-Z]{3,}\b', job_desc.lower())) - stop_words
    words_res = set(re.findall(r'\b[a-zA-Z]{3,}\b', resume_text.lower()))
    
    if not words_jd:
        return 50.0
    
    overlap = words_jd.intersection(words_res)
    ratio = len(overlap) / len(words_jd)
    return min(100.0, ratio * 100.0)

@app.get("/")
async def serve_index():
    """Serve the single-page frontend application at the root route."""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "AI Resume Screener API is running. Visit /docs for API documentation."}

@app.post("/score-resumes/")
async def score_resumes(
    job_description: str = Form(...),
    files: list[UploadFile] = File(...)
):
    """
    Score uploaded PDF resumes against a job description.
    Returns candidate fit scores (0-100) sorted descending with recommendation badges.
    """
    if not job_description or not job_description.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")
    if not files:
        raise HTTPException(status_code=400, detail="At least one PDF resume must be uploaded.")

    results = []

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            results.append({
                "filename": file.filename,
                "candidate_name": clean_candidate_name(file.filename),
                "score": 0.0,
                "recommendation": "Error: Non-PDF File",
                "fit_category": "Low Fit",
                "word_count": 0,
                "char_count": 0
            })
            continue

        contents = await file.read()
        resume_text = extract_text_from_pdf(contents)

        if not resume_text:
            results.append({
                "filename": file.filename,
                "candidate_name": clean_candidate_name(file.filename),
                "score": 0.0,
                "recommendation": "Error: Unreadable PDF",
                "fit_category": "Low Fit",
                "word_count": 0,
                "char_count": 0
            })
            continue

        word_count = len(resume_text.split())
        char_count = len(resume_text)

        # 1. DistilBERT Model Inference
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
            raw_output = outputs.logits.item() if outputs.logits.numel() == 1 else outputs.logits[0][0].item()

            if 0.0 <= raw_output <= 1.0:
                model_score = raw_output * 100.0
            elif 0.0 <= raw_output <= 100.0:
                model_score = raw_output
            else:
                model_score = torch.sigmoid(outputs.logits).item() * 100.0

        # 2. Keyword Alignment Score
        keyword_score = compute_keyword_overlap(job_description, resume_text)

        # 3. Hybrid Fit Score (50% Model NLP + 50% Keyword Match)
        raw_final_score = (0.5 * model_score) + (0.5 * keyword_score)
        
        # Scaling adjustment to provide clear separation
        if keyword_score > 65.0:
            raw_final_score = max(76.0, raw_final_score + 10.0)
        elif keyword_score > 35.0:
            raw_final_score = max(52.0, raw_final_score)

        score = max(0.0, min(100.0, round(raw_final_score, 2)))

        # Assign Recommendation Category
        if score >= 75.0:
            recommendation = "Shortlist (High Fit)"
            fit_category = "High Fit"
        elif score >= 50.0:
            recommendation = "Moderate Fit"
            fit_category = "Moderate Fit"
        else:
            recommendation = "Low Fit"
            fit_category = "Low Fit"

        results.append({
            "filename": file.filename,
            "candidate_name": clean_candidate_name(file.filename),
            "score": score,
            "recommendation": recommendation,
            "fit_category": fit_category,
            "word_count": word_count,
            "char_count": char_count
        })

    # Sort descending by match score
    results.sort(key=lambda x: x["score"], reverse=True)

    return {
        "status": "success",
        "total_analyzed": len(results),
        "candidates": results
    }