import streamlit as st
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import pandas as pd
import os
from dotenv import load_dotenv
import re

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="AI Career Assistant", layout="wide")

# -------------------------
# PASTEL UI
# -------------------------
st.markdown("""
<style>

.stApp{
background-color:#FFF8E7;
}

h1,h2,h3{
color:#2F5D50;
}

.stButton>button{
background-color:#A8D5BA;
color:black;
border-radius:8px;
padding:8px;
}

.skillbox{
background:#E8F5E9;
padding:10px;
border-radius:10px;
margin:5px;
}

.quizcard{
background:#ffffff;
border-radius:12px;
padding:15px;
box-shadow:0px 2px 6px rgba(0,0,0,0.1);
margin-bottom:15px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# ENV
# -------------------------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=api_key)

# -------------------------
# MODEL
# -------------------------
st.info("Loading AI model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------
# TITLE
# -------------------------
st.title("AI Career Assistant 🚀")

st.write("Resume Analysis • Skill Gap Detection • Learning Plan • Quiz • Mock Interview")

# -------------------------
# FUNCTIONS
# -------------------------

def extract_pdf_text(file):
    try:
        return extract_text(file)
    except:
        return ""


def similarity_score(resume, jd):

    e1 = model.encode([resume])
    e2 = model.encode([jd])

    score = cosine_similarity(e1, e2)[0][0]

    return score


def extract_skills(text):

    skills = [
        "python","sql","machine learning","deep learning",
        "docker","kubernetes","aws","azure","gcp",
        "pandas","numpy","tableau","power bi",
        "git","linux","flask","django","react","node"
    ]

    found = []

    for s in skills:
        if s in text.lower():
            found.append(s)

    return found


def skill_gap(resume, jd):

    resume_skills = extract_skills(resume)
    job_skills = extract_skills(jd)

    missing = list(set(job_skills) - set(resume_skills))
    matched = list(set(job_skills) & set(resume_skills))

    return matched, missing


def ai_report(resume, jd):

    prompt=f"""
Analyze this resume and job description.

Resume:
{resume}

Job Description:
{jd}

Give:

1 Resume evaluation with scores out of 5
2 Resume improvement suggestions
3 Career advice
"""

    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"user","content":prompt}]
    )

    return r.choices[0].message.content


def recommend_jobs(resume):

    prompt=f"""
Based on this resume recommend top 5 job roles.

Resume:
{resume}
"""

    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"user","content":prompt}]
    )

    return r.choices[0].message.content


def generate_learning_plan(skill):

    prompt=f"Create a 30 day learning roadmap for {skill}"

    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"user","content":prompt}]
    )

    return r.choices[0].message.content


def generate_quiz(skill):

    prompt=f"""
Create 5 quiz questions about {skill}.
Give:

Question
Answer
"""

    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"user","content":prompt}]
    )

    return r.choices[0].message.content


def generate_interview(role):

    prompt=f"""
Give 5 interview questions for {role} with answers.
"""

    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"user","content":prompt}]
    )

    return r.choices[0].message.content


# -------------------------
# INPUT
# -------------------------

st.subheader("Upload Resume")

resume_file = st.file_uploader("Upload PDF Resume", type=["pdf"])

job_desc = st.text_area("Paste Job Description")

if st.button("Analyze Resume"):

    if resume_file and job_desc:

        resume_text = extract_pdf_text(resume_file)

        ats = similarity_score(resume_text, job_desc)

        report = ai_report(resume_text, job_desc)

        matched, missing = skill_gap(resume_text, job_desc)

        jobs = recommend_jobs(resume_text)

        # -------------------------
        # DASHBOARD
        # -------------------------

        st.subheader("Resume Score Dashboard")

        col1,col2 = st.columns(2)

        with col1:
            st.metric("ATS Score", round(ats,2))

        with col2:
            st.metric("Resume Strength", round((ats*0.8),2))

        df = pd.DataFrame({
            "Scores":[ats, ats*0.8],
            "Type":["ATS Score","Resume Score"]
        })

        st.bar_chart(df.set_index("Type"))

        # -------------------------
        # SKILL GAP
        # -------------------------

        st.subheader("Skill Gap Detection")

        col1,col2 = st.columns(2)

        with col1:

            st.write("### Matching Skills")

            for s in matched:
                st.markdown(f"<div class='skillbox'>{s}</div>", unsafe_allow_html=True)

        with col2:

            st.write("### Missing Skills")

            for s in missing:
                st.markdown(f"<div class='skillbox'>{s}</div>", unsafe_allow_html=True)

        # -------------------------
        # JOB ROLES
        # -------------------------

        st.subheader("Recommended Job Roles")

        st.write(jobs)

        # -------------------------
        # REPORT
        # -------------------------

        st.subheader("Resume Improvement Suggestions")

        st.write(report)

        st.download_button(
            "Download Report",
            report,
            file_name="resume_report.txt"
        )

# -------------------------
# CAREER TOOLS
# -------------------------

st.divider()

st.subheader("AI Career Tools")

tool = st.selectbox(
"Select Tool",
["Learning Plan","Skill Quiz","Mock Interview"]
)

# -------------------------
# LEARNING PLAN
# -------------------------

if tool=="Learning Plan":

    skill = st.text_input("Enter skill")

    if st.button("Generate Plan"):

        plan = generate_learning_plan(skill)

        st.write(plan)

# -------------------------
# QUIZ
# -------------------------

if tool=="Skill Quiz":

    skill = st.text_input("Skill")

    if st.button("Generate Quiz"):

        quiz = generate_quiz(skill)

        questions = quiz.split("\n")

        for q in questions:

            if q.strip()!="":

                st.markdown(f"<div class='quizcard'>{q}</div>", unsafe_allow_html=True)

# -------------------------
# MOCK INTERVIEW
# -------------------------

if tool=="Mock Interview":

    role = st.text_input("Job Role")

    if st.button("Start Interview"):

        interview = generate_interview(role)

        qs = interview.split("\n")

        for q in qs:

            if q.strip()!="":

                st.markdown(f"<div class='quizcard'>{q}</div>", unsafe_allow_html=True)

