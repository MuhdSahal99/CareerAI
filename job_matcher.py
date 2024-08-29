import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from utils.text_processing import preprocess_text
from utils.resume_parser import parse_resume

class JobMatcher:
    def __init__(self, job_listings_path=None):
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        if job_listings_path:
            self.jobs_df = pd.read_csv(job_listings_path)
            self.jobs_df['processed_description'] = self.jobs_df['description'].apply(preprocess_text)
            self.job_embeddings = self.model.encode(self.jobs_df['processed_description'].tolist())

        # Initialize Gemini
        genai.configure(api_key='AIzaSyA2Yh9iY_NxpWIBOjSGjLiT2mfXapPuM4E')
        self.gemini = genai.GenerativeModel('gemini-pro')

    def process_resume(self, resume_file):
        resume_text = parse_resume(resume_file)
        processed_resume = preprocess_text(resume_text)
        resume_embedding = self.model.encode([processed_resume])[0]

        similarities = cosine_similarity([resume_embedding], self.job_embeddings)[0]
        top_indices = similarities.argsort()[-5:][::-1]

        recommendations = []
        for idx in top_indices:
            job_title = self.jobs_df.iloc[idx]['title']
            job_description = self.jobs_df.iloc[idx]['description']
            similarity = similarities[idx]

            # Use Gemini to generate a personalized job description
            prompt = f"""Based on the following job description, provide concise, bullet-point information for a potential candidate. Include:
            1. Key skills the candidate should highlight
            2. Potential interview questions they might face
            4. Tips for tailoring their application

            Job Description: {job_description}

            Format the response as bullet points for easy readability."""

            response = self.gemini.generate_content(prompt)
            candidate_info = response.text

            recommendations.append((job_title, similarity, candidate_info))

        return recommendations

    def match_candidates(self, job_description, resumes):
        processed_job = preprocess_text(job_description)
        job_embedding = self.model.encode([processed_job])[0]

        processed_resumes = [preprocess_text(resume) for resume in resumes]
        resume_embeddings = self.model.encode(processed_resumes)

        similarities = cosine_similarity([job_embedding], resume_embeddings)[0]
        top_indices = similarities.argsort()[-5:][::-1]

        matches = []
        for idx in top_indices:
            resume = resumes[idx]
            similarity = similarities[idx]

            # Use Gemini to generate a match summary
            prompt = f"Based on the following job description and resume, provide a concise summary of why this candidate might be a good match:\n\nJob Description: {job_description}\n\nResume: {resume}"
            response = self.gemini.generate_content(prompt)
            match_summary = response.text

            matches.append((resume, similarity, match_summary))

        return matches