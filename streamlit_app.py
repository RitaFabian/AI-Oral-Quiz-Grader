import streamlit as st
import whisper
from sentence_transformers import SentenceTransformer, util
import sounddevice as sd
from scipy.io.wavfile import write
import os
import random
import numpy as np
import logging

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Oral Quiz Grader", page_icon="🎤", layout="centered")

# Cache the heavy AI models to prevent reloading on every interaction.
# This significantly improves app performance after the initial load.
@st.cache_resource
def load_models():
    print("Loading AI models...")
    stt_model = whisper.load_model("small")
    nlp_model = SentenceTransformer("all-MiniLM-L6-v2")
    return stt_model, nlp_model

# --- CORE LOGIC ---
class AudioAutoGrader:
    def __init__(self):
        """Initialize the grader with cached AI models and the question bank."""
        self.stt_model, self.nlp_model = load_models()
        
        # In a real production app, this would likely come from a database or JSON file.
        self.questions = [
            {
                "id": 1, "question": "Explain the process by which plants make their own food.",
                "reference": "Plants use photosynthesis to convert sunlight, water, and carbon dioxide into energy and food.",
                "keywords": "photosynthesis, sunlight, carbon dioxide, chlorophyll, glucose"
            },
            {
                "id": 2, "question": "What is the largest organ in the human body and what is its purpose?",
                "reference": "The skin is the largest organ in the human body, acting as a protective barrier against the environment.",
                "keywords": "skin, organ, human body, protection, barrier"
            },
            {
                "id": 3, "question": "Describe the basic building blocks of life.",
                "reference": "Cells are the fundamental building blocks of all living organisms and carry out all life processes.",
                "keywords": "cells, building blocks, organisms, microscopic"
            },
            {
                "id": 4, "question": "Which part of a cell houses the genetic material or DNA?",
                "reference": "The nucleus is the cell organelle that contains DNA and coordinates cell activities.",
                "keywords": "nucleus, DNA, genetic material, organelle, chromosomes"
            },
            {
                "id": 5, "question": "Explain the process of transpiration in plants.",
                "reference": "Transpiration is the process where water travels through a plant and evaporates from the leaves.",
                "keywords": "transpiration, evaporation, leaves, xylem, water movement"
            },
            {
                "id": 6, "question": "What is a carbohydrate and why does the body need it?",
                "reference": "Carbohydrates are organic compounds like sugars and starches that provide the body with energy.",
                "keywords": "carbohydrate, sugar, starch, energy, glucose"
            },
            {
                "id": 7, "question": "What is the primary function of the heart in the human body?",
                "reference": "The heart acts as a pump that circulates blood throughout the body to deliver oxygen and nutrients.",
                "keywords": "heart, pump, blood, circulation, oxygen, nutrients"
            },
            {
                "id": 8, "question": "Briefly describe the process of binary fission.",
                "reference": "Binary fission is a type of asexual reproduction where a single cell divides into two identical daughter cells.",
                "keywords": "binary fission, asexual reproduction, division, identical, bacteria"
            },
            {
                "id": 9, "question": "Define the scientific study of biology.",
                "reference": "Biology is the branch of science that deals with the study of living organisms and their life processes.",
                "keywords": "biology, science, living organisms, life processes"
            },
            {
                "id": 10, "question": "What are some of the key characteristics that define a living thing?",
                "reference": "Living things are defined by their ability to grow, reproduce, move, and respond to stimuli in their environment.",
                "keywords": "growth, reproduction, movement, stimuli, metabolism"
            }
        ]

    def get_random_question(self):
        return random.choice(self.questions)

    def record_audio(self, duration=10, filename="student_response.wav", fs=44100):
        """Captures audio from the default input device and saves to disk."""
        try:
            # Record raw audio data as float32 for better compatibility
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
            sd.wait()
            write(filename, fs, recording)
            return filename
        except Exception as e:
            logging.error(f"Audio recording failed: {e}")
            return None

    def transcribe(self, audio_path, context_keywords=""):
        """
        Uses OpenAI Whisper to convert speech to text.
        'context_keywords' provides a hint to the model for domain-specific terms.
        """
        try:
            result = self.stt_model.transcribe(audio_path, fp16=False, initial_prompt=context_keywords)
            return result["text"].strip()
        except Exception as e:
            logging.error(f"Transcription failed: {e}")
            return ""

    def grade_response(self, student_text, reference_answer, threshold=0.65):
        """
        Grades the response by calculating the semantic similarity between the 
        student's answer and the reference key.
        """
        if not student_text:
            return {"grade": 0, "similarity_score": 0, "status": "FAIL"}

        # Convert text to vector embeddings to capture semantic meaning
        embedding_1 = self.nlp_model.encode(student_text.lower(), convert_to_tensor=True)
        embedding_2 = self.nlp_model.encode(reference_answer.lower(), convert_to_tensor=True)

        # Calculate Cosine Similarity (direction match between vectors)
        cosine_scores = util.cos_sim(embedding_1, embedding_2)
        score = float(cosine_scores[0][0])
        
        # Convert 0-1 score to percentage
        final_grade = max(0, score) * 100
        
        # Determine pass/fail based on the strictness threshold
        status = "PASS" if score >= threshold else "FAIL"
        
        return {
            "grade": round(final_grade, 2), 
            "similarity_score": round(score, 4), 
            "status": status
        }

# --- STREAMLIT UI ---
def main():
    st.title("🎤 AI Oral Quiz Grader")
    st.markdown("### Interactive Oral Exam System")
    st.markdown("Click **Get Question**, record your answer, and get instant AI feedback.")

    # Sidebar for Settings
    with st.sidebar:
        st.header("⚙️ Settings")
        pass_threshold = st.slider("Pass Threshold", 0.0, 1.0, 0.65)
        st.caption("Adjust how strict the AI grading is.")

    # Initialize the Grader class in Session State to persist across re-runs
    if 'grader' not in st.session_state:
        with st.spinner("Loading AI Models..."):
            st.session_state.grader = AudioAutoGrader()
    
    # Track the current question so it doesn't change when buttons are clicked
    if 'current_question' not in st.session_state:
        st.session_state.current_question = None

    # Handler for generating a new question
    if st.button("📝 Get New Question"):
        st.session_state.current_question = st.session_state.grader.get_random_question()
        # Reset previous grading results
        if 'last_result' in st.session_state:
            del st.session_state.last_result

    # Main Interface Area
    if st.session_state.current_question:
        q = st.session_state.current_question
        st.markdown(f"### ❓ Question: \n> **{q['question']}**")
        
        duration = st.slider("⏱️ Recording Duration (seconds)", 5, 30, 10)
        
        if st.button("🔴 Record Answer"):
            with st.spinner(f"Recording for {duration} seconds... Speak now!"):
                audio_file = st.session_state.grader.record_audio(duration=duration)
            
            if audio_file and os.path.exists(audio_file):
                st.success("Recording complete!")
                # Allow user to verify their recording
                st.audio(audio_file, format="audio/wav")
            
            with st.spinner("Transcribing and Grading..."):
                if audio_file:
                    student_text = st.session_state.grader.transcribe(audio_file, context_keywords=q['keywords'])
                    result = st.session_state.grader.grade_response(student_text, q['reference'], threshold=pass_threshold)
                else:
                    student_text = ""
                    result = {"grade": 0, "status": "ERROR", "similarity_score": 0}
                
                st.divider()
                st.write(f"**You said:** *\"{student_text}\"*")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Grade", f"{result['grade']}/100")
                with col2:
                    if result['status'] == "PASS":
                        st.success(f"Result: {result['status']}")
                    else:
                        st.error(f"Result: {result['status']}")
                
                with st.expander("Show Reference Answer"):
                    st.write(q['reference'])

if __name__ == "__main__":
    main()
