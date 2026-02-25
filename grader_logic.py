import sounddevice as sd
from scipy.io.wavfile import write
from sentence_transformers import util
import json
import random

class AudioAutoGrader:
    def __init__(self, stt_model, nlp_model, questions_file="questions.json"):
        self.stt_model = stt_model
        self.nlp_model = nlp_model
        self.questions = self.load_questions(questions_file)

    def load_questions(self, filepath):
        """Load questions from the JSON file. Use a default if it fails."""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading questions: {e}")
            # Return a simple backup question so the app doesn't crash
            return [{
                "id": 0,
                "question": "Sample Question: What is Biology?",
                "reference": "Biology is the study of life and living organisms.",
                "keywords": "study, life, organisms"
            }]

    def get_random_question(self):
        return random.choice(self.questions)

    def record_audio(self, duration=10, filename="student_response.wav"):
        """Records audio from the microphone."""
        try:
            fs = 44100 # Sample rate (standard for audio)
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
            sd.wait()
            write(filename, fs, recording)
            return filename
        except Exception as e:
            print(f"Recording error: {e}")
            return None

    def transcribe(self, audio_path, context_keywords=""):
        """Converts audio to text using AI."""
        try:
            result = self.stt_model.transcribe(audio_path, fp16=False, initial_prompt=context_keywords)
            return result["text"].strip()
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""

    def grade_response(self, student_text, reference_answer, threshold=0.65):
        """Grades the answer by comparing meanings (Semantic Similarity)."""
        if not student_text:
            return {"grade": 0, "similarity_score": 0, "status": "FAIL"}

        # 1. Convert text to numbers (Embeddings)
        emb1 = self.nlp_model.encode(student_text.lower(), convert_to_tensor=True)
        emb2 = self.nlp_model.encode(reference_answer.lower(), convert_to_tensor=True)

        # 2. Calculate how close the numbers are (Cosine Similarity)
        # .item() just extracts the single number from the result
        score = util.cos_sim(emb1, emb2).item()
        
        # 3. Convert to percentage
        grade = max(0, score) * 100
        status = "PASS" if score >= threshold else "FAIL"
        
        return {
            "grade": round(grade, 2), 
            "similarity_score": round(score, 4), 
            "status": status
        }
