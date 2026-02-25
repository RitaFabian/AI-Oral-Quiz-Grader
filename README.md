# AI Oral Quiz Grader 🎤

## Overview
The **AI Oral Quiz Grader** is an automated assessment tool designed to conduct oral examinations. It generates questions, records student audio responses, transcribes speech to text, and grades the answer based on semantic similarity to a reference key.

This project demonstrates the application of advanced Data Science techniques—specifically **Automatic Speech Recognition (ASR)** and **Natural Language Processing (NLP)**—to solve real-world educational challenges.

## Features
- **Live Audio Recording**: Captures student responses directly via microphone.
- **Speech-to-Text**: Uses OpenAI's Whisper model for high-accuracy transcription.
- **Semantic Grading**: Grades answers based on *meaning*, not just keyword matching.
- **Interactive UI**: Built with Streamlit for a user-friendly experience.
- **Configurable Difficulty**: Adjustable grading threshold.

---

## The Data Science Behind It

This project relies on three core Data Science pillars:

### 1. Automatic Speech Recognition (ASR)
We utilize **OpenAI Whisper**, a state-of-the-art deep learning model trained on 680,000 hours of multilingual data. 
- **Role**: Converts raw audio waveforms (`.wav`) into text strings.
- **Why Whisper?**: It is robust against accents, background noise, and technical jargon, making it ideal for diverse student inputs.

### 2. Vector Embeddings (NLP)
Traditional grading checks for exact word matches (e.g., if the student says "create" instead of "make", a keyword check fails). We use **Sentence-Transformers (SBERT)** to solve this.
- **Process**: The model (`all-MiniLM-L6-v2`) converts the student's sentence and the teacher's reference answer into **384-dimensional dense vectors**.
- **Concept**: These vectors represent the *semantic meaning* of the sentence in a geometric space. Sentences with similar meanings are located close together in this space.

### 3. Cosine Similarity (Linear Algebra)
To calculate the grade, we compute the **Cosine Similarity** between the student's vector ($\mathbf{A}$) and the reference vector ($\mathbf{B}$):

$$ \text{similarity} = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|} $$

- **Result**: A score between -1 (opposite meaning) and 1 (identical meaning).
- **Application**: This score is normalized to a percentage (0-100%) to provide the final grade.

---

## Installation

### Prerequisites
- Python 3.8 or higher
- FFmpeg (required for audio processing)

### Install Dependencies
Run the following command to install the required Python libraries:

```bash
pip install streamlit openai-whisper sentence-transformers sounddevice scipy numpy
```

*Note: You may need to install PyTorch separately depending on your OS.*

---

## Usage

1. **Launch the Application**:
   Open your terminal and run the Streamlit server:
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Take the Quiz**:
   - Click **"Get New Question"** to receive a random biology question.
   - Adjust the **Recording Duration** slider if you need more time.
   - Click **"Record Answer"** and speak clearly into your microphone.

3. **View Results**:
   - The system will transcribe your audio.
   - It will display your **Grade** (0-100) and **Status** (PASS/FAIL).
   - You can expand the "Reference Answer" section to compare your response.

---

## For Lecturers: How to Add Questions

You can easily add, remove, or edit questions without touching any code.

1.  **Open the `questions.json` file** in a simple text editor (like Notepad on Windows or TextEdit on Mac).
2.  The file contains a list of questions. Each question is a block of text enclosed in curly braces `{ }`.

**To add a new question:**

- Copy an existing question block (from a `{` to a `}`).
- Paste it at the end of the list, **before** the closing square bracket `]`.
- **Important**: Add a comma `,` after the preceding question's closing brace `}`.
- Change the `"id"`, `"question"`, `"reference"`, and `"keywords"` for your new question.

### Example: Adding a question

```json
[
    {
        "id": 1,
        "question": "Existing question...",
        "reference": "...",
        "keywords": "..."
    },
    {
        "id": 2,
        "question": "This is my new question?",
        "reference": "This is the perfect answer for my new question.",
        "keywords": "new, question, keywords"
    }
]
```

---

## Project Structure

```
├── streamlit_app.py    # Main application entry point containing UI and Logic
├── README.md           # Project documentation
└── requirements.txt    # List of dependencies
```

### Key Classes
- **`AudioAutoGrader`**: The main controller class.
  - `load_models()`: Caches heavy AI models for performance.
  - `record_audio()`: Handles hardware interaction.
  - `transcribe()`: Wraps the Whisper inference logic.
  - `grade_response()`: Performs the vector embedding and similarity calculation.
