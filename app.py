import os
import tempfile
import requests
import streamlit as st
from bs4 import BeautifulSoup
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from txtai.embeddings import Embeddings

# Load URLs and Fetch Content
def load_urls(file_path="collected_urls.txt"):
    with open(file_path, "r") as file:
        return [line.strip() for line in file.readlines()]

def fetch_content(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            return soup.get_text()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
    return ""

def load_and_chunk_content(file_path="collected_urls.txt", chunk_size=500):
    urls = load_urls(file_path)
    chunks = []
    for url in urls:
        content = fetch_content(url)
        if content:
            words = content.split()
            chunks.extend([" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)])
    return chunks

# Initialize Hugging Face Pipelines
def load_huggingface_pipelines():
    audio_to_text = pipeline("automatic-speech-recognition", model="openai/whisper-base")
    text_to_audio = pipeline("text-to-speech", model="espnet/kan-bayashi_ljspeech_vits")
    return audio_to_text, text_to_audio

# Create Embeddings Index with txtai
def create_txtai_index(chunks):
    embeddings = Embeddings({"path": "sentence-transformers/all-MiniLM-L6-v2"})
    embeddings.index([(uid, chunk, None) for uid, chunk in enumerate(chunks)])
    return embeddings

# Retrieve Answer
def retrieve_answer(embeddings, question_text, chunks):
    results = embeddings.search(question_text, 1)
    best_chunk_idx = results[0][0]
    return chunks[best_chunk_idx]

# Streamlit App
st.title("Urdu Literature Voice Chatbot")
st.write("Upload an audio file (in .wav format) to ask your question in Urdu or English!")

audio_to_text, text_to_audio = load_huggingface_pipelines()

# File Uploader for audio input
uploaded_file = st.file_uploader("Upload a .wav audio file", type=["wav"])
if uploaded_file is not None:
    # Convert audio to text
    question_text = audio_to_text(uploaded_file)["text"]
    st.write("Transcribed question:", question_text)
    
    # Retrieve answer
    chunks = load_and_chunk_content("collected_urls.txt")
    embeddings = create_txtai_index(chunks)
    answer_text = retrieve_answer(embeddings, question_text, chunks)
    st.write("Answer:", answer_text)
    
    # Convert answer text to speech
    answer_audio = text_to_audio(answer_text)
    answer_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with open(answer_audio_path.name, "wb") as f:
        f.write(answer_audio["waveform"])
    st.audio(answer_audio_path.name)

    # Clean up
    os.remove(answer_audio_path.name)

 

