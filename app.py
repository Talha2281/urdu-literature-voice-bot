import os
import tempfile
import requests
import faiss
import streamlit as st
import pyttsx3
import whisper
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader

# Step 1: Load URLs from a .txt file and scrape content
def load_urls(file_path):
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

def load_and_chunk_content(file_path, chunk_size=500):
    urls = load_urls(file_path)
    chunks = []
    for url in urls:
        content = fetch_content(url)
        if content:
            words = content.split()
            chunks.extend([" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)])
    return chunks

# Step 2: Embed data and create FAISS index
def create_faiss_index(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return FAISS(index=index, embeddings=embeddings, docs=chunks, metadatas=None)

# Step 3: Set up retrieval system
def setup_retriever(index):
    retriever = index.as_retriever()
    qa_chain = RetrievalQA(retriever=retriever, llm=HuggingFacePipeline.from_pretrained("google/flan-t5-small"))
    return qa_chain

# Step 4: Add Voice-to-Voice using Whisper and pyttsx3
def transcribe_audio(file_path):
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    return result["text"]

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.setProperty("voice", "com.apple.speech.synthesis.voice.sara")
    engine.setProperty("rate", 125)  # Set the speed for clearer speech
    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    engine.save_to_file(text, temp_audio_file.name)
    engine.runAndWait()
    return temp_audio_file.name

# Step 5: Deploy on Streamlit
st.title("Urdu Literature Voice Chatbot")
st.write("Ask questions in Urdu or English by voice!")

uploaded_file = st.file_uploader("Upload your voice question", type=["wav", "mp3"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_audio_file:
        temp_audio_file.write(uploaded_file.getvalue())
        temp_audio_file_path = temp_audio_file.name
    
    # Transcribe voice input
    question_text = transcribe_audio(temp_audio_file_path)
    st.write("Transcribed question:", question_text)
    
    # Retrieve answer
    chunks = load_and_chunk_content("collected_urls.txt")
    index = create_faiss_index(chunks)
    qa_chain = setup_retriever(index)
    answer_text = qa_chain.run({"query": question_text})
    st.write("Answer:", answer_text)
    
    # Convert answer to speech and play
    answer_audio_path = text_to_speech(answer_text)
    st.audio(answer_audio_path)

    # Clean up temporary files
    os.remove(temp_audio_file_path)
    os.remove(answer_audio_path)
