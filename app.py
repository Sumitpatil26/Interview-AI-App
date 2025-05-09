
import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# Load pretrained models from Hugging Face Hub
@st.cache_resource
def load_classifier():
    tokenizer = BertTokenizer.from_pretrained("sumit2603/bert-sports-interview-classifier")
    model = BertForSequenceClassification.from_pretrained("sumit2603/bert-sports-interview-classifier")
    return tokenizer, model

@st.cache_resource
def load_gpt2():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

# Load embedding data
@st.cache_data
def load_embeddings():
    return pd.read_csv("embedding_data.csv")

# Label Encoder setup
label_encoder = LabelEncoder()
label_encoder.classes_ = [
    'contract_talk',
    'injury_update',
    'playoff_expectations',
    'post_game_reaction',
    'team_dynamics'
]

# Classification function
def classify_transcript(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return label_encoder.inverse_transform([pred])[0]

# Text generation
def generate_response(prompt, gpt2_tokenizer, gpt2_model, device, max_length=100):
    inputs = gpt2_tokenizer(prompt, return_tensors="pt").to(device)
    outputs = gpt2_model.generate(
        **inputs,
        max_new_tokens=max_length,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
        pad_token_id=gpt2_tokenizer.eos_token_id
    )
    return gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True).split("A:")[-1].strip()

# Visualization
def show_plot(df):
    fig = px.scatter(
        df,
        x="x", y="y",
        color="label",
        hover_data=["sample_text"],
        title="Transcript Embedding Clusters"
    )
    st.plotly_chart(fig, use_container_width=True)

# Load everything once
tokenizer, model = load_classifier()
gpt2_tokenizer, gpt2_model, gpt2_device = load_gpt2()
embeddings_df = load_embeddings()

# Streamlit UI
st.title("🏟️ Sports Interview AI Dashboard")

section = st.sidebar.radio("Choose Feature", ["Transcript Classification", "Q&A Generator", "Visualization"])

if section == "Transcript Classification":
    st.header("📌 Classify Interview Transcript")
    text_input = st.text_area("Paste full transcript:")
    if st.button("Classify"):
        if text_input.strip():
            pred = classify_transcript(text_input, tokenizer, model)
            st.success(f"Predicted Category: {pred}")
        else:
            st.warning("Please enter transcript text.")

elif section == "Q&A Generator":
    st.header("🧠 AI Interview Response Generator")
    category = st.selectbox("Select Category", ["post_game_reaction", "injury_update", "contract_talk", "team_dynamics", "playoff_expectations"])
    question = st.text_input("Enter your question:")
    if st.button("Generate Answer"):
        if question.strip():
            prompt = f"Category: {category}\nQ: {question}\nA:"
            answer = generate_response(prompt, gpt2_tokenizer, gpt2_model, gpt2_device)
            st.markdown("**💬 AI Response:**")
            st.write(answer)
        else:
            st.warning("Please enter a question.")

elif section == "Visualization":
    st.header("📊 Transcript Embedding Explorer")
    show_plot(embeddings_df)
