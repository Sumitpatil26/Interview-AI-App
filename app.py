import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import plotly.express as px
import joblib

# Load models and tokenizer
@st.cache_resource
def load_classifier():
    tokenizer = BertTokenizer.from_pretrained("./saved_bert_model")
    model = BertForSequenceClassification.from_pretrained("./saved_bert_model")
    return tokenizer, model

tokenizer_cls, model_cls = load_classifier()

@st.cache_resource
def load_generative_model():
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-rw-1b")
    model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-rw-1b")
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return tokenizer, model

tokenizer_gen, model_gen = load_generative_model()

# Load embedding data
@st.cache_data
def load_embedding_data():
    df = pd.read_csv("embedding_data.csv")
    return df

embedding_df = load_embedding_data()

# Label decoder
label_map = {
    0: "post_game_reaction",
    1: "injury_update",
    2: "contract_talk",
    3: "team_dynamics",
    4: "playoff_expectations",
    5: "training_commentary",
    6: "strategy_discussion",
    7: "personal_reflection"
}

# Transcript classification
def classify_transcript(text):
    inputs = tokenizer_cls(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model_cls(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return label_map.get(pred, "Unknown")

# Prompt generation
def build_prompt(category, question):
    readable = category.replace("_", " ")
    return f"You are a professional athlete giving a {readable} interview.\nQ: {question}\nA:"

def generate_response(prompt, max_length=100):
    inputs = tokenizer_gen(prompt, return_tensors="pt").to(model_gen.device)
    outputs = model_gen.generate(
        **inputs,
        max_new_tokens=max_length,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.85,
        pad_token_id=tokenizer_gen.eos_token_id
    )
    decoded = tokenizer_gen.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("A:")[-1].strip()

# Streamlit UI
st.title("🏟️ Sports Interview AI App")
section = st.sidebar.radio("Choose a Feature", ["Transcript Classification", "Q&A Generator", "Embedding Explorer"])

if section == "Transcript Classification":
    st.header("🗂️ Classify a Transcript")
    user_text = st.text_area("Paste interview transcript:")
    if st.button("Classify"):
        if user_text.strip():
            result = classify_transcript(user_text)
            st.success(f"Predicted Category: {result}")
        else:
            st.warning("Please enter text.")

elif section == "Q&A Generator":
    st.header("💬 Generate Interview Answer")
    category = st.selectbox("Select Category", list(label_map.values()))
    question = st.text_input("Enter your question:")
    if st.button("Generate"):
        if question.strip():
            prompt = build_prompt(category, question)
            answer = generate_response(prompt)
            st.markdown("**AI Response:**")
            st.write(answer)
        else:
            st.warning("Enter a question.")

elif section == "Embedding Explorer":
    st.header("📊 Explore Interview Topics")
    fig = px.scatter(
        embedding_df,
        x="x", y="y",
        color="label",
        hover_data=["sample_text"],
        title="Transcript Embeddings (t-SNE or PCA)"
    )
    st.plotly_chart(fig, use_container_width=True)
