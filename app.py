import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

st.set_page_config(page_title="Fake News Detection", layout="centered")

st.title("📰 Fake News Detection App")
st.write("Paste a news article and check whether it is **Fake** or **Real**")

@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

news_text = st.text_area("📝 Enter News Text", height=200)

if st.button("🔍 Predict"):
    if news_text.strip() == "":
        st.warning("Please enter some news text.")
    else:
        inputs = tokenizer(
            news_text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        with torch.no_grad():
            outputs = model(**inputs)

        prediction = torch.argmax(outputs.logits).item()

        if prediction == 1:
            st.success("✅ This looks like REAL News")
        else:
            st.error("❌ This looks like FAKE News")
