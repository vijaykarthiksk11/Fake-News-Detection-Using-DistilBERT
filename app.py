import streamlit as st
import torch
import torch.nn.functional as F
import pickle
from huggingface_hub import hf_hub_download

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detection App")
st.write("Enter a news article to check whether it is **Fake** or **Real**.")

# -------------------------------------------------
# LOAD MODEL & TOKENIZER (CPU SAFE)
# -------------------------------------------------
@st.cache_resource
def load_model_and_tokenizer():
    # Download from Hugging Face
    model_path = hf_hub_download(
        repo_id="vijaykarthik11/fake-news-model",
        filename="fake_news_model.pkl"
    )

    tokenizer_path = hf_hub_download(
        repo_id="vijaykarthik11/fake-news-model",
        filename="fake_news_tokenizer.pkl"
    )

    # Load model (IMPORTANT: CPU only)
    model = torch.load(model_path, map_location=torch.device("cpu"))
    model.eval()

    # Load tokenizer
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer


model, tokenizer = load_model_and_tokenizer()

# -------------------------------------------------
# USER INPUT
# -------------------------------------------------
news_text = st.text_area(
    "📝 Paste News Content Here",
    height=200,
    placeholder="Enter news article text..."
)

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
if st.button("🔍 Check News"):
    if news_text.strip() == "":
        st.warning("Please enter some news text.")
    else:
        inputs = tokenizer(
            news_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0][prediction].item()

        if prediction == 1:
            st.success(f"✅ Real News ({confidence * 100:.2f}% confidence)")
        else:
            st.error(f"❌ Fake News ({confidence * 100:.2f}% confidence)")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & NLP")
