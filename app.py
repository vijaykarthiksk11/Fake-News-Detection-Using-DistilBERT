import streamlit as st
import pickle
import torch
import torch.nn.functional as F

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
# LOAD MODEL & TOKENIZER
# -------------------------------------------------
@st.cache_resource
def load_model_and_tokenizer():
    with open("fake_news_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("fake_news_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    model.eval()
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
            st.success(f"✅ **Real News** ({confidence*100:.2f}% confidence)")
        else:
            st.error(f"❌ **Fake News** ({confidence*100:.2f}% confidence)")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & NLP")
