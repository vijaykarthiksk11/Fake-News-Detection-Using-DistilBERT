import streamlit as st
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# -------------------------------------------------
# HEADER / UI
# -------------------------------------------------
st.markdown(
    """
    <h1 style="text-align:center;">📰 Fake News Detection System</h1>
    <p style="text-align:center; color:grey;">
    NLP-based web application using DistilBERT to classify news articles
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# -------------------------------------------------
# MODEL LOADING (WITH SPINNER)
# -------------------------------------------------
@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained(
        "distilbert-base-uncased"
    )
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )
    model.eval()
    return tokenizer, model

try:
    with st.spinner("🔄 Loading AI model, please wait..."):
        tokenizer, model = load_model()
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error("❌ Error while loading the model")
    st.stop()

# -------------------------------------------------
# USER INPUT
# -------------------------------------------------
news_text = st.text_area(
    "📝 Enter News Article",
    placeholder="Paste the complete news article text here...",
    height=220
)

# -------------------------------------------------
# PREDICTION LOGIC
# -------------------------------------------------
if st.button("🔍 Analyze News"):
    if news_text.strip() == "":
        st.warning("⚠ Please enter some text before clicking Analyze.")
    else:
        try:
            with st.spinner("🤖 Analyzing the news content..."):
                inputs = tokenizer(
                    news_text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                )

                with torch.no_grad():
                    outputs = model(**inputs)
                    probabilities = F.softmax(outputs.logits, dim=1)

                prediction = torch.argmax(probabilities).item()
                confidence = probabilities[0][prediction].item() * 100

            st.divider()

            # -------------------------------------------------
            # RESULT DISPLAY (WITH CONFIDENCE)
            # -------------------------------------------------
            if prediction == 1:
                st.success(
                    f"✅ **REAL NEWS**\n\n"
                    f"🔐 Confidence Score: **{confidence:.2f}%**"
                )
            else:
                st.error(
                    f"❌ **FAKE NEWS**\n\n"
                    f"🔐 Confidence Score: **{confidence:.2f}%**"
                )

        except Exception as e:
            st.error("❌ An unexpected error occurred during prediction.")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.divider()
st.markdown(
    """
    **Model:** DistilBERT  
    **Domain:** Natural Language Processing (NLP)  
    **Deployment:** Streamlit Community Cloud
    """
)
