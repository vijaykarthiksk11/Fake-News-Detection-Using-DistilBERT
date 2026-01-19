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
# HEADER
# -------------------------------------------------
st.markdown(
    """
    <h1 style="text-align:center;">📰 Fake News Detection System</h1>
    <p style="text-align:center; color:grey;">
    NLP-based web application using a fine-tuned DistilBERT model
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# -------------------------------------------------
# LOAD TRAINED MODEL (LOCAL)
# -------------------------------------------------
MODEL_PATH = "C:\Users\11vij\Downloads\544d63c1-320c-4ac6-9f8e-1d8c1191432d\544d63c1-320c-4ac6-9f8e-1d8c1191432d.tmp"

@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

try:
    with st.spinner("🔄 Loading trained model..."):
        tokenizer, model = load_model()
    st.success("✅ Trained model loaded successfully")
except Exception as e:
    st.error("❌ Model loading failed. Please check model files.")
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
# PREDICTION
# -------------------------------------------------
if st.button("🔍 Analyze News"):
    if news_text.strip() == "":
        st.warning("⚠ Please enter some text.")
    else:
        try:
            with st.spinner("🤖 Analyzing news content..."):
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

                prediction = torch.argmax(probs).item()
                confidence = probs[0][prediction].item() * 100

            st.divider()

            if prediction == 1:
                st.success(
                    f"✅ **REAL NEWS**\n\n"
                    f"🔐 Confidence: **{confidence:.2f}%**"
                )
            else:
                st.error(
                    f"❌ **FAKE NEWS**\n\n"
                    f"🔐 Confidence: **{confidence:.2f}%**"
                )

        except Exception as e:
            st.error("❌ Error during prediction.")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.divider()
st.markdown(
    """
    **Model:** Fine-tuned DistilBERT  
    **Deployment:** Streamlit Community Cloud
    """
)
