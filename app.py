from huggingface_hub import hf_hub_download
import pickle

model_path = hf_hub_download(
    repo_id="vijaykarthik11/fake-news-model",
    filename="fake_news_model.pkl"
)

tokenizer_path = hf_hub_download(
    repo_id="vijaykarthik11/fake-news-model",
    filename="fake_news_tokenizer.pkl"
)

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)
