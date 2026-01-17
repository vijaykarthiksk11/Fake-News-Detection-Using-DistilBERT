# Fake-News-Detection-Using-DistilBERT


Fake news has become a major challenge in today’s digital world, spreading misinformation rapidly through social media and online news platforms. This project aims to automatically classify news articles as **Fake News** or **Real News** using **Natural Language Processing (NLP)** and a **Transformer-based deep learning model (DistilBERT)**.

The model is fine-tuned on a labeled news dataset and achieves high accuracy in detecting fake news.

---

## 📌 Project Objective

The main objectives of this project are:
- To build an automated system for fake news detection
- To apply NLP techniques using transformer-based models
- To reduce misinformation using AI-based solutions

---

## 🧠 Model Description

- **Model Used:** DistilBERT (distilbert-base-uncased)
- **Model Type:** Transformer-based language model
- **Task:** Binary text classification
- **Classes:**
  - `0` → Fake News
  - `1` → Real News

DistilBERT is a lighter and faster version of BERT that maintains strong language understanding while reducing computational cost.

---

## 📂 Dataset Description

The dataset consists of two CSV files:

### 🔹 Fake.csv
- Contains fake news articles
- Columns: `title`, `text`

### 🔹 True.csv
- Contains real news articles
- Columns: `title`, `text`

Each article is assigned a label:
- Fake News → `0`
- True News → `1`

**Dataset Source:** Kaggle Fake News Dataset

---

## ⚙️ Technologies Used

- Python  
- PyTorch  
- Hugging Face Transformers  
- Google Colab (GPU enabled)  
- Pandas  
- Matplotlib  

---

## 🚀 Project Workflow

1. Load and preprocess the dataset
2. Assign labels to fake and real news
3. Tokenize text using DistilBERT tokenizer
4. Create a custom PyTorch Dataset
5. Split the dataset into training and testing sets
6. Fine-tune the DistilBERT model
7. Evaluate the model using accuracy
8. Save the trained model and tokenizer
9. Perform inference on new news articles

---

## 🖥️ How to Run the Project (Google Colab)

1. Open the notebook in Google Colab  
2. Enable GPU  
   - Runtime → Change runtime type → GPU
3. Upload dataset files:
   - `Fake.csv`
   - `True.csv`
4. Run all cells in order
5. View accuracy results and predictions

---

## 📊 Model Performance

- **Training Accuracy:** ~99%
- **Testing Accuracy:** ~99%
- **Loss:** Decreases consistently across epochs

> High accuracy is achieved due to the quality of the dataset and the strong contextual understanding capability of DistilBERT.

---

## 🔍 Sample Prediction

**Input:**
Breaking: Scientists discover a new element on Mars!

makefile
Copy code

**Output:**
Fake News

yaml
Copy code

---

## 💾 Model Saving

After training, the following files are saved:
- Fine-tuned DistilBERT model
- Tokenizer configuration files

These can be reused for inference or deployment.

---

## 🎯 Applications

- Social media misinformation detection  
- News credibility verification  
- Content moderation systems  
- Media analytics platforms  

---

## 🔮 Future Enhancements

- Deploy as a Streamlit web application  
- Add Precision, Recall, and F1-score metrics  
- Support multilingual fake news detection  
- Integrate real-time news APIs  

---

## 👨‍💻 Author

- **Project Title:** Fake News Detection Using DistilBERT  
- **Project Type:** Academic / Final Year Project  
- **Domain:** Data Science / AIML/ NLP
- Vijay Karthik
