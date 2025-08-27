# Semantic Textual Similarity (STS) Model API

This project implements a **Semantic Textual Similarity (STS) model** that quantifies how semantically close two text inputs are on a continuous scale from **0 (highly dissimilar)** to **1 (highly similar)**.  

---

## 🔍 Model Used
- **LaBSE (Language-agnostic BERT Sentence Embedding)**  
- Generates embeddings for both text inputs and computes cosine similarity.

---

## 🚀 Features
- Quantifies **semantic similarity** between two text inputs
- Built using **transformer-based sentence embeddings**
- Lightweight **REST API endpoint** using **FastAPI** (can also run with Uvicorn/Flask)
- **JSON input/output format** for easy integration
- Deployed on **Hugging Face Spaces** (free & accessible)

---

## 📦 Input and Output Format

### Request
{
"text1": "nuclear body seeks new technology ...",
"text2": "terror suspects face arrest ..."
}

### Response
{
"similarity score": 0.2
}

---

## ⚙️ Running Locally

1. **Clone this repository**
https://github.com/KumarSachin-DS/STS.git
cd STS

2. **Install dependencies**
pip install -r requirements.txt

3. **Run the API**
uvicorn app:app --host 0.0.0.0 --port 7860

4. **Test with `cURL` or Postman**
curl -X POST "https://Sachin-kumar-STS.hf.space/predict"
-H "Content-Type: application/json"
-d '{"text1": "This is a sample.", "text2": "This is another sample."}'

---

## ☁️ Deployment

- **Platform Used:** Hugging Face Spaces  
- **Why Hugging Face?**
- Heroku requires credit card details for free credits  
- Render’s free memory tier was insufficient to build the container  
- Hugging Face offers a **free, credit-free, hassle-free option**

---

## 📌 Summary
This API allows fast and simple **Semantic Textual Similarity (STS) computation** between any two text inputs.  
Ideal for applications like:  
- Document matching  
- Duplicate detection  
- Paraphrase identification  
- Smart search systems  

---

## 🛠️ Tech Stack
- **Python**
- **FastAPI** / Uvicorn
- **Transformers (LaBSE)**
- **Hugging Face Spaces**

---

## 👨‍💻 Author
Developed by **<Sachin-Kumar>**

