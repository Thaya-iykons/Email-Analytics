# 📧 Conversational Analytics over Client Emails

This project provides a FastAPI-based service that performs **conversational analytics** on client emails received on a specific day or over a custom date range. The system connects to a MySQL database, fetches email data, cleans it, and uses NLP models to extract insights like **sentiment**, **intent**, and **topic**, then generates action-oriented tasks for each email.

---

## 🚀 Features

- 🔍 Clean email bodies (HTML, signatures, quoted replies)
- 📅 Analyze emails by date or date range
- 😄 Predict sentiment (positive or negative)
- 🧠 Identify email intent (e.g., inquiry, complaint)
- 🗂️ Extract topics using BERTopic
- 📝 Generate follow-up tasks based on insights
- 💾 Save results to `email_analytics.json`

---

## 🛠️ Tech Stack

- Python 3.8+
- FastAPI
- MySQL
- Transformers (HuggingFace)
- SentenceTransformers
- BERTopic
- UMAP + HDBSCAN
- BeautifulSoup

---

## 📦 Requirements

Ensure you have Python 3.8+ installed. Then install required packages:

```bash
pip install -r requirements.txt
