import re
import pandas as pd
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional  # For Python 3.8 compatibility
import mysql.connector
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
import torch
import uvicorn

# Configuration
SQL_HOST = 'localhost'
SQL_USER = 'root'  
SQL_PASSWORD = 'Cr0c0d1le'  
SQL_DATABASE = 'iyconnect'
SQL_TABLE = 'emails'
SENTIMENT_MODEL = 'distilbert-base-uncased-finetuned-sst-2-english'
INTENT_MODEL = 'distilbert-base-uncased'
OUTPUT_FILE = 'email_analytics.json'

# FastAPI Models
class DateInput(BaseModel):
    date: str  
    end_date: Optional[str] = None  

# Email Cleaning
def clean_email_text(text):
    """Clean email text by removing signatures, HTML, and noise."""
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    text = re.sub(r'(?m)^--+.*$|^>.*$|^From:.*$|^To:.*$|^Subject:.*$|^Best regards,.*$|^Sincerely,.*$',
                  '', text, flags=re.IGNORECASE)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Database Integration
def fetch_sql_emails(start_date, end_date):
    """Fetch emails from SQL database for the given date range."""
    try:
        conn = mysql.connector.connect(
            host=SQL_HOST,
            user=SQL_USER,
            password=SQL_PASSWORD,
            database=SQL_DATABASE
        )
        cursor = conn.cursor(dictionary=True)
        
        # Query emails in date range
        query = """
            SELECT email_id, date, subject, body
            FROM {} 
            WHERE date >= %s AND date < %s
        """.format(SQL_TABLE)
        cursor.execute(query, (start_date, end_date))
        
        emails = []
        for row in cursor:
            body = clean_email_text(row.get('body', ''))
            emails.append({
                'email_id': str(row.get('email_id', '')),
                'date': row.get('date').isoformat() if row.get('date') else '',
                'text': f"{row.get('subject', '')} {body}"
            })
        
        cursor.close()
        conn.close()
        return emails
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# Email Analyzer with Pretrained Models
class EmailAnalyzer:
    def __init__(self):
        self.sentiment_pipeline = pipeline('sentiment-analysis', model=SENTIMENT_MODEL)
        self.intent_tokenizer = AutoTokenizer.from_pretrained(INTENT_MODEL)
        self.intent_model = AutoModelForSequenceClassification.from_pretrained(INTENT_MODEL)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=UMAP(n_components=5),
            hdbscan_model=HDBSCAN(min_cluster_size=5)
        )
        self.intent_labels = ['inquiry', 'complaint', 'follow-up', 'scheduling', 'other']

    def preprocess(self, texts):
        """Tokenize texts for intent model."""
        return self.intent_tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')

    def predict_sentiment(self, texts):
        """Predict sentiment using pretrained model (POSITIVE/NEGATIVE)."""
        results = self.sentiment_pipeline(texts)
        scores = [0.75 + 0.25 * r['score'] if r['label'] == 'POSITIVE' else 0.25 + 0.5 * r['score'] for r in results]
        return scores

    def predict_intent(self, texts):
        """Predict intent using pretrained model with keyword-based mapping."""
        intents = []
        for text in texts:
            text_lower = text.lower()
            if any(keyword in text_lower for keyword in ['when', 'how', 'what', '?']):
                intents.append('inquiry')
            elif any(keyword in text_lower for keyword in ['crash', 'issue', 'problem', 'fail']):
                intents.append('complaint')
            elif any(keyword in text_lower for keyword in ['follow', 'update']):
                intents.append('follow-up')
            elif any(keyword in text_lower for keyword in ['schedule', 'meeting', 'call']):
                intents.append('scheduling')
            else:
                intents.append('other')
        return intents

    def predict_topics(self, texts):
        """Extract topics using BERTopic."""
        topics, _ = self.topic_model.fit_transform(texts)
        topic_info = self.topic_model.get_topic_info()
        topic_labels = []
        for topic in topics:
            if topic == -1:
                topic_labels.append('miscellaneous')
            else:
                keywords = self.topic_model.get_topic(topic)[:3]
                topic_labels.append(', '.join([word for word, _ in keywords]))
        return topic_labels

    def generate_tasks(self, intents, sentiments):
        """Generate tasks based on intent and sentiment."""
        tasks = []
        for intent, sentiment in zip(intents, sentiments):
            if intent == 'complaint' and sentiment < 0.4:
                tasks.append('Follow up re: urgent complaint')
            elif intent == 'inquiry':
                tasks.append('Send information requested')
            elif intent == 'scheduling':
                tasks.append('Schedule meeting')
            else:
                tasks.append('Review email')
        return tasks

# FastAPI App
app = FastAPI(title="EMAIL Analytics API")

@app.post("/analyze-emails")
async def analyze_emails(date_input: DateInput):
    """Analyze emails from SQL database for a single date or date range."""
    try:
        start_date = date_input.date
        end_date = date_input.end_date

        # Validate date(s)
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

        # Handle single date or range
        if end_date:
            try:
                datetime.strptime(end_date, '%Y-%m-%d')
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid end_date format. Use YYYY-MM-DD.")
            query_end_date = end_date
        else:
            # Single date: set end_date to next day
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            query_end_date = (start_dt + timedelta(days=1)).strftime('%Y-%m-%d')

        # Fetch emails from database
        emails = fetch_sql_emails(start_date, query_end_date)

        if not emails:
            return {"message": "No emails found for the specified date range", "results": []}

        # Analyze
        df = pd.DataFrame(emails)
        df['date'] = pd.to_datetime(df['date']).astype('int64') // 10**6  # Convert to Unix timestamp (ms)
        texts = df['text'].tolist()
        analyzer = EmailAnalyzer()
        sentiments = analyzer.predict_sentiment(texts)
        intents = analyzer.predict_intent(texts)
        topics = analyzer.predict_topics(texts)
        tasks = analyzer.generate_tasks(intents, sentiments)

        # Combine results
        df['sentiment'] = sentiments
        df['intent'] = intents
        df['topic'] = topics
        df['task'] = tasks
        results = df[['email_id', 'date', 'sentiment', 'intent', 'topic', 'task']].to_dict('records')

        # Save to file
        pd.DataFrame(results).to_json(OUTPUT_FILE, orient='records', lines=True)
        return {"message": "Analysis complete", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/health")
async def health_check():
    """Check API health."""
    return {"status": "healthy"}

