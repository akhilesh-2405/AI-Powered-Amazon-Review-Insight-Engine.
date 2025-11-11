# AI-Powered-Amazon-Review-Insight-Engine.
AI-Powered Amazon Review Insight Engine
Multilingual NLP Pipeline for Sentiment, Topic, and Emotion Analysis

This project is a comprehensive Natural Language Processing (NLP) pipeline built to extract meaningful insights from Amazon product reviews.
It automates language translation, sentiment classification, topic modeling, and emotion detection, generating an intelligent report that summarizes customer opinions across multiple dimensions.

🚀 Key Features
🌍 1. Multilingual Review Processing

Automatically detects the language of each review (supports Hindi, French, Spanish, etc.)

Translates non-English text to English using Google Translate API

Cleans, tokenizes, and standardizes all reviews for analysis

🧹 2. Advanced Text Preprocessing

Stopword removal, lemmatization, and normalization using NLTK and spaCy

Word frequency and WordCloud visualizations

POS tagging and Named Entity Recognition for linguistic insight

🎭 3. Sentiment Analysis (Hybrid Approach)

Lexicon-based (VADER): Interpretable rule-based scoring

Machine Learning (Logistic Regression + Naive Bayes): Predictive modeling via TF-IDF vectors

Comparison of accuracy between both models

🧩 4. Topic Modeling & Semantic Analysis

Latent Semantic Analysis (LSA) for broad theme discovery

Latent Dirichlet Allocation (LDA) for probabilistic topic-word distributions

Word2Vec training for semantic similarity exploration

Produces interpretable topics and top keywords per theme

💬 5. Aspect-Based Sentiment & Emotion Detection

Detects key product aspects (price, quality, service, experience)

Maps sentiments to emotional tones using a simplified NRC-style lexicon

Generates aspect-sentiment heatmaps and emotion distribution plots

📊 6. Automated Report Generation

Creates a professional PDF report (Quantiphi_Final_Report.pdf)

Includes charts for sentiment, aspects, and emotions

Summarizes major findings and actionable insights

🛠️ Tech Stack
Category	Tools & Libraries
Data Processing	pandas, numpy, datasets
NLP & Translation	nltk, spacy, googletrans, langdetect, textblob
Visualization	matplotlib, seaborn, wordcloud
Machine Learning	scikit-learn, gensim
Reporting	fpdf
📦 Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/yourusername/ai-powered-amazon-review-insight-engine.git
cd ai-powered-amazon-review-insight-engine

2️⃣ Install Dependencies
pip install -r requirements.txt


(Or install selectively within Google Colab cells as shown in the notebook.)

3️⃣ Run the Script / Notebook
python ai_powered_amazon_review_insight_engine.py


Alternatively, open it in Google Colab to visualize intermediate results interactively.

📈 Output Overview
Output File	Description
enriched_reviews.csv	Cleaned & translated dataset
wordcloud.png	Word cloud visualization of processed text
topic_keywords.json	Top keywords per topic (LSA & LDA)
Quantiphi_Final_Report.pdf	Final summarized insights report
🧭 Analytical Flow

Data Enrichment → Multilingual cleaning and translation

Preprocessing → Lemmatization, POS tagging, stopword removal

Visualization → Word frequency, sentiment distribution

Sentiment Modeling → Lexicon + ML-based comparison

Topic Modeling → LSA, LDA, and Word2Vec for semantic exploration

Aspect & Emotion Detection → Fine-grained emotion tracking

Report Generation → Automated business-ready PDF report

🧪 Example Insights

“Joy” and “Trust” dominate overall emotional tone.

Product Quality and Price emerge as the most discussed aspects.

Service and Delivery attract more negative emotions.

Lexicon model achieves ~81% accuracy on sentiment classification.

LDA reveals hidden topics across categories like books, music, and electronics.

Developed by [Akhilesh Mohorir]
Originally designed in Google Colab for NLP experimentation and report automation.

Special thanks to open-source libraries powering modern NLP:
NLTK, spaCy, Gensim, Scikit-learn, and FPDF.
