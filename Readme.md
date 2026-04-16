# 🛡️ AI-Powered Fake Review Detection & Seller Trust Scoring

## 📌 Overview

Fake reviews distort customer trust, influence purchasing decisions, and reduce the credibility of online marketplaces. This project builds an end-to-end AI system that detects suspicious reviews, identifies coordinated review campaigns, and computes trust scores for sellers.

The system evolves from a basic classifier into a **real-time fraud detection platform** with:

- Machine Learning-based review classification  
- Behavioral feature engineering  
- Suspicious reviewer clustering  
- Seller trust scoring  
- Explainable AI outputs  
- FastAPI real-time inference  
- Streamlit interactive dashboard  

---

## 🚀 Key Features

### 🔍 Fake Review Detection
- TF-IDF + Logistic Regression
- Combines text + behavioral features

**Outputs:**
- Fake probability  
- Predicted label (Real / Fake)  
- Authenticity score (0–100)

---

### 🧠 Behavioral Analysis
Captures real-world fraud patterns using:

- Review length & word count  
- Exclamation count & uppercase ratio  
- Sentiment polarity  
- Verified purchase behavior  
- Reviewer activity patterns  
- Daily review frequency (bursts)  
- Rating deviation from product average  
- Extreme rating detection  
- Spam phrase detection  
- Repeated word ratio  

---

### 🕸️ Coordinated Campaign Detection
- Uses **DBSCAN clustering**
- Groups reviewers based on behavior
- Detects potential spam campaigns

**Outputs:**
- `cluster_id`  
- `suspicious_cluster_flag`

---

### ⭐ Seller Trust Scoring
Aggregates review-level signals into seller-level insights.

**Factors:**
- Authenticity score  
- Verified purchase ratio  
- Fake probability trends  
- Suspicious cluster activity  

**Outputs:**
- Trust score (0–100)  
- Seller grade (A–F)  
- Trust explanation  

---

### 💬 Explainable AI
Each prediction includes a human-readable explanation.

**Explains:**
- Why a review is suspicious or authentic  
- Behavioral and linguistic signals  
- Model confidence patterns  

---

### 🚨 Input Validation
- Detects gibberish or invalid review text  
- Prevents unrealistic predictions  
- Improves system robustness  

---

### 🔌 Real-Time API (FastAPI)


**Returns:**
- Fake probability  
- Predicted label  
- Authenticity score  
- Seller trust score  
- Seller grade  
- Review explanation  
- Seller explanation  

---

### 🖥️ Interactive Dashboard (Streamlit)

- Real-time review analysis  
- Clean UI with metrics  
- Interactive visualizations  

**Includes:**
- Fake probability gauge  
- Authenticity score gauge  
- Seller trust score chart  
- Risk profile visualization  

---

## 🏗️ Project Structure

fake-review-detector/
│
├── data/
│ ├── raw/
│ └── processed/
│
├── src/
│ ├── data_preprocessing.py
│ ├── features.py
│ ├── model.py
│ ├── clustering.py
│ ├── scoring.py
│ ├── explanation.py
│ ├── utils.py
│ ├── api.py
│ └── app.py
│
├── outputs/
│ ├── predictions.csv
│ └── seller_scores.csv
│
├── main.py
├── predict_one.py
├── requirements.txt
└── README.md


---

## ⚙️ How It Works


Raw Data
↓
Data Preprocessing
↓
Feature Engineering
↓
Fake Review Model
↓
Clustering (Campaign Detection)
↓
Seller Trust Scoring
↓
Explainability Layer
↓
API / Dashboard Output


---

## 📊 Input Data

The dataset includes:

- `review_text`  
- `rating`  
- `label`  
- `product_id`  
- `reviewer_id`  
- `seller_id`  
- `timestamp`  
- `verified_purchase`  

Additional features are engineered during processing.

---

## 📈 Outputs

### 📄 predictions.csv
- Fake probability  
- Authenticity score  
- Predicted label  
- Cluster detection  
- Behavioral features  
- Explanation  

### 📄 seller_scores.csv
- Seller trust score (0–100)  
- Seller grade (A–F)  
- Aggregated metrics  
- Trust explanation  

---

## 🧮 Trust Score Formula


40% → Review authenticity
25% → Verified purchase ratio
20% → Cluster trust
15% → Fake probability adjustment


---

## 🛠️ Technologies Used

- Python  
- Pandas & NumPy  
- Scikit-learn  
- TF-IDF  
- Logistic Regression  
- DBSCAN  
- TextBlob  
- FastAPI  
- Streamlit  
- Plotly  

---

## ▶️ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
2. Run batch pipeline
python main.py
3. Run single prediction
python predict_one.py
4. Start API
python -m uvicorn api:app --reload

Open:

http://127.0.0.1:8000/docs
5. Launch dashboard
python -m streamlit run app.py
🎯 Progress by Phase
Phase 1
Basic ML fake review classifier
TF-IDF + Logistic Regression
Phase 2
Behavioral feature engineering
Clustering (DBSCAN)
Seller trust scoring
Explainable AI
Single review prediction
Phase 3
FastAPI real-time API
Streamlit dashboard
Input validation
Interactive charts
Full system integration
💡 Key Insight

This project goes beyond classification by combining:

Text analysis
Behavioral intelligence
Network-based clustering
Explainability
Real-time deployment

to create a complete trust evaluation system.

🔮 Future Improvements
Transformer models (BERT)
Graph-based fraud detection
Batch upload analysis
SHAP / LIME explainability
Cloud deployment
Admin analytics dashboard
🏁 Conclusion

This project evolved into a full AI-powered fraud detection platform that:

Detects fake reviews
Identifies coordinated fraud
Explains decisions
Scores seller credibility
Provides real-time insights

It can be adapted for real-world platforms like Amazon, Yelp, and Flipkart.