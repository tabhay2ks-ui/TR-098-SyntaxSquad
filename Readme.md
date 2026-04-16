🛡️ AI-Powered Fake Review Detection & Seller Trust Scoring (Phase 2)
📌 Overview

Fake reviews distort customer trust and influence purchasing decisions across e-commerce platforms. This project builds an end-to-end AI pipeline to detect fake reviews, identify coordinated review campaigns, and compute a trust score for sellers.

In Phase 2, the system has been significantly enhanced to incorporate behavioral intelligence, clustering-based fraud detection, and explainability, making it closer to a real-world fraud detection system.

🚀 Key Features
🔍 Fake Review Detection
Uses TF-IDF + Logistic Regression
Combines text + behavioral features
Outputs:
Fake probability
Authenticity score (0–100)
Predicted label (Real/Fake)
🧠 Behavioral Analysis (NEW in Phase 2)

Enhanced feature engineering to capture real-world fraud patterns:

Review length & word count
Rating deviation from product average
Reviewer activity patterns
Daily review frequency (review bursts)
Extreme rating detection (1★ or 5★)
Verified purchase behavior
🕸️ Coordinated Campaign Detection (Enhanced)
Uses DBSCAN clustering
Groups reviewers based on behavior patterns
Detects potential review spam campaigns
Outputs:
cluster_id
suspicious_cluster_flag
⭐ Seller Trust Scoring (Improved)

Aggregates review-level signals into seller-level insights:

Incorporates:
Authenticity scores
Verified purchase ratio
Fake probability trends
Suspicious cluster participation
Outputs:
Trust score (0–100)
Seller grade (A–F)
Human-readable explanation
💬 Explainable AI (NEW)

Every review now includes a human-readable explanation:

Explains why a review is suspicious or authentic
Uses feature-based reasoning such as:
Short review length
High reviewer activity
Rating anomalies
Cluster membership
🏗️ Project Structure
fake-review-detector/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data_preprocessing.py
│   ├── features.py           # Enhanced behavioral features
│   ├── model.py              # Improved ML pipeline (scaled + balanced)
│   ├── clustering.py         # DBSCAN-based campaign detection
│   ├── scoring.py            # Seller trust scoring
│   ├── explanation.py        # Explainable AI (NEW)
│   └── utils.py
│
├── outputs/
│   ├── predictions.csv
│   └── seller_scores.csv
│
├── main.py                   # Full pipeline execution
├── predict_one.py            # Single review prediction (NEW)
├── requirements.txt
└── README.md
⚙️ How It Works
Raw Data
   ↓
Data Preprocessing
   ↓
Feature Engineering (Text + Behavioral)
   ↓
Fake Review Model (ML)
   ↓
Clustering (Campaign Detection)
   ↓
Seller Trust Scoring
   ↓
Explainability Layer
   ↓
Outputs
📊 Input Data

The dataset includes:

review_text
rating
label (fake or real)
product_id
reviewer_id
seller_id
timestamp
verified_purchase

Additional features are engineered during processing.

📈 Outputs
📄 predictions.csv

Review-level insights:

Fake probability
Authenticity score
Predicted label
Cluster detection
Behavioral features
Explanation (NEW)
📄 seller_scores.csv

Seller-level insights:

Seller trust score (0–100)
Seller grade (A–F)
Aggregated behavior metrics
Trust explanation (NEW)
🧮 Trust Score Formula

The seller trust score is computed using:

40% → Review authenticity
25% → Verified purchase ratio
20% → Cluster trust (penalty for suspicious groups)
15% → Fake probability adjustment
🛠️ Technologies Used
Python
Pandas & NumPy
Scikit-learn
DBSCAN (Clustering)
TF-IDF (Text Vectorization)
TextBlob (Sentiment Analysis)
▶️ How to Run
Install dependencies:
pip install -r requirements.txt
Run full pipeline:
python main.py
Run single review prediction:
python predict_one.py
Outputs will be saved in:
outputs/
🎯 Phase 2 Improvements Summary

Compared to Phase 1, this version introduces:

Behavioral feature engineering
Scaled and balanced ML model
Coordinated campaign detection via clustering
Seller trust scoring improvements
Explainable AI for both reviews and sellers
Interactive single-review prediction
🔮 Future Improvements
Deep learning models (BERT / Transformers)
Graph-based fraud detection
Real-time API (FastAPI)
Dashboard (Streamlit / Tableau)
Advanced explainability (SHAP / LIME)
💡 Key Insight

This project moves beyond simple classification by integrating:

Text analysis
Behavioral intelligence
Network-based clustering
Explainability

to deliver a comprehensive trust evaluation system.

🏁 Conclusion

In Phase 2, we transformed a basic fake review classifier into a multi-layer fraud detection system that:

Detects fake reviews more reliably
Identifies coordinated fraudulent behavior
Provides transparent explanations
Quantifies seller credibility

This system is scalable and can be adapted for real-world platforms like Amazon, Yelp, and Flipkart.