# 🛡️ AI-Powered Fake Review Detection & Seller Trust Scoring

## 📌 Overview

Fake reviews distort customer trust and influence purchasing decisions across e-commerce platforms. This project builds an end-to-end AI pipeline to detect fake reviews, identify coordinated review campaigns, and compute a trust score for sellers.

Our system combines **machine learning, behavioral analysis, and clustering techniques** to provide both review-level and seller-level insights.

---

## 🚀 Key Features

### 🔍 Fake Review Detection

* Uses **TF-IDF + Logistic Regression**
* Outputs:

  * Fake probability
  * Authenticity score (0–100)

### 🧠 Behavioral Analysis

* Detects unusual patterns such as:

  * Review bursts
  * Reviewer activity frequency
  * Rating deviations

### 🕸️ Coordinated Campaign Detection

* Uses **DBSCAN clustering**
* Identifies groups of reviewers acting together
* Flags suspicious clusters

### ⭐ Seller Trust Scoring

* Aggregates review-level signals into seller-level insights
* Outputs:

  * Trust score (0–100)
  * Grade (A–F)

---

## 🏗️ Project Structure

```
fake-review-detector/
│
├── data/
│   ├── raw/                  # Original dataset
│   └── processed/            # Cleaned dataset
│
├── src/
│   ├── data_preprocessing.py # Data cleaning & preparation
│   ├── features.py           # Feature engineering
│   ├── model.py              # ML model for fake detection
│   ├── clustering.py         # Campaign detection
│   ├── scoring.py            # Seller trust scoring
│   └── utils.py              # Helper functions
│
├── outputs/
│   ├── predictions.csv       # Review-level predictions
│   └── seller_scores.csv     # Seller-level trust scores
│
├── main.py                   # Runs full pipeline
├── requirements.txt
└── README.md
```

---

## ⚙️ How It Works

```
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
Outputs
```

---

## 📊 Input Data

The dataset includes:

* `review_text`
* `rating`
* `label` (fake or real)
* `product_id`
* `reviewer_id` (generated)
* `seller_id` (derived)
* `timestamp` (generated)
* `verified_purchase` (simulated)

We used a Kaggle Amazon Fake Review dataset and standardized it into a structured format for our pipeline.

---

## 📈 Outputs

### 📄 predictions.csv

Contains review-level insights:

* Fake probability
* Authenticity score
* Predicted label
* Cluster detection
* Behavioral features

### 📄 seller_scores.csv

Contains seller-level insights:

* Seller trust score (0–100)
* Seller grade (A–F)
* Aggregated behavior metrics

---

## 🧮 Trust Score Formula

The seller trust score is computed using:

* 50% → Review authenticity
* 25% → Verified purchase ratio
* 15% → Suspicious activity penalty
* 10% → Fake probability adjustment

---

## 🛠️ Technologies Used

* Python
* Pandas & NumPy
* Scikit-learn
* DBSCAN (Clustering)
* TF-IDF (Text Vectorization)

---

## ▶️ How to Run

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Run the pipeline:

```
python main.py
```

3. Outputs will be saved in:

```
outputs/
```

---

## 🎯 Future Improvements

* Add deep learning models (BERT)
* Improve sentiment analysis
* Real-time API deployment (FastAPI)
* Interactive dashboard (Streamlit)
* Explainable AI (why a review is fake)

---

## 💡 Key Insight

This project goes beyond simple classification by combining:

* Text analysis
* Behavioral signals
* Network-based clustering

to deliver a **complete trust evaluation system**.

---

## 🏁 Conclusion

We developed a scalable and modular system that:

* Detects fake reviews
* Identifies coordinated fraud
* Quantifies seller credibility

This approach can be applied to platforms like Amazon, Yelp, and Flipkart to improve trust and transparency in online marketplaces.

---
