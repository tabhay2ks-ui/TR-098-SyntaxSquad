import pandas as pd

df = pd.read_csv("data/raw/fake reviews dataset.csv")

df = df.rename(columns={
    'text_': 'review_text',
    'rating': 'rating',
    'label': 'label',
    'category': 'product_id'
})

print(df['label'].unique())

df['label'] = df['label'].map({
    'CG': 1,
    'OR': 0,
    1: 1,
    0: 0
})

import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

df['review_text'] = df['review_text'].apply(clean_text)

import numpy as np

df['reviewer_id'] = np.random.randint(1000, 2000, size=len(df))
df['seller_id'] = df['product_id']
df['timestamp'] = pd.date_range(start='2021-01-01', periods=len(df), freq='T')
df['verified_purchase'] = np.random.choice([0, 1], size=len(df))

df = df.sample(n=3000, random_state=42)

df.to_csv("data/processed/reviews.csv", index=False)