import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ AI Fake Review Detection & Seller Trust Scoring")
st.write(
    "Analyze a product review in real time using your fraud detection API. "
    "The system evaluates review authenticity, suspicious behavior, and seller trust."
)

with st.form("review_form"):
    review_text = st.text_area(
        "Review Text",
        placeholder="Enter a product review here..."
    )

    rating = st.slider("Rating", min_value=1.0, max_value=5.0, value=5.0, step=1.0)

    verified_purchase = st.selectbox(
        "Verified Purchase",
        options=[1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    product_id = st.text_input("Product ID", value="demo_product_1")
    reviewer_id = st.number_input("Reviewer ID", min_value=1, value=12345, step=1)
    seller_id = st.text_input("Seller ID", value="demo_seller_1")

    submitted = st.form_submit_button("Analyze Review")


def label_badge(label: str) -> str:
    if label.lower() == "fake":
        return "🚨 Fake"
    return "✅ Real"


if submitted:
    if not review_text.strip():
        st.error("Please enter review text.")
    else:
        payload = {
            "review_text": review_text,
            "rating": rating,
            "verified_purchase": verified_purchase,
            "product_id": product_id,
            "reviewer_id": int(reviewer_id),
            "seller_id": seller_id
        }

        try:
            with st.spinner("Analyzing review..."):
                response = requests.post(API_URL, json=payload, timeout=30)

            if response.status_code != 200:
                st.error(f"API error: {response.status_code}")
                try:
                    st.json(response.json())
                except Exception:
                    st.text(response.text)
            else:
                result = response.json()

                st.success("Analysis complete.")

                col1, col2, col3 = st.columns(3)
                col1.metric("Prediction", label_badge(result.get("predicted_label", "Unknown")))
                col2.metric("Fake Probability", f'{result.get("fake_probability", 0):.4f}')
                col3.metric("Authenticity Score", f'{result.get("authenticity_score", 0):.2f}')

                col4, col5 = st.columns(2)
                col4.metric("Seller Trust Score", f'{result.get("seller_trust_score", 0):.2f}')
                col5.metric("Seller Grade", result.get("seller_grade", "N/A"))

                st.subheader("Review Explanation")
                st.write(result.get("review_explanation", "No explanation available."))

                st.subheader("Seller Trust Explanation")
                st.write(result.get("trust_explanation", "No explanation available."))

                with st.expander("Raw API Response"):
                    st.json(result)

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to the API. Make sure your FastAPI server is running at "
                "http://127.0.0.1:8000"
            )
        except requests.exceptions.Timeout:
            st.error("The request timed out. Try again.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")