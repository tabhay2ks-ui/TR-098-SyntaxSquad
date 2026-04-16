import requests
import streamlit as st
import plotly.graph_objects as go

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="AI Fake Review Detection",
    page_icon="🛡️",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main {
        padding-top: 1.2rem;
    }

    .hero-card {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        padding: 28px;
        border-radius: 18px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.18);
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }

    .hero-subtitle {
        font-size: 1rem;
        opacity: 0.9;
        line-height: 1.5;
    }

    .section-card {
        background: rgba(17, 24, 39, 0.70);
        padding: 20px;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 4px 18px rgba(0,0,0,0.20);
        margin-bottom: 18px;
    }

    .metric-card {
        padding: 18px;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(31, 41, 55, 0.75);
        box-shadow: 0 4px 16px rgba(0,0,0,0.16);
        text-align: center;
        min-height: 110px;
    }

    .metric-label {
        font-size: 0.92rem;
        color: #cbd5e1;
        margin-bottom: 10px;
    }

    .metric-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #f8fafc;
        word-wrap: break-word;
    }

    .status-real {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.45);
        color: #a7f3d0;
        padding: 14px 18px;
        border-radius: 14px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 18px;
    }

    .status-fake {
        background: rgba(239, 68, 68, 0.14);
        border: 1px solid rgba(239, 68, 68, 0.45);
        color: #fecaca;
        padding: 14px 18px;
        border-radius: 14px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 18px;
    }

    .status-invalid {
        background: rgba(245, 158, 11, 0.16);
        border: 1px solid rgba(245, 158, 11, 0.45);
        color: #fde68a;
        padding: 14px 18px;
        border-radius: 14px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 18px;
    }

    .explanation-card {
        background: rgba(30, 41, 59, 0.80);
        padding: 18px;
        border-radius: 14px;
        border-left: 5px solid #3b82f6;
        margin-top: 8px;
        margin-bottom: 18px;
        color: #e5e7eb;
        line-height: 1.6;
    }

    .small-badge {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(99, 102, 241, 0.16);
        color: #c7d2fe;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 8px;
        border: 1px solid rgba(99, 102, 241, 0.22);
    }

    .footer-note {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-top: 12px;
    }

    div[data-testid="stPlotlyChart"] {
        background: rgba(17, 24, 39, 0.55);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def get_status_block(label: str, invalid_input: bool = False) -> str:
    label_lower = str(label).strip().lower()

    if invalid_input:
        return '<div class="status-invalid">⚠️ Invalid / Suspicious Input</div>'

    if label_lower == "fake":
        return '<div class="status-fake">🚨 Predicted as Fake Review</div>'

    return '<div class="status-real">✅ Predicted as Real Review</div>'


def get_risk_level(fake_probability: float, invalid_input: bool = False) -> str:
    if invalid_input:
        return "Invalid Input"
    if fake_probability >= 0.75:
        return "High Risk"
    if fake_probability >= 0.5:
        return "Moderate Risk"
    if fake_probability >= 0.3:
        return "Low Risk"
    return "Very Low Risk"


def grade_color(grade: str) -> str:
    grade = str(grade).upper()
    if grade == "A":
        return "#34d399"
    if grade == "B":
        return "#10b981"
    if grade == "C":
        return "#f59e0b"
    if grade == "D":
        return "#f97316"
    return "#ef4444"


def make_gauge(value, title, color_steps):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"font": {"size": 32, "color": "#f8fafc"}},
        title={"text": title, "font": {"size": 18, "color": "#e5e7eb"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
            "bar": {"color": "#60a5fa"},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": color_steps,
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=60, b=20),
        height=300
    )
    return fig


def make_seller_bar(score):
    fig = go.Figure(go.Bar(
        x=[score],
        y=["Seller Trust"],
        orientation="h",
        text=[f"{score:.2f}"],
        textposition="inside",
        marker=dict(
            color=score,
            colorscale="Blues",
            line=dict(color="#93c5fd", width=1)
        )
    ))
    fig.update_layout(
        title="Seller Trust Score",
        title_font_color="#e5e7eb",
        xaxis=dict(range=[0, 100], color="#cbd5e1"),
        yaxis=dict(color="#cbd5e1"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
        margin=dict(l=20, r=20, t=50, b=20),
        height=260
    )
    return fig


def make_radar(fake_probability, authenticity_score, seller_trust_score):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[
            fake_probability * 100,
            authenticity_score,
            seller_trust_score
        ],
        theta=[
            "Fake Risk",
            "Authenticity",
            "Seller Trust"
        ],
        fill="toself",
        line=dict(color="#60a5fa"),
        fillcolor="rgba(96,165,250,0.25)"
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(color="#cbd5e1")
            ),
            angularaxis=dict(
                tickfont=dict(color="#e5e7eb")
            )
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
        font=dict(color="#e5e7eb"),
        title="Risk Profile"
    )
    return fig


st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">🛡️ AI Fake Review Detection & Seller Trust Scoring</div>
        <div class="hero-subtitle">
            Analyze review authenticity in real time using machine learning, behavioral signals,
            suspicious cluster detection, and seller trust scoring.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

top_left, top_right = st.columns([1.15, 1], gap="large")

with top_left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Review Input")

    with st.form("review_form"):
        review_text = st.text_area(
            "Review Text",
            placeholder="Enter a product review here...",
            height=160
        )

        c1, c2 = st.columns(2)
        with c1:
            rating = st.slider("Rating", min_value=1.0, max_value=5.0, value=5.0, step=1.0)
        with c2:
            verified_purchase = st.selectbox(
                "Verified Purchase",
                options=[1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

        c3, c4, c5 = st.columns(3)
        with c3:
            product_id = st.text_input("Product ID", value="demo_product_1")
        with c4:
            reviewer_id = st.number_input("Reviewer ID", min_value=1, value=12345, step=1)
        with c5:
            seller_id = st.text_input("Seller ID", value="demo_seller_1")

        submitted = st.form_submit_button("Analyze Review", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Quick Test Examples")

    example_choice = st.selectbox(
        "Choose a sample review",
        [
            "Select an example",
            "Real review example",
            "Fake / spam review example",
            "Invalid input example"
        ]
    )

    if example_choice == "Real review example":
        st.code(
            "The quality of this product is really good. It works as expected and feels durable.",
            language=None
        )
    elif example_choice == "Fake / spam review example":
        st.code(
            "BEST PRODUCT EVER!!! BUY NOW!!! AMAZING!!!",
            language=None
        )
    elif example_choice == "Invalid input example":
        st.code(
            "ahsbhibsvhibavb",
            language=None
        )

    st.markdown('</div>', unsafe_allow_html=True)

if submitted:
    if not review_text.strip():
        st.error("Please enter review text before analyzing.")
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
                st.session_state.result = response.json()

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to the API. Make sure FastAPI is running at http://127.0.0.1:8000"
            )
        except requests.exceptions.Timeout:
            st.error("The request timed out. Please try again.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

with top_right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Analysis Results")

    if "result" not in st.session_state:
        st.info("Submit a review to view prediction results.")
    else:
        result = st.session_state.result
        invalid_input = bool(result.get("invalid_input", False))
        fake_probability = float(result.get("fake_probability", 0))
        authenticity_score = float(result.get("authenticity_score", 0))
        seller_trust_score = float(result.get("seller_trust_score", 0))
        seller_grade = result.get("seller_grade", "N/A")

        st.markdown(
            get_status_block(result.get("predicted_label", "Unknown"), invalid_input),
            unsafe_allow_html=True
        )

        row1 = st.columns(3)
        row2 = st.columns(3)

        metrics = [
            ("Prediction", result.get("predicted_label", "N/A")),
            ("Fake Probability", f"{fake_probability:.4f}"),
            ("Authenticity Score", f"{authenticity_score:.2f}"),
            ("Seller Trust Score", f"{seller_trust_score:.2f}"),
            ("Seller Grade", seller_grade),
            ("Risk Level", get_risk_level(fake_probability, invalid_input)),
        ]

        for col, (label, value) in zip(row1 + row2, metrics):
            if label == "Seller Grade":
                value_html = f'<div class="metric-value" style="color:{grade_color(value)};">{value}</div>'
            else:
                value_html = f'<div class="metric-value">{value}</div>'

            with col:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        {value_html}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.markdown(
            f"""
            <span class="small-badge">Product ID: {result.get("product_id", "")}</span>
            <span class="small-badge">Reviewer ID: {result.get("reviewer_id", "")}</span>
            <span class="small-badge">Seller ID: {result.get("seller_id", "")}</span>
            """,
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

if "result" in st.session_state:
    result = st.session_state.result
    invalid_input = bool(result.get("invalid_input", False))
    fake_probability = float(result.get("fake_probability", 0))
    authenticity_score = float(result.get("authenticity_score", 0))
    seller_trust_score = float(result.get("seller_trust_score", 0))

    chart_col1, chart_col2, chart_col3 = st.columns(3, gap="large")

    with chart_col1:
        st.plotly_chart(
            make_gauge(
                fake_probability * 100,
                "Fake Probability %",
                [
                    {"range": [0, 30], "color": "rgba(34,197,94,0.25)"},
                    {"range": [30, 60], "color": "rgba(245,158,11,0.25)"},
                    {"range": [60, 100], "color": "rgba(239,68,68,0.25)"},
                ]
            ),
            use_container_width=True
        )

    with chart_col2:
        st.plotly_chart(
            make_gauge(
                authenticity_score,
                "Authenticity Score",
                [
                    {"range": [0, 40], "color": "rgba(239,68,68,0.25)"},
                    {"range": [40, 70], "color": "rgba(245,158,11,0.25)"},
                    {"range": [70, 100], "color": "rgba(34,197,94,0.25)"},
                ]
            ),
            use_container_width=True
        )

    with chart_col3:
        st.plotly_chart(
            make_seller_bar(seller_trust_score),
            use_container_width=True
        )

    st.plotly_chart(
        make_radar(fake_probability, authenticity_score, seller_trust_score),
        use_container_width=True
    )

    bottom1, bottom2 = st.columns(2, gap="large")

    with bottom1:
        st.subheader("Review Explanation")
        st.markdown(
            f'<div class="explanation-card">{result.get("review_explanation", "No explanation available.")}</div>',
            unsafe_allow_html=True
        )

    with bottom2:
        st.subheader("Seller Trust Explanation")
        st.markdown(
            f'<div class="explanation-card">{result.get("trust_explanation", "No explanation available.")}</div>',
            unsafe_allow_html=True
        )

    with st.expander("Raw API Response"):
        st.json(result)

st.markdown(
    """
    <div class="footer-note">
        This dashboard uses a FastAPI backend with TF-IDF + Logistic Regression,
        behavioral feature engineering, suspicious cluster detection, and seller trust scoring.
    </div>
    """,
    unsafe_allow_html=True
)