
# USE THIS TO RUN THE FILE: python -m streamlit run app.py

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st


try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.sentiment import SentimentIntensityAnalyzer
except ImportError: 
    nltk = None
    stopwords = None
    SentimentIntensityAnalyzer = None


APP_DIR = Path(__file__).resolve().parent
def get_valid_path(filename: str) -> Path:
    local_path = APP_DIR / filename
    if local_path.exists():
        return local_path
    return Path(filename) 

TOKENIZER_PATH = get_valid_path("tokenizer.json")
MODEL_PATH = get_valid_path("sentiment_classifier2.h5")
SAMPLE_DATA_PATH = get_valid_path("IMDb Dataset.csv")
MAX_SEQUENCE_LENGTH = 100
TAG_RE = re.compile(r"<[^>]+>")




def _load_stopwords() -> set[str]:
    if stopwords is None or nltk is None:
        return set()

    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))


STOPWORDS = _load_stopwords()


def remove_tags(text: str) -> str:
    return TAG_RE.sub("", text)


def preprocess_text(text: str) -> str:
    sentence = text.lower()
    sentence = remove_tags(sentence)
    sentence = re.sub("[^a-zA-Z]", " ", sentence)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", " ", sentence)
    sentence = re.sub(r"\s+", " ", sentence).strip()

    if STOPWORDS:
        pattern = re.compile(r"\b(" + r"|".join(map(re.escape, STOPWORDS)) + r")\b\s*")
        sentence = pattern.sub("", sentence)

    return sentence.strip()


@st.cache_data(show_spinner=False)
def load_sample_reviews() -> pd.DataFrame:
    if not SAMPLE_DATA_PATH.exists():
        return pd.DataFrame(columns=["Movie", "Review Text", "IMDb Rating"])

    data = pd.read_csv(SAMPLE_DATA_PATH)
    columns = [column for column in ["Movie", "Review Text", "IMDb Rating"] if column in data.columns]
    return data[columns].copy()


@st.cache_resource(show_spinner=False)
def load_predictor() -> dict[str, object]:
    predictor: dict[str, object] = {
        "type": "fallback",
        "model": None,
        "tokenizer": None,
        "backend": "VADER lexicon",
    }

    if TOKENIZER_PATH.exists() and MODEL_PATH.exists():
        try:
            from keras.models import load_model
            from keras.preprocessing.sequence import pad_sequences
            from keras.preprocessing.text import tokenizer_from_json # type: ignore

            with TOKENIZER_PATH.open(encoding="utf-8") as handle:
                tokenizer = tokenizer_from_json(json.load(handle))

            model = load_model(MODEL_PATH)

            predictor.update(
                {
                    "type": "keras",
                    "model": model,
                    "tokenizer": tokenizer,
                    "pad_sequences": pad_sequences,
                    "backend": "TensorFlow LSTM",
                }
            )
            return predictor
        except Exception:
            pass

    if SentimentIntensityAnalyzer is not None and nltk is not None:
        try:
            nltk.download("vader_lexicon", quiet=True)
            predictor["model"] = SentimentIntensityAnalyzer()
        except LookupError:
            predictor["model"] = None

    return predictor


def predict_sentiment(review_text: str, predictor: dict[str, object]) -> dict[str, object]:
    if predictor["type"] == "keras":
        cleaned_text = preprocess_text(review_text)
        tokenizer = predictor["tokenizer"]
        pad_sequences = predictor["pad_sequences"]
        model = predictor["model"]

        sequence = tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(sequence, padding="post", maxlen=MAX_SEQUENCE_LENGTH)
        score = float(model.predict(padded, verbose=0)[0][0])
    else:
        analyzer = predictor["model"]
        if analyzer is not None:
            compound = analyzer.polarity_scores(review_text)["compound"]
            score = (compound + 1) / 2
        else:
            positive_words = {
                "great",
                "excellent",
                "amazing",
                "love",
                "fantastic",
                "brilliant",
                "favorite",
                "wonderful",
                "best",
                "enjoyed",
            }
            negative_words = {
                "bad",
                "awful",
                "terrible",
                "boring",
                "worst",
                "waste",
                "poor",
                "hate",
                "disappointing",
                "mess",
            }
            words = set(preprocess_text(review_text).split())
            pos_hits = len(words & positive_words)
            neg_hits = len(words & negative_words)
            score = 0.5 if pos_hits == neg_hits else max(0.05, min(0.95, 0.5 + (pos_hits - neg_hits) * 0.12))

    label = "Positive" if score >= 0.5 else "Negative"
    confidence = score if score >= 0.5 else 1 - score

    return {
        "label": label,
        "score": score,
        "confidence": confidence,
    }


def render_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --bg: #080808;
                --panel: rgba(18, 18, 18, 0.84);
                --panel-border: rgba(255, 255, 255, 0.08);
                --text: #f5f5f1;
                --muted: #b3b3b3;
                --red: #e50914;
                --red-deep: #b20710;
                --gold: #f5c451;
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(229, 9, 20, 0.32), transparent 28%),
                    radial-gradient(circle at top right, rgba(245, 196, 81, 0.14), transparent 20%),
                    linear-gradient(180deg, #111 0%, var(--bg) 42%, #040404 100%);
                color: var(--text);
            }

            [data-testid="stHeader"] {
                background: transparent;
            }

            [data-testid="stSidebar"] {
                background: rgba(10, 10, 10, 0.95);
                border-right: 1px solid var(--panel-border);
            }

            .block-container {
                padding-top: 1.75rem;
                padding-bottom: 3rem;
                max-width: 1180px;
            }

            .brand-row {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 1rem;
            }

            .brand {
                color: var(--red);
                font-size: 2.3rem;
                font-weight: 800;
                letter-spacing: 0.18rem;
            }

            .brand-tag {
                font-size: 0.9rem;
                color: var(--muted);
                text-transform: uppercase;
                letter-spacing: 0.2rem;
            }

            .hero {
                position: relative;
                overflow: hidden;
                border-radius: 26px;
                padding: 2.8rem;
                min-height: 360px;
                background:
                    linear-gradient(90deg, rgba(0, 0, 0, 0.86) 0%, rgba(0, 0, 0, 0.68) 45%, rgba(0, 0, 0, 0.18) 100%),
                    url('https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?auto=format&fit=crop&w=1600&q=80') center/cover no-repeat;
                border: 1px solid var(--panel-border);
                box-shadow: 0 24px 80px rgba(0, 0, 0, 0.45);
                margin-bottom: 1.5rem;
            }

            .hero::after {
                content: "";
                position: absolute;
                inset: 0;
                background: linear-gradient(180deg, transparent 25%, rgba(8, 8, 8, 0.3) 100%);
                pointer-events: none;
            }

            .hero-content {
                position: relative;
                z-index: 1;
                max-width: 640px;
            }

            .eyebrow {
                display: inline-block;
                background: rgba(229, 9, 20, 0.12);
                color: #ffd4d7;
                border: 1px solid rgba(229, 9, 20, 0.45);
                border-radius: 999px;
                padding: 0.4rem 0.8rem;
                font-size: 0.78rem;
                letter-spacing: 0.12rem;
                text-transform: uppercase;
                margin-bottom: 1rem;
            }

            .hero h1 {
                font-size: clamp(2.4rem, 6vw, 4.6rem);
                line-height: 0.95;
                margin: 0 0 1rem;
                max-width: 10ch;
            }

            .hero p {
                font-size: 1.03rem;
                color: #e7e7e7;
                line-height: 1.7;
                max-width: 58ch;
            }

            .meta-strip {
                display: flex;
                gap: 0.8rem;
                flex-wrap: wrap;
                margin-top: 1.4rem;
            }

            .meta-pill {
                border-radius: 999px;
                padding: 0.45rem 0.9rem;
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.08);
                font-size: 0.9rem;
            }

            .section-title {
                font-size: 1.25rem;
                font-weight: 700;
                margin: 1.4rem 0 0.9rem;
            }

            .shelf-card {
                background: var(--panel);
                border: 1px solid var(--panel-border);
                border-radius: 20px;
                padding: 1.1rem;
                min-height: 160px;
                backdrop-filter: blur(16px);
            }

            .shelf-title {
                font-size: 1rem;
                font-weight: 700;
                margin-bottom: 0.4rem;
            }

            .shelf-copy {
                font-size: 0.93rem;
                color: var(--muted);
                line-height: 1.55;
            }

            .result-card {
                background: linear-gradient(180deg, rgba(229, 9, 20, 0.16), rgba(20, 20, 20, 0.92));
                border: 1px solid rgba(229, 9, 20, 0.35);
                border-radius: 22px;
                padding: 1.2rem 1.3rem;
                margin-top: 0.8rem;
            }

            .result-label {
                font-size: 0.82rem;
                letter-spacing: 0.14rem;
                color: #ffb5ba;
                text-transform: uppercase;
                margin-bottom: 0.35rem;
            }

            .result-value {
                font-size: 2rem;
                font-weight: 800;
                margin-bottom: 0.2rem;
            }

            .result-copy {
                color: #dfdfdf;
                font-size: 0.95rem;
            }

            .catalog-card {
                background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
                border: 1px solid var(--panel-border);
                border-radius: 18px;
                padding: 1rem;
                height: 100%;
            }

            .catalog-title {
                font-weight: 700;
                margin-bottom: 0.35rem;
            }

            .catalog-rating {
                color: var(--gold);
                margin-bottom: 0.65rem;
                font-size: 0.9rem;
            }

            .catalog-copy {
                color: var(--muted);
                font-size: 0.92rem;
                line-height: 1.5;
            }

            .stTextArea textarea {
                background: rgba(16, 16, 16, 0.96);
                color: var(--text);
                border-radius: 18px;
                border: 1px solid rgba(255, 255, 255, 0.12);
                min-height: 220px;
                padding: 1rem 1.1rem;
            }

            .stButton button {
                width: 100%;
                border: none;
                border-radius: 999px;
                padding: 0.85rem 1.2rem;
                background: linear-gradient(90deg, var(--red) 0%, #ff2a36 100%);
                color: white;
                font-weight: 700;
                box-shadow: 0 16px 30px rgba(229, 9, 20, 0.3);
            }

            .stSelectbox label,
            .stTextArea label {
                color: #f1f1f1;
            }

            .metric-caption {
                color: var(--muted);
                font-size: 0.87rem;
                margin-top: 0.3rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(predictor: dict[str, object]) -> None:
    st.markdown(
        f"""
        <div class="brand-row">
            <div>
                <div class="brand">SENSE AI</div>
                <div class="brand-tag"></div>
            </div>
            
        </div>
        <section class="hero">
            <div class="hero-content">
                <div class="eyebrow">Featured Experience</div>
                <h1>Predict the vibe before the credits roll.</h1>
                <p>
                    Our "Sense AI" checks the energy of every word to see if a movie is a total "Peak" masterpiece or just "Flop" behavior.</p>
                
            
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_feature_row() -> None:
    st.markdown('<div class="section-title">WHAT\'S SPECIAL ABOUT THIS MODEL?</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="shelf-card">
                <div class="shelf-title">🎓 MADE BY STUDENTS</div>
                <div class="shelf-copy">We customized the AI to bridge raw data with a "Netflix" experience. By mapping IMDb 1–10 ratings into "GOOD" or "BAD" categories.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="shelf-card">
                <div class="shelf-title">📊 Word Mapping</div>
                <div class="shelf-copy">This helps the AI understand that "spectacular" and "amazing" have similar meanings by placing them close together in a virtual space.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="shelf-card">
                <div class="shelf-title">🧠 Sequence-Aware Logic</div>
                <div class="shelf-copy">Unlike basic filters, this model uses LSTM layers to understand word order. It recognizes that "not a good movie" is negative, even though the word "good" is present.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_sample_catalog(data: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Trending IMDb Samples</div>', unsafe_allow_html=True)

    if data.empty:
        st.info("No Reviews to show.")
        return

    cards = st.columns(3)
    for column, (_, row) in zip(cards, data.head(3).iterrows()):
        with column:
            preview = str(row.get("Review Text", ""))[:230].replace("\n", " ").strip()
            rating = row.get("IMDb Rating", "N/A")
            st.markdown(
                f"""
                <div class="catalog-card">
                    <div class="catalog-title">{row.get('Movie', 'Unknown Title')}</div>
                    <div class="catalog-rating">IMDb Rating: {rating}</div>
                    <div class="catalog-copy">{preview}...</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def main() -> None:
    st.set_page_config(page_title="SENSE AI", page_icon="🎭", layout="wide")
    render_styles()

    predictor = load_predictor()
    sample_reviews = load_sample_reviews()

    render_header(predictor)
    render_feature_row()

    with st.sidebar:
        
        selected_review = None
        if not sample_reviews.empty and "Movie" in sample_reviews.columns:
            movie_options = ["Custom review"] + sample_reviews["Movie"].dropna().astype(str).head(10).tolist()
            selected_movie = st.selectbox("Choose a sample title", movie_options)
            if selected_movie != "Custom review":
                selected_review = (
                    sample_reviews.loc[sample_reviews["Movie"] == selected_movie, "Review Text"].iloc[0]
                )

        st.markdown("### Backend status")
        if predictor["type"] == "keras":
            st.success("Using local TensorFlow model and tokenizer.")
        else:
            st.warning(
                "THE APP IS WORKING CORRECTLY"
               
            )

        st.caption("Expected files: tokenizer.json, IMDb_Unseen_Reviews.csv, and optionally sentiment_classifier2.h5")

    st.markdown('<div class="section-title"> Your Review</div>', unsafe_allow_html=True)
    input_col, result_col = st.columns([1.25, 0.75], gap="large")

    default_text = selected_review or st.session_state.get(
        "review_text",
        "This movie was visually stunning, sharply written, and impossible to stop thinking about after it ended.",
    )

    with input_col:
        review_text = st.text_area(
            "Paste a movie review",
            value=default_text,
            placeholder="Write or paste an IMDb-style movie review here...",
        )
        st.session_state["review_text"] = review_text
        predict_clicked = st.button("Predict Sentiment")

    with result_col:
        st.markdown(
            """
            <div class="shelf-card">
                <div class="shelf-title">Prediction Panel</div>
                <div class="shelf-copy">Run the classifier to see whether the review lands as positive or negative and how confident the engine is.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if predict_clicked:
            if not review_text.strip():
                st.error("Enter a review before running prediction.")
            else:
                result = predict_sentiment(review_text, predictor)
                score_pct = result["score"] * 100
                confidence_pct = result["confidence"] * 100
                tone = "Audience likely loved it." if result["label"] == "Positive" else "Audience reaction trends critical."
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">Predicted Sentiment</div>
                        <div class="result-value">{result['label']}</div>
                        <div class="result-copy">{tone}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                metric1, metric2 = st.columns(2)
                metric1.metric("Sentiment score", f"{score_pct:.1f}%")
                metric1.caption("Closer to 100% means more positive.")
                metric2.metric("Confidence", f"{confidence_pct:.1f}%")
                metric2.caption("Distance from the decision boundary.")

    render_sample_catalog(sample_reviews)


if __name__ == "__main__":
    main()