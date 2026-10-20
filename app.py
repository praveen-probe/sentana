"""AI SENTIMENT ANALYZER - Streamlit interface. Run: streamlit run app.py"""
from __future__ import annotations
import streamlit as st
from sentiment_model import (
    MODEL_ID, analyze_sentiment, keyword_fallback_sentiment, validate_text)

st.set_page_config(page_title="AI Sentiment Analyzer", page_icon="💬",
                   layout="centered")
st.markdown("""
<style>
.block-container { max-width: 760px; padding-top: 2rem; }
.hero h1 { font-size: 2.2rem; margin-bottom: 0; }
.hero p.sub { font-size: 1.1rem; opacity: .75; margin-top: .2rem; }
.result-card { border-radius: 14px; padding: 1.2rem 1.4rem; margin: 1rem 0;
  border: 1px solid rgba(128,128,128,.25); }
.result-pos { background: rgba(34,197,94,.12); border-left: 6px solid #22c55e; }
.result-neg { background: rgba(239,68,68,.12); border-left: 6px solid #ef4444; }
.result-neu { background: rgba(234,179,8,.14); border-left: 6px solid #eab308; }
.result-card h2 { margin: 0 0 .3rem 0; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

RESULT_STYLE = {
    "POSITIVE": ("result-pos", "😊", "Your text expresses positive sentiment."),
    "NEGATIVE": ("result-neg", "😞", "Your text expresses negative sentiment."),
    "NEUTRAL": ("result-neu", "😐", "Your text reads as neutral / factual."),
}


@st.cache_resource(show_spinner="Loading AI model (first run downloads it)...")
def get_model_status() -> str:
    try:
        from sentiment_model import _load_model_cached
        _load_model_cached()
        return "ready"
    except Exception as exc:
        return f"unavailable: {exc}"


def render_result(result: dict, fallback: bool = False) -> None:
    sentiment = result["sentiment"]
    css, icon, explainer = RESULT_STYLE[sentiment]
    conf_pct = result["confidence"] * 100
    st.markdown(
        f'<div class="result-card {css}"><h2>{icon} {sentiment}</h2>'
        f"<p>{explainer}</p><p><b>Model confidence: {conf_pct:.1f}%</b></p></div>",
        unsafe_allow_html=True)
    if fallback:
        st.warning("Showing a clearly-labeled keyword fallback (offline mode) "
                   "- not the transformer. Reconnect + reload for AI predictions.")
    scores = result["scores"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Positive", f'{scores["positive"] * 100:.1f}%')
    c2.metric("Neutral", f'{scores["neutral"] * 100:.1f}%')
    c3.metric("Negative", f'{scores["negative"] * 100:.1f}%')
    st.write("Probability breakdown")
    st.progress(scores["positive"], text=f'Positive - {scores["positive"] * 100:.1f}%')
    st.progress(scores["neutral"], text=f'Neutral - {scores["neutral"] * 100:.1f}%')
    st.progress(scores["negative"], text=f'Negative - {scores["negative"] * 100:.1f}%')
    st.caption("Confidence = model output probabilities, not a guarantee "
               "of correctness, and may not be calibrated.")

# __PART2__
st.markdown('<div class="hero"><h1>💬 AI SENTIMENT ANALYZER</h1>'
            '<p class="sub"><i>Understand the emotion behind every message.</i></p></div>',
            unsafe_allow_html=True)
st.write("Type a sentence, tweet, or emoji-filled message and the AI classifies "
         "it as **POSITIVE** 🟢, **NEGATIVE** 🔴, or **NEUTRAL** 🟡.")

model_status = get_model_status()
if model_status != "ready":
    st.warning("Transformer unavailable this session "
               f"({model_status}). Using labeled keyword fallback.")
    use_fallback = True
else:
    use_fallback = False

def clear_inputs() -> None:
    st.session_state["user_text"] = ""
    st.session_state.pop("last_result", None)


def set_example(text: str) -> None:
    st.session_state["user_text"] = text


EXAMPLES = {
    "I love this product! It is amazing! ❤️": "Positive + emojis",
    "This product is terrible. I hate it.": "Negative",
    "The package arrived today.": "Neutral",
    "😂❤️🔥": "Emoji only",
    "It was good, but it could be better.": "Ambiguous",
}
st.subheader("Try an example")
cols = st.columns(len(EXAMPLES))
for col, (text, label) in zip(cols, EXAMPLES.items()):
    col.button(label, key=f"ex_{label}", on_click=set_example, args=(text,))

user_text = st.text_area("Your message", key="user_text", height=140,
    placeholder="Enter your tweet, sentence, or emojis here...", max_chars=2000)
b1, b2 = st.columns(2)
analyze = b1.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
b2.button("🧹 Clear", use_container_width=True, on_click=clear_inputs)

if analyze:
    cleaned, error = validate_text(user_text or "")
    if error:
        st.error(f"⚠️ {error}")
    else:
        with st.spinner("Analyzing..."):
            try:
                if use_fallback:
                    result = keyword_fallback_sentiment(cleaned)
                else:
                    try:
                        result = analyze_sentiment(cleaned)
                    except RuntimeError as exc:
                        st.warning(f"Model inference failed ({exc}). Using labeled fallback.")
                        result = keyword_fallback_sentiment(cleaned)
                        use_fallback = True
                st.session_state["last_result"] = (result, use_fallback)
            except ValueError as exc:
                st.error(f"⚠️ {exc}")

if "last_result" in st.session_state:
    st.subheader("SENTIMENT RESULT")
    result, was_fallback = st.session_state["last_result"]
    render_result(result, fallback=was_fallback)

st.divider()
st.caption(f"🤖 Powered by `{MODEL_ID}` (RoBERTa transformer, Hugging Face). "
           "Text is processed locally after the one-time model download.")
