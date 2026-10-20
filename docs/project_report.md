# Mini-Project Report: AI-Based Sentiment Analysis

## 1. Project Title
AI-Based Sentiment Analysis for Text, Tweets, and Emojis
## 2. Abstract
Web app (Streamlit + Python) classifying text as POSITIVE / NEGATIVE /
NEUTRAL with confidence scores, using pretrained RoBERTa transformer.
## 3. Introduction
Social media text carries opinions; manual review is slow. NLP sentiment
analysis automates this at scale.
## 4. Problem Statement
Build a beginner-friendly tool accepting sentences, tweets, emojis and
returning sentiment instantly with clear visual feedback.
## 5. Objectives
- Pretrained transformer inference (no training from scratch)
- Emoji/social-media friendly input
- Color-coded UI + confidence + probability breakdown
- Validation, error handling, automated tests, docs
## 6. Existing System
Rule/lexicon systems (e.g. word lists): fast but miss context, negation,
sarcasm, emojis.
## 7. Proposed System
Pretrained RoBERTa classifier via Hugging Face Transformers + PyTorch,
CPU inference, Streamlit UI, cached single model load.
## 8. Hardware Requirements
Student laptop, 4GB+ RAM (8GB recommended), ~1GB disk for model, CPU only.
## 9. Software Requirements
Windows 10/11, Python 3.12, Streamlit, transformers, torch (CPU), pytest.
Internet once for model download; offline afterwards.
## 10. System Architecture
User -> Streamlit UI -> validation -> tokenizer -> RoBERTa model ->
logits -> softmax -> label map -> result card + metrics.
## 11. Technologies Used
Python 3.12, Streamlit, transformers, PyTorch, pytest. See README.
## 12. Transformer-Based Sentiment Analysis (simple English)
- NLP: teaching computers to understand human language.
- Sentiment analysis: deciding if text is positive/negative/neutral.
- Transformer: neural network reading whole sentences at once.
- Attention: model focuses on important words ("not good").
- Tokenization: splitting text into pieces the model understands; emojis kept.
- Pretrained model: already trained on millions of tweets; we reuse it.
- Fine-tuning vs inference: fine-tuning = extra training; we do inference only.
- Softmax: converts raw scores (logits) to probabilities summing to 1.
- Confidence: highest probability; not a correctness guarantee.
## 13. Selected Pretrained Model
`cardiffnlp/twitter-roberta-base-sentiment-latest` - RoBERTa-based (not
original BERT), trained for tweet sentiment. Labels verified from config:
0=negative, 1=neutral, 2=positive. CPU inference, truncation at 512 tokens.
## 14. System Workflow
1. Enter text -> 2. Validate -> 3. Tokenize -> 4. Model forward pass
(eval mode, inference_mode) -> 5. Softmax -> 6. Map labels -> 7. Display.
Empty/too-long/model-failure each show friendly errors.
## 15. Implementation Details
`sentiment_model.py`: validate_text, normalize_label/id2label, build_result,
softmax, _load_model_cached (lru_cache), analyze_sentiment, labeled
keyword_fallback_sentiment. `app.py`: cached load, examples, analyze/clear,
result card, metrics, progress bars.
## 16. User Interface Description
Title AI SENTIMENT ANALYZER, subtitle, intro, example buttons, text area
(placeholder given), Analyze/Clear buttons, result card (green/red/yellow +
icon + label), confidence %, 3 metrics + progress bars, model footer.
## 17. Testing Methodology
Pytest unit tests (no model download) + opt-in integration tests (real
model). See tests/test_sentiment.py. Ran `python -m pytest -q`.
## 18. Actual Test Results
`python -m pytest -q` (20-Oct-2026): **23 passed, 0 failed**
(21 unit + 2 integration). Syntax check via py_compile: clean.
Streamlit app serves HTTP 200 at http://localhost:8501.
Integration demo (real model, CPU):
- "I love this product." -> POSITIVE (98.42%)
- "I hate this product." -> NEGATIVE (93.78%)
- "The package arrived today." -> POSITIVE (70.29%) - note: the
  tweet-trained model leans positive here; illustrative, not a failure.
- emoji-only -> POSITIVE (97.58%)
No accuracy % claimed (no labeled dataset evaluated).
## 19. Advantages
Real transformer AI, fast local CPU inference, emoji friendly, honest
confidence, tested, easy demo.
## 20. Limitations
Confidence uncalibrated; sarcasm/negation errors; emoji-only less reliable;
no accuracy % claimed; first run needs internet (~500MB download).
## 21. Future Enhancements
Batch CSV analysis, charts/history, multilingual model, sarcasm handling,
feedback loop, deployment.
## 22. Conclusion
Objective met: working, tested, documented mini-project ready for demo.
## 23. References
- Model card: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
- Transformers docs: https://huggingface.co/docs/transformers/index
- Streamlit docs: https://docs.streamlit.io/
