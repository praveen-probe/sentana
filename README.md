# 💬 AI-Based Sentiment Analysis for Text, Tweets & Emojis

Classify any sentence, tweet, or emoji message as **POSITIVE** 🟢,
**NEGATIVE** 🔴, or **NEUTRAL** 🟡 using a pretrained RoBERTa transformer.

> *"Understand the emotion behind every message."*

## Features
- Streamlit web app with color-coded result cards + confidence %
- Pretrained model `cardiffnlp/twitter-roberta-base-sentiment-latest`
  (RoBERTa transformer, inference only — nothing trained from scratch)
- Emoji / hashtag / mention friendly; emojis preserved for the tokenizer
- Input validation (empty, whitespace, max 2000 chars)
- Probability breakdown for all 3 classes
- Clearly-labeled keyword fallback ONLY when the transformer is offline
- Pytest suite runnable without downloading the model

## Tech stack
| Layer | Choice |
|---|---|
| Language | Python 3.12 |
| Frontend | Streamlit |
| Model | cardiffnlp/twitter-roberta-base-sentiment-latest |
| Libraries | transformers, torch (CPU), pytest |

## Installation (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
If activation is blocked: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
then retry activation (process-only, safest option).

## Run
```powershell
streamlit run app.py
```
First launch downloads the model (~500MB, needs internet); later runs are offline.
Open the URL Streamlit prints (usually http://localhost:8501).

## Tests
```powershell
python -m pytest -q
python -m pytest -q -m "not integration"   # unit tests only (no download)
python -m pytest -q -m integration -s      # real-model demo (needs internet)
```

## Troubleshooting
| Problem | Fix |
|---|---|
| `transformers/torch` missing | reinstall requirements |
| Model download fails | check internet, retry; proxy users set HF proxy env |
| Torch install slow | CPU wheel is large (~200MB); be patient |
| Port busy | `streamlit run app.py --server.port 8502` |

## Limitations
- Confidence = softmax probabilities, not calibrated correctness.
- Emoji-only inputs are attempted but less reliable.
- Sarcasm/negation can fool the model; no accuracy % claimed.
- Fallback (if ever shown) is keyword-based and always labeled.

## Project files
`app.py` (UI) · `sentiment_model.py` (model/inference) · `tests/`
· `docs/project_report.md` · `docs/viva_questions.md`
