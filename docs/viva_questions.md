# Viva Questions & Answers (25)

## 1. What is sentiment analysis?
Finding whether text expresses positive, negative, or neutral opinion.
## 2. What is NLP?
Natural Language Processing - computers understanding human language.
## 3. Why Python?
Rich AI libraries (transformers, torch, streamlit), beginner friendly.
## 4. Why Streamlit?
Fast Python-only web UI; ideal for ML demos.
## 5. What is BERT?
Bidirectional transformer encoder by Google (2018) for language tasks.
## 6. What is RoBERTa?
Robustly-optimized BERT (Meta): better-trained BERT variant, no NSP loss.
## 7. Is RoBERTa the same as BERT?
No - same family, different training recipe and performance.
## 8. What transformer did we use?
cardiffnlp/twitter-roberta-base-sentiment-latest (RoBERTa-based).
## 9. What is a transformer?
Neural network using self-attention to read whole sentences at once.
## 10. What is tokenization?
Splitting text into model-readable pieces (subwords); emojis preserved.
## 11. What is a pretrained model?
Model already trained on huge data; we reuse it for inference.
## 12. Did we train the model?
No - inference only with a pretrained model. No training/fine-tuning.
## 13. How does prediction work?
Tokenize -> model outputs logits -> softmax -> highest-probability label.
## 14. What is confidence?
The winning class probability (e.g. 0.94 = 94%); not a correctness guarantee.
## 15. What is softmax?
Function converting raw scores into probabilities summing to 1.
## 16. How are emojis handled?
Passed to the tokenizer unchanged; model learned emoji meanings from tweets.
## 17. Positive vs negative?
Positive = praise/happiness; negative = criticism/anger/dislike.
## 18. What is neutral?
Factual/no strong opinion, e.g. "The package arrived today."
## 19. Rule-based vs transformer?
Rules use fixed word lists; transformers learn context from data.
## 20. Limitations?
Sarcasm, negation, emoji-only inputs, uncalibrated confidence.
## 21. How to improve?
Bigger/multilingual models, sarcasm data, user feedback, batch analysis.
## 22. If model fails to load?
Friendly error shown; clearly-labeled keyword fallback offered; no fake AI claim.
## 23. How is the model loaded once?
functools.lru_cache in backend + st.cache_resource in UI; eval + inference_mode.
## 24. How are labels mapped?
From model config id2label (0=negative,1=neutral,2=positive), verified not assumed.
## 25. Privacy?
Local inference after download; no storage, no API keys, input never executed.
