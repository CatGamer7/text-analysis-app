"""Реестр моделей прототипа.

Ключ — короткое название модели, которое используется в запросах (поле "model").
Если в запросе модель не указана, берётся модель по умолчанию для пары (задача, язык).
"""

MODELS = {
    # ---------------- NER ----------------
    "ner-bert-ru": {
        "hf": "Gherman/bert-base-NER-Russian",
        "task": "ner", "languages": ["ru"], "license": "MIT",
        "note": "ФИО и адреса (FIRST_NAME, LAST_NAME, CITY, STREET...), организаций не находит",
    },
    "ner-wikineural-multi": {
        "hf": "Babelscape/wikineural-multilingual-ner",
        "task": "ner", "languages": ["ru", "en"], "license": "CC BY-NC-SA 4.0",
        "note": "PER / LOC / ORG / MISC, 9 языков",
    },
    # ------------- Тональность -------------
    "sentiment-rubert-tiny-ru": {
        "hf": "cointegrated/rubert-tiny-sentiment-balanced",
        "task": "sentiment", "languages": ["ru"], "license": "не указана",
        "note": "negative / neutral / positive",
    },
    "sentiment-xlmr-multi": {
        "hf": "clapAI/roberta-base-multilingual-sentiment",
        "task": "sentiment", "languages": ["ru", "en"], "license": "MIT",
        "note": "мультиязычная (16+ языков), XLM-RoBERTa-base, F1 81.8",
    },
    "sentiment-twitter-roberta-en": {
        "hf": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "task": "sentiment", "languages": ["en"], "license": "MIT",
        "note": "negative / neutral / positive, обучена на твитах",
    },
    # ---------------- Эмоции ----------------
    "emotion-rubert-tiny2-cedr-ru": {
        "hf": "cointegrated/rubert-tiny2-cedr-emotion-detection",
        "task": "emotion", "languages": ["ru"], "license": "не указана",
        "multi_label": True,
        "note": "no_emotion / joy / sadness / surprise / fear / anger",
    },
    "emotion-xlmr-multi": {
        "hf": "proxy3d/multi-motions-28",
        "task": "emotion", "languages": ["ru", "en"], "license": "MIT",
        "multi_label": True,
        "note": "мультиязычная RU+EN, XLM-RoBERTa, 27 эмоций + neutral (GoEmotions)",
    },
    "emotion-mdeberta-multi": {
        "hf": "catgamer1/mdeberta-ru-en-emotion",
        "task": "emotion", "languages": ["ru", "en"], "license": "MIT",
        "multi_label": True,
        "note": "мультиязычная RU+EN, mDeBERTa-v3, 6 эмоций Экмана + neutral, macro F1 0.70",
    },
    "emotion-distilroberta-en": {
        "hf": "j-hartmann/emotion-english-distilroberta-base",
        "task": "emotion", "languages": ["en"], "license": "Apache 2.0",
        "note": "anger / disgust / fear / joy / neutral / sadness / surprise",
    },
}

# Модель по умолчанию для каждой пары (задача, язык)
DEFAULTS = {
    ("ner", "ru"): "ner-bert-ru",
    ("ner", "en"): "ner-wikineural-multi",
    ("sentiment", "ru"): "sentiment-rubert-tiny-ru",
    ("sentiment", "en"): "sentiment-twitter-roberta-en",
    ("emotion", "ru"): "emotion-rubert-tiny2-cedr-ru",
    ("emotion", "en"): "emotion-distilroberta-en",
}
