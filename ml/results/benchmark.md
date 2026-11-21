# Результаты бенчмарка (9 моделей)

MacBook (Apple Silicon), Python 3.9, torch 2.8.0, transformers 4.57.6. batch = 1, 5 прогревов + 30 замеров, медиана, мс.

| Модель | Задача | Язык | CPU 128 | CPU 512 | GPU 128 | GPU 512 | Веса (GPU), МБ |
|---|---|---|---:|---:|---:|---:|---:|
| ner-bert-ru | NER | RU | 37.4 | 133.6 | 20.3 | 64.2 | 676 |
| ner-wikineural-multi | NER | RU+EN | 38.4 | 139.5 | 21.7 | 67.0 | 677 |
| sentiment-rubert-tiny-ru | Тональность | RU | 2.7 | 11.5 | 4.6 | 7.1 | 46 |
| sentiment-xlmr-multi | Тональность | RU+EN | 40.0 | 136.4 | 18.0 | 60.8 | 1062 |
| sentiment-twitter-roberta-en | Тональность | EN | 40.0 | 137.4 | 20.4 | 64.3 | 476 |
| emotion-rubert-tiny2-cedr-ru | Эмоции | RU | 2.5 | 10.0 | 4.6 | 7.0 | 112 |
| emotion-xlmr-multi | Эмоции | RU+EN | 38.0 | 136.3 | 18.0 | 60.8 | 1061 |
| emotion-mdeberta-multi | Эмоции | RU+EN | 65.6 | 244.3 | 39.8 | 165.6 | 1064 |
| emotion-distilroberta-en | Эмоции | EN | 18.9 | 67.6 | 10.6 | 32.2 | 314 |
