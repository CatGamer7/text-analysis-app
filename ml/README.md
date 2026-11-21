# NLP Prototype

Прототип для анализа текстов на русском и английском языках:

- **NER** — поиск именованных сущностей (люди, места, организации);
- **Sentiment** — тональность (positive / neutral / negative);
- **Emotion** — эмоции (радость, грусть, гнев и т.д.).

Модель загружается с Hugging Face, принимает текст и возвращает метки с оценками.
Если видеокарта недоступна или на ней произошла ошибка, прототип автоматически переходит на CPU.

## Структура

| Файл | Что внутри |
|---|---|
| `models.py` | Реестр из 9 моделей (4 мультиязычные) и модели по умолчанию для каждой пары «задача + язык» |
| `schema.json` | Формат запроса и ответа + примеры |
| `model_runner.py` | Прототип: загрузка модели, инференс, fallback GPU → CPU, обработка ошибок |
| `benchmark.py` | Замер скорости и памяти на 128 и 512 токенах, CPU и GPU |
| `tests/` | Тесты прототипа (работают без скачивания моделей) |
| `docs/` | Отчёт и лог бенчмарка всех моделей |

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

При первом запуске каждая модель скачивается с Hugging Face (от 50 до 700 МБ), дальше берётся из кэша.

## Запуск

```bash
python model_runner.py                     # демо: NER, тональность, эмоции
python model_runner.py --list              # список моделей
python model_runner.py --task sentiment --language ru --text 'Отличный сервис!'
python model_runner.py --task ner --language en --text 'Elon Musk founded SpaceX' --model ner-wikineural-multi
python model_runner.py --task emotion --language ru --text 'Я так рад!' --device cuda -v   # видно fallback
```

В терминале macOS (zsh) текст с `!` пишите в одинарных кавычках.

## Формат запроса и ответа

Запрос:

```json
{"task": "sentiment", "language": "ru", "text": "Отличный сервис, очень доволен!"}
```

Необязательные поля: `model` (название из `models.py`) и `device` (`auto` / `cpu` / `cuda` / `mps`).

Ответ:

```json
{
  "task": "sentiment", "language": "ru", "model": "sentiment-rubert-tiny-ru",
  "device": "cpu", "fallback": false, "inference_time_ms": 3.1,
  "results": [{"label": "positive", "score": 0.9978}, {"label": "neutral", "score": 0.0019}, {"label": "negative", "score": 0.0003}]
}
```

Для NER `results` — список сущностей: `{"entity": "CITY", "word": "Москве", "start": 20, "end": 26, "score": 0.998}`.

При ошибке: `{"error": {"code": "MODEL_NOT_FOUND", "message": "..."}}`.
Коды: `INVALID_INPUT`, `MODEL_NOT_FOUND`, `MODEL_LOAD_FAILED`, `INFERENCE_FAILED`.

## Модели

| Название | Модель на Hugging Face | Задача | Язык |
|---|---|---|---|
| `ner-bert-ru` * | Gherman/bert-base-NER-Russian | NER | RU |
| `ner-wikineural-multi` * | Babelscape/wikineural-multilingual-ner | NER | RU, EN (мультиязычная) |
| `sentiment-rubert-tiny-ru` * | cointegrated/rubert-tiny-sentiment-balanced | Тональность | RU |
| `sentiment-xlmr-multi` | clapAI/roberta-base-multilingual-sentiment | Тональность | RU, EN (мультиязычная) |
| `sentiment-twitter-roberta-en` * | cardiffnlp/twitter-roberta-base-sentiment-latest | Тональность | EN |
| `emotion-rubert-tiny2-cedr-ru` * | cointegrated/rubert-tiny2-cedr-emotion-detection | Эмоции | RU |
| `emotion-xlmr-multi` | proxy3d/multi-motions-28 | Эмоции | RU, EN (мультиязычная) |
| `emotion-mdeberta-multi` | catgamer1/mdeberta-ru-en-emotion | Эмоции | RU, EN (мультиязычная) |
| `emotion-distilroberta-en` * | j-hartmann/emotion-english-distilroberta-base | Эмоции | EN |

\* модель по умолчанию для своей задачи и языка. `ner-wikineural-multi` — по умолчанию для NER на английском.

Мультиязычные модели обрабатывают и русский, и английский одной моделью; их можно выбрать полем `model` в запросе.

## Fallback GPU → CPU

- Устройства перебираются по порядку: CUDA (NVIDIA) → MPS (Apple) → CPU.
- Запрошенное устройство недоступно → сразу CPU, в ответе `"fallback": true`.
- Модель не запустилась на видеокарте (проверяется пробным прогоном) → CPU.
- Ошибка видеокарты во время обработки (например, нехватка памяти) → модель переносится на CPU, текст обрабатывается заново.

## Бенчмарк

```bash
python benchmark.py                                   # все модели
python benchmark.py --models sentiment-rubert-tiny-ru --runs 10
```

Результаты сохраняются в `results/benchmark.csv` и `results/benchmark.md`.
Готовый отчёт по всем 9 моделям (MacBook, CPU и GPU Apple) — `docs/benchmark_report.pdf`, сводная таблица — `results/benchmark.md`.

## Тесты

```bash
pytest -v
```

Тесты подменяют модель поддельной, поэтому проверяют fallback и обработку ошибок без скачивания весов и без видеокарты.
