"""Бенчмарк: скорость инференса и память на последовательностях 128 и 512 токенов, CPU и GPU.

Методика:
  * вход — ровно N токенов (текст повторяется и обрезается токенизатором модели);
  * один текст за раз (batch = 1);
  * 5 прогревочных прогонов (не считаются), затем 30 замеров; итог — медиана и среднее;
  * для GPU перед остановкой таймера ждём окончания вычислений (synchronize);
  * меряется только модель (forward), без токенизации;
  * память: CPU — прирост RSS процесса при загрузке модели (psutil), GPU — пик видеопамяти.

Запуск:
    python benchmark.py                                   # все модели, CPU + доступный GPU
    python benchmark.py --models sentiment-rubert-tiny-ru --runs 10
    python benchmark.py --devices cpu
Результаты: results/benchmark.csv и results/benchmark.md
"""
import argparse
import csv
import gc
import os
import statistics
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")

import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification  # noqa: E402

from model_runner import available_devices, load_tokenizer  # noqa: E402
from models import MODELS  # noqa: E402

SAMPLE = {
    "ru": "Сегодня наконец-то съездили в новый парк на набережной, и это просто восторг! Дорожки чистые, "
          "много скамеек, есть кафе с нормальными ценами. Единственное — парковка забита уже к десяти утра. ",
    "en": "Finally visited the new riverside park today and honestly it is amazing! The paths are clean, "
          "there are plenty of benches and the cafe prices are reasonable. The only problem is parking. ",
}


def rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**20
    except ImportError:
        return float("nan")


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def make_input(tokenizer, lang: str, seq_len: int, device: str) -> dict:
    text = SAMPLE[lang]
    while len(tokenizer(text, verbose=False)["input_ids"]) < seq_len:
        text += SAMPLE[lang]
    enc = tokenizer(text, truncation=True, max_length=seq_len, return_tensors="pt")
    assert enc["input_ids"].shape[1] == seq_len
    return {k: v.to(device) for k, v in enc.items()}


def bench_model(name: str, device: str, seq_lens, warmup: int, runs: int) -> list:
    spec = MODELS[name]
    cls = AutoModelForTokenClassification if spec["task"] == "ner" else AutoModelForSequenceClassification
    gc.collect()
    rss_before = rss_mb()
    t0 = time.perf_counter()
    tokenizer = load_tokenizer(spec["hf"])
    model = cls.from_pretrained(spec["hf"]).eval().to(device)
    sync(device)
    load_s = time.perf_counter() - t0
    model_ram = rss_mb() - rss_before

    rows = []
    for seq_len in seq_lens:
        inputs = make_input(tokenizer, spec["languages"][0], seq_len, device)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        times = []
        with torch.inference_mode():
            for _ in range(warmup):
                model(**inputs)
            sync(device)
            for _ in range(runs):
                start = time.perf_counter()
                model(**inputs)
                sync(device)
                times.append((time.perf_counter() - start) * 1000)

        gpu_mb = None
        if device == "cuda":
            gpu_mb = torch.cuda.max_memory_allocated() / 2**20
        elif device == "mps":
            gpu_mb = torch.mps.current_allocated_memory() / 2**20

        row = {
            "model": name, "task": spec["task"], "device": device, "tokens": seq_len,
            "median_ms": round(statistics.median(times), 2),
            "mean_ms": round(statistics.mean(times), 2),
            "p95_ms": round(sorted(times)[min(len(times) - 1, int(len(times) * 0.95))], 2),
            "texts_per_s": round(1000 / statistics.median(times), 1),
            "load_s": round(load_s, 2),
            "model_ram_mb": round(model_ram, 0),
            "gpu_mem_mb": round(gpu_mb, 0) if gpu_mb is not None else "",
        }
        print(f"  {name:30s} {device:4s} {seq_len:4d} ток.  медиана {row['median_ms']:8.2f} мс")
        rows.append(row)

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
    return rows


def save(rows: list, out_dir: Path) -> None:
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "benchmark.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    head = ["Модель", "Устройство", "Токенов", "Медиана, мс", "Среднее, мс", "p95, мс", "Текстов/с",
            "Загрузка, с", "RAM модели, МБ", "GPU память, МБ"]
    lines = ["# Результаты бенчмарка", "",
             f"torch {torch.__version__}, устройства: {', '.join(available_devices())}", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        lines.append("| " + " | ".join(str(v) for k, v in r.items() if k != "task") + " |")
    (out_dir / "benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nСохранено: {out_dir / 'benchmark.csv'}, {out_dir / 'benchmark.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Бенчмарк моделей: 128 и 512 токенов, CPU и GPU")
    parser.add_argument("--models", help="через запятую (по умолчанию все из models.py)")
    parser.add_argument("--devices", help="через запятую: cpu,cuda,mps (по умолчанию все доступные)")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    names = args.models.split(",") if args.models else list(MODELS)
    devices = args.devices.split(",") if args.devices else available_devices()
    rows = []
    for name in names:
        for device in devices:
            try:
                rows += bench_model(name, device, [128, 512], args.warmup, args.runs)
            except Exception as e:  # noqa: BLE001
                print(f"  {name} на {device}: ошибка — {e}")
    if rows:
        save(rows, Path(__file__).parent / "results")


if __name__ == "__main__":
    main()
