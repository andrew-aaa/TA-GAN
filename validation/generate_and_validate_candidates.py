# validation/generate_and_validate_candidates.py
"""
Модуль массовой генерации, биофизической валидации и ранжирования кандидатов антидотов.

Скрипт выполняет полный конвейер (pipeline) пост-процессинга и оценки качества:
1. Загружает обученную модель генератора и референсные FASTA-файлы.
2. Проводит стохастический инференс при различных температурах (от жесткого детерминированного 
   ArgMax до высокоэнтропийного сэмплирования) для покрытия максимального пространства признаков.
3. Рассчитывает кастомную целевую метрику качества (Candidate Score), штрафующую за 
   структурные аномалии (длинные монотонные повторы, отклонения от целевой длины).
4. Вычисляет независимые биофизические параметры (pI, гидрофобность, амфипатичность) 
   и сохраняет структурированный CSV-отчет для последующего отбора топ-кандидатов.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from tqdm import tqdm

# Динамическое подключение корня проекта в пути поиска модулей
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from config import (
    DEVICE,
    AMINO_ACIDS,
    TOXIN_FASTA_PATH,
    ANTITOXIN_FASTA_PATH,
    GENERATION_TEMPERATURES,
    TOXIN_EMBEDDINGS_PATH,
)

from models.generator import Generator
from utils import decode_sequence

# Множество валидных символов аминокислот для быстрой O(1) проверки корректности строк
VALID_AA = set(AMINO_ACIDS)


def parse_fasta(path: str | Path) -> list[tuple[str, str]]:
    """
    Выполняет синтаксический анализ (парсинг) файлов формата FASTA без сторонних зависимостей.

    Args:
        path (str | Path): Путь к файлу последовательностей.

    Returns:
        list[tuple[str, str]]: Список кортежей вида (идентификатор, аминокислотная строка).
    """
    path = Path(path)
    records: list[tuple[str, str]] = []
    if not path.exists():
        return records
        
    with open(path, "r", encoding="utf-8") as f:
        cur_id = None
        chunks = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if cur_id is not None:
                    # Объединяем накопленные куски строк белка в одну строку верхнего регистра
                    records.append((cur_id, "".join(chunks).upper()))
                # Извлекаем первый токен заголовка в качестве чистого ID
                cur_id = line[1:].split()[0]
                chunks = []
            elif line:
                chunks.append(line)
        if cur_id is not None:
            records.append((cur_id, "".join(chunks).upper()))
    return records


def compute_bio_metrics(seq: str) -> dict[str, float | int]:
    """
    Вычисляет базовые физико-химические дескрипторы для одной текстовой последовательности.

    Определяет эмпирическую гидрофобность (на основе индекса Kyte-Doolittle / Эйзенберга),
    интегральный заряд и простейшую оценку изоэлектрической точки (pI).

    Args:
        seq (str): Очищенная строка аминокислот.

    Returns:
        dict[str, float | int]: Словарь вычисленных био-метрик (hydrophobicity, pI, net_charge).
    """
    if not seq:
        return {"hydrophobicity": 0.0, "pI": 7.0, "net_charge": 0.0}

    # Внутренний гидрофобный словарь (аналогичен заложенному в training/bio_losses.py)
    hydro_map = {
        'A': 0.70, 'C': 0.50, 'D': -1.00, 'E': -1.00, 'F': 1.00,
        'G': 0.00, 'H': -0.40, 'I': 1.00, 'K': -1.00, 'L': 1.00,
        'M': 0.80, 'N': -0.60, 'P': -0.50, 'Q': -0.70, 'R': -1.00,
        'S': -0.30, 'T': -0.20, 'V': 0.90, 'W': 0.80, 'Y': 0.40
    }
    
    # Накопление суммарной гидрофобности с фильтрацией неизвестных символов
    h_sum = sum(hydro_map.get(aa, 0.0) for aa in seq)
    avg_hydro = h_sum / len(seq)

    # Подсчет заряженных функциональных групп
    c = Counter(seq)
    acidic = c['D'] + c['E']        # Отрицательно заряженные (Аспарагиновая, Глутаминовая)
    basic = c['K'] + c['R'] + c['H'] # Положительно заряженные (Лизин, Аргинин, Гистидин)
    net_charge = basic - acidic

    # Линейно-пропорциональная аппроксимация pI относительно нейтральной среды (pH=7)
    est_pi = 7.0 + 6.5 * (net_charge / len(seq))

    return {
        "hydrophobicity": round(avg_hydro, 4),
        "pI": round(est_pi, 2),
        "net_charge": float(net_charge)
    }


def get_candidate_score(seq: str, pred_len: int) -> tuple[float, dict[str, float | int]]:
    """
    Вычисляет комплексный штрафной балл кандидата (чем ниже score, тем качественнее белок).

    Критерий оптимизации (Candidate Score) учитывает:
    1. Длину: Модель жестко штрафуется за генерацию пустых строк или сильное отклонение 
       от таргетной длины, выданной блоком Length Predictor.
    2. Валидность: Обнаружение символов не из стандартного алфавита (B, Z, X) влечет максимальный штраф.
    3. Повторы: Скользящее окно замеряет непрерывный "ран" (run) одинаковых символов. 
       Зацикливание генератора (коллапс мод) пенализируется экспоненциально.

    Args:
        seq (str): Проверяемая аминокислотная строка.
        pred_len (int): Оптимальная длина антидота, предсказанная нейросетью.

    Returns:
        tuple[float, dict[str, float | int]]: 
            - Итоговое значение score (float).
            - Словарь мета-параметров (длина, разнообразие, максимальная серия повторов).
    """
    # 1. Жесткий штраф за пустой или некорректный инференс
    if not seq or len(seq) < 2:
        return 999.0, {"ngram3_diversity": 0.0, "max_run": 0, "len_err": abs(pred_len)}

    # Проверка на алфавитную чистоту
    if not all(aa in VALID_AA for aa in seq):
        return 999.0, {"ngram3_diversity": 0.0, "max_run": 99, "len_err": abs(len(seq) - pred_len)}

    # 2. Анализ локального разнообразия через триграммы (3-grams)
    ngrams = [seq[i:i+3] for i in range(len(seq)-2)]
    div3 = len(set(ngrams)) / len(ngrams) if ngrams else 1.0

    # 3. Алгоритм подсчета максимальной непрерывной серии одинаковых аминокислот
    max_run = 1
    current_run = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i-1]:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 1

    # 4. Вычисление компонент функции штрафа
    len_err = abs(len(seq) - pred_len)
    
    # Слагаемое за отклонение от ожидаемой длины
    length_penalty = float(len_err) * 0.1
    
    # Штраф за низкое разнообразие триграмм (коллапс структуры)
    div_penalty = (1.0 - div3) * 10.0
    
    # Экспоненциальный рост штрафа при монотонных повторах длины более 3х символов
    run_penalty = 0.0
    if max_run > 3:
        run_penalty = float(max_run - 3) ** 2.0

    # Сборка финального взвешенного значения
    total_score = length_penalty + div_penalty + run_penalty

    meta = {
        "ngram3_diversity": round(div3, 4),
        "max_run": max_run,
        "len_err": len_err
    }
    return total_score, meta


def main():
    """
    Главная управляющая функция. Организует парсинг аргументов, загрузку весов, 
    генерацию батчей и формирование финального датафрейма результатов.
    """
    parser = argparse.ArgumentParser(description="TA-GAN Candidate Generation & Validation Pipeline")
    parser.add_argument("--weights", type=str, required=True, help="Путь к файлу чекпоинта генератора (.pt)")
    parser.add_argument("--num_per_toxin", type=int, default=10, help="Количество генерируемых вариантов на 1 токсин")
    parser.add_argument("--out_dir", type=str, default="validation_outputs", help="Директория сохранения CSV отчета")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[init] Загрузка референсных обучающих данных...")
    real_toxins = parse_fasta(TOXIN_FASTA_PATH)
    real_antitoxins = parse_fasta(ANTITOXIN_FASTA_PATH)

    # Формируем кэш обучающего множества антитоксинов, чтобы детектировать плагиат (копирование из трейна)
    antitoxin_train_set = set(seq for _, seq in real_antitoxins)
    print(f"[init] База известных антитоксинов содержит {len(antitoxin_train_set)} уникальных структур.")

    # Инициализация архитектуры генератора и перевод в режим инференса (eval)
    print(f"[init] Инициализация модели генератора на устройстве: {DEVICE}")
    generator = Generator().to(DEVICE)
    checkpoint = torch.load(args.weights, map_location=DEVICE)
    
    # Поддержка извлечения весов как из чистого состояния, так и из словарей EMA/генератора
    if "model_state_dict" in checkpoint:
        generator.load_state_dict(checkpoint["model_state_dict"])
    elif "generator_state_dict" in checkpoint:
        generator.load_state_dict(checkpoint["generator_state_dict"])
    else:
        generator.load_state_dict(checkpoint)
    generator.eval()

    # Загрузка предвычисленных ESM-эмбеддингов токсинов для ускорения инференса
    print(f"[init] Чтение предвычисленных эмбеддингов ESM-2...")
    if not Path(TOXIN_EMBEDDINGS_PATH).exists():
        raise FileNotFoundError(f"Критическая ошибка: Файл эмбеддингов {TOXIN_EMBEDDINGS_PATH} не найден. Запустите precompute_toxin_embeddings.py")
        
    emb_data = torch.load(TOXIN_EMBEDDINGS_PATH, map_location=DEVICE)
    emb_sequences = emb_data["sequences"]
    emb_vectors = emb_data["embeddings"]

    # Построение отображения: Сырая_Строка_Токсина -> Его_Тензорный_Эмбеддинг
    seq_to_emb = {seq: emb_vectors[idx] for idx, seq in enumerate(emb_sequences)}

    results = []

    print(f"\n[generation] Запуск конвейера валидации для {len(real_toxins)} токсинов...")
    # Итерация по всем парам токсинов с индикатором прогресса tqdm
    for t_id, t_seq in tqdm(real_toxins):
        if t_seq not in seq_to_emb:
            continue
            
        # Извлекаем эмбеддинг и добавляем батч-размерность [1, EMBED_DIM]
        toxin_emb = seq_to_emb[t_seq].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            # 1. Шаг: Предсказание идеальной длины антидота встроенной головой Length Predictor
            pred_len_logits = generator.length_predictor(toxin_emb)
            pred_len = int(torch.argmax(pred_len_logits, dim=-1).item())

            candidates = []
            
            # 2. Шаг: Генерация кандидатов при разных температурах для баланса между точностью и новизной
            for temp in GENERATION_TEMPERATURES:
                # Генерируем пачку токенов сразу под целевой объем num_per_toxin
                # Для этого дублируем вектор эмбеддинга токсина по первой размерности
                repeated_emb = toxin_emb.expand(args.num_per_toxin, -1, -1) if toxin_emb.dim() == 3 else toxin_emb.expand(args.num_per_toxin, -1)
                
                ids, _, _ = generator._autoregressive_generate(
                    repeated_emb,
                    temperature=temp,
                    hard=True,
                    differentiable=False
                )

                # Декодируем и оцениваем каждого полученного кандидата в подбатче
                for i in range(args.num_per_toxin):
                    seq = decode_sequence(ids[i])
                    score, meta = get_candidate_score(seq, pred_len)
                    candidates.append({
                        "sequence": seq,
                        "temperature": temp,
                        "candidate_score": score,
                        "meta": meta
                    })
            
            # Сортируем сгенерированный пул по возрастанию штрафа (лучшие — в начале)
            candidates.sort(key=lambda x: x["candidate_score"])
            
            # Оставляем только топ-N кандидатов для данного токсина, вычисляя для них расширенную биофизику
            for rank, cand in enumerate(candidates[:args.num_per_toxin]):
                bio = compute_bio_metrics(cand["sequence"])
                results.append({
                    "toxin_id": t_id,
                    "rank": rank + 1,
                    "sequence": cand["sequence"],
                    "length": len(cand["sequence"]),
                    "pred_len": pred_len,
                    "score": cand["candidate_score"],
                    "temp": cand["temperature"],
                    # Флаг новизны: True, если последовательности нет в обучающей выборке
                    "is_new": cand["sequence"] not in antitoxin_train_set,
                    **bio,
                    **cand["meta"]
                })

    # Запись итоговой сводной таблицы в CSV-формате
    csv_path = out_dir / "candidates_report.csv"
    if results:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[done] Валидационный отчет успешно сохранен в: {csv_path}")


if __name__ == "__main__":
    main()