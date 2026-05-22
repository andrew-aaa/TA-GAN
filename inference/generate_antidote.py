# inference/generate_antidote.py
"""
Модуль инференса и точечной генерации аминокислотных последовательностей антидотов.

Скрипт обеспечивает загрузку предобученных весов генератора (предпочтительно сглаженных 
с помощью EMA), считывание структуры целевого токсина из FASTA-файла, условную авторегрессионную 
генерацию пула кандидатов при различных стохастических режимах (температурах), их оценку 
и сохранение оптимальной последовательности в выходной FASTA-файл.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Tuple, Dict, Any

# Динамическое добавление корня проекта в PATH для корректного импорта внутренних модулей
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from config import (
    DEVICE,
    GENERATION_TEMPERATURES,
    NUM_GENERATION_ATTEMPTS,
)

# АРХИТЕКТУРНОЕ ОБОСНОВАНИЕ: Приоритет отдается EMA (Exponential Moving Average) чекпоинту,
# так как сглаженные веса генератора обеспечивают более стабильное распределение логитов 
# и минимизируют появление аномальных аминокислотных повторов на этапе инференса.
try:
    from config import EMA_BEST_PATH
except ImportError:
    EMA_BEST_PATH = None

from models.generator import Generator
from utils import decode_sequence

DEFAULT_TOXIN_FASTA = PROJECT_ROOT / "data" / "target_toxin.fasta"
DEFAULT_OUTPUT_FASTA = PROJECT_ROOT / "outputs" / "generated_antidote.fasta"


def read_fasta(path: str | Path) -> List[Tuple[str, str]]:
    """
    Парсер биологических файлов формата FASTA для извлечения целевых последовательностей.

    Args:
        path (str | Path): Путь к файлу FASTA.

    Returns:
        List[Tuple[str, str]]: Список кортежей вида (идентификатор_белка, аминокислотная_строка).

    Raises:
        FileNotFoundError: Если целевой файл отсутствует по указанному пути.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {path}")

    records: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        current_id: str | None = None
        current_seq: List[str] = []
        for line in f:
            line = line.strip()
            if not line: 
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records.append((current_id, "".join(current_seq).upper()))
                current_id = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id is not None:
            records.append((current_id, "".join(current_seq).upper()))
    return records


def candidate_score(seq: str, target_length: int) -> Tuple[float, Dict[str, Any]]:
    """
    Эвристическая функция многокритериальной оценки сгенерированного белка-кандидата.

    Вычисляет штрафные баллы на основе анализа первичной структуры. Модель штрафуется 
    за отклонение от идеальной предсказанной длины и за наличие нереалистичных 
    гомополимерных участков (длинных непрерывных повторов одной аминокислоты).

    Args:
        seq (str): Аминокислотная последовательность сгенерированного антидота.
        target_length (int): Оптимальная длина последовательности, предсказанная нейросетью.

    Returns:
        Tuple[float, Dict[str, Any]]: Кортеж, содержащий:
            - score (float): Итоговый штрафной балл (чем ниже, тем качественнее белок).
            - meta (Dict[str, Any]): Словарь промежуточных биологических метрик для логирования.
    """
    if not seq:
        return 999.0, {"len_diff": 999, "max_run": 999, "unique_ratio": 0.0}

    # ИНЖЕНЕРНЫЙ КОНТРОЛЬ: Расчет максимального непрерывного подряда одинаковых аминокислот.
    # Длинные повторы (например, ...AAAAA...) биологически нефункциональны для антидотов.
    max_run = 1
    current_run = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i-1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1

    len_diff = abs(len(seq) - target_length)
    unique_aa = len(set(seq))
    unique_ratio = unique_aa / len(seq)

    # МАТЕМАТИЧЕСКАЯ ФОРМУЛА ШТРАФА: Линейная комбинация структурных ограничений
    score = (len_diff * 1.5) + (max_run * 2.0) - (unique_ratio * 10.0)
    
    meta = {
        "len_diff": len_diff,
        "max_run": max_run,
        "unique_ratio": unique_ratio
    }
    return score, meta


def main() -> None:
    """
    Основной конвейер инференса: инициализация модели, авторегрессионное сэмплирование 
    и селекция лучшего физико-химического кандидата.
    """
    parser = argparse.ArgumentParser(description="TA-GAN single antidote generation script")
    parser.add_argument("--fasta", type=str, default=str(DEFAULT_TOXIN_FASTA), help="Path to input toxin FASTA")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_FASTA), help="Path to output antidote FASTA")
    args = parser.parse_args()

    print(f"[inference] Инициализация целевого устройства: {DEVICE}")
    
    # Считывание целевого биологического контекста (токсина)
    try:
        records = read_fasta(args.fasta)
    except Exception as e:
        print(f"[error] Ошибка чтения входного FASTA: {e}")
        return

    if not records:
        print(f"[error] Входной файл FASTA пуст или поврежден: {args.fasta}")
        return

    toxin_id, toxin_seq = records[0]
    print(f"[inference] Целевой токсин: {toxin_id} | Длина: {len(toxin_seq)} AA")

    # Инициализация генератора и загрузка наиболее стабильного чекпоинта весов
    generator = Generator().to(DEVICE)
    
    ckpt_path = EMA_BEST_PATH or (PROJECT_ROOT / "checkpoints" / "generator_best.pt")
    if Path(ckpt_path).exists():
        print(f"[inference] Загрузка сохраненных весов модели: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        if "model_state_dict" in checkpoint:
            generator.load_state_dict(checkpoint["model_state_dict"])
        else:
            generator.load_state_dict(checkpoint)
    else:
        print(f"[warn] Предупреждение: Чекпоинт {ckpt_path} не найден. Используются случайные веса!")

    generator.eval()

    # НАУЧНОЕ ОБОСНОВАНИЕ: Предвычисление скрытого условного вектора (контекста) токсина.
    # Код использует энкодер ESM-2 для проецирования первичной структуры токсина в биологическое пространство признаков.
    with torch.no_grad():
        toxin_emb = generator.toxin_encoder([toxin_seq])
        
        # Интегрированный в генератор перцептрон предсказывает адаптивную длину будущего антидота
        len_logits = generator.length_predictor(toxin_emb)
        pred_len_idx = torch.argmax(len_logits, dim=-1).item()
        
    print(f"[inference] Модель предсказала оптимальную длину антидота: {pred_len_idx} аминокислотных остатков")

    best_overall_score = float("inf")
    best_overall_seq = ""
    best_temp = 1.0
    best_meta: Dict[str, Any] = {}
    
    target_length = pred_len_idx
    final_target_len = target_length

    # КОНВЕЙЕР МНОГОЭТАПНОГО СЭМПЛИРОВАНИЯ:
    # Перебор различных температур позволяет управлять энтропией распределения вероятностей.
    # Низкие температуры (0.4) генерируют консервативные, высоковероятные последовательности,
    # высокие (1.0) — расширяют биологическое разнообразие структуры (Diversity).
    with torch.no_grad():
        for temp in GENERATION_TEMPERATURES:
            print(f"\n[inference] Генерация с температурой {temp}...")
            
            for attempt in range(NUM_GENERATION_ATTEMPTS):
                # Вызов низкоуровневой авторегрессионной генерации токенов (без вычисления градиентов)
                ids, _, _ = generator._autoregressive_generate(
                    toxin_emb, 
                    temperature=temp, 
                    hard=True, 
                    differentiable=False
                )
                
                # Детокенизация: перевод целочисленных индексов тензора PyTorch обратно в строку аминокислот
                seq = decode_sequence(ids[0])
                
                # Валидация текущей попытки по комплексной эвристической шкале
                score, meta = candidate_score(seq, target_length)

                # Сохранение глобально лучшего кандидата
                if score < best_overall_score:
                    best_overall_score = score
                    best_overall_seq = seq
                    best_temp = temp
                    best_meta = meta

                print(f"  Попытка {attempt+1}/{NUM_GENERATION_ATTEMPTS} | Score: {score:.2f} | Len: {len(seq)}", end="\r")
            print()

    # Формирование и запись результатов в формате FASTA, соответствующем международным биоинформатическим стандартам
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f">generated_antidote_for_{toxin_id}_target_len_{final_target_len}\n")
        # Форматирование строки: разбиение длинной последовательности на блоки по 80 символов для читаемости в Clustal/BLAST
        for i in range(0, len(best_overall_seq), 80):
            f.write(best_overall_seq[i:i+80] + "\n")

    print(f"\n[done] Результат успешно сохранен в: {output_path}")
    print(f"\tИтоговая аминокислотная последовательность: {best_overall_seq}")
    print(f"\tДлина белка: {len(best_overall_seq)} AA")
    print(f"\tВыбранный стохастический режим (температура): {best_temp}")
    print(f"\tМетрики селекции: {best_meta}")


if __name__ == "__main__":
    main()