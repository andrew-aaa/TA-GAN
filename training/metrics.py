# training/metrics.py
"""
Модуль валидационных метрик для оценки качества генерации аминокислотных последовательностей.

Данный модуль содержит специализированные функции для мониторинга состояния 
генератора во время состязательного обучения (GAN). Метрики позволяют оценивать:
1. Лингвистическое разнообразие и локальную сложность белков (n-gram diversity, repeat ratio).
2. Склонность модели к генерации пустых строк или застреванию в токенах-заглушках (nonempty ratio).
3. Точность предсказания длины молекулы (MAE) встроенным блоком Length Predictor.
"""

from __future__ import annotations

import torch

from config import PAD_IDX, EOS_IDX, AA_START_IDX


def _trim_ids(row: torch.Tensor) -> list[int]:
    """
    Очищает сырой вектор индексов токенов от служебных символов заполнения и разметки.

    Удаляет токены `PAD_IDX`, отсекает последовательность при достижении токена 
    конца строки `EOS_IDX` и оставляет только валидные индексы аминокислот.

    Args:
        row (torch.Tensor): Одномерный тензор (вектор) индексов токенов.

    Returns:
        list[int]: Очищенный список целочисленных индексов аминокислот.
    """
    out = []
    for x in row.tolist():
        if x in (PAD_IDX,):
            continue
        if x == EOS_IDX:
            break
        if x >= AA_START_IDX:
            out.append(int(x))
    return out


def nonempty_ratio(fake_ids: torch.Tensor) -> float:
    """
    Вычисляет долю непустых сгенерированных последовательностей в батче.

    Помогает отслеживать проблему, когда генератор сразу же выдает токен `EOS_IDX`,
    избегая построения структуры антитоксина.

    Args:
        fake_ids (torch.Tensor): Матрица сгенерированных индексов токенов 
            размерностью [BATCH_SIZE, SEQ_LEN].

    Returns:
        float: Значение в диапазоне [0.0, 1.0] — доля содержательных строк.
    """
    vals = [1.0 if len(_trim_ids(row)) > 0 else 0.0 for row in fake_ids]
    return float(sum(vals) / max(1, len(vals)))


def repeat_ratio(fake_ids: torch.Tensor) -> float:
    """
    Определяет среднюю долю непосредственных аминокислотных повторов (1-gram дубликатов).

    Метрика крайне важна для GAN в дискретных пространствах: высокий коэффициент (> 0.3) 
    сигнализирует о деградации модели (коллапсе мод), когда генератор зацикливается 
    на производстве одной и той же аминокислоты (например, "...GGGGGG...").

    Args:
        fake_ids (torch.Tensor): Матрица сгенерированных индексов [BATCH_SIZE, SEQ_LEN].

    Returns:
        float: Средняя частота повторений смежных токенов по всему батчу.
    """
    values = []
    for row in fake_ids:
        seq = _trim_ids(row)
        if len(seq) < 2:
            values.append(0.0)
            continue
        # Подсчет количества шагов, где текущий токен равен предыдущему
        rep = sum(1 for i in range(1, len(seq)) if seq[i] == seq[i - 1]) / (len(seq) - 1)
        values.append(rep)
    return float(sum(values) / max(1, len(values)))


def ngram_diversity(fake_ids: torch.Tensor, n: int = 2) -> float:
    """
    Вычисляет коэффициент уникальности n-грамм (по умолчанию биграмм) внутри последовательностей.

    Измеряет отношение количества уникальных n-грамм к их общему числу. 
    Низкое разнообразие свидетельствует об однообразном внутреннем паттерне генерации.

    Args:
        fake_ids (torch.Tensor): Матрица сгенерированных индексов [BATCH_SIZE, SEQ_LEN].
        n (int): Длина окна n-грамм для анализа (обычно 2 или 3).

    Returns:
        float: Средний коэффициент уникальности n-грамм по батчу [0.0, 1.0].
    """
    scores = []
    for row in fake_ids:
        seq = _trim_ids(row)
        if len(seq) < n:
            scores.append(0.0)
            continue
        # Нарезка списка аминокислот на скользящие n-граммы
        ngrams = [tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)]
        # Отношение количества уникальных n-грамм (set) к общему числу n-грамм в строке
        scores.append(len(set(ngrams)) / len(ngrams))
    return float(sum(scores) / max(1, len(scores)))


def predicted_lengths(fake_ids: torch.Tensor) -> torch.Tensor:
    """
    Вычисляет фактические длины сгенерированных последовательностей до токена EOS.

    Args:
        fake_ids (torch.Tensor): Матрица сгенерированных индексов [BATCH_SIZE, SEQ_LEN].

    Returns:
        torch.Tensor: Тензор вещественных чисел (float32), содержащий 
            длины очищенных последовательностей [BATCH_SIZE].
    """
    lengths = []
    for row in fake_ids:
        length = 0
        for x in row.tolist():
            if x == EOS_IDX:
                break
            if x >= AA_START_IDX:
                length += 1
        lengths.append(length)
    return torch.tensor(lengths, dtype=torch.float32)


def length_mae(fake_ids: torch.Tensor, target_lengths: torch.Tensor) -> float:
    """
    Вычисляет среднюю абсолютную ошибку (MAE) между сгенерированной и целевой длинами.

    Показывает, насколько точно модель научилась интерпретировать таргетные 
    значения длин, поступающие из предсказательного модуля (`length_predictor`).

    Args:
        fake_ids (torch.Tensor): Матрица сгенерированных индексов [BATCH_SIZE, SEQ_LEN].
        target_lengths (torch.Tensor): Тензор истинных (желаемых) длин из датасета [BATCH_SIZE].

    Returns:
        float: Значение ошибки MAE в количестве аминокислотных остатков.
    """
    pred = predicted_lengths(fake_ids).to(target_lengths.device)
    mae = torch.abs(pred - target_lengths.float()).mean()
    return float(mae.item())