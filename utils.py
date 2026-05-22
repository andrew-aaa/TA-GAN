# utils.py
"""
Модуль вспомогательных утилит и математических преобразований для пайплайна TA-GAN.

Содержит биоинформатические функции предобработки, токенизации и декодирования 
аминокислотных последовательностей, методы фиксации стохастической воспроизводимости (seeding), 
а также критически важные дифференцируемые операции над дискретными распределениями 
(Gumbel-Softmax репараметризация) для сквозного обучения генеративно-состязательной сети.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from config import (
    VOCAB,
    MAX_AA_LEN,
    MAX_LEN,
    PAD_IDX,
    PAD_TOKEN,
    BOS_IDX,
    BOS_TOKEN,
    EOS_IDX,
    EOS_TOKEN,
)

# Формирование статических словарей прямого и обратного маппинга токенов
aa_to_idx = {aa: i for i, aa in enumerate(VOCAB)}
idx_to_aa = {i: aa for aa, i in aa_to_idx.items()}


def set_seed(seed: int) -> None:
    """
    Фиксирует генераторы случайных чисел (PRNG) во всех задействованных библиотеках.

    Необходима для обеспечения строгой воспроизводимости результатов экспериментов,
    стабилизации инициализации весов и детерминизма на GPU.

    Args:
        seed (int): Численное значение инициализирующего зерна.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_sequence(seq: str) -> str:
    """
    Выполняет синтаксическую очистку и нормализацию аминокислотной строки.

    Переводит символы в верхний регистр, удаляет пробельные символы и полностью 
    отфильтровывает любые технические токены (BOS, EOS, PAD), оставляя только 
    валидные биологические остатки.

    Args:
        seq (str): Сырая аминокислотная последовательность.

    Returns:
        str: Очищенная биологическая последовательность.
    """
    seq = seq.strip().upper()
    return ''.join(aa for aa in seq if aa in aa_to_idx and aa not in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN))


def encode_sequence(seq: str, max_aa_len: int = MAX_AA_LEN):
    """
    Преобразует строковый белок в индексированные массивы для обучения Трансформера.

    Формирует сдвинутые на один шаг последовательности: `decoder_input` (начинается с BOS) 
    и `target` (заканчивается на EOS) для авторегрессионного обучения с учителем (Teacher Forcing).

    Args:
        seq (str): Очищенная аминокислотная строка.
        max_aa_len (int): Максимально допустимая длина физической цепи.

    Returns:
        tuple[list[int], list[int], int]: Кортеж, содержащий:
            - decoder_input: Список индексов, дополненный PAD-токенами до MAX_LEN.
            - target: Целевой список индексов (shift right), дополненный PAD-токенами до MAX_LEN.
            - aa_length: Реальная физическая длина последовательности без учета технических токенов.
    """
    seq = clean_sequence(seq)[:max_aa_len]
    aa_length = len(seq)
    aa_ids = [aa_to_idx[aa] for aa in seq]

    # Сдвиг последовательностей для классической задачи Seq2Seq Auto-regression
    decoder_input = [BOS_IDX] + aa_ids
    target = aa_ids + [EOS_IDX]

    # Обрезка до жесткого лимита архитектуры, если это необходимо
    decoder_input = decoder_input[:MAX_LEN]
    target = target[:MAX_LEN]

    # Добавление PAD-токенов справа до фиксированной длины кадра (Padding Alignment)
    if len(decoder_input) < MAX_LEN:
        decoder_input += [PAD_IDX] * (MAX_LEN - len(decoder_input))
    if len(target) < MAX_LEN:
        target += [PAD_IDX] * (MAX_LEN - len(target))

    aa_length = min(aa_length, MAX_AA_LEN)
    return decoder_input, target, aa_length


def decode_sequence(indices: Iterable[int]) -> str:
    """
    Декодирует тензор/массив целочисленных индексов обратно в биологическую строку.

    Игнорирует паддинги и стартовые маркеры, останавливает сборку при встрече 
    стоп-кодона (EOS). Используется при валидации и логировании результатов.

    Args:
        indices (Iterable[int]): Итерируемый объект, содержащий ID токенов словаря.

    Returns:
        str: Результирующая аминокислотная последовательность белка.
    """
    out = []
    for idx in indices:
        idx = int(idx)
        if idx in (PAD_IDX, BOS_IDX, EOS_IDX):
            if idx == EOS_IDX:  # Терминальный токен обрывает генерацию
                break
            continue
        aa = idx_to_aa.get(idx)
        if aa is not None and aa not in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN):
            out.append(aa)
    return ''.join(out)


def to_one_hot(seq: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """
    Преобразует дискретные индексы токенов в плотные One-hot векторы.

    Args:
        seq (torch.Tensor): Тензор индексов произвольной размерности.
        vocab_size (int): Полный размер используемого словаря (размерность эмбеддинга).

    Returns:
        torch.Tensor: Вещественный тензор с дополнительной размерностью [..., vocab_size].
    """
    return F.one_hot(seq, num_classes=vocab_size).float()


def gumbel_softmax(logits: torch.Tensor, temperature: float = 1.0, hard: bool = False) -> torch.Tensor:
    """
    Выполняет репараметризацию Gumbel-Softmax над логитами токенов.

    КРИТИЧЕСКАЯ ОПЕРАЦИЯ ДЛЯ GAN: Позволяет аппроксимировать дискретный выбор аминокислоты 
    непрерывным дифференцируемым распределением. Это дает возможность градиентам от Дискриминатора 
    проходить обратно в веса Генератора в процессе состязательного обучения.

    Args:
        logits (torch.Tensor): Сырые выходы модели (предсказания вероятностей токенов).
        temperature (float): Степень сглаживания (чем ближе к 0, тем ближе к One-hot).
        hard (bool): Флаг STE (Straight-Through Estimator). Если True, на прямом проходе 
                     выдается чистый One-hot, а на обратном — пропускаются градиенты мягкого распределения.

    Returns:
        torch.Tensor: Дифференцируемое непрерывное распределение вероятностей по словарю.
    """
    return F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)


def build_valid_mask_from_lengths(lengths: torch.Tensor, seq_len: int) -> torch.Tensor:
    """
    Генерирует бинарную маску валидности позиций на основе векторов длин последовательностей.

    Используется для исключения влияния элементов паддинга при расчете биологических 
    функций потерь или метрик на батче белков переменной длины.

    Args:
        lengths (torch.Tensor): Тензор реальных длин последовательностей в батче [BATCH_SIZE].
        seq_len (int): Максимальная длина временного окна (размер кадра).

    Returns:
        torch.Tensor: Булевая или числовая маска размерности [BATCH_SIZE, seq_len], 
                      где True (или 1) указывает на реальный аминокислотный остаток.
    """
    positions = torch.arange(seq_len, device=lengths.device).unsqueeze(0)
    return positions <= lengths.unsqueeze(1)


def write_metrics_row(csv_path: str, fieldnames: list[str], row: dict) -> None:
    """
    Осуществляет потоковую дозапись строки метрик в CSV-файл логирования.

    Автоматически инициализирует файл и записывает заголовки, если файл не существовал. 
    Используется для мониторинга динамики состязательного обучения на каждой эпохе.

    Args:
        csv_path (str): Путь к результирующему CSV-файлу на диске.
        fieldnames (list[str]): Список имен полей (колонок) в фиксированном порядке.
        row (dict): Словарь значений текущей эпохи вида {имя_метрики: значение}.
    """
    p = Path(csv_path)
    # Проверка необходимости записи шапки (заголовков) файла
    file_exists = p.exists() and p.stat().st_size > 0

    with open(p, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)