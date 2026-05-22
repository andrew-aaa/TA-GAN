# training/bio_losses.py
"""
Модуль вычисления специализированных биологических функций потерь (Bio-Losses).

Этот модуль реализует дифференцируемые аппроксимации физико-химических свойств 
белка (гидрофобность, изоэлектрическая точка, амфипатичность), позволяя интегрировать 
биологические ограничения непосредственно в градиентный спуск генератора через 
непрерывные "мягкие" (soft) распределения аминокислот, полученные с помощью Gumbel-Softmax.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from config import (
    VOCAB,
    BIO_HYDRO_WEIGHT,
    BIO_PI_WEIGHT,
    BIO_AMP_WEIGHT,
)

from utils import build_valid_mask_from_lengths

# Шкала гидрофобности аминокислотных остатков (нормализованная).
# Положительные значения соответствуют гидрофобным свойствам, отрицательные — гидрофильным.
HYDROPHOBICITY = {
    'A': 0.70,
    'C': 0.50,
    'D': -1.00,
    'E': -1.00,
    'F': 1.00,
    'G': 0.00,
    'H': -0.40,
    'I': 1.00,
    'K': -1.00,
    'L': 1.00,
    'M': 0.80,
    'N': -0.60,
    'P': -0.50,
    'Q': -0.70,
    'R': -1.00,
    'S': -0.30,
    'T': -0.20,
    'V': 0.90,
    'W': 0.80,
    'Y': 0.40,
}


def _build_hydro_vector(device) -> torch.Tensor:
    """
    Создает статический вектор гидрофобности, согласованный со словарем VOCAB.

    Векторизация необходима для быстрого вычисления скалярного произведения 
    между вероятностями аминокислот на каждом шаге и их физико-химическими свойствами.

    Args:
        device (torch.device): Устройство (cpu/cuda), на котором будут производиться вычисления.

    Returns:
        torch.Tensor: Тензор значений гидрофобности размера [VOCAB_SIZE].
    """
    vec = torch.zeros(len(VOCAB), device=device)
    for aa, val in HYDROPHOBICITY.items():
        if aa in VOCAB:
            vec[VOCAB.index(aa)] = val
    return vec


def hydrophobicity_loss(fake_onehot: torch.Tensor, target_lengths: torch.Tensor, threshold: float = 0.45, weight: float = 0.05) -> torch.Tensor:
    """
    Вычисляет штраф за выход средней гидрофобности белка за установленный порог.

    Предотвращает генерацию избыточно гидрофобных (склонных к неспецифической агрегации)
    или избыточно гидрофильных последовательностей.

    Args:
        fake_onehot (torch.Tensor): Мягкое распределение токенов [BATCH_SIZE, SEQ_LEN, VOCAB_SIZE].
        target_lengths (torch.Tensor): Целевые длины белков в батче [BATCH_SIZE].
        threshold (float): Пороговое значение гидрофобности.
        weight (float): Коэффициент масштабирования лосса.

    Returns:
        torch.Tensor: Скалярный тензор функции потерь гидрофобности.
    """
    device = fake_onehot.device
    hydro_vec = _build_hydro_vector(device)

    # Вычисление профиля гидрофобности на каждый шаг генерации
    # [BATCH_SIZE, SEQ_LEN]
    aa_hydro = torch.matmul(fake_onehot, hydro_vec)

    # Исключение влияния токенов заполнения (PAD) с помощью бинарной маски
    valid_mask = build_valid_mask_from_lengths(target_lengths, fake_onehot.size(1))
    aa_hydro = aa_hydro * valid_mask.float()

    # Усреднение гидрофобности только по значащим аминокислотам
    sum_hydro = aa_hydro.sum(dim=1)
    denom = valid_mask.float().sum(dim=1).clamp(min=1.0)
    mean_hydro = sum_hydro / denom

    # Штрафуется только превышение порога (использование ReLU)
    penalty = F.relu(mean_hydro - threshold)

    return weight * penalty.mean()


def amphipathicity_loss(fake_onehot: torch.Tensor, target_lengths: torch.Tensor, window_size: int = 7, weight: float = 0.08) -> torch.Tensor:
    """
    Оценивает и максимизирует амфипатичность последовательности с помощью скользящего окна.

    Амфипатичность (чередование гидрофобных и гидрофильных участков) критически важна 
    для формирования вторичной структуры (альфа-спиралей и бета-слоев) антидотов, 
    взаимодействующих с токсинами. Лосс минимизирует дисперсию гидрофобности внутри окна,
    сглаживая экстремальные локальные аномалии.

    Args:
        fake_onehot (torch.Tensor): Мягкое распределение токенов [BATCH_SIZE, SEQ_LEN, VOCAB_SIZE].
        target_lengths (torch.Tensor): Целевые длины белков [BATCH_SIZE].
        window_size (int): Размер окна сканирования первичной структуры.
        weight (float): Коэффициент масштабирования лосса.

    Returns:
        torch.Tensor: Скалярный лосс амфипатичности.
    """
    device = fake_onehot.device
    hydro_vec = _build_hydro_vector(device)
    
    # Проекция распределения аминокислот в скалярные значения гидрофобности
    aa_hydro = torch.matmul(fake_onehot, hydro_vec).unsqueeze(1)  # [BATCH_SIZE, 1, SEQ_LEN]

    # Сверточный фильтр для эффективного вычисления скользящего среднего по батчу
    kernel = torch.ones(1, 1, window_size, device=device) / window_size
    
    # Вычисление локального математического ожидания гидрофобности в окне
    local_mean = F.conv1d(aa_hydro, kernel, padding=window_size // 2)
    
    # Локальная дисперсия (штраф за резкую кластеризацию однородных аминокислот)
    local_var = torch.square(aa_hydro - local_mean)

    valid_mask = build_valid_mask_from_lengths(target_lengths, fake_onehot.size(1)).unsqueeze(1)
    local_var = local_var * valid_mask.float()

    sum_var = local_var.sum(dim=(1, 2))
    denom = valid_mask.float().sum(dim=(1, 2)).clamp(min=1.0)
    mean_var = sum_var / denom

    return weight * mean_var.mean()


def approximate_pi(seq_onehot: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
    """
    Математическая экспресс-аппроксимация изоэлектрической точки (pI) белка.

    Вместо решения классического нелинейного уравнения Хендерсона-Хассельбаха,
    которое недифференцируемо, применяется линеаризованная корреляционная модель:
    pI базово привязан к нейтральной точке (7.0) и смещается пропорционально 
    плотности чистого заряда (net charge) молекулы на единицу длины.

    Args:
        seq_onehot (torch.Tensor): Распределение токенов [BATCH_SIZE, SEQ_LEN, VOCAB_SIZE].
        target_lengths (torch.Tensor): Физические длины белковых цепей [BATCH_SIZE].

    Returns:
        torch.Tensor: Вектор предсказанных значений pI для каждого белка в батче [BATCH_SIZE].
    """
    # Выделение индексов кислых (отрицательно заряженных) и основных (положительно заряженных) остатков
    acidic_idxs = [VOCAB.index(aa) for aa in "DE"]
    basic_idxs = [VOCAB.index(aa) for aa in "KRH"]

    # Суммирование вероятностей присутствия заряженных групп по всей длине цепи
    acidic = seq_onehot[..., acidic_idxs].sum(dim=(1, 2))
    basic = seq_onehot[..., basic_idxs].sum(dim=(1, 2))

    # Вычисление результирующего электростатического заряда
    net_charge = basic - acidic
    valid_len = target_lengths.float().clamp(min=1.0)

    # Эмпирическое уравнение зависимости pH от плотности заряда
    return 7.0 + 6.5 * (net_charge / valid_len)


def pi_balance_loss(fake_onehot: torch.Tensor, target_lengths: torch.Tensor, pi_min: float = 5.2, pi_max: float = 9.8, weight: float = 0.05) -> torch.Tensor:
    """
    Штрафует модель за генерацию белков с экстремальными значениями изоэлектрической точки.

    Удержание pI в физиологическом диапазоне (например, 5.2 - 9.8) гарантирует,
    что сгенерированный антидот останется растворимым в водных средах организма 
    и не выпадет в осадок при нейтральном pH.

    Args:
        fake_onehot (torch.Tensor): Мягкое распределение токенов [BATCH_SIZE, SEQ_LEN, VOCAB_SIZE].
        target_lengths (torch.Tensor): Валидные длины последовательностей [BATCH_SIZE].
        pi_min (float): Нижний допустимый порог pI.
        pi_max (float): Верхний допустимый порог pI.
        weight (float): Весовой коэффициент компоненты потерь в общем функционале.

    Returns:
        torch.Tensor: Скалярный тензор функции потерь pI.
    """
    pi_est = approximate_pi(fake_onehot, target_lengths)

    # Двухсторонний штраф с использованием ReLU (выход за левую или правую границу)
    penalty = (F.relu(pi_min - pi_est) + F.relu(pi_est - pi_max))

    return weight * penalty.mean()


def bio_loss_generator(fake_onehot: torch.Tensor, target_lengths: torch.Tensor, hydro_weight: float | None = None, pi_weight: float | None = None, amp_weight: float | None = None) -> dict[str, torch.Tensor]:
    """
    Агрегатор и диспетчер вычисления биологических функций потерь для генератора.

    Собирает все отдельные физико-химические метрики в единый словарь, 
    подсчитывает суммарную биологическую потерю и обеспечивает интерфейс для 
    динамического изменения весов (например, во время фазы ramp-up состязательного обучения).

    Args:
        fake_onehot (torch.Tensor): Выходные мягкие вероятности генератора [BATCH_SIZE, SEQ_LEN, VOCAB_SIZE].
        target_lengths (torch.Tensor): Предсказанные/истинные длины антидотов [BATCH_SIZE].
        hydro_weight (float | None): Переопределенный вес гидрофобности (если None — берется из config).
        pi_weight (float | None): Переопределенный вес изоэлектрической точки.
        amp_weight (float | None): Переопределенный вес амфипатичности.

    Returns:
        dict[str, torch.Tensor]: Словарь, содержащий детализированные компоненты потерь 
            и итоговую агрегированную потерю по ключу 'bio_total'.
    """
    if hydro_weight is None:
        hydro_weight = BIO_HYDRO_WEIGHT
    if pi_weight is None:
        pi_weight = BIO_PI_WEIGHT
    if amp_weight is None:
        amp_weight = BIO_AMP_WEIGHT

    # Посегментный расчет физико-химических штрафов
    h_loss = hydrophobicity_loss(fake_onehot, target_lengths, weight=hydro_weight)
    p_loss = pi_balance_loss(fake_onehot, target_lengths, weight=pi_weight)
    a_loss = amphipathicity_loss(fake_onehot, target_lengths, weight=amp_weight)

    # Суммирование в единый оптимизируемый критерий
    total_bio = h_loss + p_loss + a_loss

    return {
        "bio_total": total_bio,
        "bio_hydro": h_loss,
        "bio_pi": p_loss,
        "bio_amphipathic": a_loss
    }