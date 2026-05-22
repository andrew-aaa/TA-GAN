# training/losses.py
"""
Модуль базовых функций потерь для состязательного и авторегрессионного обучения.

Этот модуль реализует математический аппарат, необходимый для обучения 
генеративно-состязательной сети на дискретных биологических последовательностях.
Включает в себя:
1. Градиентный штраф (Gradient Penalty) для аппроксимации метрики Earth Mover's
   Distance (дистанция Васерштейна) в рамках архитектуры WGAN-GP.
2. Кросс-энтропию с поддержкой сглаживания меток (Label Smoothing) для 
   стабилизации фазы предварительного обучения генератора (Teacher Forcing).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from config import (
    PAD_IDX,
    LABEL_SMOOTHING,
)


def gradient_penalty(
    discriminator,
    toxin_emb: torch.Tensor,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    target_lengths: torch.Tensor,
    device: torch.device | str
) -> torch.Tensor:
    """
    Вычисляет градиентный штраф (Gradient Penalty) для оптимизации дискриминатора.

    Математически штрафует отклонение L2-нормы градиента дискриминатора от 1 
    вдоль случайных линейных интерполяций между распределениями реальных и 
    сгенерированных (мягких) аминокислотных последовательностей. 
    Это обеспечивает выполнение условия 1-Липшицевости: ||∇D(x)|| <= 1, 
    что предотвращает исчезновение или взрыв градиентов, свойственные ванильным GAN.

    Args:
        discriminator (nn.Module): Оценивающая нейросеть (Критик).
        toxin_emb (torch.Tensor): Контекстный эмбеддинг управляющего токсина [BATCH_SIZE, CONDITION_DIM].
        real_samples (torch.Tensor): One-hot тензор реальных аминокислот [BATCH_SIZE, SEQ_LEN, VOCAB_SIZE].
        fake_samples (torch.Tensor): Soft/One-hot тензор сгенерированных аминокислот [BATCH_SIZE, SEQ_LEN, VOCAB_SIZE].
        target_lengths (torch.Tensor): Тензор истинных длин последовательностей в батче [BATCH_SIZE].
        device (torch.device | str): Устройство выполнения вычислений (cpu/cuda).

    Returns:
        torch.Tensor: Скалярный тензор величины градиентного штрафа (GP).
    """
    batch_size = real_samples.size(0)

    # Генерация случайного коэффициента смешивания для каждого элемента в батче
    # [BATCH_SIZE, 1, 1] для корректного вещания (broadcasting) по длине и словарю
    alpha = torch.rand(batch_size, 1, 1, device=device)

    # Линейная интерполяция (смешивание реального пространства признаков и фейкового)
    # [BATCH_SIZE, SEQ_LEN, VOCAB_SIZE]
    interpolated = (alpha * real_samples + (1.0 - alpha) * fake_samples)

    # Включение отслеживания градиентов для интерполированных точек
    # Это необходимо, так как мы будем дифференцировать выход дискриминатора по этому тензору
    interpolated.requires_grad_(True)

    # Прогон интерполированных образцов через дискриминатор при заданном условии токсина
    d_interpolated = discriminator(toxin_emb, interpolated, target_lengths)

    # Вычисление градиентов: d(d_interpolated) / d(interpolated)
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(
            d_interpolated
        ),
        create_graph=True,  # Сохраняем граф вычислений для вычисления градиентов от градиентов на backward
        only_inputs=True,
    )[0]

    # Выпрямление тензора градиентов для каждого подлинного шага
    # [BATCH_SIZE, SEQ_LEN * VOCAB_SIZE]
    gradients = gradients.reshape(gradients.size(0), -1)

    # Вычисление L2-нормы вектора градиентов с добавлением эпсилон (1e-12) против деления на ноль
    gradients_norm = torch.sqrt(gradients.pow(2).sum(dim=1) + 1e-12)

    # Штраф за отклонение нормы от единицы (двухсторонний штраф Васерштейна)
    gp = ((gradients_norm - 1.0) ** 2).mean()

    return gp


def token_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    """
    Вычисляет сглаженную кросс-энтропию (Cross-Entropy Loss) для токенов аминокислот.

    Используется на фазе максимизации правдоподобия (учительского форсирования / Pretraining),
    а также как регуляризатор в состязательном режиме. Игнорирует токены заполнения (PAD_IDX),
    чтобы модель не штрафовалась за выравнивание хвостов последовательностей разной длины.

    Args:
        logits (torch.Tensor): Ненормированные выходы генератора [BATCH_SIZE, SEQ_LEN, VOCAB_SIZE]
            или уже сплющенные до [BATCH_SIZE * SEQ_LEN, VOCAB_SIZE].
        target (torch.Tensor): Истинные целочисленные индексы токенов [BATCH_SIZE, SEQ_LEN]
            или [BATCH_SIZE * SEQ_LEN].

    Returns:
        torch.Tensor: Скалярный тензор усредненных потерь кросс-энтропии.
    """
    # Преобразование 3D тензора логитов в 2D матрицу для совместимости с F.cross_entropy
    # [BATCH_SIZE * SEQ_LEN, VOCAB_SIZE]
    logits = logits.reshape(-1, logits.size(-1))

    # Преобразование целевых индексов в одномерный вектор
    # [BATCH_SIZE * SEQ_LEN]
    target = target.reshape(-1)

    # Вычисление стандартной функции потерь с маскированием PAD_IDX и сглаживанием распределения меток
    loss = F.cross_entropy(
        logits,
        target,
        ignore_index=PAD_IDX,
        label_smoothing=LABEL_SMOOTHING
    )

    return loss