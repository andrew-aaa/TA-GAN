# models/discriminator.py
"""
Модуль архитектуры Дискриминатора (Критика) для условной генеративно-состязательной сети TA-GAN.

Реализует трансформерную нейросетевую модель, оценивающую степень биологической 
правдоподобности (степень "реальности") сгенерированных аминокислотных последовательностей 
антидотов с учетом контекста целевого токсина и предсказанной длины белка.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import (
    MAX_AA_LEN, MAX_LEN, VOCAB_SIZE, CONDITION_DIM,
    DISC_EMBED_DIM, DISC_NUM_HEADS, DISC_NUM_LAYERS,
    FF_MULT, DISC_DROPOUT, PROJECTION_DIM,
)
from utils import build_valid_mask_from_lengths


class Discriminator(nn.Module):
    """
    Класс Дискриминатора (Критика) на базе архитектуры Transformer Encoder.

    Выполняет нелинейную проекцию one-hot представлений аминокислот, учитывает 
    позиционное кодирование и вычисляет оценку WGAN (Wasserstein distance) для пары 
    "Токсин + Сгенерированный/Реальный антидот". Поддерживает условную генерацию 
    через конкатенацию признаков длины и эмбеддинга токсина.
    """

    def __init__(self):
        """
        Инициализация внутренних слоев, эмбеддингов и блоков трансформера дискриминатора.
        """
        super().__init__()
        
        # Проекция дискретного/вероятностного пространства аминокислот в непрерывный векторный эмбеддинг
        self.seq_proj = nn.Linear(VOCAB_SIZE, DISC_EMBED_DIM)
        
        # Слой нормализации для стабилизации распределения признаков на входе и выходе трансформера
        self.input_norm = nn.LayerNorm(DISC_EMBED_DIM)
        self.transformer_norm = nn.LayerNorm(DISC_EMBED_DIM)
        
        # Обуславливание по длине: дискретное признаковое пространство длин белков
        self.length_embedding = nn.Embedding(MAX_AA_LEN + 1, DISC_EMBED_DIM)
        
        # Позиционное кодирование для сохранения информации о порядке аминокислот в первичной структуре
        self.pos_embedding = nn.Parameter(torch.randn(1, MAX_LEN, DISC_EMBED_DIM) * 0.02)

        # Конфигурация базового слоя кодировщика (энкодера) трансформера
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=DISC_EMBED_DIM,
            nhead=DISC_NUM_HEADS,
            dim_feedforward=DISC_EMBED_DIM * FF_MULT,
            dropout=DISC_DROPOUT,
            batch_first=True,
            activation='gelu',
            norm_first=True,  # Pre-LayerNorm для повышения стабильности обучения в GAN-архитектурах
        )
        
        # Сборка глубокого трансформера из последовательности базовых слоев
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=DISC_NUM_LAYERS
        )
        
        # Проекционный слой объединенного контекста (эмбеддинг ESM-2 токсина + эмбеддинг длины)
        self.cond_proj = nn.Sequential(
            nn.Linear(CONDITION_DIM + DISC_EMBED_DIM, DISC_EMBED_DIM),
            nn.GELU(),
            nn.Dropout(DISC_DROPOUT),
        )
        
        # Голова классификации / Критика: вычисление скалярной оценки соответствия (WGAN score)
        self.adv_head = nn.Sequential(
            nn.Linear(DISC_EMBED_DIM, PROJECTION_DIM),
            nn.GELU(),
            nn.Dropout(DISC_DROPOUT),
            nn.Linear(PROJECTION_DIM, 1)
        )

    def masked_mean_pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Выполняет усреднение скрытых представлений с учетом маски паддинга (Masked Average Pooling).

        Исключает влияние технических токенов заполнения (PAD) на итоговый векторный
        репрезентант последовательности, гарантируя инвариантность к длине паддинга.

        Args:
            x (torch.Tensor): Выходные тензоры трансформера [BATCH_SIZE, SEQ_LEN, DISC_EMBED_DIM].
            mask (torch.Tensor): Бинарная маска валидных аминокислот [BATCH_SIZE, SEQ_LEN].

        Returns:
            torch.Tensor: Сглаженный вектор признаков белка [BATCH_SIZE, DISC_EMBED_DIM].
        """
        maskf = mask.unsqueeze(-1).float()
        x = x * maskf
        summed = x.sum(dim=1)
        denom = maskf.sum(dim=1).clamp(min=1e-6)  # Защита от деления на ноль при пустых последовательностях
        return summed / denom

    def forward(self, toxin_emb: torch.Tensor, antidote_onehot: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход модели: вычисление состязательной оценки для переданного кандидата.

        Args:
            toxin_emb (torch.Tensor): Предвычисленный эмбеддинг управляющего токсина [BATCH_SIZE, CONDITION_DIM].
            antidote_onehot (torch.Tensor): Распределение или one-hot матрица антидота [BATCH_SIZE, SEQ_LEN, VOCAB_SIZE].
            target_lengths (torch.Tensor): Тензор истинных или предсказанных длин антидотов [BATCH_SIZE].

        Returns:
            torch.Tensor: Скалярная оценка критика для каждого элемента батча [BATCH_SIZE, 1].
        """
        # Инженерный контроль границ: ограничение диапазона длин во избежание выхода за пределы Embedding
        target_lengths = target_lengths.clamp(min=0, max=MAX_AA_LEN)
        
        # Генерация маски внимания: инвертируется далее для корректной работы src_key_padding_mask в PyTorch
        valid_mask = build_valid_mask_from_lengths(target_lengths, antidote_onehot.size(1))

        # СТАБИЛИЗАЦИЯ ГРАДИЕНТОВ: Фильтрация возможных аномалий (NaN/Inf) во внешних эмбеддингах ESM-2
        toxin_emb = torch.nan_to_num(toxin_emb, nan=0.0, posinf=3.0, neginf=-3.0)
        toxin_emb = torch.clamp(toxin_emb, -5.0, 5.0)

        # Этап 1: Подготовка признаков антидота и применение позиционного кодирования
        x = self.seq_proj(antidote_onehot)
        x = self.input_norm(x)
        x = x + self.pos_embedding[:, :antidote_onehot.size(1), :]
        
        # Этап 2: Извлечение контекстных зависимостей внутри последовательности с помощью механизма Self-Attention
        x = self.transformer(x, src_key_padding_mask=~valid_mask)
        x = self.transformer_norm(x)
        
        # Этап 3: Агрегация пространственных признаков белка в единый вектор
        seq_repr = self.masked_mean_pool(x, valid_mask)
        
        # Этап 4: Формирование условного вектора (условия) GAN
        length_repr = self.length_embedding(target_lengths)
        cond = torch.cat([toxin_emb, length_repr], dim=-1)
        cond_repr = self.cond_proj(cond)
        
        # МАТЕМАТИКА СОСТЯЗАТЕЛЬНОЙ ОЦЕНКИ (cGAN / WGAN-GP):
        # Реализуется скалярное произведение (модуляция) признаков сгенерированного антидота (seq_repr)
        # и условий внешней среды (cond_repr), дополненное аддитивной безусловной компонентой.
        unconditional = self.adv_head(seq_repr)
        conditional = (seq_repr * cond_repr).sum(dim=-1, keepdim=True)
        
        # Итоговый лог-коэффициент правдоподобия
        return unconditional + conditional