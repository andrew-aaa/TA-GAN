# models/generator.py
"""
Модуль архитектуры Генератора для условной генеративно-состязательной сети TA-GAN.

Реализует авторегрессионную трансформерную модель (Decoder-only), которая принимает 
на вход биологические эмбеддинги целевого токсина от модели ESM-2, предсказывает 
оптимальную длину цепи антидота и генерирует дифференцируемым или стохастическим 
образом валидные аминокислотные последовательности.
"""

from __future__ import annotations
from typing import List
import torch
import torch.nn as nn

from config import (
    MAX_AA_LEN, MAX_LEN, VOCAB_SIZE, EMBED_DIM,
    NUM_HEADS, NUM_LAYERS, FF_MULT, DROPOUT,
    BOS_IDX, EOS_IDX, PAD_IDX, CONDITION_DIM, LATENT_DIM,
)
from utils import gumbel_softmax
from esm_utils import ToxinESMEncoder


class Generator(nn.Module):
    """
    Класс Генератора на базе модифицированной архитектуры Transformer Decoder.

    Модель решает задачу условного макромолекулярного дизайна (conditional de novo design), 
    транслируя признаки структуры токсина в терапевтическую последовательность антидота.
    Обеспечивает сквозную дифференцируемость по дискретным токенам за счет 
    Gumbel-Softmax репараметризации.
    """

    def __init__(self):
        """
        Инициализация структурных компонентов нейросети, проекционных слоев и предикторов.
        """
        super().__init__()
        # Биологический энкодер на основе предобученной языковой модели белка ESM-2
        self.toxin_encoder = ToxinESMEncoder()

        # Таблица эмбеддингов для генерируемых токенов аминокислот
        self.token_embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        
        # Линейные проекции для согласования размерностей условий внешней среды и шума
        self.condition_proj = nn.Linear(CONDITION_DIM, EMBED_DIM)
        self.noise_proj = nn.Linear(LATENT_DIM, EMBED_DIM)
        
        # Обуславливание по длине и обучаемое позиционное кодирование последовательности
        self.length_embedding = nn.Embedding(MAX_AA_LEN + 1, EMBED_DIM)
        self.pos_embedding = nn.Parameter(torch.randn(1, MAX_LEN, EMBED_DIM) * 0.02)

        # Многослойный перцептрон (MLP) для априорного предсказания длины молекулы антидота
        self.length_predictor = nn.Sequential(
            nn.Linear(CONDITION_DIM, EMBED_DIM * 2),
            nn.LayerNorm(EMBED_DIM * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(EMBED_DIM * 2, MAX_AA_LEN + 1)
        )

        # Конфигурация базового слоя декодера трансформера с Pre-LayerNorm
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=EMBED_DIM * FF_MULT,
            dropout=DROPOUT,
            batch_first=True,
            activation='gelu',
            norm_first=True
        )
        
        # Сборка стека авторегрессионных слоев трансформера
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=NUM_LAYERS)
        self.output_norm = nn.LayerNorm(EMBED_DIM)
        
        # Выходной классификатор для отображения скрытых векторов в логиты словаря аминокислот
        self.token_head = nn.Linear(EMBED_DIM, VOCAB_SIZE)

    def forward_teacher_forcing(self, decoder_input: torch.Tensor, toxin_emb: torch.Tensor, z: torch.Tensor | None = None, target_lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Прямой проход в режиме Teacher Forcing (используется на этапе Pre-training).

        Модель обучается предсказывать следующий токен на основе истинной (подсмотренной) 
        предыстории, что кардинально ускоряет сходимость кросс-энтропийной функции потерь.

        Args:
            decoder_input (torch.Tensor): Матрица индексов токенов [BATCH_SIZE, SEQ_LEN].
            toxin_emb (torch.Tensor): Эмбеддинги управляющих токсинов [BATCH_SIZE, CONDITION_DIM].
            z (torch.Tensor | None): Вектор латентного случайного шума [BATCH_SIZE, LATENT_DIM].
            target_lengths (torch.Tensor | None): Истинные длины целевых белков [BATCH_SIZE].

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Пары тензоров:
                - token_logits (torch.Tensor): Логиты аминокислот [BATCH_SIZE, SEQ_LEN, VOCAB_SIZE].
                - len_logits (torch.Tensor): Логиты предсказания длин [BATCH_SIZE, MAX_AA_LEN + 1].
        """
        bsz, seq_len = decoder_input.size()
        device = decoder_input.device

        # Шаг 1: Расчет логитов для предиктора длины на основе скрытого контекста токсина
        len_logits = self.length_predictor(toxin_emb)
        if target_lengths is None:
            target_lengths = torch.argmax(len_logits, dim=-1)
        target_lengths = target_lengths.clamp(min=0, max=MAX_AA_LEN)

        # Шаг 2: Агрегация всех условий (Контекст + Длина + Позиция + Случайный шум GAN)
        x = self.token_embedding(decoder_input)
        cond_emb = self.condition_proj(toxin_emb).unsqueeze(1)
        len_emb = self.length_embedding(target_lengths).unsqueeze(1)
        
        x = x + cond_emb + len_emb + self.pos_embedding[:, :seq_len, :]

        if z is not None:
            z_emb = self.noise_proj(z).unsqueeze(1)
            x = x + z_emb

        # МАТЕМАТИЧЕСКАЯ ИНФРАСТРУКТУРА: Создание причинно-следственной маски (Causal Mask).
        # Предотвращает "подглядывание" трансформера на будущие токены аминокислот при авторегрессии.
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)

        # Шаг 3: Прогон через трансформер и вычисление распределения вероятностей токенов
        h = self.transformer(x, mask=mask)
        h = self.output_norm(h)
        token_logits = self.token_head(h)

        return token_logits, len_logits

    def _autoregressive_generate(self, toxin_emb: torch.Tensor, z: torch.Tensor | None = None, temperature: float = 1.0, hard: bool = False, differentiable: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Внутренний низкоуровневый конвейер циклической пошаговой генерации молекулы.

        Обеспечивает два режима работы: жесткое дискретное сэмплирование для инференса 
        и непрерывное дифференцируемое приближение (Gumbel-Softmax) для состязательного 
        обучения совместно с Критиком.

        Args:
            toxin_emb (torch.Tensor): Эмбеддинги токсинов [BATCH_SIZE, CONDITION_DIM].
            z (torch.Tensor | None): Вектор латентного случайного шума [BATCH_SIZE, LATENT_DIM].
            temperature (float): Степень стохастичности (энтропии) при генерации.
            hard (bool): Флаг жесткой дискретизации (Argmax/One-hot).
            differentiable (bool): Сохранять ли граф вычислений PyTorch для обратного прохода.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Кортеж выходных тензоров:
                - generated_ids (torch.Tensor): Индексы сгенерированных аминокислот [BATCH_SIZE, MAX_LEN].
                - probs (torch.Tensor): Распределение вероятностей (one-hot/soft) [BATCH_SIZE, MAX_LEN, VOCAB_SIZE].
                - target_lengths (torch.Tensor): Адаптивные длины белков батча [BATCH_SIZE].
        """
        bsz = toxin_emb.size(0)
        device = toxin_emb.device

        # Предсказание целевых границ белков текущего батча
        len_logits = self.length_predictor(toxin_emb)
        target_lengths = torch.argmax(len_logits, dim=-1).clamp(min=0, max=MAX_AA_LEN)

        # Формирование базовых условных представлений, инвариантных к шагу генерации
        cond_emb = self.condition_proj(toxin_emb).unsqueeze(1)
        len_emb = self.length_embedding(target_lengths).unsqueeze(1)
        
        base_cond = cond_emb + len_emb
        if z is not None:
            base_cond = base_cond + self.noise_proj(z).unsqueeze(1)

        # Инициализация генерации стартовым токеном начала последовательности (BOS - Beginning of Sequence)
        current_tokens = torch.full((bsz, 1), BOS_IDX, dtype=torch.long, device=device)
        
        # Подготовка структуры для мягких (дифференцируемых) представлений токенов
        bos_onehot = torch.zeros(bsz, 1, VOCAB_SIZE, device=device)
        bos_onehot[:, :, BOS_IDX] = 1.0
        generated_soft_tokens = [bos_onehot]
        generated_tokens = [current_tokens.squeeze(1)]

        probs_steps = [bos_onehot]

        # ЦИКЛ АВТОРЕГРЕССИИ: Пошаговый синтез первичной структуры белка
        for step in range(1, MAX_LEN):
            seq_len = step
            
            # В зависимости от требований дифференцируемости выбирается эмбеддинг истории токенов
            if differentiable:
                input_soft = torch.cat(generated_soft_tokens, dim=1)
                x = torch.matmul(input_soft, self.token_embedding.weight)
            else:
                input_ids = torch.stack(generated_tokens, dim=1)
                x = self.token_embedding(input_ids)

            # Наложение контекста и динамического позиционного смещения для текущей длины истории
            x = x + base_cond + self.pos_embedding[:, :seq_len, :]

            # Причинно-следственное маскирование для предотвращения утечки информации из будущего
            mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)
            
            h = self.transformer(x, mask=mask)
            h = self.output_norm(h)
            
            # Извлечение логитов исключительно для последнего сгенерированного шага
            next_logits = self.token_head(h[:, -1, :])
            next_logits = torch.nan_to_num(next_logits, nan=0.0, posinf=5.0, neginf=-5.0)

            # РЕПАРАМЕТРИЗАЦИЯ GUMBEL-SOFTMAX: Ключевой узел сквозного дифференцирования дискретных данных
            next_probs = gumbel_softmax(next_logits, temperature=temperature, hard=hard)
            
            probs_steps.append(next_probs.unsqueeze(1))
            generated_soft_tokens.append(next_probs.unsqueeze(1))
            
            # Фиксация жестких индексов аминокислот
            next_tokens = torch.argmax(next_probs, dim=-1)
            generated_tokens.append(next_tokens.detach() if differentiable else next_tokens)

            # ИНЖЕНЕРНЫЙ КРИТЕРИЙ ОСТАНОВА: Если модель превысила целевой порог длины 
            # и все последовательности в батче выдали технические токены EOS или PAD — прекратить цикл.
            if step > target_lengths.max() and ((next_tokens == EOS_IDX).all() or (next_tokens == PAD_IDX).all()):
                break

        # Сборка разрозненных временных шагов в единые монолитные тензоры структуры белков
        generated_ids = torch.stack(generated_tokens, dim=1)
        probs = torch.cat(probs_steps, dim=1)

        # ТЕХНИЧЕСКИЙ ПАДДИНГ: Приведение матриц к фиксированной статической форме MAX_LEN
        if generated_ids.size(1) < MAX_LEN:
            pad_ids = torch.full((bsz, MAX_LEN - generated_ids.size(1)), PAD_IDX, dtype=torch.long, device=device)
            generated_ids = torch.cat([generated_ids, pad_ids], dim=1)
        if probs.size(1) < MAX_LEN:
            pad_probs = torch.zeros(bsz, MAX_LEN - probs.size(1), VOCAB_SIZE, device=device)
            pad_probs[:, :, PAD_IDX] = 1.0
            probs = torch.cat([probs, pad_probs], dim=1)

        return generated_ids, probs, target_lengths

    def generate(self, toxin_seqs: List[str], temperature: float = 1.0, hard: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Публичный метод интерфейса для стохастического инференса (генерации во время валидации/тестирования).

        Args:
            toxin_seqs (List[str]): Список аминокислотных строк целевых токсинов.
            temperature (float): Коэффициент стохастичности сэмплирования.
            hard (bool): Возвращать ли дискретные One-hot/Argmax значения.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Сгенерированные ID, вероятности и длины белков.
        """
        self.eval()
        with torch.no_grad():
            toxin_emb = self.toxin_encoder(toxin_seqs)
            return self._autoregressive_generate(toxin_emb, temperature=temperature, hard=hard, differentiable=False)

    def generate_differentiable(self, toxin_emb: torch.Tensor, z: torch.Tensor, temperature: float = 1.0, hard: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Публичный метод интерфейса для состязательного обучения (Adversarial Training).

        Сохраняет вычислительный граф (градиенты), позволяя ошибке от Дискриминатора 
        беспрепятственно проходить сквозь сгенерированные токены обратно в веса Генератора.

        Args:
            toxin_emb (torch.Tensor): Тензор эмбеддингов токсинов [BATCH_SIZE, CONDITION_DIM].
            z (torch.Tensor): Тензор латентного шума [BATCH_SIZE, LATENT_DIM].
            temperature (float): Температура Gumbel-Softmax сглаживания.
            hard (bool): Использование Straight-Through Estimator (STE) Gumbel-Softmax.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Дифференцируемые наборы ID, вероятностей и длин.
        """
        self.train()
        return self._autoregressive_generate(toxin_emb, z=z, temperature=temperature, hard=hard, differentiable=True)