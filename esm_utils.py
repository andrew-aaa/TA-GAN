# esm_utils.py
"""
Модуль извлечения и адаптации биологических признаков на базе больших языковых моделей белков.

Обеспечивает интеграцию предобученной трансформерной модели ESM-2 (Meta AI) в общий 
контур генеративно-состязательной сети TA-GAN. Модуль отвечает за токенизацию, 
вычисление скрытых представлений аминокислотных последовательностей токсинов, 
их корректное усреднение (Mean Pooling) с учетом масок и проекцию в признаковое 
пространство фиксированной размерности.

Для минимизации аппаратных затрат и предотвращения катастрофического забывания 
внедрена технология низкоранговой адаптации весов (LoRA) через интерфейс PEFT.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from transformers import EsmModel, EsmTokenizer
from peft import get_peft_model, LoraConfig, TaskType

from config import (
    DEVICE,
    ESM_MODEL_NAME,
    USE_LORA,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    ESM_OUTPUT_DIM
)


class ToxinESMEncoder(nn.Module):
    """
    Нейросетевой кодировщик признаков токсинов на основе ESM-2 и LoRA-адаптеров.

    Класс инкапсулирует в себе логику предобработки и векторизации белковых 
    последовательностей. Выступает в роли условного контекста (Conditioning), 
    направляющего авторегрессионный декодер Генератора в процессе de novo дизайна.

    Успешно обрабатывает батчи переменной длины за счет внутренней фильтрации 
    технических токенов заполнения (Pad-токенов).
    """

    def __init__(self, 
                 model_name: str | None = None,
                 output_dim: int | None = None,
                 use_lora: bool | None = None,
                 lora_r: int | None = None,
                 lora_alpha: int | None = None):
        """
        Инициализирует токенизатор, базовую модель трансформера и проекционные слои.

        Args:
            model_name (str | None): Идентификатор или локальный путь к весам ESM-2.
            output_dim (int | None): Целевая размерность выходного вектора (Condition Dim).
            use_lora (bool | None): Флаг активации эффективного fine-tuning'а.
            lora_r (int | None): Матричный ранг для адаптеров LoRA.
            lora_alpha (int | None): Коэффициент масштабирования весов LoRA.
        """
        super().__init__()
        
        # Динамическое переопределение конфигурации: приоритет у переданных аргументов,
        # fallback — на глобальные константы проекта.
        self.model_name = model_name or ESM_MODEL_NAME
        self.output_dim = output_dim or ESM_OUTPUT_DIM
        self.use_lora = use_lora if use_lora is not None else USE_LORA
        self.lora_r = lora_r or LORA_R
        self.lora_alpha = lora_alpha or LORA_ALPHA
        
        # Загрузка тяжеловесных весов из хаба Hugging Face или локального кэша
        self.tokenizer = EsmTokenizer.from_pretrained(self.model_name)
        self.model = EsmModel.from_pretrained(self.model_name)
        
        # Настройка Parameter-Efficient Fine-Tuning (PEFT)
        if self.use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=LORA_DROPOUT,
                target_modules=["query", "value", "key"],
                bias="none"
            )
            # Обертка базовой модели в прокси-класс PEFT, замораживающий исходные миллиардные веса
            self.model = get_peft_model(self.model, lora_config)
            print(f"[esm utils] LoRA в действии с ESM-2 ({self.model_name})")
        else:
            print(f"[esm utils] ESM-2 загружена без LoRA: {self.model_name}")
        
        # Проекционная голова (Projection Head)
        # Служит для сжатия/изменения признакового пространства ESM-2 (например, 640 или 1280)
        # к фиксированному внутреннему размеру CONDITION_DIM (512) генератора.
        self.projection = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.GELU(), 
            nn.Dropout(0.1) 
        )

        self.to(DEVICE)

    def forward(self, sequences: list[str]) -> torch.Tensor:
        """
        Выполняет прямой проход кодировщика: токенизацию, эмбеддинг и линейную проекцию.

        Принимает на вход сырые текстовые строки, осуществляет инференс pLMM 
        и возвращает очищенные от влияния паддинга усредненные белковые профили.

        Args:
            sequences (list[str]): Батч аминокислотных последовательностей токсинов.

        Returns:
            torch.Tensor: Плотные векторы условий размерности [BATCH_SIZE, ESM_OUTPUT_DIM].

        Raises:
            ValueError: Если на вход передан пустой список или коллекция нулевой длины.
        """
        if not sequences or len(sequences) == 0:
            raise ValueError("[eroror] Список пустых последовательностей, переданных в ToxinESMEncoder")
            
        # Выполняется токенизация батча с принудительным выравниванием.
        # Ограничение max_length=256 согласуется с глобальным лимитом MAX_AA_LEN в config.py.
        inputs = self.tokenizer(
            sequences, 
            padding=True, 
            truncation=True, 
            max_length=256,
            return_tensors="pt"
        ).to(DEVICE)

        # АРХИТЕКТУРНОЕ ОБОСНОВАНИЕ: Вычисление градиентов базовой pLMM разрешено только в том случае,
        # если модель находится в режиме обучения (train) И активирован механизм LoRA. В режиме freeze 
        # или во время валидации/инференса вычисление градиентов блокируется для экономии VRAM.
        with torch.set_grad_enabled(self.training and self.use_lora):
            outputs = self.model(**inputs)
            
            # Извлечение скрытых состояний последнего слоя модели ([Batch_Size, Seq_Len, Hidden_Size])
            last_hidden_state = outputs.last_hidden_state
            
            # Извлечение маски внимания для корректного исключения паддинг-токенов при агрегации
            attention_mask = inputs.attention_mask  # [Batch_Size, Seq_Len]
            
            # КРИТИЧЕСКИЙ ЭТАП: Mean Pooling с корректной фильтрацией паддингов.
            # МАТЕМАТИЧЕСКОЕ ОБОСНОВАНИЕ: Прямое применение метода .mean() к тензору скрытых состояний 
            # приведет к искажению биологического смысла эмбеддинга, так как в расчет попадут нулевые 
            # векторы Pad-токенов. Производится ручное маскирование: суммируются только значимые 
            # позиции, после чего сумма делится на реальную физическую длину каждого конкретного белка.
            
            # Расширяем маску [B, S] до [B, S, H], чтобы выровнять размерности с тензором скрытых состояний
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(last_hidden_state).float()
            
            # Обнуляем скрытые векторы, относящиеся к паддинг-токенам
            masked_hidden = last_hidden_state * mask_expanded
            
            # Суммируем векторы только реальных аминокислот
            summed = masked_hidden.sum(dim=1)
            
            # Считаем количество реальных аминокислот в каждой строке батча.
            # clamp(min=1e-9) защищает от деления на ноль (Zero Division Variance), если на вход попадет полностью пустая строка.
            counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
            
            # Вычисляем финальный усредненный вектор белка ([Batch_Size, Hidden_Size])
            mean_pooled = summed / counts

        # Финальная адаптация признаков: LayerNorm стабилизирует распределение выходов трансформера, 
        # а нелинейность GELU обеспечивает гладкость латентного пространства перед подачей в GAN.
        # Выходной тензор имеет строго фиксированную форму [Batch_Size, ESM_OUTPUT_DIM]
        return self.projection(mean_pooled)