# training/ema.py
"""
Модуль реализации экспоненциального скользящего среднего (Exponential Moving Average, EMA).

EMA применяется к весам генератора для повышения стабильности состязательного 
обучения (WGAN-GP). Метод аккумулирует сглаженную во времени копию параметров,
что позволяет эффективно подавлять высокочастотные осцилляции градиентов, 
минимизировать риск "взрыва" или коллапса мод (mode collapse) и генерировать 
более консервативные, биологически правдоподобные аминокислотные последовательности 
на этапе инференса.
"""

from __future__ import annotations

import copy
import torch


class EMA:
    """
    Класс для расчета и поддержки теневых (сглаженных) весов нейронной сети.

    В процессе обучения обновляет параметры копии модели по формуле:
    W_shadow = decay * W_shadow + (1 - decay) * W_current
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        """
        Инициализирует теневую копию модели и замораживает вычисление ее градиентов.

        Args:
            model (torch.nn.Module): Исходная модель (генератор), веса которой 
                будут сглаживаться.
            decay (float): Коэффициент затухания (гиперпараметр сглаживания). 
                Обычно выбирается в диапазоне [0.99, 0.9999].
        """
        self.decay = decay
        
        # Корректная обработка моделей, обернутых в контейнеры параллельного обучения (DDP/DP)
        model_to_copy = model.module if hasattr(model, 'module') else model
        self.shadow = copy.deepcopy(model_to_copy)
        
        # Перевод теневой модели в режим валидации и полная заморозка градиентов
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad = False

    def update(self, model: torch.nn.Module) -> None:
        """
        Обновляет теневые параметры на основе текущих весов обучаемой модели.

        Этот метод вызывается в конце каждой итерации (шага оптимизатора) 
        для проведения линейной интерполяции между старыми и новыми весами.

        Args:
            model (torch.nn.Module): Текущая обучаемая модель со свежими градиентами.
        """
        # Извлечение базового модуля на случай использования DistributedDataParallel
        model_for_update = model.module if hasattr(model, 'module') else model
        
        with torch.no_grad():
            shadow_dict = self.shadow.state_dict()
            for name, param in model_for_update.named_parameters():
                if name not in shadow_dict:
                    continue
        
                # Применение формулы экспоненциального сглаживания In-Place
                avg_param = shadow_dict[name]
                new_avg = avg_param.data * self.decay + param.data * (1.0 - self.decay)
                avg_param.data.copy_(new_avg)

    def state_dict(self) -> dict:
        """
        Возвращает состояние (веса) сглаженной теневой модели.

        Используется для сохранения чекпоинтов сглаженного генератора на диск.

        Returns:
            dict: Словарь состояния (`state_dict`) теневой модели.
        """
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Загружает состояние (веса) в теневую модель.

        Применяется при возобновлении процесса обучения или при инференсе 
        из сохраненного ранее EMA-чекпоинта.

        Args:
            state_dict (dict): Словарь параметров модели.
        """
        self.shadow.load_state_dict(state_dict)

    def __call__(self) -> torch.nn.Module:
        """
        Обеспечивает удобный интерфейс вызова объекта класса как функции.

        Позволяет получить прямую ссылку на объект сглаженной модели для 
        проведения валидации или генерации.

        Returns:
            torch.nn.Module: Экземпляр сглаженной нейросети (теневой генератор).
        """
        return self.shadow