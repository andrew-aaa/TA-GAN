# precompute_toxin_embeddings.py
"""
Модуль предобработки и статического расчета биологических признаков (эмбеддингов) токсинов.

Скрипт выполняет сквозной препроцессинг парных FASTA-файлов, фильтрацию аномальных 
последовательностей по длине, инференс предобученной языковой модели белка ESM-2 
и сохранение вычисленных векторов на диск. Статический предрасчет эмбеддингов позволяет 
кратно ускорить итерации обучения GAN, устраняя избыточные вычисления на GPU.
"""

from __future__ import annotations

from Bio import SeqIO
import torch
from pathlib import Path
from tqdm import tqdm

from config import (
    TOXIN_FASTA_PATH,
    ANTITOXIN_FASTA_PATH,
    TOXIN_EMBEDDINGS_PATH,
    MAX_AA_LEN,
    DEVICE
)
from utils import clean_sequence
from esm_utils import ToxinESMEncoder


@torch.no_grad()
def main():
    """
    Основная управляющая функция пайплайна предрасчета эмбеддингов.

    Считывает парные FASTA-файлы, выполняет пошаговую экстракцию признаков кодировщиком
    и упаковывает результаты в сериализованный PyTorch-контейнер.

    Raises:
        ValueError: Если количество токсинов и антитоксинов не совпадает.
        ValueError: Если в результате фильтрации не обнаружено ни одной валидной пары.
    """
    print(f"[precompute] Device: {DEVICE}")
    print(f"[precompute] Loading ToxinESMEncoder...")

    # Инициализация кодировщика признаков (внутри применяется ESM-2 с адаптерами LoRA)
    encoder = ToxinESMEncoder().to(DEVICE)
    encoder.eval()  # Принудительный перевод в режим инференса для отключения Dropout слоев

    # Чтение биоинформатических файлов (парные выборки токсин-антитоксин)
    toxins = list(SeqIO.parse(TOXIN_FASTA_PATH, 'fasta'))
    antidotes = list(SeqIO.parse(ANTITOXIN_FASTA_PATH, 'fasta'))

    # ЖЕСТКАЯ ВАЛИДАЦИЯ СТРУКТУРЫ: Индексы в обоих файлах должны строго соответствовать друг другу
    if len(toxins) != len(antidotes):
        raise ValueError(f'[error] Количество токсинов и антитоксинов не совпадает: {len(toxins)} vs {len(antidotes)}')

    sequences = []
    embeddings = []
    skipped = 0

    print(f"[precompute] Начало обработки {len(toxins)} пар...")

    # Итерирование по выборке с автоматическим отслеживанием прогресса (tqdm)
    for i, (t_record, a_record) in enumerate(tqdm(zip(toxins, antidotes), total=len(toxins)), start=1):
        # Очистка последовательностей от технических символов, пробелов и сторонних аминокислотных масок
        toxin_seq = clean_sequence(str(t_record.seq))
        antidote_seq = clean_sequence(str(a_record.seq))

        # Фильтрация пустых или некорректных записей
        if not toxin_seq or not antidote_seq:
            continue
            
        # АРХИТЕКТУРНОЕ ОГРАНИЧЕНИЕ: Если антитоксин превышает лимит MAX_AA_LEN (256), 
        # пара пропускается, так как она не поместится в фиксированную матрицу выравнивания Генератора.
        if len(antidote_seq) > MAX_AA_LEN:
            skipped += 1
            continue

        # Инференс модели белка. Выходной тензор имеет форму [1, Hidden_Size] (обычно 1, 512)
        emb = encoder([toxin_seq])
        
        # ОПТИМИЗАЦИЯ ПАМЯТИ: Тензор эмбеддинга немедленно переносится в оперативную память (.cpu()) 
        # и отсекается батч-размерность (.squeeze(0)). Это предотвращает переполнение VRAM GPU при больших датасетах.
        embeddings.append(emb.cpu().squeeze(0))
        sequences.append(toxin_seq)

        # Промежуточный логинг для контроля стабильности процесса
        if i % 50 == 0 or i == len(toxins):
            print(f"\tОбработано: {i}/{len(toxins)} | Пропущено: {skipped}")

    # Проверка на наличие финальных данных перед записью на диск
    if not sequences:
        raise ValueError("[error] Не найдено ни одной валидной пары.")

    # Подготовка целевой директории для сохранения артефактов данных
    emb_path = Path(TOXIN_EMBEDDINGS_PATH)
    emb_path.parent.mkdir(parents=True, exist_ok=True)

    # Упаковка данных: сырые строки последовательностей и объединенный тензор эмбеддингов.
    # torch.stack преобразует список векторов в монолитный тензор формы [NUM_SAMPLES, HIDDEN_SIZE].
    payload = {
        'sequences': sequences,
        'embeddings': torch.stack(embeddings, dim=0)
    }

    # Сериализация словаря на диск в бинарном формате PyTorch (.pt)
    torch.save(payload, emb_path)
    print(f"[precompute] Успешно сохранено на диск: {emb_path} | Итоговая форма тензора: {payload['embeddings'].shape}")


if __name__ == "__main__":
    main()