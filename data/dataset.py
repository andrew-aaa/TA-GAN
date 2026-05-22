# data/dataset.py
"""
Модуль подготовки данных и формирования батчей для модели TA-GAN.

Обеспечивает парсинг биологических FASTA-файлов, их валидацию, фильтрацию по длине,
а также сопоставление последовательностей токсинов с их предвычисленными 
высокоуровневыми эмбеддингами из языковой модели ESM-2.
"""

from __future__ import annotations

from Bio import SeqIO
import torch
from torch.utils.data import Dataset
from pathlib import Path

from config import MAX_AA_LEN
from utils import encode_sequence, clean_sequence


class ToxinAntitoxinDataset(Dataset):
    """
    Класс-датасет для итерации по парам аминокислотных последовательностей 'Токсин-Антитоксин'.

    Выполняет предварительную обработку данных (очистку от недопустимых символов),
    контролирует ограничение на максимальную длину последовательностей и осуществляет
    мэппинг сырых строк в предвычисленные тензорные представления ESM-2 для оптимизации 
    вычислительного графа во время обучения.
    """

    def __init__(self, toxin_fasta: str, antidote_fasta: str, toxin_embeddings_path: str = None):
        """
        Инициализация датасета и валидация структуры входных биологических данных.

        Args:
            toxin_fasta (str): Путь к файлу FASTA, содержащему последовательности токсинов.
            antidote_fasta (str): Путь к файлу FASTA, содержащему последовательности антитоксинов.
            toxin_embeddings_path (str, optional): Путь к предвычисленным эмбеддингам ESM-2 (*.pt).
                Если передан, данные будут отфильтрованы в строгом соответствии с индексами эмбеддингов.

        Raises:
            ValueError: Если количество токсинов и антитоксинов в исходных файлах не совпадает.
        """
        # Считывание сырых данных с использованием библиотеки Biopython
        toxins = list(SeqIO.parse(toxin_fasta, 'fasta'))
        antidotes = list(SeqIO.parse(antidote_fasta, 'fasta'))

        # КРИТИЧЕСКАЯ ВАЛИДАЦИЯ: Нарушение парности делает невозможным условное обучение (cGAN)
        if len(toxins) != len(antidotes):
            raise ValueError(f'[error] Число токсинов и антитоксинов не совпадает: {len(toxins)} vs {len(antidotes)}')

        self.toxin_seqs = []
        self.antitoxin_seqs = []
        skipped_too_long = 0

        # Фильтрация и очистка данных на этапе загрузки в оперативную память
        for t, a in zip(toxins, antidotes):
            toxin_seq = clean_sequence(str(t.seq))
            antitoxin_seq = clean_sequence(str(a.seq))
            
            # Игнорируем пустые или поврежденные записи после чистки
            if not toxin_seq or not antitoxin_seq:
                continue
                
            # НАУЧНОЕ ОГРАНИЧЕНИЕ: Слишком длинные белки вызывают квадратичный рост памяти в Transformer Attention
            if len(antitoxin_seq) > MAX_AA_LEN:
                skipped_too_long += 1
                continue
                
            self.toxin_seqs.append(toxin_seq)
            self.antitoxin_seqs.append(antitoxin_seq)

        self.toxin_embeddings = None

        # Интеграция предвычисленных контекстных представлений (Обусловливание)
        if toxin_embeddings_path:
            try:
                emb_path = Path(toxin_embeddings_path)
                if emb_path.exists():
                    data = torch.load(emb_path, map_location='cpu')
                    if 'sequences' in data and 'embeddings' in data:
                        emb_seqs = data['sequences']
                        embs = data['embeddings']
                        
                        # Перестроение выборок: гарантируем точное соответствие последовательности её эмбеддингу
                        filtered_toxin = []
                        filtered_anti = []
                        indices = []
                        
                        for t_s, a_s in zip(self.toxin_seqs, self.antitoxin_seqs):
                            if t_s in emb_seqs:
                                idx = emb_seqs.index(t_s)
                                filtered_toxin.append(t_s)
                                filtered_anti.append(a_s)
                                indices.append(idx)
                        
                        # Фиксация эмбеддингов в тензоре для мгновенного доступа в __getitem__
                        self.toxin_embeddings = embs[indices]
                        self.toxin_seqs = filtered_toxin
                        self.antitoxin_seqs = filtered_anti
                        print(f"[dataset] Загружено {len(embs)} предвычисленных эмбеддингов")
                    else:
                        print("[error] Структура файла эмбеддингов некорректна (ожидаются 'sequences' и 'embeddings')")
                else:
                    print(f"[error] Файл эмбеддингов не найден: {emb_path}")
            except Exception as e:
                print(f"[error] Не удалось загрузить предвычисленные эмбеддинги: {e}")

        # Логирование процесса сборки датасета для контроля качества фильтрации
        print(f'[dataset] Пропущено слишком длинных антитоксинов: {skipped_too_long} (лимит {MAX_AA_LEN})')
        print(f'[dataset] Валидных пар: {len(self.toxin_seqs)}')

    def __len__(self) -> int:
        """Возвращает общее количество валидных биологических пар в датасете."""
        return len(self.toxin_seqs)

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Формирует один тренировочный кортеж данных по индексу.

        Выполняет токенизацию текстовой аминокислотной строки антитоксина, добавляет 
        служебные токены начала (BOS) и конца (EOS) генерации.

        Args:
            idx (int): Индекс элемента в выборке.

        Returns:
            tuple: Компоненты батча:
                - toxin_seq (str): Исходная строка аминокислот токсина.
                - decoder_input (Tensor): Токенизированный вход декодера (с BOS токеном, смещен влево).
                - target (Tensor): Целевые индексы для подсчета кросс-энтропии (с EOS токеном).
                - aa_length (Tensor): Истинная длина белка без учета технического паддинга.
        """
        toxin_seq = self.toxin_seqs[idx]
        decoder_input, target, aa_length = encode_sequence(self.antitoxin_seqs[idx])

        return (
            toxin_seq,
            torch.tensor(decoder_input, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
            torch.tensor(aa_length, dtype=torch.long),
        )

    def collate_fn(self, batch: list) -> dict:
        """
        Объединяет отдельные элементы данных в тензорные батчи (Batching) фиксированного размера.

        Применяется DataLoader'ом для динамического выравнивания длин тензоров внутри пакета
        и извлечения соответствующего пакета предвычисленных эмбеддингов ESM-2.

        Args:
            batch (list): Список кортежей, возвращенных методом __getitem__.

        Returns:
            map: Сформированный пакет данных:
                - toxin_seqs (list[str]): Список строк токсинов.
                - decoder_inputs (Tensor): Батч входов декодера [BATCH_SIZE, MAX_LEN].
                - targets (Tensor): Батч таргетов для Loss [BATCH_SIZE, MAX_LEN].
                - aa_lengths (Tensor): Вектор истинных длин белков в батче [BATCH_SIZE].
                - batch_embs (Tensor или None): Тензор эмбеддингов токсинов [BATCH_SIZE, EMB_DIM].
        """
        toxin_seqs = [item[0] for item in batch]
        decoder_inputs = torch.stack([item[1] for item in batch], dim=0)
        targets = torch.stack([item[2] for item in batch], dim=0)
        aa_lengths = torch.stack([item[3] for item in batch], dim=0)

        # Вырезаем срез эмбеддингов для текущего батча, если они были инициализированы
        batch_embs = None
        if self.toxin_embeddings is not None:
            # Механизм автоматической индексации PyTorch собирает под-тензор батча в GPU-friendly структуру
            indices = [item[4] for item in batch if item[4] is not None]
            if len(indices) == len(batch):
                batch_embs = torch.stack([self.toxin_embeddings[idx] for idx in indices], dim=0)

        return {
            'toxin_seqs': toxin_seqs,
            'decoder_inputs': decoder_inputs,
            'targets': targets,
            'aa_lengths': aa_lengths,
            'toxin_embedding': batch_embs
        }