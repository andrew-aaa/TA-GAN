# validation/select_top_candidates.py
"""
Модуль финального многофакторного ранжирования и экспорта топ-кандидатов антидотов.

Скрипт выполняет аналитическую фильтрацию сводного отчета генерации:
1. Загружает сформированный ранее файл `candidates_report.csv`.
2. Очищает данные от ультракоротких артефактов генерации (длиной менее 5 аминокислот).
3. Вычисляет интегральный критерий жизнеспособности (Combined Ranking Score), 
   гармонично объединяющий сырой штраф модели, инвертированную диверсификацию триграмм 
   и длину непрерывных монотонных повторов.
4. Выполняет группировку кандидатов по идентификаторам токсинов (`toxin_id`) 
   и экспортирует Top-K лучших последовательностей в формате CSV (с метриками) 
   и в стандартном биологическом формате FASTA для валидации методами in silico / в мокрой лаборатории.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd

# Динамическое подключение корня проекта в пути поиска модулей для сквозной переносимости
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    """
    Выполняет конвейер фильтрации, переранжирования и экспорта топ-кандидатов.

    Алгоритм работы:
    1. Парсинг параметров: считывание путей ввода-вывода и гиперпараметра Top-K.
    2. Фильтрация: отсечение слишком коротких строк, которые физически не могут 
       сформировать стабильный глобулярный белок или альфа-спираль.
    3. Комбинированное ранжирование: расчет `combined_ranking_score`. Чем ниже данный балл,
       тем более перспективным является кандидат с точки зрения физико-химической стабильности.
    4. Экспорт: сохранение результатов в структурированную таблицу и целевой FASTA-файл.
    """
    parser = argparse.ArgumentParser(description="Фильтрация и многофакторный отбор топ-кандидатов антидотов.")
    parser.add_argument("--input_csv", type=str, required=True, help="Путь к исходному файлу candidates_report.csv")
    parser.add_argument("--output_prefix", type=str, default="top_candidates", help="Префикс для выходных файлов (*_metrics.csv, *.fasta)")
    parser.add_argument("--top_k", type=int, default=10, help="Количество лучших кандидатов, отбираемых для каждого токсина")
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"[error] Ошибка: Исходный файл {args.input_csv} не найден.")
        return

    # Загрузка результатов массовой генерации в DataFrame
    df = pd.read_csv(input_path)
    if df.empty:
        print("[error] Ошибка: Исходный CSV-файл пуст.")
        return

    # Гарантируем строковый тип данных для последовательностей и заполняем возможные NaN пустой строкой
    df['sequence'] = df['sequence'].fillna('').astype(str)

    initial_count = len(df)
    
    # Жесткая фильтрация структурных артефактов (длина последовательности должна быть >= 5 аминокислот)
    df = df[df['sequence'].str.len() >= 5].copy()
    
    # Вычисление инвертированного разнообразия триграмм. 
    # Если диверсификация n-грамм стремится к 1.0 (идеал), то inv_div стремится к 0.0 (минимальный штраф).
    df['inv_div'] = 1.0 - df['ngram3_diversity']
    
    # МАТЕМАТИЧЕСКОЕ ОБОСНОВАНИЕ КОМБИНИРОВАННОГО БАЛЛА (Combined Ranking Score):
    # Формула взвешивает три критических дефекта генерации:
    # 1. df['score'] * 1.0     -> Базовый штраф за отклонение от предсказанной длины.
    # 2. df['inv_div'] * 5.0   -> Значительный штраф за локальную повторяемость контекста (низкое разнообразие).
    # 3. df['max_run'] * 0.5   -> Штраф за длинные монотонные "раны" из идентичных аминокислот.
    # Оптимизация направлена на минимизацию итоговой суммы (Low Score == High Quality).
    df['combined_ranking_score'] = (
        df['score'] * 1.0 +
        df['inv_div'] * 5.0 +
        df['max_run'] * 0.5
    )

    # Двухуровневая сортировка: сначала группируем по токсинам, внутри группы выстраиваем по возрастанию штрафного балла
    top_candidates = df.sort_values(['toxin_id', 'combined_ranking_score'])
    
    # Смена стратегии: выборка ровно Top-K лучших уникальных белков-кандидатов для каждого уникального биотоксина
    top_candidates = top_candidates.groupby('toxin_id').head(args.top_k).copy()

    # Определение путей для записи результатов валидации
    out_dir = input_path.parent
    metrics_out = out_dir / f"{args.output_prefix}_metrics.csv"
    fasta_out = out_dir / f"{args.output_prefix}.fasta"

    # 1. Сохранение отфильтрованной и переранжированной таблицы с метриками
    top_candidates.to_csv(metrics_out, index=False)

    # 2. Потоковый экспорт последовательностей в финальный файл формата FASTA
    with open(fasta_out, "w", encoding="utf-8") as f:
        for _, row in top_candidates.iterrows():
            # Формируем уникальный читаемый системный идентификатор кандидата
            cand_id = f"cand_{row['toxin_id']}_{row['rank']}"
            
            # Сборка информационного заголовка (header) FASTA, содержащего ключевые дескрипторы для биоинформатиков
            header = (
                f">{cand_id} rank={row['rank']} toxin={row['toxin_id']} "
                f"score={row['combined_ranking_score']:.4f} pI={row['pI']} "
                f"hydro={row['hydrophobicity']} new={row['is_new']}"
            )
            f.write(header + "\n")
            
            # Посимвольная запись аминокислотной последовательности кандидата
            f.write(row['sequence'] + "\n")

    print(f"[done] Фильтрация завершена. Из {initial_count} исходных записей отобрано {len(top_candidates)} кандидатов.")
    print(f"[done] Таблица метрик сохранена в: {metrics_out}")
    print(f"[done] Целевой FASTA-файл успешно сгенерирован: {fasta_out}")


if __name__ == "__main__":
    main()