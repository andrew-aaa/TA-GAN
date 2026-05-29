# Protein Antidote GAN (TA-GAN)

**Генеративно-состязательная сеть для создания аминокислотных последовательностей антитоксинов, обусловленных токсином.**

Проект использует **ESM-2** для извлечения представлений токсинов и Transformer-архитектуру с Gumbel-Softmax для генерации биологически правдоподобных антитоксинов.

---

## Основные возможности

- **Conditional generation** — генерация антитоксина напрямую зависит от эмбеддинга токсина
- **ESM-2 + LoRA** — современные представления белков
- **WGAN-GP** с Gradient Penalty для стабильного обучения
- **Gumbel-Softmax** — дифференцируемая работа с дискретными последовательностями
- **Length Predictor** — автоматическое предсказание оптимальной длины антитоксина
- **Биологические потери** — контроль гидрофобности, изоэлектрической точки (pI) и амфипатичности
- **EMA** (Exponential Moving Average) — сглаживание весов генератора
- **Многоэтапное обучение** с постепенным включением adversarial и bio-losses

---

## Структура проекта

```
TA-GAN/
├── data/                          # Каталог хранения и структурирования данных
│   ├── toxins_paired.fasta        # Очищенные последовательности токсинов
│   ├── antitoxins_paired.fasta    # Соответствующие нативные антитоксины
│   └── toxin_embeddings.pt        # Сериализованный тензор эмбеддингов ESM-2
├── models/                        # Модуль нейросетевых архитектур
│   ├── __init__.py
│   ├── generator.py               # Авторегрессионный Transformer Decoder + Length Predictor
│   └─── discriminator.py           # Условный Transformer Encoder (Критик Васерштейна)
├── training/                      # Модуль алгоритмического обеспечения обучения
│   ├── __init__.py
│   ├── losses.py                  # Состязательные лоссы (WGAN-GP, CE Label Smoothing)
│   ├── bio_losses.py              # Дифференцируемые гидрофобные и электростатические штрафы
│   ├── metrics.py                 # Вычисление онлайн-метрик (Diversity, Repeat Ratio)
│   └── ema.py                     # Сглаживание весов (Exponential Moving Average)
├── validation/                    # Модуль пост-генерационного анализа и фильтрации
│   ├── __init__.py
│   ├── generate_and_validate_candidates.py  # Массовый инференс и первичная биофизика
│   └── select_top_candidates.py            # Финальное многокритериальное ранжирование
├── config.py                      # Централизованный конфигурационный файл гиперпараметров
├── utils.py                       # Общие математические и токенизационные утилиты
├── esm_utils.py                   # Адаптер pLLM ESM-2 с LoRA конфигурацией
├── prepare_pairs.py               # Скрипт парсинга и сопоставления нативных пар
├── precompute_toxin_embeddings.py # Скрипт статического предрасчета признаков токсинов
├── train.py                       # Главный диспетчер (оркестратор) цикла обучения
├── requirements.txt               # Спецификация зависимостей вычислительного окружения
└── README.md                      # Документация и руководство по репликации эксперимента
```

---

## Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Подготовка данных

```bash
# 1. Подготовка пар токсин-антитоксин
python prepare_pairs.py

# 2. Предвычисление эмбеддингов токсинов (рекомендуется)
python precompute_toxin_embeddings.py
```

### 3. Обучение

```bash
python train.py
```

### 4. Генерация антитоксина

```bash
python generate_antidote.py \
  --fasta data/target_toxin.fasta \
  --output outputs/my_antidote.fasta
```

---

## Скрипты проекта

| Скрипт | Назначение |
|-------|----------|
| `train.py` | Основное обучение модели |
| `generate_antidote.py` | Генерация одного лучшего антитоксина |
| `precompute_toxin_embeddings.py` | Предрасчёт ESM-эмбеддингов |
| `prepare_pairs.py` | Создание пар токсин-антитоксин |
| `validation/generate_and_validate_candidates.py` | Массовое генерирование кандидатов |
| `validation/select_top_candidates.py` | Отбор лучших кандидатов |

---

## Конфигурация

Все основные параметры находятся в `config.py`:

- `MAX_AA_LEN = 256` — максимальная длина антитоксина
- `BATCH_SIZE`, `EPOCHS`, learning rates
- Параметры биологических потерь
- Пути к данным и чекпоинтам

---

## Результаты

Модель обучается в несколько фаз:
1. **Pretraining** генератора (Cross-Entropy)
2. **Warm-up** биологических потерь
3. **Adversarial training** (WGAN-GP)

Ключевые метрики валидации:
- Token Cross-Entropy
- Length MAE / Exact Match
- Repeat Ratio
- 3-gram Diversity
- Biological plausibility scores

---

## Технологический стек

- **PyTorch 2.3+**
- **Transformers + ESM-2**
- **PEFT (LoRA)**
- **Biopython**
- **WGAN-GP + Gumbel-Softmax**

---

## Лицензия

MIT License

---

**Автор:** https://github.com/andrew-aaa  
**Цель проекта:** Разработка ИИ-инструментов для дизайна белковых антитоксинов нового поколения.

---