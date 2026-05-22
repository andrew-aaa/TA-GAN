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
├── data/                          # данные (FASTA + эмбеддинги)
├── models/
│   ├── generator.py
│   ├── discriminator.py
│   └── esm_utils.py
├── training/
│   ├── losses.py
│   ├── bio_losses.py
│   ├── metrics.py
│   └── ema.py
├── utils.py
├── config.py
├── train.py                       # основной скрипт обучения
├── generate_antidote.py           # генерация антитоксина
├── validation/
│   ├── generate_and_validate_candidates.py
│   └── select_top_candidates.py
├── prepare_pairs.py
├── precompute_toxin_embeddings.py
└── requirements.txt
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