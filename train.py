# train.py
"""
Модуль сквозного состязательного обучения генеративно-состязательной сети TA-GAN.

Скрипт координирует весь пайплайн обучения модели дизайна с нуля антидотов:
1. Инициализирует и настраивает компоненты Генератора и Дискриминатора.
2. Реализует многофазное расписание: предварительное обучение Генератора на максимум 
   правдоподобия (Teacher Forcing Cross-Entropy) и последующую фазу GAN-обучения.
3. Интегрирует комбинированную функцию потерь: состязательную потерю (WGAN-GP), 
   линейную кросс-энтропию длин и биофизические штрафы (гидрофобность, изоэлектрическая точка).
4. Поддерживает механизм скользящего среднего весов (EMA) для стабилизации инференса.
"""

from __future__ import annotations

import os
# Форсированная блокировка асинхронности CUDA для получения точных трейсбэков при отладке
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from config import *
from utils import set_seed, to_one_hot, decode_sequence
from data.dataset import ToxinAntitoxinDataset
from models.generator import Generator
from models.discriminator import Discriminator
from training.ema import EMA
from training.losses import gradient_penalty, token_ce_loss
from training.bio_losses import bio_loss_generator
from training.metrics import repeat_ratio, ngram_diversity


def get_adv_weight(epoch_idx: int) -> float:
    """
    Вычисляет динамический вес состязательного лосса (Adversarial Weight Ramp-up).

    Обеспечивает плавное включение градиентов Дискриминатора после завершения 
    фазы предварительного обучения (pre-training) Генератора.

    Args:
        epoch_idx (int): Индекс текущей эпохи обучения.

    Returns:
        float: Масштабирующий коэффициент состязательных потерь в диапазоне [0.0, ADV_WEIGHT_MAX].
    """
    if epoch_idx < GENERATOR_PRETRAIN_EPOCHS:
        return 0.0
    ramp = max(1, EPOCHS - GENERATOR_PRETRAIN_EPOCHS)
    return ADV_WEIGHT_MAX * min(1.0, (epoch_idx - GENERATOR_PRETRAIN_EPOCHS + 1) / ramp)


def get_bio_weight(epoch_idx: int) -> float:
    """
    Вычисляет динамический вес биологических штрафов (Bio-loss Ramp-up).

    Плавный ввод ограничений по гидрофобности и pI позволяет Генератору сначала 
    освоить базовый синтаксис распределения аминокислот, а затем скорректировать 
    физико-химический профиль цепи.

    Args:
        epoch_idx (int): Индекс текущей эпохи обучения.

    Returns:
        float: Текущий вес биологических функций потерь.
    """
    if epoch_idx < BIO_LOSS_RAMP_EPOCH:
        return 0.0
    return min(1.0, (epoch_idx - BIO_LOSS_RAMP_EPOCH + 1) / 10.0)


def get_tau(epoch_idx: int) -> float:
    """
    Реализует линейный отжиг (Annealing) температуры Gumbel-Softmax распределения.

    Высокая температура на старте стимулирует исследование латентного пространства, 
    низкая температура под конец фазы сближает распределение с жестким дискретным выбором.

    Args:
        epoch_idx (int): Индекс текущей эпохи обучения.

    Returns:
        float: Значение температуры сглаживания распределения токенов.
    """
    progress = min(1.0, epoch_idx / max(1, EPOCHS - 1))
    return TAU_START + progress * (TAU_END - TAU_START)


def evaluate(generator: Generator, loader: DataLoader, epoch_idx: int, device: str) -> dict[str, float]:
    """
    Выполняет валидационный проход, рассчитывает метрики качества и логирует примеры генерации.

    Args:
        generator (Generator): Текущий инстанс модели Генератора.
        loader (DataLoader): Валидационный подмножество данных.
        epoch_idx (int): Номер текущей эпохи для текстового вывода.
        device (str): Целевое вычислительное устройство.

    Returns:
        dict[str, float]: Агрегированные метрики качества валидационной выборки.
    """
    generator.eval()
    total_ce = 0.0
    total_len_loss = 0.0
    all_fake_ids = []
    all_target_lengths = []
    
    sample_logged = False

    with torch.no_grad():
        for batch in loader:
            toxin_seqs = batch['toxin_seqs']
            decoder_inputs = batch['decoder_inputs'].to(device)
            targets = batch['targets'].to(device)
            aa_lengths = batch['aa_lengths'].to(device)
            
            toxin_emb = batch['toxin_embedding']
            if toxin_emb is not None:
                toxin_emb = toxin_emb.to(device)

            # Валидационный форвард в режиме Teacher Forcing для оценки сходимости
            logits, pred_len_logits = generator.forward_teacher_forcing(
                decoder_inputs, toxin_emb, z=None, target_lengths=aa_lengths
            )
            
            # Накопление классических потерь валидации
            loss_ce = token_ce_loss(logits, targets)
            loss_len = F.cross_entropy(pred_len_logits, aa_lengths)
            
            total_ce += loss_ce.item() * toxin_emb.size(0)
            total_len_loss += loss_len.item() * toxin_emb.size(0)

            # Стохастическая генерация с нуля (без подсказок со стороны реальной последовательности)
            fake_ids, _, pred_lengths = generator._autoregressive_generate(
                toxin_emb, temperature=1.0, hard=True, differentiable=False
            )
            all_fake_ids.append(fake_ids.cpu())
            all_target_lengths.append(aa_lengths.cpu())

            # Логирование первого попавшегося примера из батча для визуальной оценки качества синтаксиса белков
            if not sample_logged and len(batch['toxin_seqs']) > 0:
                real_str = decode_sequence(targets[0].cpu().tolist())
                fake_str = decode_sequence(fake_ids[0].cpu().tolist())
                print(f"\n--- [epoch {epoch_idx}] Валидационный пример ---")
                print(f"Целевой токсин: {batch['toxin_seqs'][0][:50]}...")
                print(f"Реальный антидот: {real_str[:60]}")
                print(f"Сгенерирован de novo: {fake_str[:60]}")
                print(f"Физическая длина (Реальная: {aa_lengths[0].item()} | Предсказанная: {pred_lengths[0].item()})")
                print("------------------------------------------")
                sample_logged = True

    # Агрегация собранных тензоров со всей валидационной выборки
    all_fake_ids = torch.cat(all_fake_ids, dim=0)
    all_target_lengths = torch.cat(all_target_lengths, dim=0)
    num_samples = len(loader.dataset)

    # Вычисление лингвистических и структурных метрик белковых последовательностей
    metrics = {
        "val_loss_ce": total_ce / num_samples,
        "val_loss_len": total_len_loss / num_samples,
        "val_repeat_ratio": repeat_ratio(all_fake_ids),
        "val_diversity": ngram_diversity(all_fake_ids, n=3),
    }
    return metrics


def main():
    """
    Главная точка входа управляющего скрипта обучения.
    Конфигурирует циклы оптимизации, распределяет шаг обучения между сетями и сохраняет чекпоинты.
    """
    set_seed(42)
    device = DEVICE
    print(f"[train] Инициализация пайплайна на устройстве: {device}")

    # Загрузка и разбиение белкового датасета в пропорции 90% обучение / 10% валидация
    full_dataset = ToxinAntitoxinDataset(
        toxin_fasta=TOXIN_FASTA_PATH,
        antidote_fasta=ANTITOXIN_FASTA_PATH,
        toxin_embeddings_path=TOXIN_EMBEDDINGS_PATH
    )
    val_size = int(len(full_dataset) * 0.1)
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # Конструирование вычислительных графов Генератора и Дискриминатора
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Настройка независимых оптимизаторов для состязательной пары
    optimizer_G = optim.AdamW(generator.parameters(), lr=LR_G, betas=(0.5, 0.9), weight_decay=1e-4)
    optimizer_D = optim.AdamW(discriminator.parameters(), lr=LR_D, betas=(0.5, 0.9), weight_decay=1e-4)

    # Развертывание экспоненциального сглаживания (EMA) над весами Генератора для повышения стабильности
    ema = EMA(generator, decay=EMA_DECAY)

    best_val_ce = float('inf')
    fieldnames = ["epoch", "loss_D", "loss_G", "loss_ce", "loss_len", "loss_adv", "loss_bio", "val_loss_ce", "val_repeat_ratio", "val_diversity"]

    # ГЛАВНЫЙ ЦИКЛ ОБУЧЕНИЯ (Epoch Loop)
    for epoch in range(1, EPOCHS + 1):
        generator.train()
        discriminator.train()

        # Динамический пересчет параметров расписания для текущей эпохи
        adv_w = get_adv_weight(epoch)
        bio_w = get_bio_weight(epoch)
        tau = get_tau(epoch)

        pbar = tqdm(train_loader, desc=f"Эпоха {epoch}/{EPOCHS} [adv_w={adv_w:.2f}, tau={tau:.2f}]")
        
        for batch in pbar:
            decoder_inputs = batch['decoder_inputs'].to(device)
            aa_lengths = batch['aa_lengths'].to(device)
            
            toxin_emb = batch['toxin_embedding']
            if toxin_emb is not None:
                toxin_emb = toxin_emb.to(device)
            
            real_ids = batch['targets'].to(device) # [BATCH_SIZE, MAX_LEN]

            real_onehot = to_one_hot(real_ids, VOCAB_SIZE) # [BATCH_SIZE, MAX_LEN, VOCAB_SIZE]
            
            # Генерация латентного вектора шума для состязательного макромолекулярного дизайна
            z = torch.randn(BATCH_SIZE, LATENT_DIM, device=device)

            # ====================================================================
            # ФАЗА 1: Оптимизация Дискриминатора (выполняется только после pre-train)
            # ====================================================================
            if epoch > GENERATOR_PRETRAIN_EPOCHS:
                optimizer_D.zero_grad()

                with torch.no_grad():
                    # Получение распределений логитов от генератора без накопления его градиентов
                    logits_fake, _ = generator.forward_teacher_forcing(
                        decoder_inputs, toxin_emb, z=z, target_lengths=aa_lengths
                    )
                    fake_onehot_d = F.gumbel_softmax(logits_fake, tau=tau, hard=False, dim=-1)

                # Вычисление оценок Дискриминатора для реального и синтезированного распределений
                d_real = discriminator(toxin_emb, real_onehot, aa_lengths)
                d_fake = discriminator(toxin_emb, fake_onehot_d, aa_lengths)

                # Расчет штрафа за градиент (WGAN-GP) для обеспечения Липшицева ограничения функции
                gp = gradient_penalty(discriminator, toxin_emb, real_onehot, fake_onehot_d, aa_lengths, device)
                
                # Итоговая потеря Дискриминатора: максимизация расстояния Вассерштейна + штраф регуляризации
                loss_D = d_fake.mean() - d_real.mean() + GP_WEIGHT * gp
                
                loss_D.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), GRAD_CLIP)
                optimizer_D.step()
            else:
                loss_D = torch.tensor(0.0, device=device)

            # ====================================================================
            # ФАЗА 2: Оптимизация Генератора
            # ====================================================================
            optimizer_G.zero_grad()

            # Прямой проход Генератора (Teacher Forcing)
            logits, pred_len_logits = generator.forward_teacher_forcing(
                decoder_inputs, toxin_emb, z=z, target_lengths=aa_lengths
            )

            # Числовой хак (Стабилизация): Защита от взрыва градиентов и потенциальных NaN в полуавтоматическом режиме
            logits = torch.nan_to_num(logits, nan=0.0, posinf=5.0, neginf=-5.0)

            # Расчет базовой кросс-энтропии (лингвистическая точность) и ошибки предсказания длин белков
            loss_ce = token_ce_loss(logits, real_ids)
            loss_len = F.cross_entropy(pred_len_logits, aa_lengths)

            # Формирование мягкого дифференцируемого One-hot распределения через Gumbel-Softmax для прохода в D
            fake_onehot_g = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)

            # Состязательная потеря: Генератор пытается заставить Дискриминатор выставить максимальную оценку фейкам
            loss_adv = -discriminator(toxin_emb, fake_onehot_g, aa_lengths).mean()
            
            # Извлечение и масштабирование биологических штрафов за нарушение законов фолдинга/структуры
            bio_dict = bio_loss_generator(fake_onehot_g, aa_lengths)
            loss_bio = bio_dict["bio_total"] * bio_w

            # Агрегация многокомпонентной целевой функции Генератора
            loss_G = loss_ce + LENGTH_REG_WEIGHT * loss_len + adv_w * loss_adv + loss_bio

            loss_G.backward()
            # Ограничение нормы градиента во избежание коллапса латентного пространства трансформера
            torch.nn.utils.clip_grad_norm_(generator.parameters(), GRAD_CLIP)
            optimizer_G.step()

            # Синхронизация и обновление скользящих тензоров EMA
            ema.update(generator)

            # Обновление интерактивного прогресс-бара консоли
            pbar.set_postfix({
                "D": f"{loss_D.item():.3f}",
                "G": f"{loss_G.item():.3f}",
                "CE": f"{loss_ce.item():.3f}",
                "Adv": f"{loss_adv.item():.3f}"
            })

        # Валидационный этап по окончании каждой эпохи
        val_metrics = evaluate(generator, val_loader, epoch, device)
        print(f"[epoch {epoch}] Val CE: {val_metrics['val_loss_ce']:.4f} | Diversity: {val_metrics['val_diversity']:.3f}")

        # Сборка комплексного лога для записи в текстовый архив метрик
        row_log = {
            "epoch": epoch,
            "loss_D": loss_D.item(),
            "loss_G": loss_G.item(),
            "loss_ce": loss_ce.item(),
            "loss_len": loss_len.item(),
            "loss_adv": loss_adv.item(),
            "loss_bio": loss_bio.item() if isinstance(loss_bio, float) else loss_bio.item(),
            **val_metrics
        }
        from utils import write_metrics_row
        write_metrics_row(METRICS_CSV_PATH, fieldnames, row_log)

        # Контрольное сохранение весов (Чекпоинты): Лучшие и Последние
        torch.save(generator.state_dict(), GENERATOR_LAST_PATH)
        torch.save(ema.state_dict(), EMA_LAST_PATH)

        if val_metrics['val_loss_ce'] < best_val_ce:
            best_val_ce = val_metrics['val_loss_ce']
            torch.save(generator.state_dict(), GENERATOR_BEST_PATH)
            torch.save(ema.state_dict(), EMA_BEST_PATH)
            print(f"\t[checkpoint] Найдена лучшая модель по Cross-Entropy. Сохранено.")


if __name__ == "__main__":
    main()