import os
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    f1_score, matthews_corrcoef, balanced_accuracy_score,
    precision_score, recall_score, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold
import numpy as np
import csv
from tqdm import tqdm
from AB_Data import AB_Data
from ADCNet import PredictModel
import traceback
import sys


class Config:
    """Centralized configuration for 5-fold CV training"""

    CUDA_DEVICE = '0'

    DATA_FILE_PATH = "data/data.xlsx"
    EMBEDDING_PATHS = [
        "Embeddings/antibinder_heavy.pkl",
        "Embeddings/Light.pkl",
        "Embeddings/Antigen.pkl"
    ]

    PRETRAINED_MODEL_PATH = "model_weights/modified_model.pth"
    CHECKPOINT_DIR = "ckpts_cv"
    RESULTS_CSV = "ADCNet_5fold_cv_full_metrics.csv"
    LOG_DIR = "ADCNET_LOGS_CV"

    NUM_LAYERS = 6
    D_MODEL = 256
    DFF = 512
    NUM_HEADS = 8
    VOCAB_SIZE = 18
    
    N_FOLDS = 5
    SEEDS = 49
    USE_SEPARATE_TEST_SET = False
    TEST_RATIO = 0.1

    BATCH_SIZE = 8
    MAX_EPOCHS = 200
    MIN_CHECKPOINT_EPOCH = 20
    PATIENCE = 30

    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP_NORM = 1.0

    SCHEDULER_MODE = 'max'
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 5
    SCHEDULER_MIN_LR = 1e-7

    USE_POS_WEIGHT = True
    POS_WEIGHT_VALUE = 0.30

    DECISION_THRESHOLD = 0.5
    PRIMARY_METRIC = 'auc'
    SECONDARY_METRIC = 'acc'

    VERBOSE = True
    SHOW_PROGRESS_BARS = True
    PROGRESS_BAR_WIDTH = 30

    PRINT_FULL_TRACEBACK = True
    CONTINUE_ON_ERROR = True


def set_seed(seed):
    """Set seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_all_metrics(all_labels, all_outputs, threshold=0.5):
    """
    Compute all classification metrics

    Returns dict with: SE, SP, ACC, AUC, PRAUC, F1, PPV, MCC, BA, NPV
    """
    predictions = [1 if x >= threshold else 0 for x in all_outputs]

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(all_labels, predictions).ravel()

    # Basic metrics
    accuracy = accuracy_score(all_labels, predictions)
    auc = roc_auc_score(all_labels, all_outputs)
    prauc = average_precision_score(all_labels, all_outputs)
    f1 = f1_score(all_labels, predictions, zero_division=0)
    mcc = matthews_corrcoef(all_labels, predictions)
    balanced_acc = balanced_accuracy_score(all_labels, predictions)
    ppv = precision_score(all_labels, predictions, zero_division=0)

    # Sensitivity (Recall / True Positive Rate)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    # NPV (Negative Predictive Value)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        'SE': sensitivity,
        'SP': specificity,
        'ACC': accuracy,
        'AUC': auc,
        'PRAUC': prauc,
        'F1': f1,
        'PPV': ppv,
        'MCC': mcc,
        'BA': balanced_acc,
        'NPV': npv
    }


def evaluate_model_with_threshold(model, data_loader, device, criterion, threshold=0.5):
    """
    Evaluate model and compute all metrics

    Returns: loss, metrics_dict, labels, outputs
    """
    model.eval()
    total_loss = 0
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for x1, x2, t1, t2, t3, t4, labels in data_loader:
            x1, x2 = x1.to(device), x2.to(device)
            t1, t2, t3, t4 = t1.to(device), t2.to(device), t3.to(device), t4.to(device)
            labels = labels.to(device)

            outputs = model(x1, mask1=None, adjoin_matrix1=None,
                            x2=x2, mask2=None, adjoin_matrix2=None,
                            t1=t1, t2=t2, t3=t3, t4=t4)

            loss = criterion(outputs, labels.float())
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            all_outputs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    metrics = compute_all_metrics(all_labels, all_outputs, threshold)

    return avg_loss, metrics, all_labels, all_outputs


def train_one_fold(fold_idx, train_loader, val_loader, seed, config, device):
    """Train model on one fold with full metrics tracking"""
    set_seed(seed)

    # Setup CSV logging for this fold
    os.makedirs(config.LOG_DIR, exist_ok=True)
    csv_log_path = os.path.join(config.LOG_DIR, f'seed_{seed}_fold_{fold_idx}_log.csv')

    with open(csv_log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 
                         'train_loss', 'train_acc', 'train_auc', 'train_prauc', 'train_f1',
                         'val_loss', 'val_se', 'val_sp', 'val_acc', 'val_auc', 'val_prauc',
                         'val_f1', 'val_ppv', 'val_mcc', 'val_ba', 'val_npv',
                         'learning_rate', 'status'])

    # Model setup
    model = PredictModel(
        num_layers=config.NUM_LAYERS,
        d_model=config.D_MODEL,
        dff=config.DFF,
        num_heads=config.NUM_HEADS,
        vocab_size=config.VOCAB_SIZE
    ).to(device)

    # Load pretrained weights
    if os.path.exists(config.PRETRAINED_MODEL_PATH):
        checkpoint = torch.load(config.PRETRAINED_MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint, strict=False)

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=config.SCHEDULER_MODE,
        factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE,
        min_lr=config.SCHEDULER_MIN_LR
    )

    # Loss function
    if config.USE_POS_WEIGHT:
        pos_weight = torch.tensor([config.POS_WEIGHT_VALUE]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    # Tracking variables - store best metrics
    best_metrics = {
        'val_loss': 999.0,
        'val_SE': 0.0,
        'val_SP': 0.0,
        'val_ACC': 0.0,
        'val_AUC': 0.0,
        'val_PRAUC': 0.0,
        'val_F1': 0.0,
        'val_PPV': 0.0,
        'val_MCC': 0.0,
        'val_BA': 0.0,
        'val_NPV': 0.0,
    }
    best_epoch = -1
    stopping_monitor = 0

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'seed_{seed}_fold_{fold_idx}_best.pth')

    # Training loop
    if config.SHOW_PROGRESS_BARS:
        epoch_pbar = tqdm(range(config.MAX_EPOCHS), desc=f'  Fold {fold_idx+1}',
                          leave=False, ncols=100)
    else:
        epoch_pbar = range(config.MAX_EPOCHS)

    for epoch in epoch_pbar:
        # Training phase
        model.train()
        train_loss = 0
        train_outputs = []
        train_labels = []

        for x1, x2, t1, t2, t3, t4, labels in train_loader:
            x1, x2 = x1.to(device), x2.to(device)
            t1, t2, t3, t4 = t1.to(device), t2.to(device), t3.to(device), t4.to(device)
            labels = labels.to(device)

            outputs = model(x1, mask1=None, adjoin_matrix1=None,
                            x2=x2, mask2=None, adjoin_matrix2=None,
                            t1=t1, t2=t2, t3=t3, t4=t4)

            loss = criterion(outputs, labels.float())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP_NORM)
            optimizer.step()

            train_loss += loss.item()
            probs = torch.sigmoid(outputs)
            train_outputs.extend(probs.detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        # Calculate training metrics (basic only for speed)
        train_loss = train_loss / len(train_loader)
        train_metrics = compute_all_metrics(train_labels, train_outputs, config.DECISION_THRESHOLD)

        # Validation phase with full metrics
        val_loss, val_metrics, _, _ = evaluate_model_with_threshold(
            model, val_loader, device, criterion, threshold=config.DECISION_THRESHOLD
        )

        # Get primary metric for scheduler
        primary_val = val_metrics[config.PRIMARY_METRIC.upper()]
        scheduler.step(primary_val)
        current_lr = optimizer.param_groups[0]['lr']

        # Checkpoint logic
        status = ''
        if epoch >= config.MIN_CHECKPOINT_EPOCH:
            is_better = False
            primary_metric_upper = config.PRIMARY_METRIC.upper()

            if primary_metric_upper in ['AUC', 'PRAUC', 'ACC', 'F1', 'BA', 'SE', 'SP', 'PPV', 'NPV', 'MCC']:
                # Metrics to maximize
                is_better = (val_metrics[primary_metric_upper] > best_metrics[f'val_{primary_metric_upper}'] or 
                            (val_metrics[primary_metric_upper] == best_metrics[f'val_{primary_metric_upper}'] and 
                             val_metrics['ACC'] > best_metrics['val_ACC']))
            else:
                # Loss to minimize
                is_better = val_loss < best_metrics['val_loss']

            if is_better:
                # Update all best metrics
                best_metrics['val_loss'] = val_loss
                for key, value in val_metrics.items():
                    best_metrics[f'val_{key}'] = value
                best_epoch = epoch
                stopping_monitor = 0
                torch.save(model.state_dict(), checkpoint_path)
                status = 'BEST'
            else:
                stopping_monitor += 1
                status = 'NO_IMPROVE'
        else:
            status = 'WARMUP'

        # CSV logging
        with open(csv_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, 
                f'{train_loss:.6f}', f'{train_metrics["ACC"]:.4f}', 
                f'{train_metrics["AUC"]:.4f}', f'{train_metrics["PRAUC"]:.4f}',
                f'{train_metrics["F1"]:.4f}',
                f'{val_loss:.6f}', 
                f'{val_metrics["SE"]:.4f}', f'{val_metrics["SP"]:.4f}',
                f'{val_metrics["ACC"]:.4f}', f'{val_metrics["AUC"]:.4f}',
                f'{val_metrics["PRAUC"]:.4f}', f'{val_metrics["F1"]:.4f}',
                f'{val_metrics["PPV"]:.4f}', f'{val_metrics["MCC"]:.4f}',
                f'{val_metrics["BA"]:.4f}', f'{val_metrics["NPV"]:.4f}',
                f'{current_lr:.2e}', status
            ])

        if config.SHOW_PROGRESS_BARS and hasattr(epoch_pbar, 'set_postfix'):
            epoch_pbar.set_postfix({
                'auc': f'{val_metrics["AUC"]:.3f}',
                'acc': f'{val_metrics["ACC"]:.3f}'
            })

        if stopping_monitor >= config.PATIENCE:
            break

    # Load best model
    if best_epoch == -1:
        return None

    model.load_state_dict(torch.load(checkpoint_path))

    # Add epoch info to best_metrics
    best_metrics['best_epoch'] = best_epoch
    best_metrics['fold'] = fold_idx
    best_metrics['model_path'] = checkpoint_path

    return best_metrics


def train_cv_one_seed(seed, config, device):
    """Run 5-fold cross-validation for one seed with full metrics"""
    set_seed(seed)

    if config.VERBOSE:
        print(f"\n[INFO] Seed {seed} | Loading data...")

    # Load full dataset
    dataset_loader = AB_Data(config.DATA_FILE_PATH, config.EMBEDDING_PATHS)
    dataset_loader.get_dataloaders(seed=seed, batch_size=config.BATCH_SIZE)

    # Get the full dataset
    if config.USE_SEPARATE_TEST_SET:
        from torch.utils.data import random_split
        full_dataset = dataset_loader.train_dataset.dataset
        total_size = len(full_dataset)
        test_size = int(config.TEST_RATIO * total_size)
        cv_size = total_size - test_size

        generator = torch.Generator().manual_seed(seed)
        cv_dataset, test_dataset = random_split(full_dataset, [cv_size, test_size], generator=generator)
    else:
        cv_dataset = dataset_loader.train_dataset.dataset
        test_dataset = None

    # Get labels for stratification
    labels = [cv_dataset[i][-1].item() for i in range(len(cv_dataset))]

    # Create stratified K-fold
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=seed)

    fold_results = []

    print(f"[INFO] Seed {seed} | Running {config.N_FOLDS}-fold cross-validation...")

    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        if config.VERBOSE:
            print(f"\n{'='*60}")
            print(f"Seed {seed} | Fold {fold_idx + 1}/{config.N_FOLDS}")
            print(f"{'='*60}")

        # Create train and validation subsets
        train_subset = Subset(cv_dataset, train_indices)
        val_subset = Subset(cv_dataset, val_indices)

        # Create dataloaders
        train_loader = DataLoader(train_subset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=config.BATCH_SIZE, shuffle=False)

        # Train on this fold
        fold_result = train_one_fold(fold_idx, train_loader, val_loader, seed, config, device)

        if fold_result is not None:
            fold_results.append(fold_result)

            if config.VERBOSE:
                print(f"Fold {fold_idx+1} | Epoch: {fold_result['best_epoch']} | "
                      f"AUC: {fold_result['val_AUC']:.4f} | ACC: {fold_result['val_ACC']:.4f} | "
                      f"F1: {fold_result['val_F1']:.4f} | MCC: {fold_result['val_MCC']:.4f}")

    if len(fold_results) == 0:
        return None

    # Calculate average metrics across folds
    metric_names = ['SE', 'SP', 'ACC', 'AUC', 'PRAUC', 'F1', 'PPV', 'MCC', 'BA', 'NPV']

    avg_results = {
        'seed': seed,
        'n_folds': len(fold_results),
        'avg_best_epoch': np.mean([r['best_epoch'] for r in fold_results]),
        'std_best_epoch': np.std([r['best_epoch'] for r in fold_results]),
        'avg_val_loss': np.mean([r['val_loss'] for r in fold_results]),
        'std_val_loss': np.std([r['val_loss'] for r in fold_results]),
    }

    # Add mean and std for all metrics
    for metric in metric_names:
        key = f'val_{metric}'
        values = [r[key] for r in fold_results]
        avg_results[f'avg_{key}'] = np.mean(values)
        avg_results[f'std_{key}'] = np.std(values)

    avg_results['fold_results'] = fold_results

    # Optional: Evaluate on held-out test set
    if config.USE_SEPARATE_TEST_SET and test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

        # Use model from best fold (by primary metric)
        primary_key = f'val_{config.PRIMARY_METRIC.upper()}'
        best_fold = max(fold_results, key=lambda x: x[primary_key])

        model = PredictModel(
            num_layers=config.NUM_LAYERS,
            d_model=config.D_MODEL,
            dff=config.DFF,
            num_heads=config.NUM_HEADS,
            vocab_size=config.VOCAB_SIZE
        ).to(device)
        model.load_state_dict(torch.load(best_fold['model_path']))

        if config.USE_POS_WEIGHT:
            pos_weight = torch.tensor([config.POS_WEIGHT_VALUE]).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()

        test_loss, test_metrics, _, _ = evaluate_model_with_threshold(
            model, test_loader, device, criterion, threshold=config.DECISION_THRESHOLD
        )

        # Add test metrics
        avg_results['test_loss'] = test_loss
        for metric in metric_names:
            avg_results[f'test_{metric}'] = test_metrics[metric]

    return avg_results


def main():
    """Main cross-validation loop"""
    os.environ['CUDA_VISIBLE_DEVICES'] = Config.CUDA_DEVICE
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Using: {device}")

    # Create results CSV with all metrics
    with open(Config.RESULTS_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['seed', 'n_folds', 'avg_best_epoch', 'std_best_epoch', 'avg_val_loss', 'std_val_loss']

        # Add all validation metrics (mean and std)
        metric_names = ['SE', 'SP', 'ACC', 'AUC', 'PRAUC', 'F1', 'PPV', 'MCC', 'BA', 'NPV']
        for metric in metric_names:
            header.extend([f'avg_val_{metric}', f'std_val_{metric}'])

        # Add test metrics if using separate test set
        if Config.USE_SEPARATE_TEST_SET:
            header.append('test_loss')
            for metric in metric_names:
                header.append(f'test_{metric}')

        writer.writerow(header)

    print(f"\n{'='*80}")
    print(f"ADCNet {Config.N_FOLDS}-Fold Cross-Validation - FULL METRICS")
    print(f"Seeds: {Config.SEEDS}")
    print(f"Metrics: SE, SP, ACC, AUC, PRAUC, F1, PPV, MCC, BA, NPV")
    print(f"Results: {Config.RESULTS_CSV}")
    print(f"{'='*80}\n")

    seed = Config.SEEDS
    print(f"\n{'#'*80}")
    print(f"# SEED {seed}")
    print(f"{'#'*80}")

    result = train_cv_one_seed(seed, Config, device)

    if result is not None:

        # Write to CSV
        with open(Config.RESULTS_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [
                result['seed'], 
                result['n_folds'], 
                f"{result['avg_best_epoch']:.1f}",
                f"{result['std_best_epoch']:.1f}",
                f"{result['avg_val_loss']:.6f}",
                f"{result['std_val_loss']:.6f}"
            ]

            # Add all metrics
            metric_names = ['SE', 'SP', 'ACC', 'AUC', 'PRAUC', 'F1', 'PPV', 'MCC', 'BA', 'NPV']
            for metric in metric_names:
                row.append(f"{result[f'avg_val_{metric}']:.4f}")
                row.append(f"{result[f'std_val_{metric}']:.4f}")

            # Add test metrics if available
            if Config.USE_SEPARATE_TEST_SET and f'test_SE' in result:
                row.append(f"{result['test_loss']:.6f}")
                for metric in metric_names:
                    row.append(f"{result[f'test_{metric}']:.4f}")

            writer.writerow(row)

        print(f"\n[SUMMARY] Seed {seed}")
        print(f"  SE:     {result['avg_val_SE']:.4f} ± {result['std_val_SE']:.4f}")
        print(f"  SP:     {result['avg_val_SP']:.4f} ± {result['std_val_SP']:.4f}")
        print(f"  ACC:    {result['avg_val_ACC']:.4f} ± {result['std_val_ACC']:.4f}")
        print(f"  AUC:    {result['avg_val_AUC']:.4f} ± {result['std_val_AUC']:.4f}")
        print(f"  PRAUC:  {result['avg_val_PRAUC']:.4f} ± {result['std_val_PRAUC']:.4f}")
        print(f"  F1:     {result['avg_val_F1']:.4f} ± {result['std_val_F1']:.4f}")
        print(f"  PPV:    {result['avg_val_PPV']:.4f} ± {result['std_val_PPV']:.4f}")
        print(f"  MCC:    {result['avg_val_MCC']:.4f} ± {result['std_val_MCC']:.4f}")
        print(f"  BA:     {result['avg_val_BA']:.4f} ± {result['std_val_BA']:.4f}")
        print(f"  NPV:    {result['avg_val_NPV']:.4f} ± {result['std_val_NPV']:.4f}")



if __name__ == "__main__":
    main()
