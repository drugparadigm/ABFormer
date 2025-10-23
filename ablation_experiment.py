import os
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
import numpy as np
import csv
from tqdm import tqdm
from AB_Data import AB_Data
from utils import score
import traceback
import sys

import pandas as pd
import pickle
import torch.nn.functional as F
import rdkit
from rdkit import Chem
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection  import train_test_split

# ********************************************************************************
# ********************************************************************************
#                 Ablated Version of ABFormer [removed AntiBinder Component]
# *********************************************************************************
# ********************************************************************************

def scaled_dot_product_attention(q, k, v, mask=None, adjoin_matrix=None):
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))

    dk = k.shape[-1]
    scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(dk, dtype=torch.float32))

    if mask is not None:
        scaled_attention_logits += mask * -1e9

    if adjoin_matrix is not None:
        scaled_attention_logits += adjoin_matrix

    attention_weights = F.softmax(scaled_attention_logits, dim=-1)

    output = torch.matmul(attention_weights, v) 

    return output, attention_weights


class MultiHeadAttention(nn.Module):
  def __init__(self,d_model,num_heads):
    super().__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    self.depth = d_model // self.num_heads

    self.wq = nn.Linear(d_model,d_model)
    self.wk = nn.Linear(d_model,d_model)
    self.wv = nn.Linear(d_model,d_model)
    self.dense = nn.Linear(d_model,d_model)

  def forward(self,v,k,q,mask,adjoin_matrix):
    batch_size = q.shape[0]

    q = self.wq(q)
    k = self.wk(k)
    v = self.wv(v)

    q = self.split_heads(q,batch_size)
    k = self.split_heads(k,batch_size)
    v = self.split_heads(v,batch_size)

    scaled_attention,attention_weights = scaled_dot_product_attention(q,k,v,mask,adjoin_matrix)
    scaled_attention = scaled_attention.permute(0,2,1,3)
    concat_attention = scaled_attention.reshape(batch_size,-1, self.d_model)
    output = self.dense(concat_attention)
    return output,attention_weights

  def split_heads(self,x,batch_size):
    x = torch.reshape(x,(batch_size,-1,self.num_heads,self.depth))
    x = x.permute(0,2,1,3)
    return x


class FeedForwardNetwork(nn.Module):
  def __init__(self,dff,d_model):
    super().__init__()
    self.fc1 = nn.Linear(d_model,dff)
    self.fc2 = nn.Linear(dff,d_model)

  def forward(self,x):
    x = F.gelu(self.fc1(x))
    x = self.fc2(x)
    return x


class EncoderLayer(nn.Module):
  def __init__(self,d_model,num_heads,dff,rate = 0.1):
    super().__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = FeedForwardNetwork(dff,d_model)
    self.layernorm1 = torch.nn.LayerNorm(d_model)
    self.layernorm2 = torch.nn.LayerNorm(d_model)

    self.dropout1 = torch.nn.Dropout(rate)
    self.dropout2 = torch.nn.Dropout(rate)

  def forward(self,x,training,mask, adjoin_matrix):
    attn_output , attention_weights = self.mha(x,x,x,mask,adjoin_matrix)
    attn_output = self.dropout1(attn_output)
    out1 = self.layernorm1(x + attn_output)
    ffn_output = self.ffn(out1)
    ffn_output = self.dropout2(ffn_output)
    out2 = self.layernorm2(out1 + ffn_output)
    return out2,attention_weights


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
                 maximum_position_encoding, rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dff, rate) 
                                         for _ in range(num_layers)])
        self.dropout = nn.Dropout(rate)

    def forward(self, x, training, mask, adjoin_matrix):
        seq_len = x.shape[1]
        if adjoin_matrix is not None:
            adjoin_matrix = adjoin_matrix.unsqueeze(1)

        x = self.embedding(x)
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))

        x = self.dropout(x)

        for layer in self.enc_layers:
            x, attention_weights = layer(x, training, mask, adjoin_matrix)

        return x, attention_weights

class PredictModel(nn.Module):
    def __init__(self, num_layers=6, d_model=256, dff=512, num_heads=8, vocab_size=18, dropout_rate=0.5):
        super().__init__()
        self.encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            input_vocab_size=vocab_size,
            maximum_position_encoding=200,
            rate=dropout_rate
        )
        self.fc1 = nn.Linear(3073, d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(d_model,d_model)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(d_model, 1)

    def forward(self, x1, x2, t2, t3, t4, adjoin_matrix1=None, mask1=None, adjoin_matrix2=None, mask2=None, training=False):
        x1, attention_weights1 = self.encoder(x1, training=training, mask=mask1, adjoin_matrix=adjoin_matrix1)
        x1 = x1[:, 0, :] 

        x2, attention_weights2 = self.encoder(x2, training=False, mask=mask2, adjoin_matrix=adjoin_matrix2)
        x2 = x2[:, 0, :]

        if t4.dim() == 1:
            t4 = t4.unsqueeze(1)


        x = torch.cat((x1, x2,t2,t3, t4), dim=1)
        x = self.fc1(x) 
        x = self.dropout1(x)
        x = self.fc2(x) 
        x = self.dropout2(x)
        x = self.fc3(x)
        x = x.squeeze(1)
        return x 

# ********************************************************************************
# ********************************************************************************


class Config:
    """Centralized configuration for training"""
    
    CUDA_DEVICE = '0'  # GPU device ID
    
    DATA_FILE_PATH = "data/data.xlsx"  # Can be .csv or .xlsx
    EMBEDDING_PATHS = [
        "Embeddings/antibinder_heavy.pkl",
        "Embeddings/Light.pkl",
        "Embeddings/Antigen.pkl"
    ]
    
    PRETRAINED_MODEL_PATH = "model_weights/modified_model.pth"
    CHECKPOINT_DIR = "ckpts"
    
    RESULTS_CSV = "ADCNet_ablation.csv"
    LOG_DIR = "ADCNET_LOGS_ablation"
    
    NUM_LAYERS = 6
    D_MODEL = 256
    DFF = 512
    NUM_HEADS = 8
    VOCAB_SIZE = 18
    
    SEED = 49
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

def set_seed(seed):
    """Set seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_model_with_threshold(model, data_loader, device, criterion, threshold=0.5):
    """
    Evaluate with custom threshold and detailed metrics including PRAUC
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
                             t2=t2, t3=t3, t4=t4)

            loss = criterion(outputs, labels.float())
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            all_outputs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    predictions = [1 if x >= threshold else 0 for x in all_outputs]
    accuracy = accuracy_score(all_labels, predictions)

    try:
        auc_score = roc_auc_score(all_labels, all_outputs)
    except:
        auc_score = 0.0

    try:
        prauc_score = average_precision_score(all_labels, all_outputs)
    except:
        prauc_score = 0.0

    # Calculate specificity and sensitivity
    tp = sum((p == 1 and l == 1) for p, l in zip(predictions, all_labels))
    tn = sum((p == 0 and l == 0) for p, l in zip(predictions, all_labels))
    fp = sum((p == 1 and l == 0) for p, l in zip(predictions, all_labels))
    fn = sum((p == 0 and l == 1) for p, l in zip(predictions, all_labels))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return avg_loss, accuracy, auc_score, prauc_score, sensitivity, specificity, all_labels, all_outputs


def train_one_seed(seed, config, device):
    """
    Train with improved checkpointing strategy and PRAUC tracking.
    Returns dict with all metrics including best_epoch.
    """
    set_seed(seed)

    os.makedirs(config.LOG_DIR, exist_ok=True)
    csv_log_path = os.path.join(config.LOG_DIR, f'training_log_seed_{seed}.csv')

    with open(csv_log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'train_auc', 'train_prauc',
                         'val_loss', 'val_acc', 'val_auc', 'val_prauc',
                         'val_sens', 'val_spec', 'learning_rate', 'status'])

    # Load data
    if config.VERBOSE:
        print(f"[INFO] Seed {seed} | Loading data from: {config.DATA_FILE_PATH}")
    
    dataset = AB_Data(config.DATA_FILE_PATH, config.EMBEDDING_PATHS)
    train_loader, val_loader, test_loader = dataset.get_dataloaders(
        batch_size=config.BATCH_SIZE,
        seed=seed
    )
    
    if config.VERBOSE:
        print(f"[INFO] Seed {seed} | Data loaded successfully")

    # Model setup
    model = PredictModel(
        num_layers=config.NUM_LAYERS,
        d_model=config.D_MODEL,
        dff=config.DFF,
        num_heads=config.NUM_HEADS,
        vocab_size=config.VOCAB_SIZE
    ).to(device)

    # Load pretrained weights if available
    if os.path.exists(config.PRETRAINED_MODEL_PATH):
        checkpoint = torch.load(config.PRETRAINED_MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        if config.VERBOSE:
            print(f"[INFO] Seed {seed} | Loaded pretrained weights")

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # Scheduler
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
        if config.VERBOSE:
            print(f"[INFO] Seed {seed} | Using pos_weight={pos_weight.item():.3f}")
    else:
        criterion = nn.BCEWithLogitsLoss()

    # Checkpoint variables
    best_val_prauc = 0.0
    best_val_auc = 0.0
    best_val_acc = 0.0
    best_val_loss = 999.0
    best_val_sens = 0.0
    best_val_spec = 0.0
    best_epoch = -1
    stopping_monitor = 0

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'ADC_{seed}_best_model.pth')

    # Setup progress bar
    if config.SHOW_PROGRESS_BARS:
        epoch_pbar = tqdm(range(config.MAX_EPOCHS), desc=f'Seed {seed}',
                          bar_format='{l_bar}{bar:' + str(config.PROGRESS_BAR_WIDTH) + '}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                          ncols=100)
    else:
        epoch_pbar = range(config.MAX_EPOCHS)

    for epoch in epoch_pbar:
        # === Training Phase ===
        model.train()
        train_loss = 0
        train_outputs = []
        train_labels = []

        if config.SHOW_PROGRESS_BARS:
            batch_pbar = tqdm(train_loader, desc=f'  Epoch {epoch}', leave=False,
                              bar_format='{l_bar}{bar:20}| {n_fmt}/{total_fmt}',
                              ncols=90)
        else:
            batch_pbar = train_loader

        for x1, x2, t1, t2, t3, t4, labels in batch_pbar:
            x1, x2 = x1.to(device), x2.to(device)
            t1, t2, t3, t4 = t1.to(device), t2.to(device), t3.to(device), t4.to(device)
            labels = labels.to(device)

            outputs = model(x1, mask1=None, adjoin_matrix1=None,
                            x2=x2, mask2=None, adjoin_matrix2=None,
                             t2=t2, t3=t3, t4=t4)

            loss = criterion(outputs, labels.float())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP_NORM)
            optimizer.step()

            train_loss += loss.item()
            probs = torch.sigmoid(outputs)
            train_outputs.extend(probs.detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            if config.SHOW_PROGRESS_BARS and hasattr(batch_pbar, 'set_postfix'):
                batch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Calculate training metrics
        train_loss = train_loss / len(train_loader)
        train_preds = [1 if x >= config.DECISION_THRESHOLD else 0 for x in train_outputs]
        train_acc = accuracy_score(train_labels, train_preds)

        try:
            train_auc = roc_auc_score(train_labels, train_outputs)
        except:
            train_auc = 0.0

        try:
            train_prauc = average_precision_score(train_labels, train_outputs)
        except:
            train_prauc = 0.0

        # === Validation Phase ===
        val_loss, val_acc, val_auc, val_prauc, val_sens, val_spec, _, _ = evaluate_model_with_threshold(
            model, val_loader, device, criterion, threshold=config.DECISION_THRESHOLD
        )

        # Get primary metric value for scheduler
        if config.PRIMARY_METRIC == 'prauc':
            primary_val = val_prauc
        elif config.PRIMARY_METRIC == 'auc':
            primary_val = val_auc
        elif config.PRIMARY_METRIC == 'acc':
            primary_val = val_acc
        else:
            primary_val = val_prauc

        scheduler.step(primary_val)
        current_lr = optimizer.param_groups[0]['lr']

        # Checkpoint logic - save best validation metrics
        status = ''
        if epoch >= config.MIN_CHECKPOINT_EPOCH:
            # Check if current model is better
            is_better = False
            if config.PRIMARY_METRIC == 'prauc':
                is_better = (val_prauc > best_val_prauc or 
                            (val_prauc == best_val_prauc and val_acc > best_val_acc))
            elif config.PRIMARY_METRIC == 'auc':
                is_better = (val_auc > best_val_auc or 
                            (val_auc == best_val_auc and val_acc > best_val_acc))
            elif config.PRIMARY_METRIC == 'acc':
                is_better = (val_acc > best_val_acc or 
                            (val_acc == best_val_acc and val_prauc > best_val_prauc))

            if is_better:
                best_val_prauc = val_prauc
                best_val_auc = val_auc
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_val_sens = val_sens
                best_val_spec = val_spec
                best_epoch = epoch
                stopping_monitor = 0
                torch.save(model.state_dict(), checkpoint_path)
                status = 'BEST'
                if config.VERBOSE:
                    tqdm.write(f'Epoch {epoch:3d} | New best! Val PRAUC: {val_prauc:.4f}, Val Acc: {val_acc:.4f}')
            else:
                stopping_monitor += 1
                status = 'NO_IMPROVE'
        else:
            status = 'WARMUP'

        # CSV logging (epoch-wise)
        with open(csv_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f'{train_loss:.6f}', f'{train_acc:.4f}', 
                             f'{train_auc:.4f}', f'{train_prauc:.4f}',
                             f'{val_loss:.6f}', f'{val_acc:.4f}', 
                             f'{val_auc:.4f}', f'{val_prauc:.4f}',
                             f'{val_sens:.4f}', f'{val_spec:.4f}', 
                             f'{current_lr:.2e}', status])

        if config.SHOW_PROGRESS_BARS and hasattr(epoch_pbar, 'set_postfix'):
            epoch_pbar.set_postfix({'val_prauc': f'{val_prauc:.3f}', 'val_spec': f'{val_spec:.3f}'})

        if stopping_monitor >= config.PATIENCE:
            if config.VERBOSE:
                print(f"\n[INFO] Early stop at epoch {epoch}. Best Val PRAUC: {best_val_prauc:.4f} at epoch {best_epoch}.")
            break

    # === Evaluation Phase ===
    if best_epoch == -1:
        print(f"\n[WARNING] Seed {seed}: No checkpoint saved.")
        return None

    if config.VERBOSE:
        print(f"\n[EVAL] Seed {seed} | Best Epoch: {best_epoch} | Val PRAUC: {best_val_prauc:.4f}")

    model.load_state_dict(torch.load(checkpoint_path))

    # Test evaluation
    test_loss, test_acc, test_auc, test_prauc, test_sens, test_spec, test_labels, test_outputs = evaluate_model_with_threshold(
        model, test_loader, device, criterion, threshold=config.DECISION_THRESHOLD
    )

    # Convert probabilities to binary predictions before passing to score()
    test_predictions = [1 if x >= config.DECISION_THRESHOLD else 0 for x in test_outputs]
    supplementary_results = score(test_labels, test_predictions)

    # Return comprehensive results with all metrics from score()
    results = {
        'seed': seed,
        'best_epoch': best_epoch,
        # Validation metrics at best epoch
        'val_loss': best_val_loss,
        'val_acc': best_val_acc,
        'val_auc': best_val_auc,
        'val_prauc': best_val_prauc,
        'val_sensitivity': best_val_sens,
        'val_specificity': best_val_spec,
        # Test metrics - all from score() function
        'test_loss': test_loss,
        'test_se': supplementary_results.get('sensitivity', 0.0),
        'test_sp': supplementary_results.get('specificity', 0.0),
        'test_mcc': supplementary_results.get('mcc', 0.0),
        'test_acc': supplementary_results.get('accuracy', 0.0),
        'test_auc': supplementary_results.get('accuracy_score', 0.0),
        'test_f1': supplementary_results.get('f1_score', 0.0),
        'test_ba': supplementary_results.get('balanced_accuracy', 0.0),
        'test_prauc': supplementary_results.get('pr_auc', 0.0),
        'test_ppv': supplementary_results.get('ppv', 0.0),
        'test_npv': supplementary_results.get('npv', 0.0),
    }

    if config.VERBOSE:
        print(f"[RESULT] Seed {seed} | Test PRAUC: {results['test_prauc']:.4f} | Test AUC: {results['test_auc']:.4f} | Test Acc: {results['test_acc']:.4f}")

    return results


def main():
    """Main training loop"""
    os.environ['CUDA_VISIBLE_DEVICES'] = Config.CUDA_DEVICE
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Using: {device}")

    with open(Config.RESULTS_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'seed', 'best_epoch',
            'val_loss', 'val_acc', 'val_auc', 'val_prauc', 'val_sensitivity', 'val_specificity',
            'test_loss', 'test_se', 'test_sp', 'test_mcc', 'test_acc', 'test_auc', 
            'test_f1', 'test_ba', 'test_prauc', 'test_ppv', 'test_npv'
        ])

    print(f"\n{'='*80}")
    print(f"Results will be saved to: {Config.RESULTS_CSV}")
    print(f"Epoch logs will be saved to: {Config.LOG_DIR}/")
    print(f"Primary Metric: {Config.PRIMARY_METRIC.upper()}")
    print(f"{'='*80}\n")


    seed=Config.SEED
    print(f"\n{'#'*80}")
    print(f"# SEED {seed}")
    print(f"{'#'*80}")
    result = train_one_seed(seed, Config, device)

    with open(Config.RESULTS_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            result['seed'],
            result['best_epoch'],
            f"{result['val_loss']:.6f}",
            f"{result['val_acc']:.4f}",
            f"{result['val_auc']:.4f}",
            f"{result['val_prauc']:.4f}",
            f"{result['val_sensitivity']:.4f}",
            f"{result['val_specificity']:.4f}",
            f"{result['test_loss']:.6f}",
            f"{result['test_se']:.4f}",
            f"{result['test_sp']:.4f}",
            f"{result['test_mcc']:.4f}",
            f"{result['test_acc']:.4f}",
            f"{result['test_auc']:.4f}",
            f"{result['test_f1']:.4f}",
            f"{result['test_ba']:.4f}",
            f"{result['test_prauc']:.4f}",
            f"{result['test_ppv']:.4f}",
            f"{result['test_npv']:.4f}"
        ])


if __name__ == "__main__":
    main()
