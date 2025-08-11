# Standard library imports
import os
from os import path

# Third-party data manipulation libraries
import pandas as pd
import numpy as np

# Scikit-learn imports for metrics
from sklearn.metrics import (
    matthews_corrcoef,
    accuracy_score, 
    f1_score,
    precision_recall_curve,
    auc
)

class CSVLogger_my:
    def __init__(self, columns, file) :
        self.columns=columns
        self.file=file
        if not self.check_header():
            self._write_header()


    def check_header(self):
        if path.exists(self.file):
            header=True
        else:
            header=False
        return header


    def _write_header(self):
        with open(self.file,"a") as f:
            string=""
            for attrib in self.columns:
                string+="{},".format(attrib)
            string=string[:len(string)-1]
            string+="\n"
            f.write(string)
        return self


    def log(self,row) :
        if len(row)!=len(self.columns) :
            raise Exception("Mismatch between row vector and number of columns in logger")
        with open(self.file, "a") as f:
            string=""
            for attrib in row:
                string+="{},".format(attrib)
            string=string[:len(string)-1]
            string+="\n"
            f.write(string)




def score(y_true, y_pred_prob):
    y_pred = [1 if x >= 0.5 else 0 for x in y_pred_prob]

    y_true_np = np.array([x.cpu() for x in y_true]).astype(int)
    y_pred_np = np.array(y_pred).astype(int)
    y_pred_prob_np = np.array([x.cpu() for x in y_pred_prob])  # Keep as float for precision_recall

    tp = np.sum((y_pred_np == 1) & (y_true_np == 1))
    tn = np.sum((y_pred_np == 0) & (y_true_np == 0))
    fp = np.sum((y_pred_np == 1) & (y_true_np == 0))
    fn = np.sum((y_pred_np == 0) & (y_true_np == 1))

    # Sensitivity (Recall)
    se = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Specificity
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    mcc = matthews_corrcoef(y_true_np, y_pred_np)
    auc_roc = accuracy_score(y_true_np, y_pred_np)
    f1 = f1_score(y_true_np, y_pred_np)
    ba = (se + sp) / 2
    precision, recall, _ = precision_recall_curve(y_true_np, y_pred_prob_np)
    prauc = auc(recall, precision)

    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # NPV
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    score_dict = {
        'tp': tp,
        'tn': tn,
        'fn': fn,
        'fp': fp,
        'sensitivity': se,
        'specificity': sp,
        'mcc': mcc,
        'accuracy': acc,
        'accuracy_score': auc_roc,
        'f1_score': f1,
        'balanced_accuracy': ba,
        'pr_auc': prauc,
        'ppv': ppv,
        'npv': npv
    }

    return score_dict




def metrics_from_log(log_path):
  df = pd.read_csv(log_path)


  # Assuming you have already loaded your DataFrame as df
  metrics = {
      "tp": np.max(df['tp']),
      "tn": np.max(df['tn']),
      "fn": np.max(df['fn']),
      "fp": np.max(df['fp']),
      "se": np.max(df['se']),
      "sp": np.max(df['sp']),
      "mcc": np.max(df['mcc']),
      "acc": np.max(df['acc']),
      "test_acc": np.max(df['auc_roc_score']),
      "f1": np.max(df['F1']),
      "ba": np.max(df['BA']),
      "prauc": np.max(df['prauc']),
      "ppv": np.max(df['PPV']),
      "npv": np.max(df['NPV']),
  }

  metrics_mean = {
      "tp": np.mean(df['tp']),
      "tn": np.mean(df['tn']),
      "fn": np.mean(df['fn']),
      "fp": np.mean(df['fp']),
      "se": np.mean(df['se']),
      "sp": np.mean(df['sp']),
      "mcc": np.mean(df['mcc']),
      "acc": np.mean(df['acc']),
      "test_acc": np.mean(df['auc_roc_score']),
      "f1": np.mean(df['F1']),
      "ba": np.mean(df['BA']),
      "prauc": np.mean(df['prauc']),
      "ppv": np.mean(df['PPV']),
      "npv": np.mean(df['NPV']),
  }

  # print("\nMean values:")
  # for name, value in metrics_mean.items():
  #     print(f"{name}: {value}")

  # print()
  # print("Best values:")
  # for name, value in metrics.items():
  #     print(f"{name}: {value}")

  return metrics, metrics_mean

