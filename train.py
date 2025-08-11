import os
os.environ['CUDA_VISIBLE_DEVICES']='0'  # Use only one GPU

import warnings
warnings.filterwarnings('ignore')

import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np
from torch.utils.data import random_split
from adc_dataset import *
from ADCNet import *
from utils import *
import torch.nn as nn

def main(seed_range):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    logger = CSVLogger_my([
                            'Seed', 'Accuracy', 'At_Epoch',
                            'tp', 'tn', 'fn', 'fp', 
                            'sensitivity', 'specificity', 'mcc',
                            'accuracy_score', 'f1_score', 'balanced_accuracy', 
                            'pr_auc', 'ppv', 'npv'
                        ], 'logs/valid_logs/main_backup_valid_log.csv')

        
    csv_path = "data/main.csv"
    pkl_paths = ["Embeddings/antibinder_heavy.pkl","Embeddings/Light.pkl","Embeddings/Antigen.pkl"]
    score_dict = {}

    seed_iterator = tqdm(seed_range, position=0, leave=True, desc="Seeds")
    
    for seed in seed_iterator:
        print(f"Seed {seed}")
        dataset = AB_Data(csv_path, pkl_paths)
        
        train_loader,valid_loader, test_loader = dataset.get_dataloaders(
            seed,train_ratio=0.8,valid_ratio =0.1,test_ratio=0.1, batch_size=20, use_ddp=False)
        
        
        medium = {'name':'Medium','num_layers': 6, 'num_heads': 8, 'd_model': 256,'path':'medium_weights','addH':True}
        arch = medium
        trained_epoch = 20
        num_layers = arch['num_layers']
        num_heads = arch['num_heads']
        d_model = arch['d_model']
        addH = arch['addH']
        dff = d_model * 2
        vocab_size = 18
        dense_dropout = 0.1
        
        model = PredictModel(num_layers=num_layers,
                          d_model=d_model,
                          dff=dff,
                          num_heads=num_heads,
                          vocab_size=vocab_size).to(device)
        
        # Load checkpoint without rank checking
        checkpoint = torch.load('ckpts/modified_model.pth', map_location=device)
        print(model.load_state_dict(checkpoint, strict=False))
        
        # Remove DDP wrapper
        # model = DDP(model,device_ids= [local_rank])  # REMOVED
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCEWithLogitsLoss()
        print('Starting training...')
        
        epoch_iterator = tqdm(range(150), position=1, leave=True, desc="Epochs")
        
        for epoch in epoch_iterator:
            model.train()  # Set model to training mode
            res = []
            train_outputs = []
            train_labels = []
            
            batch_iterator = tqdm(train_loader, position=2, leave=False, desc="Batches")
            
            for x1, x2, t1, t2, t3, t4, labels in batch_iterator:
                x1 = x1.to(device)
                x2 = x2.to(device)
                t1 = t1.to(device)
                t2 = t2.to(device)
                t3 = t3.to(device)
                t4 = t4.to(device)
                labels = labels.to(device)
                
                outputs = model(x1, mask1=None, adjoin_matrix1=None, 
                              x2=x2, mask2=None, adjoin_matrix2=None,
                              t1=t1, t2=t2, t3=t3, t4=t4)
                
                act_outputs = torch.sigmoid(outputs)
                act_outputs = [1 if x >= 0.5 else 0 for x in act_outputs]
                train_outputs.extend(act_outputs)
                train_labels.extend(labels.cpu())
                
                loss = criterion(outputs, labels.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                res.append(loss.item())
            
            print(f"Epoch: {epoch+1}, Avg_loss: {np.mean(res)}")
            print(f"Training Accuracy: {accuracy_score(train_outputs, train_labels)}")

            # Save checkpoint every 10 epochs
            if not (epoch+1) % 10:
                torch.save(model.state_dict(), f"ckpts/Abform_{epoch+1}.pth")

        # Validation after training
        best_ckpt, best_results = model_validate(test_loader,valid_loader, device, num_layers, d_model, dff, num_heads, vocab_size)
        
        score_dict[seed] = (best_ckpt, best_results)
        logger.log([
            seed,                                    # 'Seed'
            best_results['accuracy_score'],          # 'Accuracy' 
            best_ckpt,                              # 'At_Epoch'
            best_results['tp'],                     # 'tp'
            best_results['tn'],                     # 'tn'
            best_results['fn'],                     # 'fn'
            best_results['fp'],                     # 'fp'
            best_results['sensitivity'],            # 'sensitivity'
            best_results['specificity'],            # 'specificity'
            best_results['mcc'],                    # 'mcc'
            best_results['accuracy_score'],         # 'accuracy_score'
            best_results['f1_score'],               # 'f1_score'
            best_results['balanced_accuracy'],      # 'balanced_accuracy'
            best_results['pr_auc'],                 # 'pr_auc'
            best_results['ppv'],                    # 'ppv'
            best_results['npv']                     # 'npv'
        ])

def model_validate(test_dataloader,valid_dataloader, device, num_layers, d_model, dff, num_heads, vocab_size):
    
    logger = CSVLogger_my([
        "epoch", "tp", "tn", "fn", "fp", "se", "sp", "mcc", "acc", 
        "auc_roc_score", "F1", "BA", "prauc", "PPV", "NPV"
    ], "logs/log.csv")
    
    model = PredictModel(num_layers=num_layers,
                          d_model=d_model,
                          dff=dff,
                          num_heads=num_heads,
                          vocab_size=vocab_size).to(device)

    ckpt_list = [x for x in range(1, 151) if x % 10 == 0]
    print(ckpt_list)
    
    best_acc = 0
    best_ckpt = 0
    best_results = {}
    
    for ckpt in ckpt_list:
        print(f"Loading checkpoint: {ckpt}")
        model.load_state_dict(torch.load(f"ckpts/Abform_{ckpt}.pth", map_location=device))
        model.eval()  # Set model to evaluation mode
        
        with torch.no_grad():
            main_labels = []
            main_outputs = []
            
            for x1, x2, t1, t2, t3, t4, label in tqdm(valid_dataloader, desc=f"Validating epoch {ckpt}"):
                x1 = x1.to(device)
                x2 = x2.to(device)
                t1 = t1.to(device)
                t2 = t2.to(device)
                t3 = t3.to(device)
                t4 = t4.to(device)
                
                outputs = model(x1, mask1=None, adjoin_matrix1=None, 
                              x2=x2, mask2=None, adjoin_matrix2=None,
                              t1=t1, t2=t2, t3=t3, t4=t4)
                outputs = torch.sigmoid(outputs)
                main_outputs.extend(outputs.cpu())
                main_labels.extend(label)

            results = score(main_labels, main_outputs)
            current_acc = results['accuracy_score']
            
            if current_acc > best_acc:
                best_ckpt = ckpt
                best_acc = current_acc
                best_results = results

    # Final test with best checkpoint
    print(f"Best checkpoint: {best_ckpt} with accuracy: {best_acc}")
    model.load_state_dict(torch.load(f"ckpts/Abform_{best_ckpt}.pth", map_location=device))
    model.eval()
    
    with torch.no_grad():
        main_labels = []
        main_outputs = []

        for x1, x2, t1, t2, t3, t4, label in tqdm(test_dataloader, desc="Final test"):
            x1 = x1.to(device)
            x2 = x2.to(device)
            t1 = t1.to(device)
            t2 = t2.to(device)
            t3 = t3.to(device)
            t4 = t4.to(device)
            
            outputs = model(x1, mask1=None, adjoin_matrix1=None, 
                          x2=x2, mask2=None, adjoin_matrix2=None,
                          t1=t1, t2=t2, t3=t3, t4=t4)
            outputs = torch.sigmoid(outputs)
            main_outputs.extend(outputs.cpu())
            main_labels.extend(label)

        final_results = score(main_labels, main_outputs)
            
    return best_ckpt, final_results

if __name__ == "__main__":
    main([2,7,8])
