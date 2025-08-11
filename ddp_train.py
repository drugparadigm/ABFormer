import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5'  # Use multiple GPUs

import warnings
warnings.filterwarnings('ignore', message='No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user.')

import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np
from torch.utils.data import random_split
from adc_dataset import *
from ADCNet import *
from utils import *
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def main(seed_range):
    
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    setup(local_rank, world_size)

    device = torch.device(f"cuda:{local_rank}")
    
    if local_rank == 0:
        logger = CSVLogger_my([
                                'Seed', 'Accuracy', 'At_Epoch',
                                'tp', 'tn', 'fn', 'fp', 
                                'sensitivity', 'specificity', 'mcc',
                                'accuracy_score', 'f1_score', 'balanced_accuracy', 
                                'pr_auc', 'ppv', 'npv'
                            ], 'logs/valid_logs/main_backup_valid_log.csv')

    csv_path = "main.csv"
    pkl_paths = ["Embeddings/antibinder_heavy.pkl","Embeddings/Light.pkl","Embeddings/Antigen.pkl"]
    score_dict = {}

    seed_iterator = tqdm(seed_range, position=0, leave=True, desc="Seeds") if local_rank == 0 else seed_range
    
    for seed in seed_iterator:
        if local_rank == 0:
            print(f"Seed {seed}")
        
        dataset = AB_Data(csv_path, pkl_paths)
        
        # Use DDP version of data loading
        train_loader, valid_loader, test_loader, train_sampler = dataset.get_dataloaders(
            seed, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, 
            batch_size=20, use_ddp=True)
        
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
        
        # Load checkpoint with proper rank handling
        if local_rank == 0:
            checkpoint = torch.load('ckpts/modified_model.pth', map_location=f'cuda:{local_rank}')
            print(model.load_state_dict(checkpoint, strict=False))
        else:
            checkpoint = torch.load('ckpts/modified_model.pth', map_location=f'cuda:{local_rank}')
            model.load_state_dict(checkpoint, strict=False)

        dist.barrier()  # Synchronize after loading
        
        # Wrap model with DDP
        model = DDP(model, device_ids=[local_rank])
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCEWithLogitsLoss()
        
        if local_rank == 0:
            print('Starting training...')
        
        epoch_iterator = tqdm(range(150), position=1, leave=True, desc="Epochs") if local_rank == 0 else range(150)
        
        for epoch in epoch_iterator:
            # Set epoch for proper shuffling
            train_sampler.set_epoch(epoch)
            
            model.train()
            res = []
            train_outputs = []
            train_labels = []
            
            batch_iterator = tqdm(train_loader, position=2, leave=False, desc="Batches") if local_rank == 0 else train_loader
            
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
            
            if local_rank == 0:
                print(f"Epoch: {epoch+1}, Avg_loss: {np.mean(res)}")
                print(f"Training Accuracy: {accuracy_score(train_outputs, train_labels)}")

            # Save checkpoint only from rank 0, every 10 epochs
            if local_rank == 0 and not (epoch+1) % 10:
                torch.save(model.module.state_dict(), f"ckpts/Abform_{epoch+1}.pth")

            dist.barrier()  # Synchronize after each epoch

        # Validation after training - only on rank 0
        if local_rank == 0:
            best_ckpt, best_results = model_validate(test_loader, valid_loader, local_rank, 
                                                    num_layers, d_model, dff, num_heads, vocab_size)
            
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

        dist.barrier()  # Final synchronization
                        
    cleanup()

def model_validate(test_dataloader, valid_dataloader, local_rank, num_layers, d_model, dff, num_heads, vocab_size):
    device = torch.device(f'cuda:{local_rank}')
    
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
    
    # STEP 1: Use validation set for model selection
    for ckpt in ckpt_list:
        print(f"Validating checkpoint: {ckpt}")
        model.load_state_dict(torch.load(f"ckpts/Abform_{ckpt}.pth", map_location=device))
        model.eval()
        
        with torch.no_grad():
            valid_labels = []
            valid_outputs = []
            
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
                valid_outputs.extend(outputs.cpu())
                valid_labels.extend(label)

            results = score(valid_labels, valid_outputs)
            current_acc = results['accuracy_score']
            
            if current_acc > best_acc:
                best_ckpt = ckpt
                best_acc = current_acc

    # STEP 2: Final test with best checkpoint
    print(f"Final testing with best checkpoint: {best_ckpt}")
    model.load_state_dict(torch.load(f"ckpts/Abform_{best_ckpt}.pth", map_location=device))
    model.eval()
    
    with torch.no_grad():
        test_labels = []
        test_outputs = []

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
            test_outputs.extend(outputs.cpu())
            test_labels.extend(label)

        final_results = score(test_labels, test_outputs)
            
    return best_ckpt, final_results

if __name__ == "__main__":
    main([2, 7, 8])
