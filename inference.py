import os
import sys
import json
import warnings
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import esm
from igfold import IgFoldRunner
from abnumber import Chain
from transformers import BertConfig, BertTokenizer
from transformers.tokenization_utils import Trie
from transformers.models.bert.tokenization_bert import *
from ADCNet import *
from AntiBinder.antibinder_model import *
from AntiBinder.antigen_antibody_emb import *
from AntiBinder.cfg_ab import *

# ========== ENVIRONMENT CONFIG ==========
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

torch.serialization.add_safe_globals([
    BertConfig, BertTokenizer, Trie, BasicTokenizer, WordpieceTokenizer,
])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ========== LOAD MODELS ==========
esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
esm_model = esm_model.to(device)
esm_model.eval()
igfold = IgFoldRunner()

# ========== HELPER FUNCTIONS ==========
def split_heavy_chain_regions(sequence, scheme='chothia'):
    REGION_MAP = {
        'H-FR1': 'fr1_seq', 'H-CDR1': 'cdr1_seq', 'H-FR2': 'fr2_seq',
        'H-CDR2': 'cdr2_seq', 'H-FR3': 'fr3_seq', 'H-CDR3': 'cdr3_seq', 'H-FR4': 'fr4_seq'
    }
    output = {region: '' for region in REGION_MAP}
    ab_chain = Chain(sequence, scheme=scheme)
    if ab_chain.chain_type == 'H':
        for region, attr in REGION_MAP.items():
            output[region] = getattr(ab_chain, attr) or ''
    return output

def universal_padding(sequence, max_length):
    seq_len = len(sequence)
    if seq_len > max_length:
        return sequence[:max_length]
    else:
        padding = torch.zeros(max_length - seq_len, dtype=torch.long)
        return torch.cat([sequence, padding])

def func_padding_for_esm(sequence, max_length):
    return sequence[:max_length] if len(sequence) >= max_length else sequence + '<pad>' * (max_length - len(sequence))

def region_indexing(data):
    regions = ['H-FR1','H-CDR1','H-FR2','H-CDR2','H-FR3','H-CDR3','H-FR4']
    region_map = {'FR': 1, 'CDR1': 3, 'CDR2': 4, 'CDR3': 5}
    indices = []
    for region in regions:
        region_type = region.split('-')[1]
        if 'CDR' in region_type:
            indices.extend([region_map[region_type]] * len(data[region]))
        else:
            indices.extend([region_map['FR']] * len(data[region]))
    vh = torch.tensor(indices)
    return universal_padding(vh, 149)

str2num = {'<pad>':0, 'H': 1, 'C': 2, 'N': 3, 'O': 4, 'S': 5, 'F': 6, 'Cl': 7, 'Br': 8,
           'P': 9, 'I': 10,'Na': 11,'B':12,'Se':13,'Si':14,'<unk>':15,'<mask>':16,'<global>':17}

def numerical_smiles(smiles):
    import re
    atom_pattern = r'Cl|Br|[CNOPcnoBSIFPH]|[Na]|[Se]|[Si]'
    atoms_list = re.findall(atom_pattern, smiles)
    atoms_list = ["<global>"] + atoms_list
    nums_list = [str2num.get(atom, str2num['<unk>']) for atom in atoms_list]
    return np.array(nums_list, dtype=np.int64)

def get_esm2_embeddings(data):
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]
    embeddings = []
    for i, (_, seq) in enumerate(data):
        emb = token_representations[i, 1:len(seq)+1].mean(dim=0, keepdim=True)
        embeddings.append(emb)
    return embeddings

def process_antibody_antigen(sequences, new_dict, antigen_structure):
    emb = igfold.embed(sequences=sequences)
    structure = emb.structure_embs.detach().cpu()

    heavy_seq = ''.join(new_dict[r] for r in ['H-FR1','H-CDR1','H-FR2','H-CDR2','H-FR3','H-CDR3','H-FR4'])
    padding_len = 149 - len(heavy_seq)
    if padding_len < 0:
        structure = structure[:, :149, :]
    else:
        padding = torch.zeros(1, padding_len, structure.size(-1), dtype=torch.float)
        structure = torch.cat([structure, padding], dim=1)

    antibody = torch.tensor([AminoAcid_Vocab.get(aa, 0) for aa in new_dict['t1']])
    antibody = universal_padding(antibody, 149)
    at_type = region_indexing(new_dict)

    antigen_seq_str = new_dict['t3'].strip()
    antigen = torch.tensor([AminoAcid_Vocab.get(aa, 0) for aa in antigen_seq_str if aa in AminoAcid_Vocab])
    antigen = universal_padding(antigen, 1024)

    L, D = antigen_structure.shape
    if L > 1024:
        antigen_structure = antigen_structure[:1024, :]
    else:
        antigen_structure = F.pad(antigen_structure, (0, 0, 0, 1024 - L))

    antibody_set = [antibody.unsqueeze(0), at_type.unsqueeze(0), structure]
    antigen_set = [antigen.unsqueeze(0), antigen_structure.unsqueeze(0)]
    return antibody_set, antigen_set

def DAR_feature(dar_value):
    mean_value, variance_value = 3.86845977, 1.569108443
    std_deviation = variance_value ** 0.5
    if not isinstance(dar_value, torch.Tensor):
        dar_value = torch.tensor(dar_value, dtype=torch.float32)
    standardized = (dar_value - mean_value) / std_deviation
    normalized = (standardized - 0.8) / (12 - 0.8)
    return normalized

# ========== MAIN INFERENCE ==========
def main(seed,json_path):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load models
    ADC_model = PredictModel().to(device)
    ADC_model.load_state_dict(torch.load(f"ckpts/ADC_{seed}_best_model.pth", map_location=device))
    ADC_model.eval()

    checkpoint_path = "model_weights/AntiBinder_Weights.pth"
    state_dict = torch.load(checkpoint_path, map_location=device)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    AntiBinder_model = antibinder().to(device)
    AntiBinder_model.load_state_dict(new_state_dict, strict=False)
    AntiBinder_model.eval()

    # Read single JSON sample
    with open(json_path, "r") as f:
        sample = json.load(f)

    Heavy_Chain_Sequence = sample['Antibody_Heavy_Chain_Sequence']
    Light_Chain_Sequence = sample['Antibody_Light_Chain_Sequence']
    Antigen_Sequence = sample['Antigen_Sequence']
    Payload_SMILES = sample['Payload_Isosmiles']
    Linker_SMILES = sample['Linker_Isosmiles']
    DAR = DAR_feature(sample['DAR'])

    new_dict = {'t1': Heavy_Chain_Sequence, 't2': Light_Chain_Sequence, 't3': Antigen_Sequence, 't4': DAR}
    heavy_regions = split_heavy_chain_regions(Heavy_Chain_Sequence)
    new_dict.update(heavy_regions)

    data = [('antigen', Antigen_Sequence), ('light_chain', Light_Chain_Sequence)]
    antigen_embed, lightchain_embed = get_esm2_embeddings(data)

    padded_antigen = func_padding_for_esm(Antigen_Sequence, 1024)
    padded_tokens = batch_converter([('antigen', padded_antigen)])[2].to(device)
    antigen_structure = esm_model(padded_tokens, repr_layers=[33])['representations'][33].squeeze(0)

    heavy_emb_seq = ''.join(new_dict[r] for r in ['H-FR1','H-CDR1','H-FR2','H-CDR2','H-FR3','H-CDR3','H-FR4'])
    sequences = {'H': heavy_emb_seq}
    antibody_set, antigen_set = process_antibody_antigen(sequences, new_dict, antigen_structure)
    antibody_set = [t.to(device) for t in antibody_set]
    antigen_set = [t.to(device) for t in antigen_set]

    with torch.no_grad():
        var_heavy_emb = AntiBinder_model(antibody_set, antigen_set)
        linker_encoded = numerical_smiles(Linker_SMILES)
        payload_encoded = numerical_smiles(Payload_SMILES)

        x1 = F.pad(torch.from_numpy(linker_encoded), (0, 195 - len(linker_encoded))).unsqueeze(0).to(device).int()
        x2 = F.pad(torch.from_numpy(payload_encoded), (0, 184 - len(payload_encoded))).unsqueeze(0).to(device).int()

        t1 = var_heavy_emb.to(device).float()
        t2 = lightchain_embed.to(device).float()
        t3 = antigen_embed.to(device).float()
        t4 = torch.tensor([DAR], device=device).float()

        output = ADC_model(
            x1, mask1=None, adjoin_matrix1=None,
            x2=x2, mask2=None, adjoin_matrix2=None,
            t1=t1, t2=t2, t3=t3, t4=t4
        )

        predicted_prob = torch.sigmoid(output).item()
        predicted_label = 1 if predicted_prob > 0.5 else 0

    print(f"\nPredicted Probability: {predicted_prob:.4f}")
    print(f"Predicted Label: {predicted_label}")

# ========== ENTRY POINT ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ADC prediction on one JSON input sample.")
    parser.add_argument("--seed", type=int, default=49, help="Random seed for reproducibility")
    parser.add_argument("--json", type=str, default="input.json", help="Json file path containing input")
    args = parser.parse_args()
    main(args.seed,args.json)
