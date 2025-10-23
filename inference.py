import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import esm
from igfold import IgFoldRunner
from abnumber import Chain
from transformers import BertConfig, BertTokenizer
from transformers.tokenization_utils import Trie
from transformers.models.bert.tokenization_bert import *
import warnings

from ADCNet import *
from AntiBinder.antibinder_model import *
from AntiBinder.antigen_antibody_emb import *
from AntiBinder.cfg_ab import *

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

torch.serialization.add_safe_globals([
    BertConfig, BertTokenizer, Trie, BasicTokenizer, WordpieceTokenizer,
])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
esm_model = esm_model.to(device)
esm_model.eval()

igfold = IgFoldRunner()

# --- Helper Functions ---
def split_heavy_chain_regions(sequence, scheme='chothia'):
    REGION_MAP = {
        'H-FR1': 'fr1_seq', 'H-CDR1': 'cdr1_seq', 'H-FR2': 'fr2_seq',
        'H-CDR2': 'cdr2_seq', 'H-FR3': 'fr3_seq', 'H-CDR3': 'cdr3_seq',
        'H-FR4': 'fr4_seq'
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
    seq_len = len(sequence)
    if seq_len >= max_length:
        return sequence[:max_length]
    else:
        return sequence + '<pad>' * (max_length - seq_len)

def region_indexing(data):
    regions = ['H-FR1', 'H-CDR1', 'H-FR2', 'H-CDR2', 'H-FR3', 'H-CDR3', 'H-FR4']
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

str2num = {'<pad>':0, 'H': 1, 'C': 2, 'N': 3, 'O': 4, 'S': 5, 'F': 6, 'Cl': 7, 'Br': 8, 'P': 9,
           'I': 10,'Na': 11,'B':12,'Se':13,'Si':14,'<unk>':15,'<mask>':16,'<global>':17}

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
        emb = token_representations[i, 1:len(seq) + 1].mean(dim=0, keepdim=True)
        embeddings.append(emb)
    return embeddings

def process_antibody_antigen(sequences, new_dict, antigen_structure):
    emb = igfold.embed(sequences=sequences)
    structure = emb.structure_embs.detach().cpu()

    heavy_seq = ''.join(new_dict[r] for r in ['H-FR1', 'H-CDR1', 'H-FR2', 'H-CDR2', 'H-FR3', 'H-CDR3', 'H-FR4'])
    
    padding_len = 149 - len(heavy_seq)
    if padding_len < 0:
        structure = structure[:, :149, :]
    else:
        padding = torch.zeros(1, padding_len, structure.size(-1), dtype=torch.float)
        structure = torch.cat([structure, padding], dim=1)

    antibody = torch.tensor([AminoAcid_Vocab.get(aa, 0) for aa in new_dict['t1']])
    antibody = universal_padding(sequence=antibody, max_length=149)
    at_type = region_indexing(new_dict)
    
    antigen_seq_str = new_dict['t3'].strip()
    antigen = torch.tensor([AminoAcid_Vocab.get(aa, 0) for aa in antigen_seq_str if aa in AminoAcid_Vocab])
    antigen = universal_padding(sequence=antigen, max_length=1024)

    L, D = antigen_structure.shape
    if L > 1024:
        antigen_structure = antigen_structure[:1024, :]
    else:
        antigen_structure = F.pad(antigen_structure, (0, 0, 0, 1024 - L))

    antibody_set = [antibody.unsqueeze(0), at_type.unsqueeze(0), structure]
    antigen_set = [antigen.unsqueeze(0), antigen_structure.unsqueeze(0)]
    return antibody_set, antigen_set

def DAR_feature(dar_value):
    mean_value = 3.86845977
    variance_value = 1.569108443
    std_deviation = variance_value ** 0.5
    if not isinstance(dar_value, torch.Tensor):
        dar_value = torch.tensor(dar_value, dtype=torch.float32)
    standardized = (dar_value - mean_value) / std_deviation
    normalized = (standardized - 0.8) / (12 - 0.8)

    return normalized


def main():
    with open(json_path, 'r') as f:
        data_json = json.load(f)
    
    # Extract sequences / values from JSON
    Heavy_Chain_Sequence = data_json['Heavy_Chain_Sequence']
    Light_Chain_Sequence = data_json['Light_Chain_Sequence']
    Antigen_Sequence = data_json['Antigen_Sequence']
    Linker_IsoSmile = data_json['Linker_IsoSmile']
    Payload_IsoSmile = data_json['Payload_IsoSmile']
    DAR_value = torch.tensor(data_json['DAR_value'], dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)

    # Load ADC model
    ADC_model = PredictModel().to(device)
    ADC_model.load_state_dict(torch.load("ckpts/ADC_best.pth", map_location=device))
    ADC_model.eval()

    # Load AntiBinder model
    checkpoint_path = "model_weights/AntiBinder_Weights.pth"
    state_dict = torch.load(checkpoint_path, map_location=device)
    if all(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    
    AntiBinder_model = antibinder().to(device)
    AntiBinder_model.load_state_dict(state_dict, strict=False)
    AntiBinder_model.eval()

    # Prepare input dictionary
    new_dict = {
        'x1': Linker_IsoSmile,
        'x2': Payload_IsoSmile,
        't1': Heavy_Chain_Sequence,
        't2': Light_Chain_Sequence,
        't3': Antigen_Sequence,
        't4': DAR_value
    }

    heavy_regions = split_heavy_chain_regions(Heavy_Chain_Sequence)
    new_dict.update(heavy_regions)

    data = [('protein1', Antigen_Sequence), ('protein2', Light_Chain_Sequence)]
    antigen_embed, lightchain_embed = [e.mean(dim=0, keepdim=True) for e in get_esm2_embeddings(data)]

    padded_antigen = func_padding_for_esm(Antigen_Sequence, 1024)
    with torch.no_grad():
        padded_tokens = batch_converter([('antigen', padded_antigen)])[2].to(device)
        antigen_structure = esm_model(
            padded_tokens.squeeze(1),
            repr_layers=[33],
            return_contacts=True
        )['representations'][33].squeeze(0)

    # Heavy chain concatenation
    heavy_emb_seq = new_dict['H-FR1'] + new_dict['H-CDR1'] + new_dict['H-FR2'] + new_dict['H-CDR2'] + new_dict['H-FR3'] + new_dict['H-CDR3'] + new_dict['H-FR4']
    sequences = {'H': heavy_emb_seq}
    antibody_set, antigen_set = process_antibody_antigen(sequences, new_dict, antigen_structure)
    antibody_set = [a.to(device) for a in antibody_set[:3]]
    antigen_set = [a.to(device) for a in antigen_set[:2]]

    var_heavy_emb = AntiBinder_model(antibody_set, antigen_set)

    linker_encoded = numerical_smiles(Linker_IsoSmile)
    payload_encoded = numerical_smiles(Payload_IsoSmile)
    x1 = F.pad(torch.tensor(linker_encoded), (0, 195 - len(linker_encoded))).unsqueeze(0).to(device).int()
    x2 = F.pad(torch.tensor(payload_encoded), (0, 184 - len(payload_encoded))).unsqueeze(0).to(device).int()

    t1, t2, t3 = var_heavy_emb.to(device).float(), lightchain_embed.to(device).float(), antigen_embed.to(device).float()
    t4 = DAR_value.reshape(1).to(device).float()

    output = ADC_model(
        x1, mask1=None, adjoin_matrix1=None,
        x2=x2, mask2=None, adjoin_matrix2=None,
        t1=t1, t2=t2, t3=t3, t4=t4
    )
    output = torch.sigmoid(output)
    print("ADC Activity:", output.item())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict ADC activity from JSON input.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to input JSON file")
    args = parser.parse_args()
    main(args.json_path)


