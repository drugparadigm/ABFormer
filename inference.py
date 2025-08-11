import os
import sys
from ADCNet import *
from AntiBinder.model import *
from AntiBinder.antigen_antibody_emb import *
from AntiBinder.cfg_ab import * 
from torch.utils.data import DataLoader
import esm
import torch
from igfold import IgFoldRunner
import torch.nn.functional as F
import pandas as pd 
from abnumber import Chain
from flask import request
import json
from transformers import BertConfig, BertTokenizer
from transformers.tokenization_utils import Trie
from transformers.models.bert.tokenization_bert import *
torch.serialization.add_safe_globals([
    BertConfig,
    BertTokenizer,
    Trie,
    BasicTokenizer,
    WordpieceTokenizer,
])


esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
device = torch.device('cuda:1')
esm_model = esm_model.to(device)

igfold = IgFoldRunner()


def split_heavy_chain_regions(sequence, scheme='chothia'):
    """
    Splits a heavy chain sequence into regions using abnumber.
    Returns a dict with keys: H-FR1, H-CDR1, H-FR2, H-CDR2, H-FR3, H-CDR3, H-FR4
    """
    REGION_MAP = {
        'H-FR1': 'fr1_seq',
        'H-CDR1': 'cdr1_seq',
        'H-FR2': 'fr2_seq',
        'H-CDR2': 'cdr2_seq',
        'H-FR3': 'fr3_seq',
        'H-CDR3': 'cdr3_seq',
        'H-FR4': 'fr4_seq'
    }
    output = {}
    try:
        ab_chain = Chain(sequence, scheme=scheme)
        if ab_chain.chain_type == 'H':
            for region, attr in REGION_MAP.items():
                output[region] = getattr(ab_chain, attr) or ''
        else:
            for region in REGION_MAP:
                output[region] = ''
    except Exception as error:
        for region in REGION_MAP:
            output[region] = ''
    return output

def universal_padding(sequence, max_length):
    if len(sequence) > max_length:
        return sequence[:max_length]
    else:
        return torch.cat([sequence,torch.zeros(max_length-len(sequence))]).long()


def func_padding_for_esm(sequence, max_length):
    if len(sequence) > max_length:
        return sequence[:max_length]
    else:
        return sequence+'<pad>'*(max_length-len(sequence)-2)
    

def region_indexing(data):
    HF1 = [1 for _ in range(len(data['H-FR1']))]
    HCDR1 = [3 for _ in range(len (data['H-CDR1']))]
    HF2 = [1 for _ in range(len(data['H-FR2']))]
    HCDR2 = [4 for _ in range(len(data['H-CDR2']))]
    HF3 = [1 for _ in range(len(data['H-FR3']))]
    HCDR3 = [5 for _ in range(len(data['H-CDR3']))]
    HF4 = [1 for _ in range(len(data['H-FR4']))]
    vh = torch.tensor(list(HF1+HCDR1+HF2+HCDR2+HF3+HCDR3+HF4))
    vh = universal_padding(vh,149)

    return vh


str2num = {'<pad>':0 ,'H': 1, 'C': 2, 'N': 3, 'O': 4, 'S': 5, 'F': 6, 'Cl': 7, 'Br': 8, 'P':  9,
         'I': 10,'Na': 11,'B':12,'Se':13,'Si':14,'<unk>':15,'<mask>':16,'<global>':17}
num2str =  {i:j for j,i in str2num.items()}

def numerical_smiles(smiles):
    atoms_list, _ = smiles2adjoin(smiles, explicit_hydrogens=True)  # ignore adjacency matrix
    atoms_list = ["<global>"] + atoms_list
    nums_list = [str2num.get(atom, str2num['<unk>']) for atom in atoms_list]
    x = np.array(nums_list, dtype=np.int64)
    return x


def get_esm2_embeddings(data):

    

    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]

    # Get mean embeddings for each sequence
    embeddings = []
    for i, (_, seq) in enumerate(data):
        emb = token_representations[i, 1:len(seq)+1]  # Remove CLS & padding
        emb = emb.mean(dim=0, keepdim=True)           # Mean pooling
        embeddings.append(emb)

    return embeddings


def process_antibody_antigen(sequences, new_dict, antigen_structure):

    emb = igfold.embed(sequences=sequences)
    structure = emb.structure_embs.detach().cpu()

    regions = ['H-FR1', 'H-CDR1', 'H-FR2', 'H-CDR2', 'H-FR3', 'H-CDR3', 'H-FR4']
    boundaries = [len(sum([[x] if isinstance(x, str) else x for x in [new_dict[r] for r in regions[:i]]], [])) for i in range(1, 8)]


    part1 = structure.index_select(1, torch.arange(boundaries[4]))
    part2_mean = structure.index_select(1, torch.arange(boundaries[4], boundaries[5])).mean(dim=1).view(structure.size(0), 1, -1)
    part3 = structure.index_select(1, torch.arange(boundaries[5], boundaries[6]))

    structure = torch.cat([part1, part2_mean, part3], dim=1)
    padding_len = 149 - structure.size(1)
    if padding_len > 0:
        padding = torch.zeros(1, padding_len, structure.size(-1)).float()
        structure = torch.cat([structure, padding], dim=1).squeeze(0)

    antibody = torch.tensor([AminoAcid_Vocab[aa] for aa in new_dict['t1']])
    antibody = universal_padding(sequence=antibody, max_length=149)
    at_type = region_indexing(new_dict)

    antigen = torch.tensor([
    AminoAcid_Vocab[aa]
    for aa in new_dict['t3'].strip()  # removes leading/trailing whitespace including '\n'
    if aa in AminoAcid_Vocab          # ensures only valid amino acids are processed
])

    antigen = universal_padding(sequence=antigen, max_length=1024)

    L, D = antigen_structure.shape
    if L >= 1024:
        antigen_structure = antigen_structure[:1024, :]
    else:
        antigen_structure = F.pad(antigen_structure, (0, 0, 0, 1024 - L))

    antibody_set = [antibody.unsqueeze(0), at_type.unsqueeze(0), structure.unsqueeze(0)]
    antigen_set = [antigen.unsqueeze(0), antigen_structure.unsqueeze(0)]

    return antibody_set, antigen_set


def main():

    ADC_model = PredictModel().to(device)
    ADC_model.load_state_dict(torch.load("ckpts/ABForm_wt.pth",map_location=device))
    ADC_model.eval()
    AntiBinder_model = antibinder().to(device)
    AntiBinder_model.load_state_dict(torch.load("ckpts/AntiBinder_Weights.pth",map_location=device))
    AntiBinder_model.eval()
    Antigen_Sequence = """MVSWGRFICLVVVTMATLSLARPSFSLVEDTTLEPEEPPTKYQISQPEVYVAAPGESLEVRCLLKDAAVISWTKDGVHLGPNNRTVLIGEYLQIKGATPRDSGLYACTASRTVDSETWYFMVNVTDAISSGDDEDDTDGAEDFVSENSNNKRAPYWTNTEKMEKRLHAVPAANTVKFRCPAGGNPMPTMRWLKNGKEFKQEHRIGGYKVRNQHWSLIMESVVPSDKGNYTCVVENEYGSINHTYHLDVVERSPHRPILQAGLPANASTVVGGDVEFVCKVYSDAQPHIQWIKHVEKNGSKYGPDGLPYLKVLKAAGVNTTDKEIEVLYIRNVTFEDAGEYTCLAGNSIGISFHSAWLTVLPAPGREKEITASPDYLEIAIYCIGVFLIACMVVTVILCRMKNTTKKPDFSSQPAVHKLTKRIPLRRQVTVSAESSSSMNSNTPLVRITTRLSSTADTPMLAGVSEYELPEDPKWEFPRDKLTLGKPLGEGCFGQVVMAEAVGIDKDKPKEAVTVAVKMLKDDATEKDLSDLVSEMEMMKMIGKHKNIINLLGACTQDGPLYVIVEYASKGNLREYLRARRPPGMEYSYDINRVPEEQMTFKDLVSCTYQLARGMEYLASQKCIHRDLAARNVLVTENNVMKIADFGLARDINNIDYYKKTTNGRLPVKWMAPEALFDRVYTHQSDVWSFGVLMWEIFTLGGSPYPGIPVEELFKLLKEGHRMDKPANCTNELYMMMRDCWHAVPSQRPTFKQLVEDLDRILTLTTNEEYLDLSQPLEQYSPSYPDTRSSCSSGDDSVFSPDPMPYEPCLPQYPHINGSVKT
"""

    Heavy_Chain_Sequence = "QVQLLESGGGLVQPGGSLRLSCAASGFTFSDYAMSWVRQAPGKGLEWVSTIEGDSNYIEYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARERTYSSAFDYWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKRVEPKSCDKTHTCPPCPAPELLGGPSVFLFPPKPKDTLMISRTPEVTCVVVDVSHEDPEVKFNWYVDGVEVHNAKTKPREEQYNSTYRVVSVLTVLHQDWLNGKEYKCKVSNKALPAPIEKTISKAKGQPREPQVYTLPPSREEMTKNQVSLTCLVKGFYPSDIAVEWESNGQPENNYKTTPPVLDSDGSFFLYSKLTVDKSRWQQGNVFSCSVMHEALHNHYTQKSLSLSPGK"
    Light_Chain_Sequence = "DIQMTQSPSSLSASVGDRVTITCRASQDISSDLNWYQQKPGKAPKLLIYDASNLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCHQWYSTLYTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"

    Payload_IsoSmile = "C[C@@H]1[C@@H]2C[C@]([C@@H](/C=C/C=C(/CC3=CC(=C(C(=C3)OC)Cl)N(C(=O)C[C@@H]([C@]4([C@H]1O4)C)OC(=O)[C@H](C)N(C)C(=O)CCS)C)\C)OC)(NC(=O)O2)O"
    
    Linker_IsoSmile = "C1CC(=O)N(C1=O)OC(=O)CCSSC2=CC=CC=N2"
    DAR_value = 3.5
    
    DAR_value = torch.tensor(DAR_value, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)
    new_dict = {'x1': Linker_IsoSmile, 'x2':Payload_IsoSmile,'t1':Heavy_Chain_Sequence,'t2': Light_Chain_Sequence,'t3':Antigen_Sequence,'t4':DAR_value}
    heavy_regions = split_heavy_chain_regions(Heavy_Chain_Sequence)

    new_dict = new_dict | heavy_regions
    data = [('protein 1', Antigen_Sequence),('protein2',Light_Chain_Sequence)]
    
    antigen_embed, lightchain_embed = [e.mean(dim=0, keepdim=True) for e in get_esm2_embeddings(data)]
    
    padded_antigen = func_padding_for_esm(Antigen_Sequence, 1024)
    with torch.no_grad():
        padded_tokens = batch_converter([('antigen', padded_antigen)])[2].to(device)
        antigen_structure = esm_model(padded_tokens.squeeze(1), repr_layers=[33], return_contacts=True)['representations'][33].squeeze(0)

    heavy_emb_seq = new_dict['H-FR1'] + new_dict['H-CDR1'] + new_dict['H-FR2'] + new_dict['H-CDR2'] + new_dict['H-FR3'] + new_dict['H-CDR3']+new_dict['H-FR4']
    sequences = {'H' : heavy_emb_seq}
    antibody_set, antigen_set =  process_antibody_antigen(sequences, new_dict, antigen_structure)
    antibody_set = (antibody_set[0].to(device), antibody_set[1].to(device), antibody_set[2].to(device))
    antigen_set = (antigen_set[0].to(device), antigen_set[1].to(device))

    var_heavy_emb = AntiBinder_model(antibody_set, antigen_set)
    linker_encoded = numerical_smiles(Linker_IsoSmile)
    payload_encoded = numerical_smiles(Payload_IsoSmile)
    x1 = F.pad(torch.tensor(linker_encoded), (0, 195 - len(linker_encoded))).unsqueeze(0).to(device).int()
    x2 = F.pad(torch.tensor(payload_encoded), (0, 184 - len(payload_encoded))).unsqueeze(0).to(device).int()
    
    t1, t2, t3 = var_heavy_emb.to(device).float(), lightchain_embed.to(device).float(), antigen_embed.to(device).float()
    t4 = torch.tensor(DAR_value).reshape(1).to(device).float()
    
    print(x1.shape, x2.shape, t1.shape, t2.shape, t3.shape, t4.shape)
    
    output = ADC_model(
        x1, mask1=None, adjoin_matrix1=None,
        x2=x2, mask2=None, adjoin_matrix2=None,
        t1=t1, t2=t2, t3=t3, t4=t4
    )
    output = torch.sigmoid(output)
    return int(output.item())

if _name_ == "_main_":
    main()
