import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import rdkit
from rdkit import Chem
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection  import train_test_split


str2num = {'<pad>':0 ,'H': 1, 'C': 2, 'N': 3, 'O': 4, 'S': 5, 'F': 6, 'Cl': 7, 'Br': 8, 'P':  9,
         'I': 10,'Na': 11,'B':12,'Se':13,'Si':14,'<unk>':15,'<mask>':16,'<global>':17}
num2str =  {i:j for j,i in str2num.items()}


def cover_dict(path):
  file_path = path
  with open(file_path, 'rb') as file:
    data = pickle.load(file)
  tensor_dict = {key: torch.tensor(value) for key, value in data.items()}
  new_data = {i : value for i,(key,value) in enumerate(tensor_dict.items())}
  return new_data


def DAR_feature(file_path, column_name):
    df = pd.read_excel(file_path)
    column_data = df[column_name].values.reshape(-1, 1)
    mean_value = 3.86845977
    variance_value = 1.569108443
    std_deviation = variance_value**0.5
    column_data_standardized = (column_data - mean_value) / std_deviation
    normalized_data = (column_data_standardized - 0.8) / (12 - 0.8)
    data_dict = {index: torch.tensor(value, dtype=torch.float32) for index, value in zip(df.index, normalized_data.flatten())}
    return data_dict

def smiles2adjoin(smiles,explicit_hydrogens=True,canonical_atom_order=False): # Converting molecules in SMILES format to atom lists and adjacency matrices

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
      print('error')
      return None,None

    if explicit_hydrogens:
        mol = Chem.AddHs(mol)
    else:
        mol = Chem.RemoveHs(mol)

    if canonical_atom_order:
        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        mol = rdmolops.RenumberAtoms(mol, new_order)
    num_atoms = mol.GetNumAtoms()

    atoms_list = []
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        atoms_list.append(atom.GetSymbol())


    adjoin_matrix = np.eye(num_atoms)
    num_bonds = mol.GetNumBonds()

    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        adjoin_matrix[u,v] = 1.0
        adjoin_matrix[v,u] = 1.0

    return atoms_list,adjoin_matrix


def prep_data():


    df = pd.read_excel('data.xlsx')
    # Unnecessary rows: 
    alt_rows = []
    rows = []
    for index, row in df.iterrows():
        mol1 = Chem.MolFromSmiles(row['Payload Isosmiles'])
        mol2 = Chem.MolFromSmiles(row["Linker Isosmiles"])
        if mol1 is None or mol2 is None: 
            alt_rows.append(index)
        rows.append(row)
    
    sml_list1 = df["Payload Isosmiles"].tolist()
    sml_list2 = df["Linker Isosmiles"].tolist()
    Heavy_dict = cover_dict("Embeddings/Heavy.pkl")
    Light_dict = cover_dict("Embeddings/Light.pkl")
    Antigen_dict = cover_dict("Embeddings/Antigen.pkl")
    DAR_dict = DAR_feature("data.xlsx", "DAR_val")
    
    def sp_list_list(colle):
        return_list = []
        for ele in range(len(colle)):
            if ele not in alt_rows: 
                return_list.append(colle[ele])
        return np.array(return_list)
    
    def sp_dict_list(hashable):
        return_list = []
        for key, value in hashable.items():
            if key in alt_rows: 
                continue
            else:
                return_list.append(value)
        return np.array(return_list)
    
    
    sml_list1 = sp_list_list(sml_list1)
    sml_list2 = sp_list_list(sml_list2)
    print(sml_list1.shape, sml_list2.shape)
    Heavy_list = torch.tensor(sp_dict_list(Heavy_dict))
    Light_list = torch.tensor(sp_dict_list(Light_dict))
    Antigen_list = torch.tensor(sp_dict_list(Antigen_dict))
    DAR_list = torch.tensor(sp_dict_list(DAR_dict))
    
    
    str2num = {'<pad>':0 ,'H': 1, 'C': 2, 'N': 3, 'O': 4, 'S': 5, 'F': 6, 'Cl': 7, 'Br': 8, 'P':  9,
             'I': 10,'Na': 11,'B':12,'Se':13,'Si':14,'<unk>':15,'<mask>':16,'<global>':17}
    num2str =  {i:j for j,i in str2num.items()}
    
    def encode_smiles(smile):      
        vocab = str2num
        atoms_list, adjoin_matrix = smiles2adjoin(smile, explicit_hydrogens = True)
        atoms_list = ["<global>"] + atoms_list
        nums_list = [vocab.get(atom, vocab['<unk>']) for atom in atoms_list]
        return torch.tensor(nums_list, dtype = torch.int64)
    
    print(len([encode_smiles(sml) for sml in sml_list1]))
    enc_sml1 = pad_sequence([encode_smiles(sml) for sml in sml_list1],padding_value=0).T
    enc_sml2 = pad_sequence([encode_smiles(sml) for sml in sml_list2],padding_value=0).T
    
    
    
        
    enc_sml1.shape, enc_sml2.shape, Heavy_list.shape, Light_list.shape, Antigen_list.shape, DAR_list.shape
    
    
    
    
    df.columns
    labels = df["label（100nm）"].to_list()
    labels = [labels[ele] for ele in range(len(labels)) if ele not in alt_rows]
    labels = np.array(labels)
    labels = torch.tensor(labels,dtype = torch.long)
    labels.shape
    
    print()
    from torch.utils.data import TensorDataset
    
    dataset = TensorDataset(enc_sml1, enc_sml2,Heavy_list,Light_list, Antigen_list, DAR_list,labels)
    x1, x2, t1, t2, t3, t4, y = dataset.tensors
    x1_train, x1_val, \
    x2_train, x2_val, \
    t1_train, t1_val, \
    t2_train, t2_val, \
    t3_train, t3_val, \
    t4_train, t4_val, \
    y_train, y_val = train_test_split(
        x1, x2, t1, t2, t3, t4, y, 
        test_size=0.1, random_state=42
    )
    
    train_dataset = TensorDataset(x1_train, x2_train, t1_train, t2_train, t3_train, t4_train, y_train)
    test_dataset   = TensorDataset(x1_val, x2_val, t1_val, t2_val, t3_val, t4_val, y_val)


    
    # dataloader = DataLoader(dataset, batch_size = 1,shuffle = False,pin_memory = True, sampler = DistributedSampler(dataset))
    # batch = next(iter(dataloader))
    # x1, x2, t1, t2, t3,t4,y_true = batch
    return train_dataset,test_dataset

def scaled_dot_product_attention(q, k, v, mask=None, adjoin_matrix=None):
    # Matrix multiplication with transposed k
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))  # (batch_size, num_heads, seq_len, seq_len)

    # Scale by square root of key dimension
    dk = k.shape[-1]
    scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(dk, dtype=torch.float32))

    # Apply mask (if given)
    if mask is not None:
        scaled_attention_logits += mask * -1e9

    # Apply adjacency matrix (if given)
    if adjoin_matrix is not None:
        scaled_attention_logits += adjoin_matrix

    # Compute attention weights
    attention_weights = F.softmax(scaled_attention_logits, dim=-1)

    # Multiply by values tensor
    output = torch.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len, depth)

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
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))  # ✅ Fixed this line

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
        self.fc1 = nn.Linear(5665, d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(d_model,d_model)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(d_model, 1)

    def forward(self, x1, x2, t1, t2, t3, t4, adjoin_matrix1=None, mask1=None, adjoin_matrix2=None, mask2=None, training=False):
        x1, attention_weights1 = self.encoder(x1, training=training, mask=mask1, adjoin_matrix=adjoin_matrix1)
        x1 = x1[:, 0, :] 

        x2, attention_weights2 = self.encoder(x2, training=False, mask=mask2, adjoin_matrix=adjoin_matrix2)
        x2 = x2[:, 0, :]

        if t4.dim() == 1:
            t4 = t4.unsqueeze(1)


        x = torch.cat((x1, x2, t1,t2,t3, t4), dim=1)  # [B, 4353]
        x = self.fc1(x)  # [B, d_model]
        x = self.dropout1(x)
        x = self.fc2(x)  # [B, 1]
        x = self.dropout2(x)
        x = self.fc3(x)
        x = x.squeeze(1)
        return x  # [B] - so BCEWithLogitsLoss works directly
