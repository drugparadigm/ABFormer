# ABFormer
ABFormer is a multi-channel deep learning framework designed to predict the activity of antibody-drug conjugates (ADCs). It integrates contextual antibody-antigen binding representations with molecular features of small molecules (linkers and payloads), protein sequences, and Drug-Antibody Ratio (DAR) values.


-------------

# ABFormer Architecture

![Architecture Image](https://api.drugparadigm.com/public/images/ABFormer.png)

ABFormer extends the original ADCNet by incorporating pretrained antibody-antigen binding context embeddings from AntiBinder component. These are derived using a specialized submodule that combines sequence (via ESM-2) and structural (via IgFold) information. ABFormer fuses biological and chemical features into a unified representation, which is processed by a classifier to predict ADC activity.

-------------

## Input Processing
### Protein Sequence Encoding
**ESM-2 Pre-trained Language Model** processes three protein components:
- Light chain sequences  
- Antigen sequences

Each protein sequence is encoded into high-dimensional embeddings of 1,280 dimensions, capturing rich semantic representations of protein structure and function.
### Antibody-Antigen Binding Representation
ABFormer uses a pretrained binding encoder to generate contextual embeddings from antibody and antigen sequences:

#### Sequence Pathway:

Antibody heavy-chain sequences are processed using amino acid (AA) indexing and region indexing (e.g., CDRs).
Antigen sequences are tokenized and embedded using ESM-2, a protein language model.

#### Structure Pathway:

IgFold generates 3D structural features for the antibody, which are projected into vector space.
The sequence and structure outputs are combined via element-wise addition and projected into a final binding-aware embedding.
Bidirectional attention is used to model antibody-to-antigen and antigen-to-antibody interactions for rich feature learning.

The output is a contextual binding embedding, which replaces the individual heavy/light chain vectors used in the original ADCNet.


### Small Molecule Preprocessing
**SMILES Representation Processing** handles chemical components:
- Payload molecules (cytotoxic drugs)
- Linker molecules (connecting antibody to payload)

The preprocessing pipeline:
1. Tokenizes SMILES strings into molecular tokens
2. Processes each component (payload and linker) individually

-------------

### FG-BERT Molecular Encoders
Two specialized molecular encoders, fine-tuned on ADC datasets, process the chemical components:

**Architecture per Encoder:**
- **6 Transformer Layers** - Each containing:
  - Multi-Head Attention mechanisms
  - Feed-Forward Networks
  - Layer Normalization
  - Dropout regularization
 *******
 ### Fully Connected Layers

- It is then passed in to Fully Connected Layers to get final predicted output of 0 or 1

--------------

## Model Inputs
1. Heavy Chain Sequence
2. Light Chain Sequence
3. Antigen Sequence
4. Payload Isosmile
5. Linker Isosmile
6. DAR Value

---------------------

## Model Ouptut:
 0 [Inactive ADC]
 
 1 [Active ADC]

----------------------------

 ## Script Commands:
 ### Run training on random split dataset:
`python train.py `

 
 ### Run training on unique Antgen-Antibody pairs Split: 
 `python train.py --unique_split `

 ### Run ablation variant training:
 `python ablation.py`

 ### Run 5Fold Cross Validation Test:
 `python cross_validation_test.py`

 ### Run Inference:
 `python inference.py --json_path="data.json"`
