# ABFormer: A Transformer-based Model for ADC Activity Prediction

ABFormer is a novel, multi-channel deep learning framework designed to predict the activity of Antibody-Drug Conjugates (ADCs) with enhanced accuracy. The model addresses a critical limitation in existing computational approaches by incorporating contextual antibody-antigen binding interactions rather than treating ADC components as independent features.

## Overview

Antibody-Drug Conjugates represent a sophisticated therapeutic modality that combines the targeting specificity of antibodies with the cytotoxic potency of small molecule drugs. Accurate prediction of ADC activity is essential for accelerating drug development and reducing experimental costs. ABFormer significantly advances the state of the art in this domain through its innovative architecture.

## The Problem

Previous computational models for ADC activity prediction, including ADCNet, typically process the antibody, antigen, and drug components as independent features. This approach fundamentally overlooks the dynamic interactions occurring at the antibody-antigen binding interface, which are critical determinants of an ADC's specificity and subsequent cellular internalization. The absence of contextual binding information limits the predictive accuracy of these models.

## Our Solution

ABFormer addresses this limitation by integrating contextual antibody-antigen binding features directly into the prediction workflow. The model leverages a re-purposed AntiBinder module to generate high-dimensional, interaction-aware feature representations that encode the complex molecular recognition dynamics at the binding interface. This contextual approach enables the model to capture the nuanced interplay between antibody and antigen that determines ADC efficacy.

## Performance

ABFormer demonstrates substantial improvements over baseline models on the public ADCdb dataset, achieving 85.61% accuracy compared to 72.73% for ADCNet, and 84.25% balanced accuracy compared to 70.88%. The model's robust generalization capability is further evidenced by its 100% accuracy on an independent external test set comprising 21 novel ADCs.

## Architecture

ABFormer employs a multi-modal architecture that integrates five distinct input streams to generate comprehensive feature representations:

The small molecule components, specifically the linker and payload, are encoded from their isoSMILES string representations using two separate FG-BERT encoders. Each encoder consists of a 6-layer Transformer architecture that produces a 256-dimensional embedding vector capturing the chemical properties of these molecules.

The antibody-antigen binding interface represents the core innovation of ABFormer. Rather than processing the antibody heavy chain and antigen sequences independently, these sequences are fed jointly into a pre-trained AntiBinder module. This module internally leverages both sequence information from ESM-2 and structural data from IgFold to generate a single 2592-dimensional contextual embedding that represents the binding interaction holistically.

The antibody light chain sequence is processed independently through the ESM-2 protein language model, yielding a 1280-dimensional feature vector that captures the sequence characteristics of this component.

The antigen sequence is also encoded separately using ESM-2 to produce a 1280-dimensional vector, ensuring that complete sequence information is retained beyond the binding context alone.

The Drug-to-Antibody Ratio (DAR), a critical pharmacological parameter, is incorporated as a standardized and normalized scalar value.

These five feature vectors are concatenated into a single 5665-dimensional composite representation. This unified feature vector is then processed through a Multi-Layer Perceptron consisting of fully connected layers, culminating in a sigmoid activation function that produces the binary classification output

![Architecture Image](https://api.drugparadigm.com/public/images/ABFormer.png)

## Installation

Begin by cloning the repository to your local environment:

```bash
git clone https://github.com/drugparadigm/ABFormer.git
cd ABFormer
```

Create and activate the Conda environment using the provided configuration file, which will install all necessary dependencies including Python, PyTorch, and supporting libraries:

```bash
conda env create -f ABFormer_env.yml
conda activate ABFormer
```


## Usage

The repository provides several scripts for training, validation, and inference operations.

To train the model using a standard random split of the dataset:

```bash
python train.py
```

To train using the unique Antigen-Antibody pairs split, which implements Leave-Pair-Out validation to ensure no antibody-antigen pairs in the test set appear in the training set:

```bash
python train.py --unique_split
```

To run ablation studies that train the model without the AntiBinder module, thereby validating its contribution to overall performance:

```bash
python ablation.py
```

To perform 5-Fold Cross-Validation testing:

```bash
python cross_validation_test.py
```

To run inference on new data, provide your input in JSON format:

```bash
python inference.py --json_path="data.json"
```
