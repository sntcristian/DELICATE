# DELICATE: Diachronic Entity LInking using Class And Temporal Evidence

This repository contains the software used for implementing and testing DELICATE, an Entity Linking system trained on historical documents in Italian, designed for being more sensitive to the context of provenance of a document than general-purpose Entity Linking models. An image of the DELICATE architecture is available below:

<img src="docs/delicate.jpg" alt="drawing" width="700"/>

For the implementation of the candidate retrieval component, DELICATE relies on the [BLINK](https://github.com/facebookresearch/BLINK) library from Facebook.

## Setup Environment


Since this software uses an old version of FAISS, Python=3.9 is recommended.

```
conda create -n delicate -y python=3.9 && conda activate delicate
pip install -r requirements.txt
```


## DELICATE Tutorial Notebook

A demo of the different functionalities of DELICATE is available in a [Jupyter Notebook](./tutorial_delicate.ipynb).


## Use DELICATE for Inference

To use the DELICATE pipeline on a specific input it is possible to use the following command: 
```
python pipeline.py --config_file "configs/config_dz.json" --output_dir "./test" --publication_date "1826" --text "Il poeta di Recanati Giacomo Leopardi scrisse le Operette Morali." 
```

## Train NER Model

Example:
```
python train_NER.py --dataset_path "../ENEIDE/DZ/v0.1/json_data" --output_dir "ner_model"
```


## Train Candidate Reranker

Example:
```
python train.py --dataset_path "../ENEIDE/DZ/v0.1/" --models_dir "./DELICATE_models" --block_size 50 --negatives 10 --output_dir "./"
```

## Run DELICATE on ENEIDE Dataset

Example:
```
python run_delicate.py --documents "paragraphs_test.csv" --annotations "annotations_test.csv" --config "configs/config_amd.json" ...
```

### 2. Download Pre-trained Models

Trained models and dataset will be available soon!