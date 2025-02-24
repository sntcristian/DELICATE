# ELITE: Entity Linking in Italian for long-Tail Entities

This repository contains the software used for implementing and testing ELITE. 
For the implementation of the candidate retrieval component, ELITE relies on the [BLINK](https://github.com/facebookresearch/BLINK) library from Facebook.
An image of the ELITE architecture is available below:

<img src="docs/elite.jpg" alt="drawing" width="700"/>

Pre-trained models for Named Entity Recognition and Entity Linking, as well as the ELITE's knowledge base are 
available on [Zenodo]().

## Use ELITE

### 1.  Create conda environment and install requirements

Recommended: Python=3.9
```
conda create -n elite -y python=3.9 && conda activate elite
pip install -r requirements.txt
```

### 2. Download Models and Setup Config

The ELITE software is released along with models for Named Entity Recognition and Entity Linking, trained on the 
Italian corpus for Named Entity Recognition and Linking [*ENEIDE*](). Trained models can be downloaded from [Zenodo]().

Once the models are downloaded, it is important to setup the [config.json](config.json) file with the correct paths 
to the models and knowledge base.

## ELITE Tutorial Notebook

A demo of the different functionalities of ELITE is available in a [Jupyter Notebook](./tutorial_elite.ipynb).


## Use ELITE from Command Line

To use the ELITE pipeline on a specific input it is possible to use the following command: 
```
python run_nel.py --config_file config.json --output_dir "./test" --publication_date "1826" --text "Il poeta italiano 
Giacomo Leopardi nacque a Recanati." 
```