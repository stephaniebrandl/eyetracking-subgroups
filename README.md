## Every word counts: A multilingual analysis of individual human alignment with model attention

This repository contains code to the paper "Every word counts: A multilingual analysis of individual human alignment with model attention." accepted at AACL 2022. 

Please refer to the paper for further details.

### 1 Re-running the code
#### Data
You can read more about the MECO project [here](https://meco-read.com). We are using version 1.2 (MecoL1) 
and version 1.1 (MecoL2). For GeCo English L1 we use the file 
`EnglishMaterial_corrected.csv` from https://github.com/beinborn/relative_importance.

#### Experiments
* `data_extractor_meco.py` or `data_extractor_geco.py` to compute the eyetracking features, it might be easier to run the bash script `data_extractor.sh`
* `analyze_individuals.py` to compute attention values (this might take a while) and correlate them with the eyetracking
* `skipping-speed.py,` `pos.py,` `lextale.py` perform the analyses we showed in the paper and create Figure 1-3

### 2 Adding models or new eye-tracking datasets
When including new models, make sure that you consider individual tokenization wrt separation and end of sentence 
token. You might need to adapt `word_piece_separators` in `tokenization_utils.merge_subwords`. Currently the code is 
written for PyTorch models from huggingface.

You can also include new eye-tracking datasets, you would therefore need to eiter write a new dataloader, 
similar to `data_extractor_meco.py` or `data_extractor_geco.py` or adapt the current ones.

### 3 Folder structure
- **extract_human_fixations**: code to extract the relative fixation duration from two eye-tracking corpora and average it over all subjects. The two corpora are [GECO](https://expsy.ugent.be/downloads/geco/) and [MECO](https://meco-read.com). 

- **extract_model_importance**: code to extract attention-based importance from transformer-based language models. 

- **figs**: contains all plots.

- **results**: contains intermediate results. 

### 4 Requirements
Before you start, make sure all the paths in `config.yaml` are set. You need to set `et_orig` where the raw
eye-tracking data is stored (see `extract_human_importance/README.md` for further details). Also make sure you have installed all required Python packages as listed in 
`requirements.txt`.  To run the POS-tagging, you would need to download respective models on spacy. 
We used Python 3.9.10.

### 5 Acknowledgments
Parts of the code are based on https://github.com/beinborn/relative_importance