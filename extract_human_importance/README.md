You can either run the entire pipeline from scratch on the raw eye-tracking data or use the files in `./data/` which 
have been computed by `data_extractor_meco|geco`. In both cases you need to download the data.

#### GeCo  
Download the GECO corpus from [https://expsy.ugent.be/downloads/geco/]()
and set the paths in `config.yaml`  
* `geco: filename:` `path to MonolingualReadingData.xlsx`
* `geco_nl: filename:` `path to L1ReadingData.xlsx`
* `geco_nl: material:` `path to DutchMaterials.xlsx`
* `gecoL2: filename: ` `path to L2ReadingData.xlsx`


#### MeCo  
Download the MeCo corpus from [https://osf.io/3527a/]() for the L1 data (we used version 1.2) and from 
[https://osf.io/q9h43/]() for the L2 data (we used version 1.1) and set the paths in `config.yaml` for `et_orig`.

#### Running everything from scratch  
You can run `data_extractor.sh` to compute relative fixations and the corresponding features. Before that you need to 
save `joint_data_trimmed.rda` as `joint_data_trimmed.csv` and store it in `extract_human_importance/data/meco/L1/` and 
`extract_human_importance/data/meco/L2/` respectively. Otherwise, you can also

#### Extract gaze features
Run `data_extractor.sh` to compute gaze features. You should therefore comment out 
* `read_geco_file(filename, material, et_dir)` (l.131 in `data_extractor_geco.py`) and
* `read_meco_file(filename, language, data_path, et_dir)` (l.228 in `data_extractor_meco.py`)

You can then continue the analysis by running `analyze_individuals.py`.