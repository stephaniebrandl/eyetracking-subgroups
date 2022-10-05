from os.path import join, isdir
from os import listdir, makedirs
import pandas as pd
from extract_human_importance.extract_features import extract_model_importance, compare_importance, \
    extract_human_importance, extract_all_attention_first, extract_features_individual
from utils import load_subject_info, create_attribute_file, get_color, read_config
import click
from transformers import BertForMaskedLM, BertTokenizer, MT5Model, T5Tokenizer, RobertaTokenizer, RobertaModel

color_dict, lang_dict = get_color()

@click.command()
@click.option('--data', default='mecoL1')
@click.option('--modelname', default='mbert')
def main(data, modelname):
    """Calculates attention values and relative fixation and corresponding correlation values for each participant
    individually. A pandas.DataFrame is stored continuously where individual results can be found for the entire dataset
    and for each participant individually. Fixations need to be collected before running this function with
    data_extractor_x.py

    Parameters
    ----------
    data : str, default = mecoL1
        The dataset on which attention values and relative fixations are calculated.
        So far the following options are implemented:
            mecoL1: reading data of meco participants in their native language
            mecoL2: reading data of meco participants in English as language learners
            geco: reading data of geco participants in native English
            geco_nl: reading data of geco participants in native Dutch
            gecoL2: reading data of geco participants in English as language learners
    modelname : str, default = mbert
        The model used to compute attention values
        So far the following options are implemented:
            mbert, xlmr, mt5
    """

    assert data in ['mecoL1', 'mecoL2', 'geco', 'geco_nl', 'gecoL2'],\
        'function only implemented for the following datasets: [mecoL1, mecoL2, geco, geco_nl, gecoL2]'

    languages, results_dir, attention_dir, et_dir, et_orig, _, _, MODEL_NAME, importance_type = read_config(data,
                                                                                                            modelname)
    if not isdir(attention_dir):
        makedirs(attention_dir)

    if not isdir(results_dir):
        makedirs(results_dir)

    if modelname == 'mt5':
        model = MT5Model.from_pretrained(MODEL_NAME)
        tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    elif modelname == 'xlmr':
        model = RobertaModel.from_pretrained(MODEL_NAME)
        tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    elif modelname == 'mbert':
        model = BertForMaskedLM.from_pretrained(MODEL_NAME, output_attentions=True)
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    else:
        raise NotImplementedError

    et_tokens_all = {}
    lm_tokens_all = {}

    #checks if df is already available, otherwise new df is initialized
    try:
        results = pd.read_pickle(join(results_dir, f"correlation_individuals_{data}_{modelname}_{importance_type}.pkl"))
    except FileNotFoundError:
        results = pd.DataFrame(columns=['lang', 'subj', 'correlation_mean', 'correlation_std', 'lextale', 'age',
                                        'gender'])

    if data == "mecoL2":
        subject_overview = pd.read_csv(join(et_dir, "joint_ind.csv")).set_index("uniform_id")

    for language in languages:
        et_tokens_all[language] = []
        lm_tokens_all[language] = []
        subject_info = load_subject_info(language, et_orig, data)

        for file in sorted(listdir(et_dir)):
            #checks if relative fixations are accessible
            if file.startswith(language) and file.endswith('feats.csv'):
                subj = file.split("-")[0]
                if subj not in results.index:
                    #relative fixations are collected/calculated
                    try:
                        et_tokens, human_importance = extract_human_importance(subj, dir=et_dir)
                    except FileNotFoundError:
                        extract_features_individual(file, et_dir)
                        et_tokens, human_importance = extract_human_importance(subj, dir=et_dir)

                    #attention values are collected/calculated
                    try:
                        dataset_name = data if 'geco' in data else subj
                        lm_tokens, lm_importance = extract_model_importance(dataset_name, modelname, importance_type,
                                                                            dir=attention_dir)
                    except FileNotFoundError:
                        with open(et_dir + "/" + subj + "_sentences.txt", "r") as f:
                            sentences = f.read().splitlines()
                        dataset_name = data if 'geco' in data else subj
                        outfile = attention_dir + "/" + dataset_name + "_" + modelname + "_"
                        extract_all_attention_first(modelname, model, tokenizer, sentences, outfile + "attention-first.txt")
                        lm_tokens, lm_importance = extract_model_importance(dataset_name, modelname, importance_type,
                                                                            dir=attention_dir)

                    #correlation values are calculated
                    spearman_mean, spearman_std, human, lm = compare_importance(et_tokens, human_importance,
                                                                                lm_tokens, lm_importance,
                                                                                importance_type, subj,
                                                                                modelname, data)

                    create_attribute_file(file, human_importance, lm_importance, lm_tokens, attention_dir, modelname, importance_type)
                    results.loc[subj] = [language, subj, spearman_mean, spearman_std, None, None, None]

                    if data == "mecoL2" and subj in subject_overview.index:
                        results.loc[subj, "lextale"] = subject_overview.loc[subj, "lextale"]
                    if data == 'mecoL1' and 'lextale' in subject_info and subj in subject_info.index:
                        results.loc[subj, "lextale"] = subject_info.loc[subj, "lextale"]
                    results.to_pickle(join(results_dir, f"correlation_individuals_{data}_{modelname}_{importance_type}.pkl"))

    if 'meco' in data:
        #language codes in dataset are different from the ones in the paper
        results['lang'] = results['lang'].map(lang_dict)
    print(results.sort_index().groupby('lang')[['correlation_mean']].mean().T.round(2).to_latex())


if __name__ == "__main__":
    main()
