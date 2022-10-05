import pandas as pd
import numpy as np
from os.path import join, isdir
from os import makedirs
from typing import List
import seaborn as sns
import yaml
import spacy
from extract_human_importance.data_extractor_meco import average_features


def read_config(data, modelname):
    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    languages = config[data]['languages']
    results_dir = config[data]['results_dir']
    attention_dir = config[data]['att_dir']
    et_dir = config[data]['et_dir']
    et_orig = config[data]['et_orig'] if 'et_orig' in config[data] else None
    filename = config[data]['filename'] if 'filename' in config[data] else None
    material = config[data]['material'] if 'material' in config[data] else None
    importance_type = config['importance_type']
    MODEL_NAME = config[modelname]['MODEL_NAME']

    if not isdir(attention_dir):
        makedirs(attention_dir)

    if not isdir(results_dir):
        makedirs(results_dir)

    if not isdir(et_dir):
        makedirs(et_dir)

    return languages, results_dir, attention_dir, et_dir, et_orig, filename, material, MODEL_NAME, importance_type


def get_color():
    '''https://www.colorhexa.com/'''
    all_colors = sns.color_palette("tab20", 13)
    lw = 2
    languages = ["ee", "fi", "ge", "en", "du", "it", "gr", "he", "tr", "ru", "sp", "ko", "no"]
    lang_dict = {"ee": "et",
                 "ge": "de",
                 "du": "nl",
                 "gr": "el",
                 "sp": "es",
                 "en": "en",
                 "it": "it",
                 "he": "he",
                 "tr": "tr",
                 "ru": "ru",
                 "ko": "ko",
                 "no": "no",
                 "fi": "fi"
                 }
    color_dict = {lang: all_colors[ii] for ii, lang in enumerate(sorted(lang_dict.values()))}
    return color_dict, lang_dict


def standardize_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Normalize columns to mean=0 and std=1. New column names will contain the
    suffix "_normalized".
    Args:
        df (pd.DataFrame): dataframe on which columns are to be normalized
        cols (list of str): names of columns to normalize
    Returns:
        pd.DataFrame: `df` with normalized columns
    """
    for col in cols:
        new_col = col + '_normalized'
        df[new_col] = (df[col] - np.mean(df[col])) / (np.std(df[col]))

    return df


def load_subject_info(language, data_path, data='mecoL2'):
    if data == 'mecoL2':
        if language == "ee":
            subject_info = pd.read_excel(
                join(data_path, f"primary data/individual differences data/leap-q/{language}.xlsx"),
                sheet_name=0, header=1)
            subject_info = subject_info.set_index("Code")

        elif language == "tr":
            subject_info = pd.read_excel(
                join(data_path, f"primary data/individual differences data/leap-q/{language}.xlsx"),
                sheet_name=0)
            subject_info = subject_info.set_index("participant")

        elif language in ["ge", "en", "it", "he", "ru", "sp", "no", "fi", "du", "gr"]:
            header = 1 if language in ["du", "gr"] else 0
            subject_info = pd.read_excel(
                join(data_path, f"primary data/individual differences data/leap-q/{language}.xlsx"),
                sheet_name=0, header=header)
            subject_info = subject_info.set_index("uniform_id")

        elif language == "ko":
            return None
        else:
            raise NotImplementedError()

        if 'Gender' in subject_info:
            subject_info = subject_info.rename(columns={'Gender': 'gender'})
        elif 'Sex' in subject_info:
            subject_info = subject_info.rename(columns={'Sex': 'gender'})
        elif 'sex' in subject_info:
            subject_info = subject_info.rename(columns={'sex': 'gender'})
        if 'Age' in subject_info:
            subject_info = subject_info.rename(columns={'Age': 'age'})
        elif 'Age (years)' in subject_info:
            subject_info = subject_info.rename(columns={'Age (years)': 'age'})

    elif data == 'mecoL1':
        header = 1 if language in ["du", "gr", "ko"] else 0
        subject_info_leap = pd.read_excel(
            join(data_path, f"primary data/individual differences data/data by language/{language}/{language}-leap.xlsx"),
            sheet_name=0, header=header)
        subject_info_leap = subject_info_leap.rename(columns={'Code': 'subject'})
        subject_info_diff = pd.read_excel(
            join(data_path, f"primary data/individual differences data/data by language/{language}/{language}-ind-diff.xlsx"),
            sheet_name=0)
        subject_info_diff = subject_info_diff.rename(columns={'Code': 'subject'})
        subject_info = pd.merge(subject_info_leap, subject_info_diff, on='subject')
        subject_info = subject_info.set_index('subject')
        subject_info = subject_info.rename(columns={'age ': 'age', 'Age': 'age', 'Age ': 'age', 'age_x': 'age',
                                                    'Gender': 'gender', 'sex': 'gender', 'Sex': 'gender',
                                                    'Gender_x': 'gender', 'LexizeScore': 'lextale',
                                                    'LexITA_index': 'lextale', 'lex': 'lextale',
                                                    'LexTALE_Dutch': 'lextale'})

    else:
        subject_info = pd.DataFrame() #create empty data frame so that the code also runs for GeCo files

    return subject_info


def merge_geco():
    out_dir_geco = "results_geco"
    out_dir_geco_nl = "results_geconl"
    out_dir_gecoL2 = "results_gecoL2"
    L1 = pd.read_pickle(join(out_dir_geco, f"correlation_individuals_{modelname}_{importance_type}.pkl"))
    L1['lang'] = 'L1'

    du = pd.read_pickle(join(out_dir_geco_nl, f"correlation_individuals_{modelname}_{importance_type}.pkl"))
    du['lang'] = 'du'

    L2 = pd.read_pickle(join(out_dir_gecoL2, f"correlation_individuals_{modelname}_{importance_type}.pkl"))
    L2['lang'] = 'L2'

    return pd.concat([L1, L2, du])


def create_attribute_file(file, human_importance, lm_importance, lm_tokens, attention_dir, modelname, importance_type):
    human_importance_tmp = [attr for human, lm in zip(human_importance, lm_importance)
                            for attr in human.tolist() if len(human) == len(lm)]
    lm_importance_tmp = [attr for human, lm in zip(human_importance, lm_importance)
                         for attr in lm.tolist() if len(human) == len(lm)]
    word_pos = [ii for (tokens, human, lm) in zip(lm_tokens, human_importance, lm_importance)
                for ii, _ in enumerate(tokens) if len(human) == len(lm)]
    word_pos = ['last' if word_pos[ii + 1] == 0 else pos for ii, pos in enumerate(word_pos[:-1])]
    word_pos.append('last')
    try:
        subject_importance = pd.read_pickle(join(attention_dir, file.split("-")[0] + modelname + '_attributes.pkl'))
    except FileNotFoundError:
        tokens_tmp = [token for tokens, (human, lm) in zip(lm_tokens, zip(human_importance, lm_importance))
                      for token in tokens if len(human) == len(lm)]
        sen_len = [len(tokens) for (tokens, human, lm) in zip(lm_tokens, human_importance, lm_importance)
                   for _ in tokens if len(human) == len(lm)]
        sen_num = [ii for ((ii, tokens), human, lm) in zip(enumerate(lm_tokens), human_importance, lm_importance)
                   for _ in tokens if len(human) == len(lm)]
        subject_importance = pd.DataFrame(
            data={'relfix': human_importance_tmp, importance_type: lm_importance_tmp,
                  'tokens': tokens_tmp, 'sen_len': sen_len, 'sen_num': sen_num})
    subject_importance['word_pos'] = word_pos

    if importance_type not in subject_importance.columns:
        subject_importance[importance_type] = lm_importance_tmp

    subject_importance.to_pickle(join(attention_dir, file.split("-")[0] + '_' + modelname + '_attributes.pkl'))


def extract_pos(data, modelname, language_model):
    _, results_dir, attention_dir, _, _, _, _, _, importance_type = read_config(data, modelname)
    results = pd.read_pickle(join(results_dir, f"correlation_individuals_{data}_{modelname}_{importance_type}.pkl"))

    averaged_dict = {}
    df_all = pd.DataFrame(columns=["lang", "relfix", importance_type, "pos", "tokens"])
    for language, subject_info in results.groupby("lang"):
        if data == 'mecoL1' and language in ["he", "tr", "ee"]:
            continue
        elif data == 'mecoL2':
            nlp = spacy.load(language_model['en'])
        else:
            nlp = spacy.load(language_model[language])
        sent_dict = {}
        attr_dict = {}
        print("language: ", language)
        subject_count = 0
        for subject in subject_info.index:
            try:
                df = pd.read_pickle(join(attention_dir, subject + '_' + modelname + '_attributes.pkl'))
            except FileNotFoundError:
                print(subject, "not found")
                continue
            subject_count += 1
            for sent_num, sent_data in df.groupby('sen_num'):
                relfix_vals = list(sent_data['relfix'])
                attr_vals = list(sent_data[importance_type])
                if " ".join(map(str, list(sent_data['tokens']))) not in sent_dict:
                    sent_dict[" ".join(map(str, list(sent_data['tokens'])))] = [list(sent_data['tokens']),
                                                                                [relfix_vals]]
                    attr_dict[" ".join(map(str, list(sent_data['tokens'])))] = [list(sent_data['tokens']),
                                                                                [attr_vals]]
                else:
                    sent_dict[" ".join(map(str, list(sent_data['tokens'])))][1].append(relfix_vals)

        averaged_dict[language] = average_features(sent_dict, subject_count)
        df_tmp = pd.DataFrame(columns=["lang", "relfix", importance_type, "pos", "tokens"])
        ii = 0
        for sentence, relfix in averaged_dict[language].items():
            tokens = sentence.split(" ")
            for itoken, token in enumerate(tokens):
                doc = nlp(token)
                pos = [token.pos_ for token in doc]
                df_tmp.loc[ii] = [language, relfix[1][itoken], attr_dict[sentence][1][0][itoken], pos[0], token]
                ii += 1

        df_all = pd.concat([df_all, df_tmp.dropna(subset=['relfix', importance_type])])
    color_dict, lang_dict = get_color()
    df_all['lang'] = df_all['lang'].map(lang_dict)
    df_all.to_pickle(f"./{results_dir}/df_{data}_pos.pkl")