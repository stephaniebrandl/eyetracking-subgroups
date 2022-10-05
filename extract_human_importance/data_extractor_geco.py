from os import makedirs, listdir
from os.path import join, isfile, isdir
import pandas as pd
import numpy as np
import yaml
import click
# Extract relative fixation duration from the GECO corpus


def filter_geco(df, threshold=5):
    df_new = pd.DataFrame(columns=df.columns)
    for sent_num, sentence in df.groupby('sentence_id'):
        if len(sentence) <= threshold:
            sentence['TRT'] = float(0)
            sentence['skip'] = float(0)
            sentence['relFix'] = float(0)
        df_new = pd.concat([df_new, sentence])
    return df_new


def read_geco_file(filename, material, et_dir):
    print("Reading file for GECO: ", filename)
    try:
        data = pd.read_pickle(filename[:-5]+"_reduced.pkl")
    except FileNotFoundError:
        data = pd.read_excel(filename, usecols="A,E,F,I,J,K,BB", na_filter=False)
        data.to_pickle(filename[:-5]+"_reduced.pkl")

    if material.endswith('xlsx'):
        sentence_info = pd.read_excel(material, na_filter=False)
        sentence_info = sentence_info.rename(columns={'IA_ID': "WORD_ID"})
    else:
        sentence_info = pd.read_csv(material, na_filter=False)

    subjects = pd.unique(data['PP_NR'].values)
    sentences = pd.unique(sentence_info['SENTENCE_ID'].values)
    flat_word_index = 0
    for subj in subjects:
        print(subj)
        if not isfile(join(et_dir, subj+"-relfix-feats.csv")):
            subj_data_orig = data.loc[data['PP_NR'] == subj]
            df_subj = pd.DataFrame(columns=['sentence_id', 'word_id', 'word_id_orig', 'word', 'TRT', 'relFix'])
            for j, sent in enumerate(sentences):
                word_ids = sentence_info["WORD_ID"].loc[sentence_info['SENTENCE_ID'] == sent].values
                tokens = sentence_info["WORD"].loc[sentence_info['SENTENCE_ID'] == sent].values

                for k, (w, id) in enumerate(zip(tokens, word_ids)):
                    trt = subj_data_orig['WORD_TOTAL_READING_TIME'].loc[subj_data_orig['WORD_ID'] == id].values

                    if trt.size > 0:
                        trt = 0 if trt == "." else trt
                    else:
                        trt = 0

                    df_subj.loc[flat_word_index] = [j, k, id, str(w).lower(), float(trt), 0]
                    flat_word_index += 1

            i = 0
            max_sent = df_subj['sentence_id'].max()
            while i < max_sent + 1:
                sent_data = df_subj.loc[df_subj['sentence_id'] == i]
                try:
                    # min-max scale feature  values
                    x = [float(s)/sum(sent_data['TRT'].values) for s in sent_data['TRT'].values]
                    df_subj.loc[df_subj['sentence_id'] == i, 'relFix'] = x
                    if sum(sent_data['TRT'].values) > 0:
                        df_subj.loc[df_subj['sentence_id'] == i, 'skip'] = \
                            df_subj.loc[df_subj['sentence_id'] == i, 'TRT'].map(lambda a: 1 if a == 0 else 0)
                except ValueError:
                    print(sent_data)
                i += 1
            # write CSV files for each subject
            df_subj = filter_geco(df_subj) #remove short sentences
            df_subj.to_csv(join(et_dir, subj+"-relfix-feats.csv"))

    print("ALL DONE.")


def extract_features(dir):
    # join results from all subjects
    sent_dict ={}
    for file in sorted(listdir(dir)):
        if file.startswith("pp") and file.endswith('.csv'):
            print("Reading files for subj ", file)
            subj_data = pd.read_csv(join(dir, file), delimiter=',')
            max_sent = subj_data['sentence_id'].max()
            print(max_sent, " sentences")

            # join words in sentences
            i = 0
            while i < max_sent:
                sent_data = subj_data.loc[subj_data['sentence_id'] == i]
                if " ".join(map(str, list(sent_data['word']))):
                    relfix_vals = list(sent_data['relFix'])
                    if " ".join(map(str, list(sent_data['word']))) not in sent_dict:
                        sent_dict[" ".join(map(str, list(sent_data['word'])))] = [list(sent_data['word']), [relfix_vals]]
                    else:
                        sent_dict[" ".join(map(str, list(sent_data['word'])))][1].append(relfix_vals)
                i += 1

    # average feature values for all subjects
    averaged_dict = {}

    for sent, features in sent_dict.items():
        avg_rel_fix = np.nanmean(np.array(features[1]), axis=0)
        if len(features[0]) > 1:
            averaged_dict[sent] = [features[0], avg_rel_fix]
    print(len(averaged_dict), " total sentences.")

    out_file_text = open(join(dir, "geco_sentences.txt"), "w")
    out_file_relFix = open(join(dir, "geco_relfix_averages.txt"), "w")
    for sent, feat in averaged_dict.items():
        print(sent, file=out_file_text)
        print(", ".join(map(str, feat[1])), file=out_file_relFix)


@click.command()
@click.option('--data', default='geco')
def main(data):

    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    et_dir = config[data]['et_dir']
    filename = config[data]['filename']
    material = config[data]['material']

    if not isdir(et_dir):
        makedirs(et_dir)

    read_geco_file(filename, material, et_dir)
    extract_features(et_dir)


if __name__ == "__main__":
    main()
