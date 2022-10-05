import pandas as pd
import numpy as np
import yaml
import click
from os import listdir, makedirs
from os.path import join, isfile, isdir

corrupted_sentences = []

def average_features(sent_dict, subject_count):
    # average feature values for all subjects

    averaged_dict= {}
    for sent, features in sent_dict.items():
        avg_rel_fix = np.nanmean(np.array(features[1]), axis=0)
        if len(features[0]) > 1:
            averaged_dict[sent] = [features[0], avg_rel_fix]
    print(len(averaged_dict), " total sentences from ", subject_count, " subjects.")

    return averaged_dict


def read_meco_file(filename, language, data_path, out_dir):

    #reads meco files and stores relative fixation files
    print("Reading file for MECO: ", filename)

    # Make sure that this is available
    if not isfile(filename):
        raise FileNotFoundError(
            f"you need to store joint_data_trimmed.rda as joint_data_trimmed.csv and store it in {data_path}")

    data = pd.read_csv(filename, na_filter=False, encoding='utf-8')
    data["trialid"] = data["trialid"].apply(lambda x: int(x) if x!='NA' else x)
    data["ianum"] = data["ianum"].apply(lambda x: int(x) if x != 'NA' else x)
    print(data.columns)

    stopchars = (".", ",", "?", "!", ";", ")", ":", "'")
    beginchars = ("(", '``', '`')

    languages = data.groupby('lang')
    for lang, lang_data in languages:
        print(lang)
        subjects = pd.unique(lang_data['uniform_id'].values)
        print(len(subjects), " subjects.")

    for lang, lang_data in languages:
        if lang == language:
            texts = pd.unique(lang_data['trialid'].values)
            print(texts)
            subjects = pd.unique(lang_data['uniform_id'].values)
            print(len(subjects), " subjects.")

    for subj in subjects:
        if isfile(join(data_path, f"{subj}-relfix-feats.csv")):
            continue

        else:
            flat_word_index = 1
            print(subj)
            subj_data_orig = data.loc[data['uniform_id'] == subj]

            #there is something wrong with the English data in mecoL2, this is a manual fix
            if 'L2' in data_path and language == 'en':
                subj_data_orig = subj_data_orig.drop_duplicates(
                    subset=['firstrun.refix', 'firstrun.reg.in', 'firstrun.reg.out', 'firstrun.dur', 'ia', "sentnum",
                            "ianum", "trialid"])

            print(len(subj_data_orig))
            df_subj = pd.DataFrame(columns=['text_id', 'sentence_id', 'word_id', 'word',
                                            'TRT', 'relFix', 'word_order', 'skip'])
            # split into texts
            text_dfs = subj_data_orig.groupby('trialid')
            for trialid, text in text_dfs:

                word_order = 0

                for w_id, row in text.sort_values("ianum").iterrows():

                    trt = float(row['dur']) if row['dur'] != "NA" else 0.0
                    #tokenize correctly for downstream processing
                    word = row['ia']
                    if word == "...":
                        df_subj.loc[flat_word_index] = [trialid, row['sentnum'], int(row['ianum'])+1, ".", 0.0, 0, word_order, row['skip']]
                        flat_word_index += 1
                        word_order += 1
                        df_subj.loc[flat_word_index] = [trialid, row['sentnum'], int(row['ianum'])+1, ".", 0.0, 0, word_order, 0]
                        flat_word_index += 1
                        word_order += 1
                        df_subj.loc[flat_word_index] = [trialid, row['sentnum'], int(row['ianum'])+1, ".", 0.0, 0, word_order, 0]
                        flat_word_index += 1
                        word_order += 1
                    if len(word) > 3 and word.endswith("..."):
                        new_word = word[:-3]
                        punct = word[-1]
                        df_subj.loc[flat_word_index] = [trialid, row['sentnum'], int(row['ianum']), new_word.lower(), float(trt), 0, word_order, row['skip']]
                        flat_word_index += 1
                        word_order += 1
                        df_subj.loc[flat_word_index] = [trialid, row['sentnum'], int(row['ianum'])+1, punct, 0.0, 0, word_order, 0]
                        flat_word_index += 1
                        word_order += 1
                        df_subj.loc[flat_word_index] = [trialid, row['sentnum'], int(row['ianum'])+1, punct, 0.0, 0, word_order, 0]
                        flat_word_index += 1
                        word_order += 1
                        df_subj.loc[flat_word_index] = [trialid, row['sentnum'], int(row['ianum'])+1, punct, 0.0, 0, word_order, 0]
                        flat_word_index += 1
                        word_order += 1
                    elif word.endswith(stopchars):
                        new_word = word[:-1]
                        punct = word[-1]
                        df_subj.loc[flat_word_index] = [trialid, row['sentnum'], int(row['ianum']), new_word.lower(), float(trt), 0, word_order, row['skip']]
                        flat_word_index += 1
                        word_order += 1
                        df_subj.loc[flat_word_index] = [trialid, row['sentnum'], int(row['ianum'])+1, punct, 0.0, 0, word_order, 0]
                        flat_word_index += 1
                        word_order += 1
                    elif "'" in word:
                        tok1, punct, tok2 = word.partition("'")
                        df_subj.loc[flat_word_index] = [trialid, row['sentnum'], int(row['ianum']), tok1.lower(), float(trt), 0, word_order, row['skip']]
                        flat_word_index += 1
                        word_order += 1
                        df_subj.loc[flat_word_index] = [trialid, row['sentnum'], int(row['ianum'])+1, punct, 0.0, 0, word_order, 0]
                        flat_word_index += 1
                        word_order += 1
                        df_subj.loc[flat_word_index] = [trialid, row['sentnum'], int(row['ianum']), tok2.lower(), float(trt), 0, word_order, 0]
                        flat_word_index += 1
                        word_order += 1
                    elif "-" in word:
                        tok1, punct, tok2 = word.partition("-")
                        if tok1 != "":
                            df_subj.loc[flat_word_index] = [trialid, row['sentnum'], int(row['ianum']), tok1.lower(), float(trt), 0, word_order, row['skip']]
                            flat_word_index += 1
                            word_order += 1
                        df_subj.loc[flat_word_index] = [trialid, row['sentnum'], int(row['ianum'])+1, punct, 0.0, 0, word_order, 0]
                        flat_word_index += 1
                        word_order += 1
                        if tok2!="":
                            df_subj.loc[flat_word_index] = [trialid, row['sentnum'], int(row['ianum']), tok2.lower(), float(trt), 0, word_order, 0]
                            flat_word_index += 1
                            word_order += 1
                    elif word.endswith(beginchars):
                        new_word = word[1:]
                        #print(new_word)
                        punct = word[0]
                        df_subj.loc[flat_word_index] = [trialid, row['sentnum'], int(row['ianum'])+1, punct,0.0,0, word_order, 0]
                        flat_word_index += 1
                        word_order += 1
                        df_subj.loc[flat_word_index] = [trialid, row['sentnum'], int(row['ianum']), new_word.lower(), float(trt), 0, word_order, row['skip']]
                        flat_word_index += 1
                        word_order += 1
                    else:
                        df_subj.loc[flat_word_index] = [trialid, row['sentnum'], int(row['ianum']), row['ia'].lower(), trt, 0, word_order, row['skip']]
                        flat_word_index += 1
                        word_order += 1

            sent_data = df_subj.groupby(['text_id', 'sentence_id'])
            print(len(sent_data))
            new_sent_id = 1
            for i, sentence in sent_data:
                #min-max scale feature  values
                df_subj.loc[(df_subj['text_id'] == i[0]) & (df_subj['sentence_id'] == i[1]), 'NEW_sentence_id'] = new_sent_id
                relfix = [float(s)/sum(sentence['TRT'].values) for s in sentence['TRT'].values]
                df_subj.loc[df_subj['NEW_sentence_id'] == new_sent_id, 'relFix'] = relfix
                new_sent_id +=1

            df_subj.to_csv(join(out_dir, f"{subj}-relfix-feats.csv"), encoding="utf-8")

    print("ALL DONE.")


def extract_features(data_dir, language):
    # stores sentences and average relative fixations
    # join results from all subjects
    if not isfile(join(data_dir, f"meco_{language}_relfix_averages.txt")):
        sent_dict = {}
        for file in sorted(listdir(data_dir)):
            if file.startswith(language) and file.endswith('feats.csv'):
                print("Reading files for subj ", file)
                subj_data = pd.read_csv(join(data_dir, file), delimiter=',')
                max_sent = subj_data['NEW_sentence_id'].max()
                print(max_sent, " sentences")

                # join words in sentences
                i = 0
                while i < max_sent:
                    sent_data = subj_data.loc[subj_data['NEW_sentence_id'] == i]
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
        out_file_text = open(join(data_dir, f"meco_{language}_sentences.txt"), "w")
        out_file_relFix = open(join(data_dir, f"meco_{language}_relfix_averages.txt"), "w")

        for sent, feat in averaged_dict.items():
            print(sent, file=out_file_text)
            print(", ".join(map(str, feat[1])), file=out_file_relFix)


@click.command()
@click.option('--data', default='mecoL2')
def main(data):

    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    et_dir = config[data]['et_dir']
    data_path = config[data]["et_orig"]
    languages = config[data]['languages']

    if not isdir(et_dir):
        makedirs(et_dir)

    filename = join(et_dir, "joint_data_trimmed.csv")

    #print(subj_info_all)
    for language in languages:
        read_meco_file(filename, language, data_path, et_dir)
        print("language:", language)
        extract_features(et_dir, language)


if __name__ == "__main__":
    main()
