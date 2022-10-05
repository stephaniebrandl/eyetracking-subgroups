import scipy.stats
import sklearn.metrics
from ast import literal_eval
from os.path import isdir, join, isfile
from os import makedirs
import numpy as np
import pandas as pd
from extract_model_importance.extract_attention_first import extract_attention_first
from extract_model_importance import tokenization_util

import warnings
warnings.filterwarnings("ignore")


def extract_human_importance(dataset, dir='results_meco/'):

    with open(join(dir, dataset + "_sentences.txt"), "r") as f:
        sentences = f.read().splitlines()

    # split and lowercase
    tokens = [s.split(" ") for s in sentences]
    tokens = [[t.lower() for t in tokens] for tokens in tokens]

    human_importance = []
    # try:
    with open(join(dir, dataset + "_relfix_averages.txt"), "r") as f:
        for line in f.read().splitlines():
            fixation_duration = np.fromstring(line, dtype=float, sep=',')
            human_importance.append(fixation_duration)

    return tokens, human_importance


# Importance type is either "saliency" or "attention"
def extract_model_importance(dataset, model, importance_type, dir='results_meco'):
    lm_tokens = []
    lm_salience = []
    #dataset_short = dataset.split("_")[-1]
    with open(dir + "/" + dataset + "_" + model + "_" + importance_type + ".txt", "r") as f:
        for line in f.read().splitlines():
            tokens, heat = line.split("\t")

            tokens = list(literal_eval(tokens))
            salience = np.array(literal_eval(heat))

            # remove CLR and SEP tokens, this is an experimental choice
            lm_tokens.append(tokens[1:-1])
            salience = salience[1:-1]

            # Apply softmax over remaining tokens to get relative importance
            salience = scipy.special.softmax(salience)
            lm_salience.append(salience)

    return lm_tokens, lm_salience


def extract_features_individual(file, data_path):
    if not isfile(f"{data_path}/{file.split('-')[0]}_relfix_averages.txt"):
        sent_dict = {}
        print("Reading files for subj ", file)
        subj_data = pd.read_csv(join(data_path, file), delimiter=',')
        try:
            max_sent = subj_data['NEW_sentence_id'].max()
        except KeyError:
            subj_data['NEW_sentence_id'] = subj_data['sentence_id']
            max_sent = subj_data['NEW_sentence_id'].max()
        print(max_sent, " sentences")

        # join words in sentences
        i = 0
        while i < max_sent:
            sent_data = subj_data.loc[subj_data['NEW_sentence_id'] == i]
            if " ".join(map(str, list(sent_data['word']))):
                relfix_vals = list(sent_data['relFix'])
                if " ".join(map(str, list(sent_data['word']))) not in sent_dict:
                    sent_dict[" ".join(map(str, list(sent_data['word'])))] = [list(sent_data['word']),
                                                                              [relfix_vals]]
                else:
                    sent_dict[" ".join(map(str, list(sent_data['word'])))][1].append(relfix_vals)
            i += 1

        out_file_text = open(f"{data_path}/{file.split('-')[0]}_sentences.txt", "w")
        out_file_relFix = open(f"{data_path}/{file.split('-')[0]}_relfix_averages.txt", "w")
        print(out_file_relFix)

        for sent, feat in sent_dict.items():
            print(sent, file=out_file_text)
            print(", ".join(map(str, np.array(feat[1][0]))), file=out_file_relFix)


def extract_all_attention_first(modelname, model, tokenizer, sentences, outfile):
    with open(outfile, "w") as attention_file:
        for i, sentence in enumerate(sentences):
            # print(progress
            if i%500 ==0:
                print(i, len(sentences))
            tokens, relative_attention = extract_attention_first(model, tokenizer, sentence, modelname)
            # merge word pieces if necessary and fix tokenization
            tokens, merged_attention = tokenization_util.merge_subwords(tokens, relative_attention)
            if modelname in ["mt5", 'xlmr']:
                begin_token = {"mt5": "▁", "xlmr": "Ġ"}
                tokens, merged_attention = tokenization_util.merge_albert_tokens(
                    tokens, merged_attention, begin_token[modelname])
                tokens = ['CLS'] + tokens if modelname == 'mt5' else tokens
                merged_attention = [0] + merged_attention if modelname == 'mt5' else merged_attention
            tokens, merged_attention = tokenization_util.merge_hyphens(tokens, merged_attention)
            tokens, merged_attention = tokenization_util.merge_symbols(tokens, merged_attention)
            attention_file.write(str(tokens) + "\t" + str(merged_attention) + "\n")


def compare_importance(et_tokens, human_salience, lm_tokens, lm_salience, importance_type, corpus, modelname, meco):
    count_tok_errors = 0

    spearman_correlations = []
    kendall_correlations = []
    mutual_information = []
    humans = []
    lm = []

    if meco == 1:
        out_dir = "results_meco"
    else:
        out_dir = "../results_meco2"

    if not isdir(join(out_dir, "correlations/")):
        makedirs(join(out_dir, "correlations/"))

    with open(join(out_dir, "correlations") + "/" + corpus + "_" + modelname + "_" + importance_type + "_correlations.txt", "w") as outfile:
        outfile.write("Spearman\tKendall\tMutualInformation\n")
        for i, sentence in enumerate(et_tokens):
            if len(et_tokens[i]) == len(lm_tokens[i]) == len(human_salience[i]) == len(lm_salience[i]):
                # only take into account sentences with more than 1 token:
                if len(et_tokens[i]) > 1:
                    # Calculate the correlation
                    spearman = scipy.stats.spearmanr(lm_salience[i], human_salience[i])[0]
                    spearman_correlations.append(spearman)
                    kendall = scipy.stats.kendalltau(lm_salience[i], human_salience[i])[0]
                    kendall_correlations.append(kendall)
                    humans.extend(human_salience[i])
                    lm.extend(lm_salience[i])
                    try:
                        mi_score = sklearn.metrics.mutual_info_score(lm_salience[i], human_salience[i])
                        mutual_information.append(mi_score)
                        outfile.write("{:.2f}\t{:.2f}\t{:.2f}\n".format(spearman, kendall, mi_score))
                    except ValueError:
                        continue

            else:
                # # Uncomment if you want to know more about the tokenization alignment problems
                #print("Tokenization Error:")
                #print(len(et_tokens[i]), len(lm_tokens[i]), len(human_salience[i]), len(lm_salience[i]))
                #print(et_tokens[i])
                #print(lm_tokens[i])
                #print()
                count_tok_errors += 1

    print(corpus, modelname)
    print("Token alignment errors: ", count_tok_errors)
    print("Spearman Correlation Model: Mean, Stdev")
    mean_spearman = np.nanmean(np.asarray(spearman_correlations))
    std_spearman = np.nanstd(np.asarray(spearman_correlations))
    print(mean_spearman, std_spearman)

    print("\n")

    return mean_spearman, std_spearman, humans, lm
