import pandas as pd
import numpy as np
from utils import standardize_columns, read_config, extract_pos
import yaml
import click
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from utils import get_color
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from os.path import isdir
from os import makedirs

color_dict, lang_dict = get_color()

language_model = {'fi': "fi_core_news_sm",
                  'ge': "de_core_news_sm",
                  'en': "en_core_web_sm",
                  'du': "nl_core_news_sm",
                  'it': "it_core_news_sm",
                  'gr': "el_core_news_sm",
                  'ru': "ru_core_news_sm",
                  'sp': "es_core_news_sm",
                  'ko': "ko_core_news_sm",
                  'no': "nb_core_news_sm"}

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


@click.command()
@click.option('--data', default='mecoL1')
@click.option('--modelname', default='mbert')
def main(data, modelname):
    """Extracts pos for each token, averages respective fixation times across participants and then correlates with
    attention and plots this in comparison to correlation values (Figure 2).
    You need to run analyze_individuals.py before running this script.

    Parameters
    ----------
    data : str, default = mecoL1
        The dataset on which POS tags are extracted.
        So far the following options are implemented:
            mecoL1: reading data of meco participants in their native language
            mecoL2: reading data of meco participants in English as language learners
    modelname : str, default = mbert
        The model used to compute attention values
        So far the following options are implemented:
            mbert, xlmr, mt5
    """

    assert modelname in ['mbert', 'xlmr', 'mt5'],\
        'function only implemented for the following models: [mbert, xlmr, mt5]'

    assert data in ['mecoL1', 'mecoL2'],\
        'function only implemented for the following datasets: [mecoL1, mecoL2]'

    languages, results_dir, attention_dir, et_dir, et_orig, _, _, MODEL_NAME, importance_type = read_config(data, modelname)

    #check if df with POS tags is already available, otherwise create from scratch
    try:
        df_all = pd.read_pickle(f"./{results_dir}/df_{data}_pos.pkl")
    except FileNotFoundError:
        extract_pos(data, modelname, language_model)
        df_all = pd.read_pickle(f"./{results_dir}/df_{data}_pos.pkl")

    #creates dictionaries for pos tags
    bin = 'pos'
    len_bin = {}
    corr_bert = {}
    corr_short = {}
    tokens_mean = {}
    importance_mean = {}

    for lang, df_lang in sorted(df_all.groupby('lang')):
        print(lang)
        df_lang = df_lang.dropna(subset=['relfix', importance_type])
        df_lang = standardize_columns(df_lang, ['relfix', importance_type])
        bins = sorted(df_lang[bin][~pd.isnull(df_lang[bin])].unique())
        tokens = {}
        tokens['bert'] = {}
        tokens['tsr'] = {}
        corr_bert[lang] = {}
        corr_short[lang] = []
        tokens_mean[lang] = {}
        importance_mean[lang] = {}

        tokens_tsr = []
        tokens_bert = []
        len_bin[lang] = {}

        for ind, tag in enumerate(bins):
            df_lang_tmp = df_lang.query('{}==@tag'.format(bin))
            tokens_mean[lang][tag] = df_lang_tmp.relfix_normalized.mean()
            importance_mean[lang][tag] = df_lang_tmp['attention-first_normalized'].mean()
            len_bin[lang][tag] = len(df_lang_tmp)

            tokens['bert'][tag] = df_lang_tmp[importance_type + '_normalized']
            tokens['tsr'][tag] = df_lang_tmp['relfix_normalized']

            for index, row in df_lang_tmp.iterrows():
                tokens_tsr.append(row['relfix_normalized'])
                tokens_bert.append(row[importance_type + '_normalized'])

            corr_bert[lang][tag] = spearmanr(tokens_tsr, tokens_bert)[0]
        corr_short[lang] = sorted(corr_bert[lang].items(), key=lambda x: x[1], reverse=True)[:6]

    important_pos = ['ADJ', 'ADP', 'NOUN', 'VERB', 'PRON', 'CCONJ']
    len_pos = [len_bin['en'][pos] for pos in important_pos]
    important_pos = [important_pos[ii] for ii in np.argsort(len_pos)[::-1]]
    pos_mat = np.zeros([len(corr_short.keys()), len(important_pos)])
    bar_mat = np.zeros([len(corr_short.keys()), len(important_pos)])
    bar2_mat = np.zeros([len(corr_short.keys()), len(important_pos)])
    ylabels = []
    for ilang, lang in enumerate(corr_short.keys()):
        if data=='mecoL1' and lang in ['tr', 'he']:
            continue
        ylabels.append(lang)
        for ii, pos in enumerate(important_pos):
            try:
                pos_mat[ilang, ii] = corr_bert[lang][pos]
                bar_mat[ilang, ii] = tokens_mean[lang][pos]
                bar2_mat[ilang, ii] = importance_mean[lang][pos]
            except KeyError:
                print(lang, pos)

    #plotting
    if data == 'mecoL1':
        fig, ax = plt.subplots(2, 1, figsize=(4, 4), sharex=True, gridspec_kw={'height_ratios': [6, 1]})
        width = 0.08  # the width of the bars
    else:
        fig, ax = plt.subplots(2, 1, figsize=(4, 5), sharex=True, gridspec_kw={'height_ratios': [7, 1]})
        width = 0.07  # the width of the bars
    im = ax[0].imshow(pos_mat, cmap='YlOrRd', vmin=0, vmax=0.8, aspect=.5)
    ax[0].set_xticks(np.arange(0, len(important_pos)), labels=important_pos, rotation=45)
    ax[0].set_yticks(np.arange(0, len(ylabels)), labels=ylabels)

    ax1 = ax[0]

    for i in range(pos_mat.shape[0]):
        for j in range(pos_mat.shape[1]):
            if abs(pos_mat[i, j]) > 0.5:
                ax1.text(j, i, np.around(pos_mat[i, j], decimals=2), ha="center", va="center", color="w")

            else:
                ax1.text(j, i, np.around(pos_mat[i, j], decimals=2), ha="center", va="center", color="k")

    cax = ax[0]
    axins = inset_axes(cax,
                       width="5%",  # width = 5% of parent_bbox width
                       height="100%",  # height : 50%
                       loc='lower left',
                       bbox_to_anchor=(1.05, 0., 1, 1),
                       bbox_transform=cax.transAxes,
                       borderpad=0,
                       )

    fig.colorbar(im, cax=axins)

    x = np.arange(len(important_pos))
    ax3 = ax[1]

    offset_imp = np.arange(-5.5, 6, 1)
    offset_imp = offset_imp[:len(ylabels)]
    for iii, off in enumerate(offset_imp):
        ax3.bar(x + off * width, bar_mat[iii], width, label=ylabels[iii],
                color=color_dict[ylabels[iii]])
    ax3.legend(loc='lower left', bbox_to_anchor=(1.25, 1, 0.5, 0.5))
    ax3.set_yticks([-1, 0, 1])
    plt.subplots_adjust(wspace=0, hspace=0)

    if not isdir('./figs'):
        makedirs('./figs')

    plt.savefig(f"./figs/{data}_pos_averaged_{modelname}", dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()
