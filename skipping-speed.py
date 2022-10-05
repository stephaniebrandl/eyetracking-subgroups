import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from os import makedirs
from os.path import join, isdir
from scipy.stats import spearmanr
from utils import get_color
import click
import yaml

color_dict, lang_dict = get_color()


@click.command()
@click.option('--modelname', default='mbert')
def main(modelname):
    """Collects total reading times (TRT) and skipping behaviour for each participant from meco and plots both in
    comparison to correlation values (Figure 1). You need to run analyze_individuals.py before running this script.

    Parameters
    ----------
    modelname : str, default = mbert
        The model used to compute attention values
        So far the following options are implemented:
            mbert, xlmr, mt5
    """

    assert modelname in ['mbert', 'xlmr', 'mt5'],\
        'function only implemented for the following models: [mbert, xlmr, mt5]'

    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    L1 = pd.read_pickle(join(config['mecoL1']['results_dir'],
                             f"correlation_individuals_mecoL1_{modelname}_{config['importance_type']}.pkl"))
    L2 = pd.read_pickle(join(config['mecoL2']['results_dir'],
                             f"correlation_individuals_mecoL2_{modelname}_{config['importance_type']}.pkl"))
    et_paths = [config['mecoL1']['et_dir'], config['mecoL2']['et_dir']]

    title = ['L1', 'L2']
    fig, axes = plt.subplots(2, 2, figsize=(8, 4), sharey='row', sharex='col')

    for ii, df in enumerate([L1, L2]):
        for subj, row in df.iterrows():
            try:
                subj_et = pd.read_csv(join(et_paths[ii], f"{subj}-relfix-feats.csv"))
                df.loc[subj, 'skip'] = (subj_et['skip']/len(subj_et)).sum()
                df.loc[subj, 'TRT'] = subj_et['TRT'].sum()/len(subj_et)
                df.loc[subj, 'lang'] = lang_dict[row['lang']]
            except (KeyError, FileNotFoundError):
                print(subj, "not found")
                pass

        legend = None if ii==1 else 'full'

        sns.scatterplot(data=df, x='correlation_mean', y='skip', hue='lang', ax=axes[0, ii],
                        hue_order=sorted(lang_dict.values()), legend=legend, palette=color_dict)

        sns.scatterplot(data=df, x='correlation_mean', y='TRT', hue='lang', ax=axes[1, ii],
                        hue_order=sorted(lang_dict.values()), legend=None, palette=color_dict)
        axes[0, ii].legend(loc='upper center', ncol=7, bbox_to_anchor=(1, 1.6)) if legend else None
        axes[0, ii].set_title(title[ii])
        axes[1, ii].set_xlabel('correlation')

        print("skipping", title[ii], np.around(spearmanr(df['skip'].values, df['correlation_mean'].values)[0], decimals=2),
              spearmanr(df['skip'].values, df['correlation_mean'].values)[1])
        print("TRT", title[ii], np.around(spearmanr(df['TRT'].values, df['correlation_mean'].values)[0], decimals=2),
              spearmanr(df['TRT'].values, df['correlation_mean'].values)[1])

        # # uncomment for results on individual languages
        # for lang, subdf in df.groupby("lang"):
        #     if spearmanr(subdf['skip'].values, subdf['correlation_mean'].values)[1] < 0.05:
        #         print('skipping', lang, np.around(spearmanr(subdf['skip'].values, subdf['correlation_mean'].values)[0], decimals=2))
        #     if spearmanr(subdf['TRT'].values, subdf['correlation_mean'].values)[1] < 0.05:
        #         print('TRT', lang, np.around(spearmanr(subdf['TRT'].values, subdf['correlation_mean'].values)[0], decimals=2))

    axes[0, 0].set_ylabel('skipping rate')
    axes[1, 0].set_ylabel('TRT [ms]')
    plt.subplots_adjust(wspace=0.05, hspace=0.1)

    if not isdir('./figs'):
        makedirs('./figs')

    plt.savefig(f"./figs/skipping-speed_{modelname}", dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()
