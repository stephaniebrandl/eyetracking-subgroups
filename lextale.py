import pandas as pd
from os.path import join, isdir
from os import makedirs
import seaborn as sns
import matplotlib.pyplot as plt
from utils import get_color
import click
import yaml
from scipy.stats import spearmanr
import numpy as np


@click.command()
@click.option('--modelname', default='mbert')
def main(modelname):
    """Collects lextale scores for each participant from meco and them in comparison to correlation values (Figure 3).
    You need to run analyze_individuals.py before running this script.

    Parameters
    ----------
    modelname : str, default = mbert
        The model used to compute attention values
        So far the following options are implemented:
            mbert, xlmr, mt5
    """

    assert modelname in ['mbert', 'xlmr', 'mt5'],\
        'function only implemented for the following models: [mbert, xlmr, mt5]'

    color_dict, lang_dict = get_color()

    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    L1 = pd.read_pickle(join(config['mecoL1']['results_dir'],
                             f"correlation_individuals_mecoL1_{modelname}_{config['importance_type']}.pkl"))
    L1['lang'] = L1['lang'].map(lang_dict)
    L2 = pd.read_pickle(join(config['mecoL2']['results_dir'],
                             f"correlation_individuals_mecoL2_{modelname}_{config['importance_type']}.pkl"))
    L2['lang'] = L2['lang'].map(lang_dict)

    categories = ['lextale']
    fig, axes = plt.subplots(2, 1, figsize=(4,8), sharey='row', sharex='col')

    for df, data in zip([L1, L2], ["mecoL1", "mecoL2"]):
        for cols in categories:
            if data == 'mecoL1':
                mask = df.query("lang=='fi'").index
                df.loc[mask, cols] = df.loc[mask, cols] * 100 / 88
                sns.scatterplot(x=cols,
                                y="correlation_mean",
                                hue='lang',
                                palette=color_dict,
                                hue_order=['en', 'fi', 'nl'],
                                data=df.query("lang in ['en', 'fi', 'nl']"),
                                ax=axes[0],
                                legend=None)
                axes[0].set_ylabel("correlation values")
                axes[0].set_title("L1")
            else:
                del lang_dict['ko']
                sns.scatterplot(x=cols, y="correlation_mean", hue='lang',
                                hue_order=sorted(lang_dict.values()),
                                palette=color_dict, data=df, ax=axes[1])
                axes[1].set_title("L2")
                axes[1].set_ylabel("correlation values")
                axes[1].legend(loc='lower left', bbox_to_anchor=(1.05, 0.5, 0.5, 0.5))

            # uncomment for results on individual languages
            for lang, subdf in df.groupby("lang"):
                try:
                    if spearmanr(subdf['lextale'].values, subdf['correlation_mean'].values)[1] < 0.05:
                        print(data, lang,
                              np.around(spearmanr(subdf['lextale'].values, subdf['correlation_mean'].values)[0],
                                        decimals=2))
                except TypeError:
                    pass

            df_lextale = df.dropna(subset=['lextale'])
            print(data, "lextale",
                  np.around(spearmanr(df_lextale['lextale'].values, df_lextale['correlation_mean'].values)[0], decimals=2),
                  spearmanr(df_lextale['lextale'].values, df_lextale['correlation_mean'].values)[1])

    if not isdir('./figs'):
        makedirs('./figs')

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(f"./figs/lextale_{modelname}_{config['importance_type']}.png", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
