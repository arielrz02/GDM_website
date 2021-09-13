from gdm_xgboost import xgb_for_site, z_score

import pandas as pd
import matplotlib.pyplot as plt


def get_results_and_file(features):
    features = features.astype("float")
    pred, pred_train = xgb_for_site(features)
    pred = pd.DataFrame(pred)
    pred_train = pd.DataFrame(pred_train)
    pred.to_csv("static/results.csv")
    if len(pred) == 1:
        get_res_standing(pred, pred_train)


def get_res_standing(result, train_res):
    tag = pd.read_csv("static/forsite_new_tags.csv")
    tag.index = train_res.index
    pred_tag = pd.concat([train_res, tag], axis=1)
    pos = pred_tag[pred_tag["Tag"] == 1]
    pos_under = len(pos[pos[1] < result.iloc[0, 1]])
    neg = pred_tag[pred_tag["Tag"] == 0]
    neg_under = len(pos[pos[0] < result.iloc[0, 0]])
    plot_res(pos_under, len(pos), neg_under, len(neg))


def plot_res(pos_under, pos, neg_under, neg):
    img = plt.imread("static/colorbar.png")
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.axes.yaxis.set_visible(False)
    ax1.axes.xaxis.set_visible(False)
    ax1.imshow(img)
    ax1.axvline(x=(pos_under / pos) * ax1.get_xlim()[1])
    ax1.set_title(f"Your data has a larger score than {100*pos_under/pos:.2f}"
                  f" precent of positives for GDM\n")
    ax2.axes.yaxis.set_visible(False)
    ax2.axes.xaxis.set_visible(False)
    ax2.imshow(img)
    ax2.axvline(x=(1 - (neg_under / neg)) * ax1.get_xlim()[1])
    ax2.set_title(f"And a smaller score than {100*neg_under/neg: .2f}"
                  f" precent of negatives for GDM\n")
    plt.savefig("static/bar_plot.svg")
