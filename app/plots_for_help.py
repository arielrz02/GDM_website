import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from gdm_xgboost import xgb_for_site

def plot_neg_pos():
    _, pred = xgb_for_site(pd.DataFrame(np.zeros(9)).T)
    tags = pd.read_csv("static/forsite_new_tags.csv")
    pred = pd.DataFrame(pred)
    tags.index = pred.index
    pred_tag = pd.concat([pred, tags], axis=1)
    pos = pred_tag[pred_tag["Tag"] == 1]
    neg = pred_tag[pred_tag["Tag"] == 0]
    fig, ax = plt.subplots()
    ax.hist(x=pos[1], bins=50, alpha=0.55, density=True,
            color='r', label="positives")
    ax.hist(x=neg[1], bins=50, alpha=0.55, density=True,
             color='g', label="negatives")
    ax.set_xlabel("Chance of being positive for GDM in our model")
    ax.set_ylabel("Density")
    ax.legend()
    plt.savefig("static/pos_neg_hist")


if __name__ == "__main__":
    plot_neg_pos()