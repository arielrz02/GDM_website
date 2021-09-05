from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, plot_roc_curve
from sklearn.model_selection import GroupShuffleSplit
import itertools
import json
import xgboost as xgb
import pandas as pd
import numpy as np


def z_score(df):
    # copy the dataframe
    df_std = df.copy()
    # apply the z-score method
    for column in df_std.columns:
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()
    return df_std


def prep_data(data_file, tag_file):
    data = pd.read_csv(data_file, index_col=0)
    tags = pd.read_csv(tag_file, index_col=0)
    groupid = [string.split("-")[0] for string in tags.index]
    groupid = pd.Series(groupid)
    return data, tags, groupid

def gss(data, tags, group_id, train_prec):
    gss = GroupShuffleSplit(n_splits=1, train_size=train_prec)
    train_idx, val_idx = next(gss.split(data, groups=group_id))
    train_data, train_tag = data.iloc[train_idx, :], tags.iloc[train_idx, :]
    val_data, val_tag = data.iloc[val_idx, :], tags.iloc[val_idx, :]
    return train_data, train_tag,val_data, val_tag

def stratify(data, tags, group_id, prec):
    df_reset = tags.reset_index()
    lst_pos = df_reset.index[df_reset['Tag'] == 1].tolist()
    lst_neg = df_reset.index[df_reset['Tag'] == 0].tolist()
    tags_pos = tags.iloc[lst_pos, :]
    tags_neg = tags.iloc[lst_neg, :]
    data_pos = data.iloc[lst_pos, :]
    data_neg = data.iloc[lst_neg, :]
    group_pos = group_id.to_frame().iloc[lst_pos]
    group_neg = group_id.to_frame().iloc[lst_neg]
    train_data_pos, train_tag_pos, val_data_pos, val_tag_pos = gss(data_pos, tags_pos, group_pos, prec)
    train_data_neg, train_tag_neg, val_data_neg, val_tag_neg = gss(data_neg, tags_neg, group_neg, prec)
    return train_data_pos.append(train_data_neg).to_numpy(), train_tag_pos.append(train_tag_neg).to_numpy(),\
           val_data_pos.append(val_data_neg).to_numpy(), val_tag_pos.append(val_tag_neg).to_numpy()

def xgboost_predict(data, tags, params, to_pred):
    clf = xgb.XGBClassifier(**params, n_jobs=-1, n_estimators=int(200), objective='binary:logistic')
    clf.fit(data, tags)
    to_pred = pd.DataFrame(to_pred)
    pred = clf.predict_proba(to_pred)
    pred_train = clf.predict_proba(data)
    return pred, pred_train

def xgb_for_site(to_pred):
    params = {"learning_rate": 0.1, "reg_lambda": 100, "booster": "dart", "gamma": 0.1, "colsample_bytree": 0.7}
    data_full = prep_data("static/forsite_new_data.csv", "static/forsite_new_tags.csv")
    to_pred.columns = data_full[0].columns
    data = z_score(data_full[0].append(to_pred))
    to_pred = data.iloc[-len(to_pred):]
    data = data[:-len(to_pred)]
    return xgboost_predict(data, data_full[1], params=params, to_pred=to_pred)


def xgboost_cv(data, tags, group_id, params, cv):
    auc_lst = np.zeros(shape=(cv, ))
    auc_lst_train = np.zeros(shape=(cv, ))
    for i in range(cv):
        train_data, train_tag, val_data, val_tag = stratify(data, tags, group_id, 0.8)
        clf = xgb.XGBClassifier(**params, n_jobs=-1, n_estimators=int(200), objective='binary:logistic')
        clf.fit(train_data, train_tag)
        pred_val = clf.predict_proba(val_data)[:, 1]
        pred_train = clf.predict_proba(train_data)[:, 1]
        auc_lst[i], auc_lst_train[i] = roc_auc_score(val_tag, pred_val), roc_auc_score(train_tag, pred_train)
        plot_roc_curve(clf, val_data, val_tag)
    print(auc_lst, auc_lst_train)
    return auc_lst.mean(), auc_lst.std(), auc_lst_train.mean(), auc_lst_train.std()

def gridsearch_xgb(data_file, tag_file):
    full_data = prep_data(data_file, tag_file)
    with open("xgbres.txt", 'w') as f:
        f.write("xgb results")
    param_grid = {"learning_rate": np.logspace(-3, 0, 4),
                  "reg_lambda": np.logspace(-2, 2, 5),
                  "min_child_weight": [1, 3, 5, 7],
                  "gamma": np.logspace(-1, 1, 5),
                  "colsample_bytree": [0.3, 0.4, 0.5, 0.7]}
    for values in itertools.product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), values))
        auc, std, _, _ = xgboost_cv(*full_data, params=params, cv=5)
        with open("xgbres.txt", 'a') as f:
            f.write(f"{json.dumps(params)}:{auc}\n")

if __name__=="__main__":
    #gridsearch_xgb("forsite_new_data.csv", "forsite_new_tags.csv")
    #for col in [0.3, 0.4, 0.5, 0.7]:
    params = {"learning_rate": 0.1, "reg_lambda": 100, "booster": "dart", "gamma": 0.1, "colsample_bytree": 0.7}
    data_full = prep_data("static/forsite_new_data.csv", "static/forsite_new_tags.csv")
    print(xgboost_predict(*data_full, params=params, to_pred=data_full[0][9:10]))
