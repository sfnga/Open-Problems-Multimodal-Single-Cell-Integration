import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold


def correlation_score(y_true, y_pred):
    if type(y_true) == pd.DataFrame: y_true = y_true.values
    if type(y_pred) == pd.DataFrame: y_pred = y_pred.values
    corrsum = 0
    for i in range(len(y_true)):
        corrsum += np.corrcoef(y_true[i], y_pred[i])[1, 0]
    return corrsum / len(y_true)


def rescale(x):
    return (x - np.max(x)) / (np.max(x) - np.min(x))


def get_score(weights, train_idx, oofs, labels):
    blend = np.zeros_like(oofs[0][train_idx, :])

    for oof, weight in zip(oofs[:-1], weights):
        blend += weight * oof[train_idx, :]

    blend += (1 - np.sum(weights)) * oofs[-1][train_idx, :]
    return -correlation_score(labels[train_idx, :], blend)


def get_best_weights(oofs, labels, X, meta):
    weight_list = []
    weights = np.array([1 / len(oofs) for x in range(len(oofs) - 1)])

    kf = GroupKFold(n_splits=2)
    for fold, (train_idx,
               valid_idx) in enumerate(kf.split(X.values, groups=meta.day)):
        if fold != 1:
            continue
        res = minimize(get_score,
                       weights,
                       args=(train_idx, oofs, labels),
                       method="Nelder-Mead",
                       tol=1e-6)
        print(f"fold: {fold} res.x: {res.x}")
        weight_list.append(res.x)

    mean_weight = np.mean(weight_list, axis=0)
    x = 1 - np.sum(mean_weight)
    mean_weight = np.append(mean_weight, x)
    print(f"optimized weight: {mean_weight}")
    return mean_weight


def main():
    metadata_df = pd.read_csv('/open_problems/metadata.csv',
                              index_col='cell_id')
    metadata_df = metadata_df[metadata_df.technology == "citeseq"]

    X = pd.read_hdf('/open_problems/train_cite_inputs.h5')
    metadata_df = metadata_df.reindex(X.index)

    Y = pd.read_hdf('/open_problems/train_cite_targets.h5')
    Y = Y.values
    Y -= Y.mean(axis=1).reshape(-1, 1)
    Y /= Y.std(axis=1).reshape(-1, 1)

    submission_names = [
        'nn_1', 'nn_2', 'nn_3', 'nn_4', 'nn_5', 'catboost', 'tabnet'
    ]
    oof = []
    test = []
    for name in submission_names:
        oof.append(rescale(np.load(f'/submissions/{name}/oof.npy')))
        test.append(rescale(np.load(f'/submissions/{name}/test_preds.npy')))

    best_weights = get_best_weights(oofs, Y, X, metadata_df)

    oof_preds = np.zeros_like(oofs[0])
    test_preds = np.zeros_like(test[0])
    for i in range(len(best_weights)):
        oof_preds += best_weights[i] * oofs[i]
        test_preds += best_weights[i] * test[i]

    print(f"OOF score: {correlation_score(Y,oof):.5f}")
    np.save('oof_blend.npy', oof_preds)
    np.save('test_blend.npy', test_preds)


if __name__ == "__main__":
    main()
