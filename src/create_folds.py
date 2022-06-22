import os
import yaml
import joblib
import random
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GroupKFold

from config import config
from .utils import seed_everything


def stratified_group_k_fold(X, y, groups, k, seed=None):
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std(
                [y_counts_per_fold[i][label] / y_distr[label] for i in range(k)]
            )
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)

    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices


if __name__ == "__main__":
    seed_everything(config["seed"])

    print(os.getcwd())
    train = pd.read_csv(config["train_csv"])
    test = pd.read_csv(config["test_csv"])
    sample_submission = pd.read_csv(config["sample_csv"])

    train["text"] = train["essay_id"].apply(
        lambda x: open(Path(config["train_base"]) / f"{x}.txt").read()
    )
    test["text"] = test["essay_id"].apply(
        lambda x: open(Path(config["test_base"]) / f"{x}.txt").read()
    )

    # Changing Concluding Statement to Conclusion
    train['discourse_type'] = train['discourse_type'].apply(
        lambda x: x if x != 'Concluding Statement' else 'Conclusion'
    )
    test['discourse_type'] = test['discourse_type'].apply(
        lambda x: x if x != 'Concluding Statement' else 'Conclusion'
    )

    target_map = {"Adequate": 1, "Effective": 2, "Ineffective": 0}

    train["target"] = train["discourse_effectiveness"].map(target_map)
    train = train.reset_index(drop=True)

    if config["fold_type"] == "stratified":
        # Stratified KFold
        skf = StratifiedKFold(
            n_splits=config["folds"], shuffle=True, random_state=config["folds"]
        )
        for i, (train_index, test_index) in enumerate(
            skf.split(train, train["target"])
        ):
            train.loc[test_index, "fold"] = i

    if config["fold_type"] == "group":
        # Group KFold
        gkf = GroupKFold(n_splits=config["folds"])
        for i, (train_index, test_index) in enumerate(
            gkf.split(X=train, groups=train["essay_id"])
        ):
            train.loc[test_index, "fold"] = i

    if config["fold_type"] == "stratified_group":
        # Stratified Group KFold
        for i, (train_index, test_index) in enumerate(
            stratified_group_k_fold(
                X=train,
                y=train["target"],
                groups=train["essay_id"],
                k=config["folds"],
                seed=config["seed"],
            )
        ):
            train.loc[test_index, "fold"] = i

    train["fold"] = train["fold"].astype(int)

    print(train.groupby("fold")["discourse_effectiveness"].value_counts())

    encoder = LabelEncoder()
    train["discourse_effectiveness"] = encoder.fit_transform(
        train["discourse_effectiveness"]
    )

    with open(config["label_enc"], "wb") as fp:
        joblib.dump(encoder, fp)
    train.to_csv(config["train_folds"], index=False)
