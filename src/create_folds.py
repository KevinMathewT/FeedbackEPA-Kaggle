import os
import yaml
import joblib
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GroupKFold

from config import config

if __name__ == "__main__":
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
        # Stratified KFold
        gkf = GroupKFold(n_splits=config["folds"])
        for i, (train_index, test_index) in enumerate(
            gkf.split(X=train, groups=train["essay_id"])
        ):
            train.loc[test_index, "fold"] = i

    train["fold"] = train["fold"].astype(int)

    print(train.groupby('fold')['discourse_effectiveness'].value_counts())

    encoder = LabelEncoder()
    train['discourse_effectiveness'] = encoder.fit_transform(train['discourse_effectiveness'])

    with open(config['label_enc'], "wb") as fp:
        joblib.dump(encoder, fp)
    train.to_csv(config["train_folds"], index=False)
