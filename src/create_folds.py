import os
import sys
import yaml
import joblib
import random
import difflib
from tqdm import tqdm
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GroupKFold

from transformers import AutoTokenizer

from config import config
from .utils import seed_everything

tqdm.pandas()
pd.set_option("display.max_columns", 500)
tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"])


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


def LCSubStr(X, Y):
    m = len(X)
    n = len(Y)

    result = 0
    end = 0
    length = [[0 for j in range(m)] for i in range(2)]
    currRow = 0

    for i in range(0, m + 1):
        for j in range(0, n + 1):
            if i == 0 or j == 0:
                length[currRow][j] = 0

            elif X[i - 1] == Y[j - 1]:
                length[currRow][j] = length[1 - currRow][j - 1] + 1

                if length[currRow][j] > result:
                    result = length[currRow][j]
                    end = i - 1
            else:
                length[currRow][j] = 0

        currRow = 1 - currRow

    if result == 0:
        return "-1"
    return end - result + 1, end + 1


def get_discource_context(meta):
    essay = meta["text"]
    discourse_type = meta["discourse_type"]
    discourse_text = meta["discourse_text"].strip()

    tokens = tokenizer.tokenize(
        discourse_type + " " + discourse_text + " " + tokenizer.sep_token + " " + essay
    )
    print(len(tokens))

    return essay


essay_id_blocks = []


def essay_id_apply(x):
    essay_id = x["essay_id"].to_list()[0]
    discourse_type = x["discourse_type"].tolist()
    x["discourse_type"] = x["discourse_type"].map(
        {
            "Lead": 1,
            "Position": 2,
            "Claim": 3,
            "Counterclaim": 4,
            "Rebuttal": 5,
            "Evidence": 6,
            "Concluding Statement": 7,
        }
    )
    x = x.sort_values("discourse_type")
    x['discourse_text'] = x['discourse_text'].apply(lambda x: x.strip())

    x["tokens_size"] = x["discourse_text"].apply(lambda x: len(tokenizer.tokenize(x)))
    texts = x["discourse_text"].tolist()
    tokens_sizes = x["tokens_size"].tolist()

    essays = []

    for i in range(x.shape[0]):
        l = i - 1
        r = i + 1
        b = (
            config["max_length"]
            - 2 * tokens_sizes[i]
            - len(tokenizer.tokenize(discourse_type[i]))
            - 1
        )
        essay = texts[i]

        while b > 0:
            do_break = True
            if l >= 0 and l < len(tokens_sizes) and tokens_sizes[l] <= b:
                essay = texts[l] + " " + essay
                b -= tokens_sizes[l]
                l -= 1
                do_break = False

            if r >= 0 and r < len(tokens_sizes) and tokens_sizes[r] <= b:
                essay = essay + " " + texts[r]
                b -= tokens_sizes[r]
                r += 1
                do_break = False

            if do_break:
                break

        essays.append(
            discourse_type[i] + " " + texts[i] + " " + tokenizer.sep_token + " " + essay
        )

    x["text"] = essays
    x["text_tokens_lengths"] = x["text"].apply(lambda x: len(tokenizer.tokenize(x)))

    x["discourse_type"] = x["discourse_type"].map(
        {
            1: "Lead",
            2: "Position",
            3: "Claim",
            4: "Counterclaim",
            5: "Rebuttal",
            6: "Evidence",
            7: "Concluding Statement",
        }
    )

    # if random.randint(1, 1000000) % 7 == 0:
    #     print(x)
    #     sys.exit(0)

    essay_id_blocks.append(x)


if __name__ == "__main__":
    seed_everything(config["seed"])

    train = pd.read_csv(config["train_csv"])
    test = pd.read_csv(config["test_csv"])
    sample_submission = pd.read_csv(config["sample_csv"])

    essay_id_blocks = []
    train.groupby("essay_id").apply(essay_id_apply)
    train = pd.concat(essay_id_blocks, ignore_index=True)

    essay_id_blocks = []
    test.groupby("essay_id").apply(essay_id_apply)
    test = pd.concat(essay_id_blocks, ignore_index=True)

    # Getting essay full text
    # train["text"] = train[["text", "discourse_type", "discourse_text"]].apply(
    #     get_discource_context,
    #     axis=1,
    # )
    # test["text"] = test[["text", "discourse_type", "discourse_text"]].apply(
    #     get_discource_context,
    #     axis=1,
    # )

    # Changing Concluding Statement to Conclusion
    train["discourse_type"] = train["discourse_type"].apply(
        lambda x: x if x != "Concluding Statement" else "Conclusion"
    )
    test["discourse_type"] = test["discourse_type"].apply(
        lambda x: x if x != "Concluding Statement" else "Conclusion"
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
