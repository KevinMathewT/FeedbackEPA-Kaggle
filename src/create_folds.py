import os
import re
import sys
import math
import yaml
import glob
import codecs
import joblib
import random
import difflib
from pprint import pprint
from typing import Tuple
from tqdm import tqdm
from pathlib import Path
from text_unidecode import unidecode
from collections import Counter, defaultdict

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GroupKFold

from transformers import AutoTokenizer

from config import config
from .utils import seed_everything

tqdm.pandas()
pd.set_option("display.max_columns", 500)
# sws = stopwords.words("english") + ["n't",  "'s", "'ve"]
tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"], use_fast=True)
context = "train"
essay_id_blocks = []


def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start : error.end].encode("utf-8"), error.end


def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start : error.end].decode("cp1252"), error.end


codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)


def resolve_encodings_and_normalize(text: str) -> str:
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)
    return text


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


def other_discourse_type_context(x):
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
    x["discourse_text"] = x["discourse_text"].apply(lambda x: x.strip())

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


def surrounding_context(x):
    essay_id = x["essay_id"].to_list()[0]
    essay = open(
        Path(config[context + "_base"]) / (essay_id + ".txt"), "rt", encoding="utf-8"
    ).read()
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
    x["discourse_text"] = x["discourse_text"].apply(lambda x: x.strip())

    x["tokens_size"] = x["discourse_text"].apply(lambda x: len(tokenizer.tokenize(x)))
    texts = x["discourse_text"].tolist()
    tokens_sizes = x["tokens_size"].tolist()

    essays = []

    for i in range(x.shape[0]):
        b = (
            config["max_length"]
            - 2 * tokens_sizes[i]
            - len(tokenizer.tokenize(discourse_type[i]))
            - 1
        )
        discourse = texts[i]
        search_text = discourse[
            len(discourse) // 4 : len(discourse) // 4 + len(discourse) // 2
        ]

        if search_text not in essay:
            print("*" * 10)
            print(essay_id)
            print("*" * 10)
            print(search_text)
            print("*" * 10)
            print(essay)
            print("*" * 10)
            sys.exit(0)

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


def surrounding_context(meta):
    essay_id = meta["essay_id"]
    essay = open(
        Path(config[context + "_base"]) / (essay_id + ".txt"), "rt", encoding="utf-8"
    ).read()
    discourse_type = meta["discourse_type"]
    discourse_text = meta["discourse_text"].strip()

    search_text = discourse_text[
        len(discourse_text) // 4 : len(discourse_text) // 4 + len(discourse_text) // 2
    ]
    s = essay.find(discourse_text)
    b = (
        config["max_length"]
        - 2 * len(tokenizer.tokenize(discourse_text))
        - len(tokenizer.tokenize(discourse_type))
        - 1
    )

    x = len(tokenizer.tokenize(essay[:s]))
    y = len(tokenizer.tokenize(essay[s:]))
    p = b // 2
    q = b - p

    # if x > y:

    tokens = tokenizer.tokenize(
        discourse_type + " " + discourse_text + " " + tokenizer.sep_token + " " + essay
    )
    print(len(tokens))

    return essay


def add_topics(df, train=True):
    if train:
        topic_pred_df = pd.read_csv(config['feedback_csv'])
        topic_pred_df = topic_pred_df.drop(columns={'prob'})
        topic_pred_df = topic_pred_df.rename(columns={'id': 'essay_id'})

        topic_meta_df = pd.read_csv(config['feedmeta_csv'])
        topic_meta_df = topic_meta_df.rename(columns={'Topic': 'topic', 'Name': 'topic_name'}).drop(columns=['Count'])
        topic_meta_df.topic_name = topic_meta_df.topic_name.apply(lambda n: ' '.join(n.split('_')[1:]))

        topic_pred_df = topic_pred_df.merge(topic_meta_df, on='topic', how='left')

        df = df.merge(topic_pred_df, on='essay_id', how='left')
    
    else:
        from bertopic import BERTopic
        topic_model = BERTopic.load(config['feedback_model'])

        sws = stopwords.words("english") + ["n't",  "'s", "'ve"]
        fls = glob.glob(config['test_base'] + "*.txt")
        docs = []
        for fl in tqdm(fls):
            with open(fl) as f:
                txt = f.read()
                word_tokens = word_tokenize(txt)
                txt = " ".join([w for w in word_tokens if not w.lower() in sws])
            docs.append(txt)

        topics, probs = topic_model.transform(docs)
        pred_topics = pd.DataFrame()
        dids = list(map(lambda fl: fl.split("/")[-1].split(".")[0], fls))
        pred_topics["id"] = dids
        pred_topics["topic"] = topics
        pred_topics['prob'] = probs
        pred_topics = pred_topics.drop(columns={'prob'})
        pred_topics = pred_topics.rename(columns={'id': 'essay_id'})
        pred_topics = pred_topics.merge(topic_meta_df, on='topic', how='left')
        pred_topics

        df = df.merge(pred_topics, on='essay_id', how='left')

    return df

def get_discource_context(meta):
    print(meta)
    essay_id = meta["essay_id"]
    essay = resolve_encodings_and_normalize(
        open(Path(config[context + "_base"]) / (essay_id + ".txt"), "r").read()
    )
    discourse_type = meta["discourse_type"]
    discourse_text = meta["discourse_text"].strip()
    discourse_text = re.sub(r" \Z", "", discourse_text)
    topic_name = meta['topic_name'].strip()
    text = (
        discourse_type
        + tokenizer.sep_token
        + topic_name
        + tokenizer.sep_token
        + discourse_text
        + tokenizer.sep_token
        + essay
    )

    # tokens = tokenizer.tokenize(text)
    # print(len(tokens))

    return text


if __name__ == "__main__":
    seed_everything(config["seed"])

    train = pd.read_csv(config["train_csv"])
    test = pd.read_csv(config["test_csv"], train=True)
    sample_submission = pd.read_csv(config["sample_csv"])

    train = add_topics(train)
    test = add_topics(test)

    train["discourse_text"] = train["discourse_text"].apply(
        lambda x: resolve_encodings_and_normalize(x)
    )
    test["discourse_text"] = test["discourse_text"].apply(
        lambda x: resolve_encodings_and_normalize(x)
    )

    # Other Discourse type Context
    # context = "train"
    # essay_id_blocks = []
    # train.groupby("essay_id").apply(surrounding_context)
    # train = pd.concat(essay_id_blocks, ignore_index=True)
    #
    # context = "test"
    # essay_id_blocks = []
    # test.groupby("essay_id").apply(surrounding_context)
    # test = pd.concat(essay_id_blocks, ignore_index=True)

    # Getting essay full text
    train["text"] = train[["essay_id", "discourse_type", "discourse_text", "topic_name"]].apply(
        get_discource_context,
        axis=1,
    )
    context = "test"
    test["text"] = test[["essay_id", "discourse_type", "discourse_text", "topic_name"]].apply(
        get_discource_context,
        axis=1,
    )

    # Surrounding Context
    # train["text"] = train[["essay_id", "discourse_type", "discourse_text"]].apply(
    #     surrounding_context,
    #     axis=1,
    # )
    # test["text"] = test[["essay_id", "discourse_type", "discourse_text"]].apply(
    #     surrounding_context,
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
            n_splits=config["folds"], shuffle=True, random_state=config["seed"]
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
