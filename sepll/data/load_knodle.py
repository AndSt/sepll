from typing import List, Union, Any

import os
import joblib

import numpy as np
import pandas as pd

from transformers import AutoTokenizer


def filter_small_lfs(Z: np.ndarray, T: np.ndarray, min_matches: int = 100):
    counts = Z.sum(axis=0)
    filtered_lfs = np.where(counts >= min_matches)[0]

    Z = Z[:, filtered_lfs]
    T = T[filtered_lfs, :]
    return filtered_lfs, Z, T


def convert_text_to_transformer_input(tokenizer, texts: List[str], max_length: int = 128) -> Union[Any, Any]:
    encoding = tokenizer(texts, return_tensors="jax", padding=True, truncation=True, max_length=max_length)
    input_ids = encoding.get('input_ids')
    attention_mask = encoding.get('attention_mask')

    return input_ids, attention_mask


def df_to_transformers_input(
        df: pd.DataFrame, column: str = "sample", transformers_model: str = "roberta-base", max_length: int = 128
):
    tokenizer = AutoTokenizer.from_pretrained(transformers_model)
    input_ids, attention_mask = convert_text_to_transformer_input(tokenizer, df[column].tolist(), max_length=max_length)
    return input_ids, attention_mask


def get_weak_tokenized_data(
        data_dir: str, min_matches: int = None, transformer_model: str = "roberta-base", max_length: int = 128
):
    if transformer_model is None:
        raise ValueError("Provide transformer model")

    X = pd.read_csv(os.path.join(data_dir, "train_df.csv"), sep="\t")
    input_ids, attention_mask = df_to_transformers_input(
        X, column="sample", transformers_model=transformer_model, max_length=max_length
    )

    T = joblib.load(os.path.join(data_dir, "mapping_rules_labels_t.lib"))
    Z_orig = joblib.load(os.path.join(data_dir, "train_rule_matches_z.lib"))

    assert attention_mask.shape[0] == Z_orig.shape[0]

    # load train data, s.th. for every match there is a (x, y) pair added to the train set.
    filtered_lfs = np.arange(0, Z_orig.shape[1])
    if isinstance(min_matches, int):
        filtered_lfs, Z, T = filter_small_lfs(Z_orig, T, min_matches=min_matches)
    else:
        Z = Z_orig

    train_input_ids = []
    train_attention_masks = []
    y_train = []

    unlabelled_input_ids = []
    unlabelled_attention_masks = []

    Z_labelled = []
    Z_unlabelled = []

    for i in range(Z.shape[0]):
        if Z[i].sum() == 0:
            unlabelled_input_ids.append(input_ids[i])
            unlabelled_attention_masks.append(attention_mask[i])
            Z_unlabelled.append(Z[i])
        for j in np.nonzero(Z[i])[0]:
            train_input_ids.append(input_ids[i])
            train_attention_masks.append(attention_mask[i])
            y_train.append(j)
            Z_labelled.append(Z[i])

    X_train = np.array(train_input_ids), np.array(train_attention_masks)
    y_train = np.array(y_train)

    X_unlabelled = np.array(unlabelled_input_ids), np.array(unlabelled_attention_masks)

    Z_labelled = np.array(Z_labelled)
    Z_unlabelled = np.array(Z_unlabelled)
    try:
        X_dev = pd.read_csv(os.path.join(data_dir, "dev_df.csv"), sep="\t")
        y_dev = X_dev["label"].to_numpy(dtype=np.int32)
        X_dev = df_to_transformers_input(
            X_dev, column="sample", transformers_model=transformer_model, max_length=max_length
        )
        Z_dev = joblib.load(os.path.join(data_dir, "dev_rule_matches_z.lib"))[:, filtered_lfs]
    except FileNotFoundError:
        X_dev, y_dev, Z_dev = None, None, None
    X_test = pd.read_csv(os.path.join(data_dir, "test_df.csv"), sep="\t")
    y_test = X_test["label"].to_numpy(dtype=np.int32)
    X_test = df_to_transformers_input(
        X_test, column="sample", transformers_model=transformer_model, max_length=max_length
    )
    Z_test = joblib.load(os.path.join(data_dir, "test_rule_matches_z.lib"))[:, filtered_lfs]

    return (
        X_train, y_train, X_unlabelled,
        T, (Z_labelled, Z_unlabelled),
        X_dev, y_dev, Z_dev,
        X_test, y_test.astype(np.int32), Z_test
    )
