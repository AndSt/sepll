from typing import Dict

import os
import json

import joblib
import numpy as np
import pandas as pd


def wrench_to_knodle_format(wrench_dir: str, target_dir: str):
    train_df, train_Z, train_T, col_row_dict = wrench_json_to_files(wrench_dir, "train")

    train_df.to_csv(os.path.join(target_dir, "train_df.csv"), sep="\t", index=None)
    joblib.dump(train_Z, os.path.join(target_dir, "train_rule_matches_z.lib"))

    if os.path.isfile(os.path.join(wrench_dir, "valid.json")):
        dev_df, dev_Z, dev_T, _ = wrench_json_to_files(wrench_dir, "valid", col_row_dict=col_row_dict)

        dev_df.to_csv(os.path.join(target_dir, "dev_df.csv"), sep="\t", index=None)
        joblib.dump(dev_Z, os.path.join(target_dir, "dev_rule_matches_z.lib"))
    else:
        dev_T = None

    test_df, test_Z, test_T, _ = wrench_json_to_files(wrench_dir, "test", col_row_dict=col_row_dict)

    test_df.to_csv(os.path.join(target_dir, "test_df.csv"), sep="\t", index=None)
    joblib.dump(test_Z, os.path.join(target_dir, "test_rule_matches_z.lib"))

    if dev_T is not None:
        np.testing.assert_equal(dev_T.shape, test_T.shape)
    np.testing.assert_equal(train_T, test_T)

    joblib.dump(train_T, os.path.join(target_dir, "mapping_rules_labels_t.lib"))


def wrench_json_to_files(
        wrench_dir: str, split: str, col_row_dict=None
) -> [pd.DataFrame, np.ndarray, np.ndarray, Dict]:
    with open(os.path.join(wrench_dir, f"{split}.json"), "r") as f:
        json_data = json.load(f)
    df = []
    M = []

    for idx, info in json_data.items():
        data = info["data"]
        if split == "train":
            df.append([data["text"]])
        else:
            df.append([data["text"], info["label"]])
        M.append(info["weak_labels"])

    columns = ["sample"] if split == "train" else ["sample", "label"]
    df = pd.DataFrame(df, columns=columns)

    M = np.array(M)
    z_matrix, t_matrix, col_row_dict = transform_snorkel_matrix_to_z_t(M, col_row_dict=col_row_dict)

    return df, z_matrix, t_matrix, col_row_dict


def transform_snorkel_matrix_to_z_t(
        class_matrix: np.ndarray, col_row_dict=None
) -> [np.ndarray, np.ndarray, Dict]:
    """Takes a matrix in format used by e.g. Snorkel (https://github.com/snorkel-team/snorkel)
    and transforms it to z / t matrices. Format
        - class_matrix_ij = -1, iff the rule doesn't apply
        - class_matrix_ij = k, iff the rule labels class k

    :param class_matrix: shape=(num_samples, num_weak_labellers)
    :return: Z matrix - binary encoded array of which rules matched. Shape: instances x rules.
             T matrix - mapping of rules to labels, binary encoded. Shape: rules x classes.
    """

    if col_row_dict is None:
        col_row_dict = {}
        j = 0
        for i in range(class_matrix.shape[1]):
            col = set(class_matrix[:, i].tolist()) - {-1}
            if len(col) < 1:
                raise ValueError("Column has no matching classes")
            for elt in col:
                col_row_dict[j] = {
                    "original_col": i,
                    "label": elt
                }
                j += 1

    # init T matrix
    num_cols = len(col_row_dict)
    num_classes = max([elt.get("label") for elt in col_row_dict.values()]) + 1
    t_matrix = np.zeros((num_cols, num_classes))

    for key, val in col_row_dict.items():
        t_matrix[key, val.get("label")] = 1

    # init Z matrix
    z_matrix = np.zeros((class_matrix.shape[0], num_cols))

    for j, val in col_row_dict.items():
        idx = np.where(class_matrix[:, val.get("original_col")] == val.get("label"))
        z_matrix[idx, j] = 1

    return z_matrix, t_matrix, col_row_dict
