import torch
import random
from typing import Dict
import pandas as pd
from torch.utils.data import Dataset
from humbias_set_creation.utils import _run_n_labels_sanity_check


class CustomDataset(Dataset):
    """
    Adv transformers custom dataset
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tagname_to_tagid_classification: Dict[str, int],
        tokenizer,
        max_len: int,
    ):

        self.tokenizer = tokenizer
        self.data = dataframe

        self.target_classification = None
        self.targets_adv = None

        if dataframe is None:
            self.excerpt_text = None

        elif type(dataframe) is str:
            self.excerpt_text = [dataframe]

        elif type(dataframe) is list:
            self.excerpt_text = dataframe

        elif type(dataframe) is pd.Series:
            self.excerpt_text = dataframe.tolist()

        else:
            self.excerpt_text = dataframe["excerpt"].tolist()
            self.target_classification = dataframe["target_classification"].tolist()

        self.tagname_to_tagid_classification = tagname_to_tagid_classification
        self.tagid_to_tagname_task = list(tagname_to_tagid_classification.keys())

        self.max_len = max_len

    def _get_encoded_one_targets_type(self, targets_one_excerpt, tagname_to_tag_id):
        target_indices = [tagname_to_tag_id[target] for target in targets_one_excerpt]
        encoded_one_targets_type = torch.zeros(len(tagname_to_tag_id), dtype=float)
        encoded_one_targets_type[target_indices] = 1.0

        # encoded_one_targets_type = torch.tensor(encoded_one_targets_type, dtype=float)

        return encoded_one_targets_type

    def encode_example(self, excerpt_text: str, index=None, as_batch: bool = False):

        inputs = self.tokenizer(
            excerpt_text,
            None,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        encoded = {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
        }

        if self.target_classification:
            encoded["target_classification"] = self._get_encoded_one_targets_type(
                self.target_classification[index], self.tagname_to_tagid_classification
            )

        if self.targets_adv:

            for debiasing_task, debiasing_targets_task in self.targets_adv.items():

                encoded[
                    f"targets_{debiasing_task}"
                ] = self._get_encoded_one_targets_type(
                    debiasing_targets_task[index],
                    self.tagname_to_tagid_adv[debiasing_task],
                )

                mask_adv = torch.tensor(
                    [
                        1 if len(one_adv_target) > 0 else 0
                        for one_adv_target in debiasing_targets_task
                    ],
                    dtype=torch.long,
                )

                encoded[f"mask_{debiasing_task}"] = mask_adv

        return encoded

    def __len__(self):
        return len(self.excerpt_text)

    def __getitem__(self, index):
        excerpt_text = str(self.excerpt_text[index])
        return self.encode_example(excerpt_text, index)
