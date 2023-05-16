import torch
import torch.nn as nn
from typing import List, Dict
from transformers import AutoModel
import numpy as np
from utils import _flatten


def _get_tag_id_to_layer_id(ids_each_level):
    tag_id = 0
    list_id = 0
    tag_to_list = {}
    for id_list in ids_each_level:
        for i in range(len(id_list)):
            tag_to_list.update({tag_id + i: list_id})
        tag_id += len(id_list)
        list_id += 1
    return tag_to_list


def _get_first_level_ids(tagname_to_tagid: Dict[str, int]) -> List[List[List[int]]]:
    """having list of unique labels, create the labels ids in different lists"""
    all_names = list(tagname_to_tagid.keys())
    split_names = [name.split("->") for name in all_names]

    assert np.all([len(name_list) == 3 for name_list in split_names])
    final_ids = []

    tag_id = 0
    first_level_names = list(np.unique([name_list[0] for name_list in split_names]))
    for first_level_name in first_level_names:
        first_level_ids = []
        kept_names = [
            name_list[1:]
            for name_list in split_names
            if name_list[0] == first_level_name
        ]
        second_level_names = list(np.unique([name[0] for name in kept_names]))
        for second_level_name in second_level_names:
            second_level_ids = []
            third_level_names = [
                name_list[1]
                for name_list in kept_names
                if name_list[0] == second_level_name
            ]
            for _ in range(len(third_level_names)):
                second_level_ids.append(tag_id)
                tag_id += 1
            first_level_ids.append(second_level_ids)
        final_ids.append(first_level_ids)

    return final_ids


class MultilabelArchitecture(torch.nn.Module):
    """
    base architecture, used for finetuning the transformer model.
    """

    def __init__(
        self,
        model_name_or_path,
        tagname_to_tagid: Dict[str, int],
        dropout_rate: float,
        n_freezed_layers: int,
    ):
        super().__init__()

        self.ids_each_level = _get_first_level_ids(tagname_to_tagid)

        self.n_level0_ids = len(self.ids_each_level)
        self.n_heads = len(_flatten(self.ids_each_level))
        self.tag_id_to_layer_id = _get_tag_id_to_layer_id(self.ids_each_level)

        self.backbone = AutoModel.from_pretrained(model_name_or_path)
        self.transformer_output_length = self.backbone.config.hidden_size
        self.backbone.encoder.layer = self.backbone.encoder.layer[:-1]

        # freeze embeddings
        for param in self.backbone.embeddings.parameters():
            param.requires_grad = False
        # freeze n_freezed_layers first layers
        if n_freezed_layers > 0:
            for layer in self.backbone.encoder.layer[:n_freezed_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        self.LayerNorm_specific_hidden = torch.nn.ModuleList(
            [
                torch.nn.LayerNorm(self.transformer_output_length)
                for _ in range(self.n_level0_ids)
            ]
        )

        self.dropout = torch.nn.Dropout(dropout_rate)

        self.specific_layer = torch.nn.ModuleList(
            [
                AutoModel.from_pretrained(model_name_or_path).encoder.layer[-1]
                for _ in range(self.n_level0_ids)
            ]
        )

        self.output_layer = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.transformer_output_length, len(id_one_level))
                for id_one_level in _flatten(self.ids_each_level)
            ]
        )

        # self.activation_function = nn.SELU()

    def forward(self, inputs):

        fith_layer_transformer_output = self.backbone(
            inputs["ids"],
            attention_mask=inputs["mask"],
        ).last_hidden_state

        encoder_outputs = [
            self.specific_layer[i](fith_layer_transformer_output.clone())[0][:, 0, :]
            for i in range(self.n_level0_ids)
        ]

        embedding_output = torch.cat(encoder_outputs, dim=1)

        classification_output = torch.cat(
            [
                self.output_layer[tag_id](
                    self.LayerNorm_specific_hidden[self.tag_id_to_layer_id[tag_id]](
                        self.dropout(
                            # self.activation_function(
                            encoder_outputs[self.tag_id_to_layer_id[tag_id]].clone()
                            # )
                        )
                    )
                )
                for tag_id in range(self.n_heads)
            ],
            dim=1,
        )

        return classification_output, embedding_output
