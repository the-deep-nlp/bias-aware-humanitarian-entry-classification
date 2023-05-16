import torch
from typing import Dict, List
from models.architectures.base_architecture import BaseArchitecture
from models.architectures.multilabel_architecture import MultilabelArchitecture

architecture_setups = ["base_architecture", "multiabel_architecture"]
n_possible_nb_mid_layers = [0, 1, 2]


class ModelArchitecture(torch.nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        tagname2tagid_normal: Dict[str, int],
        dropout_rate: float,
        n_freezed_layers: int,
        architecture_setup: str,
    ):
        super().__init__()

        assert (
            architecture_setup in architecture_setups
        ), f"'architecture_setup' arg must be one of {architecture_setups}, got {architecture_setup} instead."

        if architecture_setup == "base_architecture":
            self.model = BaseArchitecture(
                model_name_or_path,
                len(tagname2tagid_normal),
                dropout_rate,
                n_freezed_layers,
            )
        else:
            self.model = MultilabelArchitecture(
                model_name_or_path,
                tagname2tagid_normal,
                dropout_rate,
                n_freezed_layers,
            )

    def forward(self, inputs):
        # for explainability
        explainability_bool = type(inputs) is tuple
        if explainability_bool:
            model_device = next(self.parameters()).device
            inputs = {
                "ids": inputs[0].to(model_device),
                "mask": inputs[1].to(model_device),
            }

        classification_output, embedding_output = self.model(inputs)

        outputs = {
            "classification": classification_output,
            "embeddings": embedding_output,
        }

        return outputs
