import torch
from typing import List, Dict

n_possible_nb_mid_layers = [0, 1, 2]


def _get_probing_architecture(
    tagname2tagid_adv_one_bias_type: Dict[str, int],
    embedding_length: int,
    dropout_rate: float,
    n_mid_layers: int = 1,
):

    assert (
        n_mid_layers in n_possible_nb_mid_layers
    ), f"'n_mid_layers' arg must be one of {n_possible_nb_mid_layers}"

    midlayer_activation_function = torch.nn.SELU()
    # output_activation_function = torch.nn.Softmax()

    if n_mid_layers == 0:

        architecture = torch.nn.Sequential(
            torch.nn.Linear(embedding_length, len(tagname2tagid_adv_one_bias_type)),
            # output_activation_function,
        )

    elif n_mid_layers == 1:
        mid_layer_length = 64

        architecture = torch.nn.Sequential(
            torch.nn.Linear(embedding_length, mid_layer_length),
            torch.nn.Dropout(dropout_rate),
            midlayer_activation_function,
            torch.nn.BatchNorm1d(mid_layer_length),
            torch.nn.Linear(mid_layer_length, len(tagname2tagid_adv_one_bias_type)),
            # output_activation_function,
        )
    else:
        first_mid_layer_output_length = 128
        second_mid_layer_output_length = 32
        architecture = torch.nn.Sequential(
            torch.nn.Linear(embedding_length, first_mid_layer_output_length),
            torch.nn.Dropout(dropout_rate),
            midlayer_activation_function,
            torch.nn.BatchNorm1d(first_mid_layer_output_length),
            torch.nn.Linear(
                first_mid_layer_output_length, second_mid_layer_output_length
            ),
            torch.nn.Dropout(dropout_rate),
            midlayer_activation_function,
            torch.nn.BatchNorm1d(second_mid_layer_output_length),
            torch.nn.Linear(
                second_mid_layer_output_length, len(tagname2tagid_adv_one_bias_type)
            ),
            # output_activation_function,
        )

    return architecture
