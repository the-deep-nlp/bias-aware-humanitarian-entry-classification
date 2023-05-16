import torch
from transformers import AutoModel


class BaseArchitecture(torch.nn.Module):
    def __init__(
        self,
        model_name_or_path,
        n_output_normal,
        dropout_rate: float,
        n_freezed_layers: int,
    ):
        super().__init__()
        self.n_level0_ids = 1  # plain structure -> only one task, used for adversarial training MLP dimension
        self.backbone = AutoModel.from_pretrained(model_name_or_path)
        # freeze embeddings
        for param in self.backbone.embeddings.parameters():
            param.requires_grad = False
        # freeze n_freezed_layers first layers
        if n_freezed_layers > 0:
            for layer in self.backbone.encoder.layer[:n_freezed_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        output_length = self.backbone.config.hidden_size

        self.classification_layer = torch.nn.Sequential(
            torch.nn.LayerNorm(output_length),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(output_length, n_output_normal),
        )

    def forward(self, inputs):

        embedding_output = self.backbone(
            inputs["ids"],
            attention_mask=inputs["mask"],
        ).last_hidden_state[:, 0, :]

        classification_output = self.classification_layer(embedding_output.clone())

        return classification_output, embedding_output
