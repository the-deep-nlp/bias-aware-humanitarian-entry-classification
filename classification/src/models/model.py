import os

from tqdm.auto import tqdm
import torch.nn.functional as F
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
from transformers import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from models.data import CustomDataset
from models.architecture import ModelArchitecture
from models.loss import FocalLoss
from utils import _get_tagname_to_id, _get_target_weights

architecture_setups = ["base_architecture", "multiabel_architecture"]
protected_attributes = ["gender", "country"]
possible_setups = [
    "hypertuning_threshold",
    "testing",
    "raw_predictions",
    "model_embeddings",
]


class ClassificationTransformer(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        train_dataset: pd.DataFrame,
        val_dataset: pd.DataFrame,
        test_dataset: pd.DataFrame,
        train_params,
        val_params,
        tokenizer,
        n_freezed_layers: int,
        learning_rate: float,
        weight_decay: float,
        dropout_rate: float,
        max_len: int,
        architecture_setup: str,
        focal_loss_training: bool,
        adam_epsilon: float = 1e-7,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.adam_epsilon = adam_epsilon

        all_targets = (
            train_dataset.target_classification.tolist()
            + val_dataset.target_classification.tolist()
            + test_dataset.target_classification.tolist()
        )

        self.tagname_to_tagid_classification = _get_tagname_to_id(all_targets)
        # print("tagname to tagid classification", self.tagname_to_tagid_classification)

        self.max_len = max_len
        self.model = ModelArchitecture(
            model_name_or_path,
            self.tagname_to_tagid_classification,
            dropout_rate,
            n_freezed_layers,
            architecture_setup=architecture_setup,
        )
        self.tokenizer = tokenizer
        self.train_params = train_params
        self.val_params = val_params
        self.max_len = max_len

        self.training_loader = self._get_loaders(
            dataset=train_dataset,
            params=train_params,
        )

        self.val_loader = self._get_loaders(
            dataset=val_dataset,
            params=val_params,
        )

        if focal_loss_training:
            self.criterion_classification = FocalLoss(gamma=2)
        else:
            self.criterion_classification = nn.BCEWithLogitsLoss(
                # pos_weight=_get_target_weights(
                #    targets=all_targets,
                #    tagname_to_tagid=self.tagname_to_tagid_classification,
                # )
            )

        # nn.BCEWithLogitsLoss(
        #     # pos_weight=_get_target_weights(
        #     #    targets=all_targets,
        #     #    tagname_to_tagid=self.tagname_to_tagid_classification,
        #     # )
        # )

    def forward(self, inputs):
        return self.model(inputs)

    def _compute_loss(self, batch):
        outputs_all = self(batch)
        outputs_classification = outputs_all["classification"]

        loss = self.criterion_classification(
            outputs_classification,
            batch["target_classification"],  # .to(torch.float64),
        )

        losses_adv = []

        if len(losses_adv) > 0:
            loss += sum(losses_adv)

        return loss

    def training_step(self, batch, batch_idx):
        train_loss = self._compute_loss(batch)

        self.log(
            "train_loss",
            train_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=False,
        )

        return train_loss

    def validation_step(self, batch, batch_idx):
        val_loss = self._compute_loss(batch)

        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=False,
        )

        return {"val_loss": val_loss}

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        optimizer = AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            eps=self.adam_epsilon,
        )

        scheduler = StepLR(optimizer, step_size=1, gamma=0.4)

        scheduler = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
        }
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return self.training_loader

    def val_dataloader(self):
        return self.val_loader

    def _get_loaders(
        self,
        dataset,
        params,
    ):
        custom_set = CustomDataset(
            dataframe=dataset,
            tagname_to_tagid_classification=self.tagname_to_tagid_classification,
            tokenizer=self.tokenizer,
            max_len=self.max_len,
        )

        loader = DataLoader(custom_set, **params, pin_memory=True)

        return loader

    def custom_predict(self, test_dataset, setup: str):
        """
        1) get raw predictions
        2) postprocess them to output an output compatible with what we want in the inference
        """

        assert setup in possible_setups, f"'setup' must be one of {possible_setups}"

        test_loader = self._get_loaders(
            test_dataset,
            self.val_params,
        )

        testing_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.to(testing_device)
        self.eval()
        # self.freeze()

        predictions, embeddings, y_true = [], [], []

        with torch.no_grad():
            for batch in tqdm(
                test_loader,
                total=len(test_loader.dataset) // test_loader.batch_size,
            ):
                inputs = {
                    "ids": batch["ids"].to(testing_device),
                    "mask": batch["mask"].to(testing_device),
                }
                if setup == "hypertuning_threshold":
                    y_true.append(batch["target_classification"])

                logits = self(inputs)  # .cpu()

                predictions.append(logits["classification"].cpu())
                embeddings.append(logits["embeddings"].cpu())

        if setup == "model_embeddings":
            return torch.cat(embeddings, dim=0)

        else:
            predictions = torch.cat(predictions, dim=0)
            predictions = torch.sigmoid(predictions)

            if setup == "hypertuning_threshold":
                y_true = torch.cat(y_true)
                return predictions, y_true

            elif setup == "testing":
                final_outputs = [
                    [
                        tagname
                        for tagname, optimal_threshold in (
                            self.optimal_thresholds.items()
                        )
                        if probas_one_excerpt[
                            self.tagname_to_tagid_classification[tagname]
                        ].item()
                        >= optimal_threshold
                    ]
                    for probas_one_excerpt in predictions
                ]

                return final_outputs

            else:
                probabilities_dict = {
                    target: predictions[:, j]
                    for target, j in self.tagname_to_tagid_classification.items()
                }

                return probabilities_dict
