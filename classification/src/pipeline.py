import os
import multiprocessing
from typing import Dict, List, Set, Union
import pandas as pd
import torch
import json
import pytorch_lightning as pl
from tqdm import tqdm

tqdm.pandas()
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only
from transformers import AutoTokenizer

# debiasing methods import
from debiasing_pipelines.counterfactual_prediction_discrepency.counterfactual_prediction_discrepency import (
    get_probabilities_results,
)

# from debiasing_pipelines.probing.classifier.model import ProbingClassificationModel

# test set biases import
from humbias_set_creation.initialize_datasets import (
    _preprocess_df,
    _initialize_test_counterfactual,
    _initialize_counterfactual_dataset,
)

# base classification model import
from models.model import ClassificationTransformer
from utils import _hypertune_threshold, _generate_test_set_results

os.environ["TOKENIZERS_PARALLELISM"] = "false"

checkpoint_callback_params = {
    "save_top_k": 1,
    "verbose": True,
    "monitor": "val_loss",
    "mode": "min",
}

BIASES_TYPES = ["gender", "country"]
architecture_setups = ["base_architecture", "multiabel_architecture"]

model_huggingface_name2reffered_name = {
    "nlp-thedeep/humbert": "humbert",
    "xlm-roberta-base": "xlm-r",
    "nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large": "distil-xlm-r",
    "bert-base-multilingual-cased": "mbert",
}


class MyModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @rank_zero_only
    def _del_model(self, *_):
        pass

    def _save_model(self, *_):
        pass


class TrainingPipeline:
    def __init__(
        self,
        hyperparameters: Dict,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        architecture_setup: str,
        training_setup: str,
        backbone_name: str,
        results_dir: str,
        humbias_set_dir: str,
    ):
        self.hyperparameters = hyperparameters
        self.architecture_setup = architecture_setup
        self.training_setup = training_setup
        self.backbone_name = backbone_name

        self.counterfactual_training = training_setup in [
            "counterfactual_debiasing",
            "all_debiasing",
        ]

        # if not os.path.exists(results_dir):
        #     os.makedirs(results_dir)

        # backbone_results = os.path.join(
        #     results_dir,
        # )
        # if not os.path.exists(backbone_results):
        #     os.makedirs(backbone_results)

        self.RESULTS_DIR = os.path.join(
            results_dir,
            model_huggingface_name2reffered_name[backbone_name],
            self.training_setup,
            self.architecture_setup,
        )
        if not os.path.exists(self.RESULTS_DIR):
            os.makedirs(self.RESULTS_DIR)

        self.HUMBIAS_SET_DIR = humbias_set_dir
        if not os.path.exists(self.HUMBIAS_SET_DIR):
            os.mkdir(self.HUMBIAS_SET_DIR)

        self.results_name = f"{model_huggingface_name2reffered_name[self.backbone_name]}_{self.training_setup}_{self.architecture_setup}"
        print(
            "###########################################################################################################"
        )
        print(
            f"######################   START RUN FOR {self.results_name}   ###############################"
        )
        print(
            "###########################################################################################################"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone_name)

        self.train_df = _preprocess_df(train_df)
        self.val_df = _preprocess_df(val_df)
        self.test_df = _preprocess_df(test_df)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = 1 if torch.cuda.is_available() else 0

        self.focal_loss_training = training_setup in [
            "focal_loss_debiasing",
            "all_debiasing",
        ]
        # self._run_sanity_checks()
        self._initialize_bias_datasets()
        self._initialize_training_args()

        if training_setup != "no_finetuning":
            self.train_classification_model()

    #################################### BASE MODEL #######################################

    def _initialize_training_args(self):
        train_df, val_df, test_df = (
            self.train_df.copy(),
            self.val_df.copy(),
            self.test_df.copy(),
        )

        self.train_params = {
            "batch_size": self.hyperparameters["train_batch_size"],
            "shuffle": True,
            "num_workers": 4,
        }

        self.val_params = {
            "batch_size": self.hyperparameters["val_batch_size"],
            "shuffle": False,
            "num_workers": 4,
        }

        self.train_params_probing = {
            "batch_size": self.hyperparameters["train_batch_size"] * 8,
            "shuffle": True,
            "num_workers": 2,
        }

        self.val_params_probing = {
            "batch_size": self.hyperparameters["val_batch_size"] * 8,
            "shuffle": False,
            "num_workers": 2,
        }

        self.trainer = pl.Trainer(
            logger=None,
            callbacks=[
                # early_stopping_callback,
                MyModelCheckpoint(monitor="val_loss", mode="min"),
            ],  # self.checkpoint_callback],
            progress_bar_refresh_rate=5,
            profiler="simple",
            log_gpu_memory=True,
            weights_summary=None,
            gpus=self.n_gpu,
            precision=16,
            accumulate_grad_batches=1,
            max_epochs=self.hyperparameters["n_epochs"],
            gradient_clip_val=1,
            gradient_clip_algorithm="norm",
        )

        self.model = ClassificationTransformer(
            model_name_or_path=self.backbone_name,
            train_dataset=train_df,
            val_dataset=val_df,
            test_dataset=test_df,
            train_params=self.train_params,
            val_params=self.val_params,
            tokenizer=self.tokenizer,
            plugin="deepspeed_stage_3_offload",
            accumulate_grad_batches=1,
            max_epochs=self.hyperparameters["n_epochs"],
            dropout_rate=self.hyperparameters["dropout"],
            weight_decay=self.hyperparameters["weight_decay"],
            learning_rate=self.hyperparameters["learning_rate"],
            max_len=self.hyperparameters["max_len"],
            n_freezed_layers=self.hyperparameters["n_freezed_layers"],
            architecture_setup=self.architecture_setup,
            focal_loss_training=self.focal_loss_training,
        )

    def _initialize_bias_datasets(self):
        """
        add one column for each kword (add 2 columns for gender (male, female), n_nationalities columns for nationalities etc.)
        """

        # conterfactual test
        COUNTERFACTUAL_FOLDER = os.path.join(
            self.HUMBIAS_SET_DIR, "counterfactual_test"
        )
        if not os.path.exists(COUNTERFACTUAL_FOLDER):
            os.mkdir(COUNTERFACTUAL_FOLDER)
        self.counterfactual_test_dfs = _initialize_test_counterfactual(
            self.test_df, self.hyperparameters["max_len"], self.tokenizer
        )

        for bias_type, bias_df in self.counterfactual_test_dfs.items():
            bias_df.to_csv(
                os.path.join(COUNTERFACTUAL_FOLDER, f"{bias_type}.csv"), index=None
            )

        if self.counterfactual_training:
            self.train_df = _initialize_counterfactual_dataset(self.train_df)
            self.val_df = _initialize_counterfactual_dataset(self.val_df)
            COUNTERFACTUAL_FOLDER = os.path.join(self.HUMBIAS_SET_DIR, "counterfactual")
            if not os.path.exists(COUNTERFACTUAL_FOLDER):
                os.mkdir(COUNTERFACTUAL_FOLDER)
            self.train_df.to_csv(
                os.path.join(COUNTERFACTUAL_FOLDER, "train.csv"), index=None
            )
            self.val_df.to_csv(
                os.path.join(COUNTERFACTUAL_FOLDER, "val.csv"), index=None
            )

    def train_classification_model(self):
        self.trainer.fit(self.model)
        self.model.optimal_thresholds = _hypertune_threshold(self.model, self.val_df)

        CLASSIFICATION_RESULTS_DIR = os.path.join(self.RESULTS_DIR, "classification")
        if not os.path.exists(CLASSIFICATION_RESULTS_DIR):
            os.makedirs(CLASSIFICATION_RESULTS_DIR)

        threshold_file_name = f"threshold_values_{self.results_name}.json"
        with open(
            os.path.join(CLASSIFICATION_RESULTS_DIR, threshold_file_name), "w"
        ) as fp:
            json.dump(self.model.optimal_thresholds, fp)

        classification_file_name = (
            f"classification_results_test_df_{self.results_name}.csv"
        )
        self.test_set_results = (
            _generate_test_set_results(self.model, self.test_df)
            .sort_values(by="tag")
            .drop(columns=["positive_examples_proportion"])
        )
        self.test_set_results.to_csv(
            os.path.join(CLASSIFICATION_RESULTS_DIR, classification_file_name)
        )

        results_probabilities_one_type = get_probabilities_results(
            self.model, self.test_df.copy()
        )
        test_df_probabilities_file_name = (
            f"probabilities_results_test_df_{self.results_name}.csv"
        )
        results_probabilities_one_type.to_csv(
            os.path.join(
                CLASSIFICATION_RESULTS_DIR,
                test_df_probabilities_file_name,
            ),
            index=None,
        )

    ############################# COUNTERFACTUAL PREDICTIONS DISCREPENCY #######################################

    def get_counterfactual_predictions_discrepency_results(self):
        PROBABILITIES_RESULTS_DIR = os.path.join(
            self.RESULTS_DIR, "predictions_discrepency"
        )
        if not os.path.exists(PROBABILITIES_RESULTS_DIR):
            os.makedirs(PROBABILITIES_RESULTS_DIR)

        self.output_counterfactual_predictions_discrepency_results = {}

        for one_bias_type, one_bias_df in self.counterfactual_test_dfs.items():
            file_name = (
                f"{self.results_name}_{one_bias_type}_predictions_discrepency.csv"
            )

            results_probabilities_one_type = get_probabilities_results(
                self.model, one_bias_df
            )

            results_probabilities_one_type.to_csv(
                os.path.join(
                    PROBABILITIES_RESULTS_DIR,
                    file_name,
                ),
                index=None,
            )
