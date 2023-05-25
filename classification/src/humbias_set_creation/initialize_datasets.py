from IPython.display import display
from typing import List, Dict
import pandas as pd

from humbias_set_creation.create_countries_bias_data import (
    create_country_augmented_dataset,
    _create_country_df,
    _keep_one_country_df,
    _get_mask_one_country,
    _augment_country_df,
)

from humbias_set_creation.create_gender_bias_data import (
    create_gender_augmented_dataset,
    _create_gender_df,
    _keep_one_gender_df,
    _get_mask_one_gender,
    _augment_gender_df,
)
from humbias_set_creation.utils import _create_scraped_excerpt

from humbias_set_creation.utils import (
    _clean_biases_dataset,
    _custom_eval,
)

BIASES_TYPES = ["gender", "country"]


def _preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    classification_df = df.copy()  # .drop_duplicates()
    """
    1. clean excerpt
    2. extract gender keywords
    3. extract country keywords
    """

    classification_df["excerpt"] = classification_df["excerpt"].apply(
        lambda x: x.replace("-", " ")
        .replace("’", "'")
        .replace("`", "'")
        .replace("(", " ( ")
        .replace(")", " ) ")
        .replace("[", " [ ")
        .replace("]", " ] ")
        .replace("—", " ")
        .replace("”", "'")
    )

    classification_df.rename(columns={"targets": "target_classification"}, inplace=True)
    classification_df["target_classification"] = classification_df[
        "target_classification"
    ].apply(
        lambda x: [
            tag
            for tag in _custom_eval(x)
            if (
                "first_level_tags" in tag.lower()
                # and "secondary_tags" not in tag.lower()
                and "first_level_tags->Affected" not in tag
            )
            or "subpillars" in tag.lower()
        ]
    )

    classification_df["excerpt_type"] = "original"
    classification_df = _create_gender_df(classification_df)
    classification_df = _create_country_df(classification_df)

    return classification_df


def _initialize_test_counterfactual(
    test_df: pd.DataFrame, max_len: int, tokenizer
) -> Dict[str, pd.DataFrame]:
    gender_test_df = _keep_one_gender_df(test_df)
    country_test_df = _keep_one_country_df(test_df)

    # counterfactual
    gender_counterfactual = _clean_biases_dataset(
        create_gender_augmented_dataset(gender_test_df),
        max_len,
        tokenizer,
        task_name="gender",
    ).rename(columns={"gender_kword_type": "kword_type", "gender_keywords": "keywords"})
    # print("gender counterfactual test")
    # display(gender_counterfactual.head())
    country_counterfactual = _clean_biases_dataset(
        create_country_augmented_dataset(country_test_df),
        max_len,
        tokenizer,
        task_name="country",
    ).rename(
        columns={"country_kword_type": "kword_type", "country_keywords": "keywords"}
    )
    # print("country counterfactual test")
    # display(country_counterfactual.head())
    test_set_datasets = {
        "gender": gender_counterfactual,
        "country": country_counterfactual,
    }

    return test_set_datasets


def _initialize_counterfactual_dataset(
    df: pd.DataFrame,
):
    final_df = df.copy()
    for one_attribute in BIASES_TYPES:
        if one_attribute == "gender":
            mask_building_function = _get_mask_one_gender
            augmentation_function = _augment_gender_df

        elif one_attribute == "country":
            mask_building_function = _get_mask_one_country
            augmentation_function = _augment_country_df

        else:
            raise RuntimeError(f"attribute '{one_attribute}' not in {BIASES_TYPES}.")

        mask = mask_building_function(final_df)
        to_be_augmented_df = final_df[mask]

        augmented_df = augmentation_function(to_be_augmented_df)
        final_df = pd.concat([final_df, augmented_df])

    return final_df


######## PROBING ##########


def _run_n_labels_sanity_check(
    items_list: List[List], n_expected_items: int, task: str = ""
):
    assert all(
        [len(one_sublist) == n_expected_items] for one_sublist in items_list
    ), f"issue for {task}"


def _get_embeddings_probing(trained_classification_model, dfs: Dict[str, pd.DataFrame]):
    X_train = trained_classification_model.custom_predict(
        dfs["train"].excerpt.tolist(), setup="model_embeddings"
    )
    X_val = trained_classification_model.custom_predict(
        dfs["val"].excerpt.tolist(), setup="model_embeddings"
    )
    X_test = trained_classification_model.custom_predict(
        dfs["test"].excerpt.tolist(), setup="model_embeddings"
    )

    Y_train = dfs["train"]["target"].tolist()
    Y_val = dfs["val"]["target"].tolist()
    Y_test = dfs["test"]["target"].tolist()

    _run_n_labels_sanity_check(Y_train, 1, f"prompt {Y_train}")
    _run_n_labels_sanity_check(Y_val, 1, f"prompt {Y_val}")
    _run_n_labels_sanity_check(Y_test, 1, f"prompt {Y_test}")

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def _initialize_probing_dfs(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    output = {'gender': {'train': df, ...}, ...}
    """
    dfs = {"train": train_df, "val": val_df, "test": test_df}
    processing_function_per_attribute = {
        "gender": _keep_one_gender_df,
        "country": _keep_one_country_df,
    }
    kwords_per_attribute = {
        "gender": ["male", "female"],
        "country": ["syria", "venezuela"],
    }

    probing_datasets = {
        protected_attribute: {
            df_type: _create_scraped_df(
                _select_chosen_kwords_probing(
                    processing_function(df),
                    kwords_per_attribute[protected_attribute],
                    protected_attribute,
                ),
                protected_attribute,
            )
            for df_type, df in dfs.items()
        }
        for protected_attribute, processing_function in processing_function_per_attribute.items()
    }

    return probing_datasets


def _select_chosen_kwords_probing(df: pd.DataFrame, kwords: List[str], df_type: str):
    final_df = df.copy()
    final_df[f"{df_type}_kword_type"] = final_df[f"{df_type}_kword_type"].apply(
        lambda x: [item for item in x if item in kwords]
    )
    final_df = final_df[final_df[f"{df_type}_kword_type"].apply(lambda x: len(x) == 1)]

    return final_df


def _create_scraped_df(
    df: pd.DataFrame, protected_attribute: str
) -> pd.DataFrame:  # TODO: sanity check protected_attribute
    relevant_probing_columns = [
        "entry_id",
        "excerpt",
        "target",
        f"{protected_attribute}_keywords",
        "excerpt_type",
    ]
    new_df = df.copy()

    new_df["excerpt"] = new_df.apply(
        lambda x: _create_scraped_excerpt(x, protected_attribute), axis=1
    )
    new_df = new_df.rename(columns={f"{protected_attribute}_kword_type": "target"})[
        relevant_probing_columns
    ]
    # print(f"scraped df {protected_attribute}")
    # display(new_df.head())
    return new_df
