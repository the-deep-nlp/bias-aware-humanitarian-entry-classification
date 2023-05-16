from collections import defaultdict
from typing import List, Dict, Union
import numpy as np
import pandas as pd
from scipy import stats

output_columns = [
    "tag",
    "original",
    "counterfactual",
    "count",
    "original_values_diff_t_test_pvalue",
    "original_values_diff_mean_shift",
    "original_values_diff_absolute_shift",
    "original_values_diff_std",
    "original_values_diff_median",
    "absolute_values_diff_t_test_pvalue",
    "absolute_values_diff_mean_shift",
    "absolute_values_diff_absolute_shift",
    "absolute_values_diff_std",
    "absolute_values_diff_median",
]


def _get_stats_one_list(values: List[Union[float, Dict[str, float]]]):

    statistics = {}

    if type(values[0]) is dict:
        original_values = np.array([one_output["original"] for one_output in values])
        counterfactual_values = np.array(
            [one_output["counterfactual"] for one_output in values]
        )
        original_values_diff = counterfactual_values - original_values
        absolute_values_diff = np.abs(counterfactual_values) - np.abs(original_values)
        original_values_diff_t_test_pvalue = stats.ttest_rel(
            original_values, counterfactual_values
        ).pvalue
        absolute_values_diff_t_test_pvalue = stats.ttest_rel(
            np.abs(original_values), np.abs(counterfactual_values)
        ).pvalue

    else:
        original_values_diff = absolute_values_diff = values
        original_values_diff_t_test_pvalue = absolute_values_diff_t_test_pvalue = "-"

    statistics[
        "original_values_diff_t_test_pvalue"
    ] = original_values_diff_t_test_pvalue
    statistics["original_values_diff_mean_shift"] = np.mean(original_values_diff)
    statistics["original_values_diff_absolute_shift"] = np.mean(
        np.abs(original_values_diff)
    )
    statistics["original_values_diff_std"] = np.std(original_values_diff)
    statistics["original_values_diff_median"] = np.median(original_values_diff)

    statistics[
        "absolute_values_diff_t_test_pvalue"
    ] = absolute_values_diff_t_test_pvalue
    statistics["absolute_values_diff_mean_shift"] = np.mean(absolute_values_diff)
    statistics["absolute_values_diff_absolute_shift"] = np.mean(
        np.abs(absolute_values_diff)
    )
    statistics["absolute_values_diff_std"] = np.std(absolute_values_diff)
    statistics["absolute_values_diff_median"] = np.median(absolute_values_diff)

    statistics["count"] = len(original_values_diff)
    return statistics


def _get_stats_one_tag(results_per_tag: Dict[str, Dict[str, List[float]]]):
    """
    Input example:
    {
        Health:{
            original_female: {
                counterfactual_male: [differences],
                counterfactual_neutral: [differences]
            },
            .
            .
            .
        }
        .
        .
        .
    }
    """
    per_tag_stats = pd.DataFrame()
    for original_kword, results_per_tag_kw in results_per_tag.items():
        counterfactual_dict_results_per_tag_kw = {}
        for (
            counterfactual_kword,
            differences_per_counterfactual,
        ) in results_per_tag_kw.items():
            stats_per_counterfactual = _get_stats_one_list(
                differences_per_counterfactual
            )
            counterfactual_dict_results_per_tag_kw[
                counterfactual_kword
            ] = stats_per_counterfactual

        counterfactual_df_results_per_tag_kw = pd.DataFrame.from_dict(
            counterfactual_dict_results_per_tag_kw, orient="index"
        )
        counterfactual_df_results_per_tag_kw[
            "counterfactual"
        ] = counterfactual_df_results_per_tag_kw.index
        counterfactual_df_results_per_tag_kw["original"] = original_kword

        per_tag_stats = pd.concat([per_tag_stats, counterfactual_df_results_per_tag_kw])
    return per_tag_stats


def _get_tags_based_stats(raw_results: Dict[str, Dict[str, Dict[str, List[float]]]]):
    """
    Input example:
    {
        Health:{
            original_female: {
                counterfactual_male: [differences],
                counterfactual_neutral: [differences]
            },
            .
            .
            .
        }
        .
        .
        .
    }
    """

    final_stats_df = pd.DataFrame()

    for tag, raw_results_per_tag in raw_results.items():
        final_results_one_tag = _get_stats_one_tag(raw_results_per_tag)
        final_results_one_tag["tag"] = tag
        final_stats_df = pd.concat([final_stats_df, final_results_one_tag])

    return final_stats_df[output_columns]
