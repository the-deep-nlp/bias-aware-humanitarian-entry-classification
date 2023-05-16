from collections import defaultdict
from debiasing_pipelines.utils import _get_tags_based_stats


def get_probabilities_results(trained_model, data):
    """
    output: Dict[tag, pd.DataFrame[entry_id, excerpt, one column per keyword containing probabilities]]
    """

    results_df = data.copy()
    results_df.sort_values(by="entry_id", inplace=True)

    probability_predictions = trained_model.custom_predict(
        results_df, setup="raw_predictions"
    )
    # tags_list = list(probability_predictions.keys())

    for tag, results_one_tag in probability_predictions.items():
        # results_df[f"ratio_{tag}"] = (
        #     results_one_tag / trained_model.optimal_thresholds[tag]
        # )
        results_df[f"probability_{tag}"] = results_one_tag

    return results_df

    # per_tag_raw_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # unique_ids = data["entry_id"].unique().tolist()

    # for one_id in unique_ids:
    #     df_one_id = results_df[results_df.entry_id == one_id]
    #     row_original = df_one_id[df_one_id.excerpt_type == "original"]
    #     kword_type_original = row_original.kword_type.values[0][0]
    #     rows_counterfactual = df_one_id[df_one_id.excerpt_type != "original"]
    #     for tag in tags_list:
    #         tag_original_probability = row_original[f"probabilities_{tag}"].values[0]
    #         for i, row_counterfactual in rows_counterfactual.iterrows():
    #             one_counterfactual_proba = row_counterfactual[f"probabilities_{tag}"]
    #             one_counterfactual_kword_type = row_counterfactual.kword_type[0]

    #             per_tag_raw_results[tag][kword_type_original][
    #                 one_counterfactual_kword_type
    #             ].append(
    #                 {
    #                     "original": tag_original_probability,
    #                     "counterfactual": one_counterfactual_proba,
    #                 }
    #             )

    # total_stats = _get_tags_based_stats(per_tag_raw_results)

    # return total_stats
