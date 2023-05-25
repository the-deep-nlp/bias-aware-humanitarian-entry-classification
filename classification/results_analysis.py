import os
import pandas as pd
from ast import literal_eval
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from typing import List, Set, Dict
from scipy import stats
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

mappings = {
    "gender": {
        "N->F": ["neutral", "female"],
        "N->M": ["neutral", "male"],
        "M->F": ["male", "female"],
    },
    "country": {
        "C->V": ["canada", "venezuela"],
        "C->S": ["canada", "syria"],
        "S->V": ["syria", "venezuela"],
    },
}

mappings_bidirectional = {
    "gender": {
        "N->F": ["neutral", "female"],
        "N->M": ["neutral", "male"],
        "M->F": ["male", "female"],
        "F->N": ["female", "neutral"],
        "M->N": ["male", "neutral"],
        "F->M": ["female", "male"],
    },
    "country": {
        "C->V": ["canada", "venezuela"],
        "C->S": ["canada", "syria"],
        "S->V": ["syria", "venezuela"],
        "V->C": ["venezuela", "canada"],
        "S->C": ["syria", "canada"],
        "V->S": ["venezuela", "syria"],
    },
}

mapping_training_type_to_name = {
    "no_debiasing": "Base",
    "counterfactual_debiasing": "CDA",
}

mapping_training_architecture2name = {
    "multiabel_architecture": "OURS",
    "base_architecture": "Base",
}

tags = [
    "first_level_tags->pillars_1d->Casualties",
    "first_level_tags->pillars_1d->Context",
    "first_level_tags->pillars_1d->Covid-19",
    "first_level_tags->pillars_1d->Displacement",
    "first_level_tags->pillars_1d->Humanitarian access",
    "first_level_tags->pillars_1d->Information and communication",
    "first_level_tags->pillars_1d->Shock/event",
    "first_level_tags->pillars_2d->At risk",
    "first_level_tags->pillars_2d->Capacities & response",
    "first_level_tags->pillars_2d->Humanitarian conditions",
    "first_level_tags->pillars_2d->Impact",
    "first_level_tags->pillars_2d->Priority interventions",
    "first_level_tags->pillars_2d->Priority needs",
    "first_level_tags->sectors->Agriculture",
    "first_level_tags->sectors->Cross",
    "first_level_tags->sectors->Education",
    "first_level_tags->sectors->Food security",
    "first_level_tags->sectors->Health",
    "first_level_tags->sectors->Livelihoods",
    "first_level_tags->sectors->Logistics",
    "first_level_tags->sectors->Nutrition",
    "first_level_tags->sectors->Protection",
    "first_level_tags->sectors->Shelter",
    "first_level_tags->sectors->Wash",
]

sorted_columns = [
    "Measurement",
    "LLM",
    "Model",
    # "N->F",
    # "N->M",
    # "M->F",
    "G-Avg",
    # "C->V",
    # "C->S",
    # "S->V",
    "C-Avg",
    "T-Avg",
]


# Classification results
classification_tags = {
    "mean->first_level_tags": "Avg",
    "mean->first_level_tags->pillars_1d": "pillars 1d",
    "mean->subpillars_1d": "subpillars 1d",
    "mean->first_level_tags->pillars_2d": "pillars 2d",
    "mean->subpillars_2d": "subpillars 2d",
    "mean->first_level_tags->sectors": "sectors",
}
results_metrics = ["precision", "f_score"]

ordered_classification_results = [
    "Measurement",
    "LLM",
    "Model",
    "sectors precision",
    "sectors f_score",
    "pillars 1d precision",
    "pillars 1d f_score",
    "subpillars 1d precision",
    "subpillars 1d f_score",
    "pillars 2d precision",
    "pillars 2d f_score",
    "subpillars 2d precision",
    "subpillars 2d f_score",
    "Avg precision",
    "Avg f_score",
]

clean_classification_task_names = {
    "sectors": "Sectors",
    "pillars_1d": "Pillars 1D",
    "pillars_2d": "Pillars 2D",
}

label_ticks = {
    "pillars_1d": sorted(
        [
            "Covid-19",
            "Displacement",
            "Context",
            "Casualties",
            "Information and\ncommunication",
            "Humanitarian\naccess",
            "Shock/event",
        ]
    ),
    "pillars_2d": sorted(
        [
            "Impact",
            "Humanitarian\nconditions",
            "Capacities &\nresponse",
            "At risk",
            "Priority\ninterventions",
            "Priority needs",
        ]
    ),
    "sectors": sorted(
        [
            "Logistics",
            "Health",
            "Cross",
            "Education",
            "Shelter",
            "Food security",
            "Protection",
            "Livelihoods",
            "Wash",
            "Agriculture",
            "Nutrition",
        ]
    ),
}

treated_level0_tags = ["sectors", "pillars_1d", "pillars_2d"]


def get_classification_results(
    llm_models: List[str],
    training_setups: List[str],
    training_architectures: List[str],
    folder_name: str,
):
    final_results_df = pd.DataFrame()

    for one_llm in llm_models:
        for one_training_setup in training_setups:
            for one_training_architecture in training_architectures:
                metrics_one_row = {
                    "Measurement": "Classification Results",
                    "LLM": one_llm,
                    "Model": f"{mapping_training_architecture2name[one_training_architecture]}-{mapping_training_type_to_name[one_training_setup]}",
                }
                df = pd.read_csv(
                    os.path.join(
                        folder_name,
                        one_llm,
                        one_training_setup,
                        one_training_architecture,
                        "classification",
                        f"classification_results_test_df_{one_llm}_{one_training_setup}_{one_training_architecture}.csv",
                    )
                ).drop(columns=["Unnamed: 0"])

                for (
                    one_results_col,
                    procesed_results_col,
                ) in classification_tags.items():
                    metrics_one_tag = df[df.tag == one_results_col]
                    for one_metic in ["precision", "f_score"]:
                        metrics_one_row[
                            f"{procesed_results_col} {one_metic}"
                        ] = metrics_one_tag[one_metic].iloc[0]

                final_results_df = final_results_df.append(
                    metrics_one_row, ignore_index=True
                )

    final_results_df = final_results_df[ordered_classification_results]

    return final_results_df


def _multisets_intersection(set_list: List[Set]):
    return list(set.intersection(*set_list))


def _generate_heatmaps(
    df: pd.DataFrame,
    max_bound: Dict[str, float],
    fig_name: str = None,
    heatmap_col: str = "median_shift",
    multiplication_factor: int = 1,
):
    original_df = df.copy().sort_values(by=["tag", "original->couterfactual"])
    # n_base_kwords = 7
    f, axes = plt.subplots(
        2,
        len(treated_level0_tags),
        sharey=False,
        sharex=False,
        figsize=(20, 5),
        gridspec_kw={
            "width_ratios": [1.4, 0.9, 1.1],
            # "height_ratios": [1, 1, 1, 0.2, 1, 1, 1],
        },
    )
    # axes[3, 0].set_visible(False)
    # axes[3, 1].set_visible(False)
    # axes[3, 2].set_visible(False)

    bias_attributes = ["gender", "country"]
    for i, one_bias_attribute in enumerate(bias_attributes):
        processed_df = original_df[original_df.bias_attribute == one_bias_attribute]

        # one_kw_df[
        #     one_kw_df.index.to_series().apply(
        #         lambda x: f"first_level_tags->{one_tag}" in x
        #     )
        # ]

        processed_df = processed_df[
            processed_df.tag.apply(
                lambda x: any(
                    [
                        f"first_level_tags->{one_tag}" in x
                        for one_tag in treated_level0_tags
                    ]
                )
            )
        ].sort_values(by=["original->couterfactual", "tag"])

        processed_df[heatmap_col] = (
            multiplication_factor * processed_df[heatmap_col]
        ).round(3)

        # for i, one_base_kw in enumerate(base_kwords):
        # final_axis_label = i + 3 * bias_attr_count
        # if bias_attr_count:
        #     final_axis_label += 1
        # one_kw_df = processed_df[processed_df.base_kw == one_base_kw]
        # min_val = min(one_kw_df[heatmap_col].min(), -1e-7)
        # max_val = one_kw_df[heatmap_col].max()
        # norm = mcolors.TwoSlopeNorm(vmin=min_val, vcenter=0, vmax=max_val)

        one_kw_df = processed_df.pivot(
            index="tag", columns="original->couterfactual", values=heatmap_col
        )
        for j, one_tag in enumerate(treated_level0_tags):
            one_tag_results = (
                one_kw_df[
                    one_kw_df.index.to_series().apply(
                        lambda x: f"first_level_tags->{one_tag}" in x
                    )
                ]
                .copy()
                .T
            )

            xticks = label_ticks[one_tag]
            # print(xticks)
            # bottom = {0: 20, 1: 10}

            # cbar_ax = f.add_axes([0.91, 0.15, 0.03, 0.7])
            # cbar_ax = f.add_axes([0.91, bottom[bias_attr_count], 0.03, 0.7])
            g = sns.heatmap(
                one_tag_results,
                cmap="coolwarm",
                cbar=True if j == len(treated_level0_tags) - 1 else False,
                # cbar_ax=cbar_ax,
                ax=axes[i, j],
                # vmin=-max_abs if min_val<0 else 0,
                vmin=-max_bound[one_bias_attribute],
                vmax=max_bound[one_bias_attribute],
                # norm=norm,
                annot=True,
                xticklabels=xticks,
                # linewidths=0.1,
                annot_kws={"fontsize": 10},
            )
            if j != 0:
                g.get_yaxis().set_visible(False)
            if i != 1:
                g.get_xaxis().set_visible(False)

            g.set_ylabel("")
            g.tick_params(labelsize=15)
            if i == 0:
                g.set_xlabel(clean_classification_task_names[one_tag], fontsize=18)
            else:
                g.set_xlabel("")

            plt.yticks(rotation=0)

    plt.subplots_adjust(wspace=0.05, hspace=0.2)

    if fig_name is not None:
        plt.savefig(fig_name, bbox_inches="tight")


def generate_probability_tag_results(
    llm_model: str,
    training_setup: str,
    training_architecture: str,
    folder_name: str,
    one_protected_attribute: str,
):
    df = pd.read_csv(
        os.path.join(
            folder_name,
            llm_model,
            training_setup,
            training_architecture,
            "predictions_discrepency",
            f"{llm_model}_{training_setup}_{training_architecture}_{one_protected_attribute}_predictions_discrepency.csv",
        )
    )

    df["kword_type"] = df["kword_type"].apply(lambda x: literal_eval(x)[0])

    keywords_one_protected_attribute = mappings[one_protected_attribute]

    shifts = defaultdict(lambda: defaultdict(list))

    for i in range(0, len(df), 3):
        df_one_entry = df.iloc[i : i + 3]
        assert df_one_entry.entry_id.nunique() == 1, "issue with df entry id"

        if not df_one_entry.isnull().values.any():
            for (
                name,
                kwords,
            ) in keywords_one_protected_attribute.items():
                base_kword = kwords[0]
                counterfactual_kword = kwords[1]

                base_df = df_one_entry[df_one_entry.kword_type == base_kword]
                counterfactual_df = df_one_entry[
                    df_one_entry.kword_type == counterfactual_kword
                ]
                for one_tag in tags:
                    dist = 100 * (
                        counterfactual_df[f"probability_{one_tag}"].values[0]
                        - base_df[f"probability_{one_tag}"].values[0]
                    )

                    shifts[name][one_tag].append(dist)

    shifts_df = pd.DataFrame()
    for abbrev_name, tagwise_distances in shifts.items():
        real_names = [
            one_real_name.capitalize()
            for one_real_name in mappings[one_protected_attribute][abbrev_name]
        ]

        for tag, all_distances_one_tag in tagwise_distances.items():
            median_distance = np.median(all_distances_one_tag)
            shifts_df = shifts_df.append(
                {
                    "tag": tag,
                    "base_kw": real_names[0],
                    "original->couterfactual": "->".join(real_names),
                    "median_shift": median_distance,
                },
                ignore_index=True,
            )
            shifts_df = shifts_df.append(
                {
                    "tag": tag,
                    "base_kw": real_names[1],
                    "original->couterfactual": "->".join(real_names[::-1]),
                    "median_shift": -median_distance,
                },
                ignore_index=True,
            )

    return shifts_df


all_visualized_methods = {
    "probability_discrepancy": generate_probability_tag_results,
}


# def _get_vizu_bounds(
#     folder_name,
#     llm_models,
#     training_architectures,
#     protected_attributes,
#     heatmap_col="median_shift",
# ):
#     training_setup = "no_debiasing"

#     for llm_model in llm_models:
#         for one_training_architecture in training_architectures:
#             for protected_attr in protected_attributes:
#                 df = pd.read_csv(
#                     os.path.join(
#                         folder_name,
#                         llm_model,
#                         training_setup,
#                         one_training_architecture,
#                         "predictions_discrepency",
#                         f"{llm_model}_{training_setup}_{one_training_architecture}_{protected_attr}_predictions_discrepency.csv",
#                     )
#                 )
#                 df = df[
#                     df.tag.apply(
#                         lambda x: any(
#                             [
#                                 f"first_level_tags->{one_tag}" in x
#                                 for one_tag in treated_level0_tags
#                             ]
#                         )
#                     )
#                 ]
#                 name = f"{llm_model}_{one_training_architecture}_{protected_attr}"
#                 val = df[heatmap_col].max()
#                 max_bounds[name] = val

#     return max_bounds


def generate_tagwise_results(
    llm_models: List[str],
    training_setups: List[str],
    training_architectures: List[str],
    protected_attributes: List[str],
    visualized_methods: List[str],
    folder_name: str,
    save_vizus: bool = True,
):
    results_df_all = pd.DataFrame()
    # max_bounds = _get_vizu_bounds(
    #     folder_name,
    #     llm_models,
    #     training_architectures,
    #     protected_attributes,
    # )
    max_bounds = defaultdict(dict)
    with tqdm(
        total=len(llm_models)
        * len(training_setups)
        # * len(protected_attributes)
        * len(visualized_methods)
        * len(training_architectures)
    ) as pbar:
        for llm_model in llm_models:
            for training_setup in training_setups:
                for one_training_architecture in training_architectures:
                    for method in visualized_methods:
                        results_generation_function = all_visualized_methods[method]
                        results_df_one_run = pd.DataFrame()
                        for protected_attr in protected_attributes:
                            visu_file = os.path.join(
                                folder_name,
                                "visualizations",
                                llm_model,
                                training_setup,
                                one_training_architecture,
                                method,
                            )
                            os.makedirs(visu_file, exist_ok=True)
                            results_df_one_method = results_generation_function(
                                llm_model=llm_model,
                                training_setup=training_setup,
                                training_architecture=one_training_architecture,
                                folder_name=folder_name,
                                one_protected_attribute=protected_attr,
                            )
                            results_df_one_method["bias_attribute"] = protected_attr
                            results_df_one_method["bias_method"] = method
                            results_df_one_method[
                                "training_architecture"
                            ] = one_training_architecture
                            results_df_one_method["trainig_setup"] = training_setup
                            results_df_one_method["llm_model"] = llm_model

                            if training_setup == "no_debiasing":
                                df = results_df_one_method.copy()
                                df = df[
                                    df.tag.apply(
                                        lambda x: any(
                                            [
                                                f"first_level_tags->{one_tag}" in x
                                                for one_tag in treated_level0_tags
                                            ]
                                        )
                                    )
                                ]
                                name = f"{llm_model}_{one_training_architecture}"
                                val = df["median_shift"].max()
                                max_bounds[name][protected_attr] = val

                            results_df_one_run = pd.concat(
                                [results_df_one_run, results_df_one_method]
                            )
                            results_df_all = pd.concat(
                                [results_df_all, results_df_one_method]
                            )

                        if save_vizus:
                            figname = os.path.join(
                                visu_file,
                                f"{llm_model}_{training_setup}_{one_training_architecture}_{method}.png",
                            )
                        else:
                            figname = None

                        max_bound = max_bounds[
                            f"{llm_model}_{one_training_architecture}"
                        ]
                        _generate_heatmaps(
                            results_df_one_run,
                            max_bound,
                            figname,
                        )

                        pbar.update(1)

    return results_df_all
