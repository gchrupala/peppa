import argparse
import ast
import os
from collections import Counter

import torch
from plotnine import ggplot, aes, geom_boxplot, ggtitle, ggsave, theme, element_text, xlab, geom_bar, ylab
from scipy.stats import pearsonr

from generate_targeted_triplets_eval_sets import load_data, get_lemmatized_words, WORDS_NAMES, FRAGMENTS, POS_TAGS
from pig.evaluation import load_best_model

import pytorch_lightning as pl
import logging
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import seaborn as sns

from pig.metrics import batch_triplet_accuracy
from pig.models import PeppaPig
from run import default_config

import matplotlib.pyplot as plt

from pig.targeted_triplets import collate_triplets, PeppaTargetedTripletCachedDataset

BATCH_SIZE = 8
NUM_WORKERS = 8

RESULT_DIR = "results/targeted_triplets"


def evaluate(model, version):
    """Compute the targeted triplets score for the given model. """
    gpus = None
    if torch.cuda.is_available():
        gpus = 1
    trainer = pl.Trainer(logger=False, gpus=gpus)
    for fragment_type in FRAGMENTS:
        for pos in POS_TAGS:
            per_sample_results = targeted_triplet_score(fragment_type, pos, model, trainer)
            row = dict(
                fragment_type=fragment_type,
                pos=pos,
                targeted_triplet_acc=np.mean(per_sample_results),
                version=version,
                hparams_path=f"lightning_logs/version_{version}/hparams.yaml"
            )
            print(row)

            # Save per-sample results for detailed analysis
            eval_info_file = f"data/eval/eval_set_{row['fragment_type']}_{row['pos']}.csv"
            results_data = pd.read_csv(eval_info_file, index_col="id")

            assert len(results_data) == len(per_sample_results), \
                f"Number of samples in eval set ({len(per_sample_results)}) doesn't match CSV info from " \
                f"{eval_info_file} ({len(results_data)})"

            results_data["result"] = per_sample_results
            path = f"{RESULT_DIR}/version_{version}/targeted_triplets_{row['fragment_type']}_{row['pos']}.csv"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            results_data.to_csv(path)

            yield row


def targeted_triplet_score(fragment_type, pos, model, trainer):
    ds = PeppaTargetedTripletCachedDataset(fragment_type, pos, force_cache=False,
                                           target_size=model.config["data"]["target_size"],
                                           audio_sample_rate=model.config["data"]["audio_sample_rate"])
    loader = DataLoader(ds, collate_fn=collate_triplets, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
    if len(ds) == 0:
        return []

    results = []
    for batch in trainer.predict(model, loader):
        results.extend(r.item() for r in batch_triplet_accuracy(batch))

    return results


def get_all_results_df(version, pos_tags, per_word_results=False):
    results_data_all = []
    for pos in pos_tags:
        for fragment_type in FRAGMENTS:
            results_data_fragment = pd.read_csv(f"{RESULT_DIR}/version_{version}/targeted_triplets_{fragment_type}_{pos}.csv",
                                                converters={"tokenized": ast.literal_eval})
            results_data_all.append(results_data_fragment)

    results_data_all = pd.concat(results_data_all, ignore_index=True)

    if per_word_results:
        # Duplicate results and introduce "word" (either target or distractor word) column
        results_data_words_1 = results_data_all.copy()
        results_data_words_1["word"] = results_data_words_1["target_word"]
        results_data_words_2 = results_data_all.copy()
        results_data_words_2["word"] = results_data_words_2["distractor_word"]
        results_data_all = pd.concat([results_data_words_1, results_data_words_2], ignore_index=True)

    results_data_all["duration"] = results_data_all["clipEnd"] - results_data_all["clipStart"]
    return results_data_all


def create_duration_results_plots(version):
    results_data_all = get_all_results_df(version, POS_TAGS)

    results_data_all["clipDuration"] = results_data_all["clipEnd"] - results_data_all["clipStart"]
    results_data_all["clipDuration"] = pd.qcut(results_data_all["clipDuration"], 5)

    results_data_all["num_tokens"] = results_data_all.tokenized.apply(len)
    results_data_all["num_tokens"] = pd.cut(results_data_all["num_tokens"], 3)

    g = ggplot(results_data_all) + geom_bar(aes(x='clipDuration')) + xlab("") + ylab("# samples") + theme(
        axis_text_x=element_text(angle=85)) + ggtitle(
        f"Number of samples per duration")
    ggsave(g, f"{RESULT_DIR}/num_samples_per_duration.pdf")

    g = ggplot(results_data_all) + geom_bar(aes(x='num_tokens')) + xlab("") + ylab("# samples") + theme(
        axis_text_x=element_text(angle=85)) + ggtitle(
        f"Number of samples per number of tokens")
    ggsave(g, f"{RESULT_DIR}/num_samples_per_num_tokens.pdf")

    results_boot = bootstrap_scores_for_column(results_data_all, "clipDuration")

    g = ggplot(results_boot, aes(x="clipDuration", y="score")) + geom_boxplot() + xlab("") + theme(
        axis_text_x=element_text(angle=85)) + ggtitle(
        f"Accuracy per test clip duration for model ID: {version}")
    ggsave(g, f"{RESULT_DIR}/results_per_duration_version_{version}.pdf")

    results_boot = bootstrap_scores_for_column(results_data_all, "num_tokens")

    g = ggplot(results_boot, aes(x="num_tokens", y="score")) + geom_boxplot() + xlab("") + theme(
        axis_text_x=element_text(angle=85)) + ggtitle(
        f"Accuracy per number of tokens for model ID: {version}")
    ggsave(g, f"{RESULT_DIR}/results_per_num_tokens_version_{version}.pdf")


def get_bootstrapped_scores(values, n_resamples=100):
    for n in range(n_resamples):
        result = values.sample(len(values), replace=True).mean()
        yield result


def bootstrap_scores_for_column(results, column_name):
    results_boot = []
    for value in results[column_name].unique():
        results_data_word = results[results[column_name] == value].result
        result_bootstrapped = get_bootstrapped_scores(results_data_word)
        items = [{"score": res, column_name: value} for res in result_bootstrapped]
        results_boot.extend(items)

    return pd.DataFrame.from_records(results_boot)


def create_per_word_result_plots(version):
    for pos in POS_TAGS:
        results_data_words = get_all_results_df(version, [pos], per_word_results=True)
        if len(results_data_words) > 0:
            results_boot = bootstrap_scores_for_column(results_data_words, "word")

            g = ggplot(results_boot, aes(x='reorder(word, score)', y="score")) + geom_boxplot() + xlab("") \
                + theme(axis_text_x=element_text(angle=85), figure_size=(15, 6)) \
                + ggtitle(f"Per-word targeted triplets accuracy for model ID: {version} | POS: {pos}")
            ggsave(g, f"{RESULT_DIR}/results_per_word_version_{version}_{pos}.pdf")

            num_samples_per_word = results_data_words["word"].value_counts(ascending=True).index.tolist()
            word_cat = pd.Categorical(results_data_words['word'], categories=num_samples_per_word)
            results_data_words = results_data_words.assign(word_cat=word_cat)
            g = ggplot(results_data_words) + geom_bar(aes(x='word_cat')) + xlab("") + ylab("# samples") + theme(
                axis_text_x=element_text(angle=85), figure_size=(15, 6)) + ggtitle(
                f"Number of samples: {pos}")
            ggsave(g, f"{RESULT_DIR}/num_samples_per_word_{pos}.pdf")


def create_correlation_results_plots(version):
    word_concreteness_ratings = get_word_concreteness_ratings()
    dataset_word_frequencies = get_dataset_word_frequencies()

    results_data_words_all = get_all_results_df(version, POS_TAGS, per_word_results=True)
    mean_acc = results_data_words_all.groupby("word")["result"].agg("mean")

    # Correlate performance with word frequency in train split
    word_frequencies = [dataset_word_frequencies[w] for w in mean_acc.keys()]
    word_frequencies = [np.log(f) for f in word_frequencies]
    word_accuracies = mean_acc.values
    pearson_corr = pearsonr(word_frequencies, word_accuracies)
    plt.figure()
    s1 = sns.scatterplot(word_frequencies, word_accuracies, marker="x")
    plt.title(f"pearson r={pearson_corr[0]:.2f} (p={pearson_corr[1]:.3f})")
    plt.xlabel("Log Frequency")
    plt.ylabel("Accuracy")
    # Named labels in scatterplot:
    for i in range(len(word_frequencies)):
        s1.text(word_frequencies[i] + 0.01, word_accuracies[i],
                mean_acc.keys()[i], horizontalalignment='left',
                size='small', color='black')
    plt.savefig(f"{RESULT_DIR}/correlation_frequency_acc_version_{version}", dpi=300)
    print(f"Pearson correlation frequency-acc: ", pearson_corr)

    # Correlate performance with word concreteness
    word_concretenesses = [get_word_concreteness(w, word_concreteness_ratings) for w in mean_acc.keys()]
    pearson_corr = pearsonr(word_concretenesses, word_accuracies)
    plt.figure()
    s2 = sns.scatterplot(word_concretenesses, word_accuracies, marker="x")
    plt.title(f"pearson r={pearson_corr[0]:.2f} (p={pearson_corr[1]:.3f})")
    plt.xlabel("Concreteness")
    plt.ylabel("Accuracy")
    # Named labels in scatterplot:
    for i in range(len(word_frequencies)):
        s2.text(word_concretenesses[i] + 0.01, word_accuracies[i],
                mean_acc.keys()[i], horizontalalignment='left',
                size='small', color='black')
    plt.savefig(f"{RESULT_DIR}/correlation_concreteness_acc_version_{version}", dpi=300)
    print(f"Pearson correlation concreteness-acc: ", pearson_corr)


def get_dataset_word_frequencies():
    _, data_tokens = load_data()

    all_words = get_lemmatized_words(data_tokens, "train", fragments=["dialog"])

    return Counter(all_words)


def get_word_concreteness_ratings():
    # Use concreteness ratings from Brysbaert, Warriner, & Kuperman, 2014
    # https://link.springer.com/article/10.3758/s13428-013-0403-5
    data = pd.read_csv("data/eval/13428_2013_403_MOESM1_ESM.csv")
    data.set_index("Word", inplace=True)
    return data["Conc.M"].to_dict()


def get_word_concreteness(word, word_concreteness_ratings):
    if word in word_concreteness_ratings:
        return word_concreteness_ratings[word]
    else:
        if word == "mr":
            # Correct spelling of "mister"
            return word_concreteness_ratings["mister"]
        elif word in WORDS_NAMES:
            # Assume that persons are maximally concrete
            return 5
        else:
            print(f"Warning: concreteness rating not found for '{word}'. Setting to 3/5.")
            return 3


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--versions", type=str, nargs="+")
    parser.add_argument("--plot-only", action="store_true", default=False,
                        help="Only plot results, do not re-run evaluation")

    parser.add_argument("--correlate-predictors", action="store_true", default=False)

    return parser.parse_args()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    args = get_args()

    os.makedirs(RESULT_DIR, exist_ok=True)
    rows = []
    for version in args.versions:
        logging.info(f"Evaluating version {version}")

        if not args.plot_only:
            if version == "BASELINE":
                net = PeppaPig(default_config)
                path = ""
            else:
                net, path = load_best_model(f"lightning_logs/version_{version}/")

            result_rows = evaluate(net, version)
            rows.extend(result_rows)

        create_per_word_result_plots(version)
        create_duration_results_plots(version)
        if args.correlate_predictors:
            create_correlation_results_plots(version)

    if len(rows) > 0:
        scores = pd.DataFrame.from_records(rows)
        scores.to_csv("{RESULT_DIR}/scores_targeted_triplets.csv", index=False, header=True)
