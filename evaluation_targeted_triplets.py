import argparse
import ast
import os
from collections import Counter

import torch
import yaml
from plotnine import ggplot, aes, geom_boxplot, ggsave, theme, element_text, xlab, geom_bar, ylab
from scipy.stats import pearsonr

from generate_targeted_triplets_eval_sets import load_data, get_lemmatized_words, WORDS_NAMES, FRAGMENTS, POS_TAGS
from pig.data import DEFAULT_SAMPLE_RATE
from pig.evaluation import load_best_model

import pytorch_lightning as pl
import logging
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import seaborn as sns

from pig.metrics import batch_triplet_accuracy

import matplotlib.pyplot as plt

from pig.targeted_triplets import collate_triplets, PeppaTargetedTripletCachedDataset, get_eval_set_info

BATCH_SIZE = 8
NUM_WORKERS = 8

RESULT_DIR = "results/targeted_triplets"


def evaluate(model):
    """Compute the targeted triplets score for the given model. """
    gpus = None
    if torch.cuda.is_available():
        gpus = 1
    trainer = pl.Trainer(logger=False, gpus=gpus)

    results_all = []
    for fragment_type in FRAGMENTS:
        for pos in POS_TAGS:
            per_sample_results = targeted_triplet_score(fragment_type, pos, model, trainer)
            print(f"Mean acc: {np.mean(per_sample_results)}")

            # Save per-sample results for detailed analysis
            results_data = get_eval_set_info(fragment_type, pos)

            assert len(results_data) == len(per_sample_results), \
                f"Number of samples in eval set ({len(per_sample_results)}) doesn't match CSV info from " \
                f"eval set CSV file: ({len(results_data)})"

            results_data["result"] = per_sample_results
            results_data["target_pos"] = pos
            results_all.append(results_data)

    results_all = pd.concat(results_all, ignore_index=True)
    return results_all


def targeted_triplet_score(fragment_type, pos, model, trainer):
    audio_sample_rate = model.config["data"].get("audio_sample_rate", DEFAULT_SAMPLE_RATE)
    ds = PeppaTargetedTripletCachedDataset(fragment_type, pos, force_cache=False,
                                           target_size=model.config["data"]["target_size"],
                                           audio_sample_rate=audio_sample_rate)
    loader = DataLoader(ds, collate_fn=collate_triplets, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
    if len(ds) == 0:
        return []

    results = []
    for batch in trainer.predict(model, loader):
        results.extend(r.item() for r in batch_triplet_accuracy(batch))

    return results


def get_all_results_df(version, pos_tags, per_word_results=False, min_samples=None):
    results_data_all = pd.read_csv(f"{RESULT_DIR}/version_{version}/minimal_pairs_scores.csv", converters={"tokenized": ast.literal_eval})
    results_data_all = results_data_all[results_data_all.target_pos.isin(pos_tags)]

    if min_samples:
        counts = results_data_all.target_word.value_counts()
        words_enough_samples = counts[counts > min_samples].keys().to_list()
        if len(words_enough_samples) == 0:
            print(f"No words with enough samples (>{min_samples}) for POS tags {pos_tags} found.")
        results_data_all = results_data_all[
            results_data_all.target_word.isin(words_enough_samples) | results_data_all.distractor_word.isin(words_enough_samples)]

    if per_word_results:
        # Duplicate results and introduce "word" (either target or distractor word) column
        results_data_words_1 = results_data_all.copy()
        results_data_words_1["word"] = results_data_words_1["target_word"]
        results_data_words_2 = results_data_all.copy()
        results_data_words_2["word"] = results_data_words_2["distractor_word"]
        results_data_all = pd.concat([results_data_words_1, results_data_words_2], ignore_index=True)

    results_data_all["duration"] = results_data_all["clipEnd"] - results_data_all["clipStart"]
    return results_data_all


def create_duration_results_plots(condition, versions):
    results_data_duration = []
    results_data_num_tokens = []
    for version in versions:
        results_data = get_all_results_df(version, POS_TAGS)
        if len(results_data) > 0:
            results_data["duration"] = pd.qcut(results_data["duration"], 3)
            results_bootstrapped_duration = bootstrap_scores_for_column(results_data, "duration")
            results_data["num_tokens"] = results_data.tokenized.apply(len)
            results_data["num_tokens"] = pd.cut(results_data["num_tokens"], 3)

            results_bootstrapped_num_tokens = bootstrap_scores_for_column(results_data, "num_tokens")

            results_data_duration.append(results_bootstrapped_duration)
            results_data_num_tokens.append(results_bootstrapped_num_tokens)

    results_data_duration = pd.concat(results_data_duration, ignore_index=True)
    results_data_num_tokens = pd.concat(results_data_num_tokens, ignore_index=True)

    g = ggplot(results_data_duration, aes(x="duration", y="score")) + geom_boxplot() + xlab("") + theme(
        axis_text_x=element_text(angle=85))
    ggsave(g, f"{RESULT_DIR}/condition_{condition}/acc_per_duration.pdf")

    g = ggplot(results_data_num_tokens, aes(x="num_tokens", y="score")) + geom_boxplot() + xlab("") + theme(
        axis_text_x=element_text(angle=85))
    ggsave(g, f"{RESULT_DIR}/condition_{condition}/acc_per_num_tokens.pdf")


def get_bootstrapped_scores(values, n_resamples=100):
    for n in range(n_resamples):
        result = np.random.choice(values, size=len(values), replace=True).mean()
        yield result


def bootstrap_scores_for_column(results, column_name):
    results_boot = []
    for value in results[column_name].unique():
        results_data_word = results[results[column_name] == value].result
        result_bootstrapped = get_bootstrapped_scores(results_data_word.values)
        items = [{"score": res, column_name: value} for res in result_bootstrapped]
        results_boot.extend(items)

    return pd.DataFrame.from_records(results_boot)


def get_average_result_bootstrapping(version):
    results_data_words_all = get_all_results_df(version, POS_TAGS)
    result_bootstrapped = list(get_bootstrapped_scores(results_data_words_all.result.values))
    mean_results, std_results = np.mean(result_bootstrapped), np.std(result_bootstrapped)
    print(f"Average result: {mean_results} +/-{std_results}")
    return mean_results, std_results


def create_per_word_result_plots(condition, versions, min_samples):
    results_pos = {pos: [] for pos in POS_TAGS}
    for version in versions:
        for pos in POS_TAGS:
            results_data_words = get_all_results_df(version, [pos], per_word_results=True, min_samples=min_samples)
            if len(results_data_words) > 0:
                results_bootstrapped = bootstrap_scores_for_column(results_data_words, "word")
                results_pos[pos].append(results_bootstrapped)

    for pos in ["NOUN", "VERB"]:
        results_data_words = pd.concat(results_pos[pos], ignore_index=True)
        if len(results_data_words) > 0:
            if pos == "NOUN":
                figsize = (15, 6)
            else:
                figsize  = (8, 6)
            g = ggplot(results_data_words, aes(x='reorder(word, score)', y="score")) + geom_boxplot(outlier_shape='') + xlab("") \
                + theme(axis_text_x=element_text(angle=85), figure_size=figsize)
            path = f"{RESULT_DIR}/condition_{condition}/acc_per_word_{pos}.pdf"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            ggsave(g, path)


def create_correlation_results_plots(version, min_samples):
    word_concreteness_ratings = get_word_concreteness_ratings()
    dataset_word_frequencies = get_dataset_word_frequencies()

    results_data_words_all = get_all_results_df(version, POS_TAGS, per_word_results=True, min_samples=min_samples)
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
    plt.savefig(f"{RESULT_DIR}/version_{version}/correlation_frequency_acc", dpi=300)
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
    plt.savefig(f"{RESULT_DIR}/version_{version}/correlation_concreteness_acc", dpi=300)
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
    parser.add_argument("--run", action="store_true", default=False,
                        help="Run evaluation")
    parser.add_argument("--versions", type=str, nargs="+")

    parser.add_argument("--plot", action="store_true", default=False,
                        help="Plot results")
    parser.add_argument("--conditions", type=str, default="conditions_minimal_pairs.yaml")

    parser.add_argument(
        "--min-samples",
        type=int,
        default=100,
        help="Minimum number of test samples for a word to be included",
    )
    parser.add_argument("--correlate-predictors", action="store_true", default=False)

    return parser.parse_args()


def add_hparams(record):
    config = yaml.safe_load(open(f"hparams_{record['condition']}.yaml"))
    record['jitter'] = config['data']['train']['jitter']
    record['static'] = config['video'].get('static', False)
    record['audio_pretrained'] = config['audio']['pretrained']
    record['video_pretrained'] = config['video']['pretrained']
    record['resolution'] = 'x'.join(map(str, config['data']['target_size']))
    record['freeze_wav2vec'] = config['audio']['freeze_feature_extractor'] \
                               and config['audio']['freeze_encoder_layers'] == 12
    record['sample_rate'] = str(config['data'].get('audio_sample_rate', DEFAULT_SAMPLE_RATE))
    return record


def create_results_table(conditions):
    data = []
    for condition, versions in conditions.items():
        results_pos = {pos: [] for pos in POS_TAGS}
        for version in versions:
            for pos in POS_TAGS:
                results_data_words_all = get_all_results_df(version, [pos])
                results_bootstrapped = list(get_bootstrapped_scores(results_data_words_all.result.values))
                results_pos[pos].extend(results_bootstrapped)

        record = {"condition": condition}
        for pos in POS_TAGS:
            score = f"{np.mean(results_pos[pos]).round(2):.2f}" + "±" +f"{np.std(results_pos[pos]).round(2):.2f}"
            record[f"minimal_pairs_score_{pos}"] = score

        record = add_hparams(record)
        data.append(record)

    data = pd.DataFrame.from_records(data)
    data = data.fillna(dict(scrambled_video=False))

    data["finetune_wav2vec"] = ~data["freeze_wav2vec"]
    data["temporal"] = ~data["static"]
    data[['finetune_wav2vec', 'jitter', 'temporal',
          'minimal_pairs_score_NOUN', 'minimal_pairs_score_VERB']] \
        .replace(True, "\checkmark").replace(False, "") \
        .rename(columns=dict(jitter='Jitt',
                             temporal='Tmp',
                             finetune_wav2vec="Finet",
                             minimal_pairs_score_NOUN='Nouns',
                             minimal_pairs_score_VERB='Verbs', )) \
        .to_latex(buf=f"{RESULT_DIR}/minimal_pairs.tex",
                  index=False,
                  escape=False,
                  float_format="%.3f")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    args = get_args()

    os.makedirs(RESULT_DIR, exist_ok=True)
    if args.run:
        for version in args.versions:
            logging.info(f"Evaluating version {version}")
            net, path = load_best_model(f"lightning_logs/version_{version}/")

            result = evaluate(net)
            result_path = f"{RESULT_DIR}/version_{version}/minimal_pairs_scores.csv"
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            result.to_csv(result_path, index=False)

    if args.plot:
        conditions = yaml.safe_load(open(args.conditions))
        create_results_table(conditions)
        for condition, versions in conditions.items():
            create_per_word_result_plots(condition, versions, args.min_samples)
            create_duration_results_plots(condition, versions)
