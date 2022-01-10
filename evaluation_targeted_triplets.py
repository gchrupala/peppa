import argparse
import ast
import os
from collections import Counter

import torch
from scipy.stats import pearsonr

from generate_targeted_triplets_eval_sets import load_data, get_lemmatized_words, WORDS_NAMES
from pig.evaluation import load_best_model

import pytorch_lightning as pl
import logging
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

from pig.metrics import batch_triplet_accuracy
from pig.triplet import collate_triplets

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pig.targeted_triplets import PeppaTargetedTripletDataset

BATCH_SIZE = 8
NUM_WORKERS = 8

MIN_DURATION = 0.29

FRAGMENTS = ['narration']

def score(model):
    """Compute all standard scores for the given model. """
    gpus = None
    if torch.cuda.is_available():
        gpus = 1
    trainer = pl.Trainer(logger=False, gpus=gpus)
    for fragment_type in FRAGMENTS:
        for pos in ["ADJ", "NOUN", "VERB"]:
            per_sample_results = targeted_triplet_score(fragment_type, pos, model, trainer)
            yield dict(fragment_type=fragment_type,
                       pos=pos,
                       targeted_triplet_acc=np.mean(per_sample_results)
                       ), per_sample_results
        

def targeted_triplet_score(fragment_type, pos, model, trainer):
    ds = PeppaTargetedTripletDataset.load(f"data/out/val_{fragment_type}_targeted_triplets_{pos}")
    loader = DataLoader(ds, collate_fn=collate_triplets, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
    results = []
    for batch in trainer.predict(model, loader):
        results.extend(r.item() for r in batch_triplet_accuracy(batch))

    return results


def get_all_results_df(results_dir):
    results_data_all = []
    for pos in ["ADJ", "VERB", "NOUN"]:
        for fragment_type in FRAGMENTS:
            results_data_fragment = pd.read_csv(f"{results_dir}/targeted_triplets_{fragment_type}_{pos}.csv",
                                                converters={"tokenized": ast.literal_eval})
            results_data_all.append(results_data_fragment)

    results_data_all = pd.concat(results_data_all, ignore_index=True)
    results_data_all["duration"] = results_data_all["clipEnd"] - results_data_all["clipStart"]
    results_data_all = results_data_all[results_data_all["duration"] > MIN_DURATION]
    return results_data_all


def create_duration_results_plots(results_data_all, results_dir, version):
    results_data_all["clipDuration"] = results_data_all["clipEnd"] - results_data_all["clipStart"]
    results_data_all["clipDuration"] = results_data_all["clipDuration"].round(1)

    plt.figure()
    results_data_all.groupby("clipDuration").size().plot.bar()
    plt.title(f"Number of samples: per duration")
    plt.xticks(rotation=75)
    plt.savefig(os.path.join(results_dir, f"num_samples_vs_duration"), dpi=300)

    plt.figure()
    results_data_all["num_tokens"] = results_data_all.tokenized.apply(len)
    results_data_all.groupby("num_tokens").size().plot.bar()
    plt.title(f"Number of samples: per number of tokens")
    plt.xticks(rotation=75)
    plt.savefig(os.path.join(results_dir, f"num_samples_vs_num_tokens"), dpi=300)

    plt.figure()
    sns.barplot(data=results_data_all, x="clipDuration", y="result")
    plt.axhline(y=0.5, color="black", linestyle='--')
    plt.title(f"Version: {version} | Accuracy: per duration")
    plt.xticks(rotation=75)
    plt.savefig(os.path.join(results_dir, f"results_clip_duration"), dpi=300)

    plt.figure()
    sns.barplot(data=results_data_all, x="num_tokens", y="result")
    plt.axhline(y=0.5, color="black", linestyle='--')
    plt.title(f"Version: {version} | Accuracy: per number of tokens")
    plt.xticks(rotation=75)
    plt.savefig(os.path.join(results_dir, f"results_num_tokens"), dpi=300)


def create_per_word_result_plots(results_dir, version, args):
    results_data_words_all = []
    for pos in ["ADJ", "VERB", "NOUN"]:
        results_data = []
        for fragment_type in FRAGMENTS:
            results_data_fragment = pd.read_csv(f"{results_dir}/targeted_triplets_{fragment_type}_{pos}.csv", converters={"tokenized":ast.literal_eval})
            results_data_fragment["duration"] = results_data_fragment["clipEnd"] - results_data_fragment["clipStart"]
            results_data_fragment = results_data_fragment[results_data_fragment["duration"] > MIN_DURATION]
            results_data.append(results_data_fragment)
        results_data = pd.concat(results_data, ignore_index=True)

        results_data_words_1 = results_data.copy()
        results_data_words_1["word"] = results_data_words_1["target_word"]
        results_data_words_2 = results_data.copy()
        results_data_words_2["word"] = results_data_words_2["distractor_word"]
        results_data_words = pd.concat([results_data_words_1, results_data_words_2], ignore_index=True)

        plt.figure(figsize=(15, 8))
        results_data_words.groupby("word").size().plot.bar()
        plt.title(f"Number of samples: {pos}")
        plt.xticks(rotation=75)
        plt.axhline(y=args.min_samples, color="black", linestyle='--')
        plt.savefig(os.path.join(results_dir, f"num_samples_{pos}_word"), dpi=300)

        words_enough_data = [w for w, occ in results_data_words.groupby("word").size().items() if
                             occ > args.min_samples]
        if len(words_enough_data) == 0:
            print(f"No words with enough samples (>{args.min_samples}) found for POS: {pos}")
            continue

        results_data_words = results_data_words[results_data_words.word.isin(words_enough_data)]
        results_data_words_all.append(results_data_words)

        plt.figure(figsize=(15, 8))
        mean_acc = results_data_words.groupby("word")["result"].agg("mean")
        order = mean_acc.sort_values()
        sns.barplot(data=results_data_words, x="word", y="result", order=order.index)
        plt.title(f"Per-word targeted triplets accuracy for model ID: {version} | POS: {pos}")
        plt.xticks(rotation=75)
        plt.ylabel("Accuracy")
        plt.ylim((0, 1))
        plt.subplots_adjust(bottom=0.1)
        plt.axhline(y=0.5, color="black", linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"results_{pos}_word"), dpi=300)

    if args.correlate_predictors:
        word_concreteness_ratings = get_word_concreteness_ratings()
        dataset_word_frequencies = get_dataset_word_frequencies()

        results_data_words_all = pd.concat(results_data_words_all, ignore_index=True)
        mean_acc = results_data_words_all.groupby("word")["result"].agg("mean")

        # Correlate performance with word frequency in train split
        word_frequencies = [dataset_word_frequencies[w] for w in mean_acc.keys()]
        word_frequencies = [np.log(f) for f in word_frequencies]
        word_accuracies = mean_acc.values
        plt.figure()
        s1 = sns.scatterplot(word_frequencies, word_accuracies, marker="x")
        plt.xlabel("Log Frequency")
        plt.ylabel("Accuracy")
        # Named labels in scatterplot:
        for i in range(len(word_frequencies)):
            s1.text(word_frequencies[i] + 0.01, word_accuracies[i],
                    mean_acc.keys()[i], horizontalalignment='left',
                    size='small', color='black')
        plt.savefig(os.path.join(results_dir, f"results_correlation_frequency_acc"), dpi=300)
        print(f"Pearson correlation frequency-acc: ", pearsonr(word_frequencies, word_accuracies))

        # Correlate performance with word concreteness
        word_concretenesses = [get_word_concreteness(w, word_concreteness_ratings) for w in mean_acc.keys()]
        plt.figure()
        s2 = sns.scatterplot(word_concretenesses, word_accuracies, marker="x")
        plt.xlabel("Concreteness")
        plt.ylabel("Accuracy")
        # Named labels in scatterplot:
        for i in range(len(word_frequencies)):
            s2.text(word_concretenesses[i] + 0.01, word_accuracies[i],
                    mean_acc.keys()[i], horizontalalignment='left',
                    size='small', color='black')
        plt.savefig(os.path.join(results_dir, f"results_correlation_concreteness_acc"), dpi=300)
        print(f"Pearson correlation concreteness-acc: ", pearsonr(word_concretenesses, word_accuracies))


def get_dataset_word_frequencies():
    _, data_tokens = load_data()

    all_words = get_lemmatized_words(data_tokens, "train")

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
        if word.endswith("'s"):
            # Concreteness for possisive pronouns his/her is around 3
            return 3
        else:
            if word in WORDS_NAMES:
                # Assume that persons are maximally concrete
                return 5
            else:
                print(f"Warning: concreteness rating not found for '{word}'. Setting to 3/5.")
                return 3


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--versions", type=str, nargs="+")
    parser.add_argument("--min-samples", type=int, default=100)

    parser.add_argument("--correlate-predictors", action="store_true", default=False)

    return parser.parse_args()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    args = get_args()

    rows = []
    for version in args.versions:
        logging.info(f"Evaluating version {version}")
        net, path = load_best_model(f"lightning_logs/version_{version}/")
        results_dir = f"results/version_{version}"

        for row, per_sample_results in score(net):
            row['version'] = version
            row['path']    = path
            row['audio_pretrained'] = net.config['audio']['pretrained']
            row['video_pretrained'] = net.config['video']['pretrained']
            row['audio_pooling'] = net.config['audio']['pooling']
            row['video_pooling'] = net.config['video']['pooling']
            print(row)
            rows.append(row)

            # Save per-sample results for detailed analysis
            eval_info_file = f"data/eval/eval_set_{row['fragment_type']}_{row['pos']}.csv"
            results_data = pd.read_csv(eval_info_file, index_col="id")

            assert len(results_data) == len(per_sample_results), \
                f"Number of samples in eval set ({len(per_sample_results)}) doesn't match CSV info from " \
                f"{eval_info_file} ({len(results_data)})"

            results_data["result"] = per_sample_results
            os.makedirs(results_dir, exist_ok=True)
            results_data.to_csv(f"{results_dir}/targeted_triplets_{row['fragment_type']}_{row['pos']}.csv")

        create_per_word_result_plots(results_dir, version, args)
        all_results = get_all_results_df(results_dir)
        create_duration_results_plots(all_results, results_dir, version)

        print("Average accuracy: ", all_results["result"].mean())

    scores = pd.DataFrame.from_records(rows)
    scores.to_csv("results/scores_targeted_triplets.csv", index=False, header=True)
