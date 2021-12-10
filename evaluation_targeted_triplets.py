import argparse
import ast
import os

import torch

from pig.evaluation import load_best_model, pretraining
import pig.data
import pytorch_lightning as pl
import logging
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pig.targeted_triplets import PeppaTargetedTripletDataset

BATCH_SIZE = 1
NUM_WORKERS = 1


def score(model):
    """Compute all standard scores for the given model. """
    gpus = None
    if torch.cuda.is_available():
        gpus = 1
    trainer = pl.Trainer(logger=False, gpus=gpus)
    for fragment_type in ['dialog', 'narration']:
        for pos in ["ADJ", "VERB", "NOUN"]:
            per_sample_results = targeted_triplet_score(fragment_type, pos, model, trainer)
            yield dict(fragment_type=fragment_type,
                       pos=pos,
                       targeted_triplet_accs=np.mean(per_sample_results)
                       ), per_sample_results
        

def targeted_triplet_score(fragment_type, pos, model, trainer):
    ds = PeppaTargetedTripletDataset.load(f"data/out/val_{fragment_type}_targeted_triplets_{pos}")
    loader = DataLoader(ds, collate_fn=pig.data.collate_triplets, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
    results = [pig.metrics.batch_triplet_accuracy(batch).item() for batch in trainer.predict(model, loader)]

    return results


def format_results_to_tex():
    data = pd.read_csv("results/scores_targeted_triplets.csv")
    for fragment_type in ['dialog', 'narration']:
        table = data.query(f"fragment_type=='{fragment_type}'")
        table['pretraining'] = table.apply(pretraining, axis=1)

        table[['version', 'pretraining',
               'pos', 'targeted_triplet_acc']]\
            .rename(columns=dict(version='ID',
                                 pretraining='Pretraining',
                                 targeted_triplet_acc='Targeted Triplet Acc',
                                 pos='POS'))\
            .to_latex(buf=f"results/scores_{fragment_type}.tex",
                      index=False,
                      float_format="%.3f")


def create_result_plots(args):
    results_plots_dir = "results/plots"
    os.makedirs(results_plots_dir, exist_ok=True)

    results_data_all = []
    for pos in ["ADJ", "VERB", "NOUN"]:
        results_data = []
        for fragment_type in ['dialog', 'narration']:
            results_data_fragment = pd.read_csv(f"results/targeted_triplets_{fragment_type}_{pos}.csv", converters={"tokenized":ast.literal_eval})
            results_data.append(results_data_fragment)
            results_data_all.append(results_data_fragment)
        results_data = pd.concat(results_data, ignore_index=True)

        results_data_words_1 = results_data.copy()
        results_data_words_1["word"] = results_data_words_1["target_word"]
        results_data_words_2 = results_data.copy()
        results_data_words_2["word"] = results_data_words_2["distractor_word"]
        results_data_words = pd.concat([results_data_words_1, results_data_words_2], ignore_index=True)

        words_enough_data = [w for w, occ in results_data_words.groupby("word").size().items() if occ > args.min_samples]
        results_data_words = results_data_words[results_data_words.word.isin(words_enough_data)]
        plt.figure()
        sns.barplot(data=results_data_words, x="word", y="result")
        plt.title(f"{pos}")
        plt.xticks(rotation=75)
        plt.axhline(y=0.5, color="black", linestyle='--')
        plt.savefig(os.path.join(results_plots_dir, f"results_{pos}_word"), dpi=300)

    results_data_all = pd.concat(results_data_all, ignore_index=True)
    results_data_all["clipDuration"] = results_data_all["clipEnd"] - results_data_all["clipStart"]
    results_data_all["clipDuration"] = results_data_all["clipDuration"].round(1)

    plt.figure()
    sns.barplot(data=results_data_all, x="clipDuration", y="result")
    plt.axhline(y=0.5, color="black", linestyle='--')
    plt.xticks(rotation=75)
    plt.savefig(os.path.join(results_plots_dir, f"results_clip_duration"), dpi=300)

    plt.show()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--versions", type=str, nargs="+")

    parser.add_argument("--min-samples", type=int, default=10)

    return parser.parse_args()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    args = get_args()

    rows = []
    for version in [args.versions]:
        logging.info(f"Evaluating version {version}")
        net, path = load_best_model(f"lightning_logs/version_{version}/")

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
                f"Number of samples in eval set {len(per_sample_results)} doesn't match CSV info from {eval_info_file} ({len(results_data)})"

            results_data["result"] = per_sample_results
            results_data.to_csv(f"results/targeted_triplets_{row['fragment_type']}_{row['pos']}.csv")

    scores = pd.DataFrame.from_records(rows)
    scores.to_csv("results/scores_targeted_triplets.csv", index=False, header=True)

    format_results_to_tex()

    create_result_plots(args)
