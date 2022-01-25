import argparse
import itertools
import json
import os
import re
from collections import Counter

import pandas as pd
from spacy.tokens import Doc
from tqdm import tqdm

import spacy

from pig.data import SPLIT_SPEC

DATA_DIR = "data/out/180x100/"
REALIGNED_DATA_DIR = "data/out/realign/"
DATA_EVAL_DIR = "data/eval/"

FRAGMENTS = ["narration"]
POS_TAGS = ["ADJ", "VERB", "NOUN"]

WORDS_NAMES = [
    "chloe",
    "danny",
    "george",
    "pedro",
    "peppa",
    "rebecca",
    "richard",
    "susie",
    "suzy",
]

SYNONYMS_REPLACE = {"granddad": "grandpa", "mommy": "mummy", "grandma": "granny"}

# Ignore some words that have been mistagged by the POS-tagger (partly because of poor pre-tokenization):
WORDS_IGNORE = {
    "VERB": ["they're", "we're", "what's", "can't"],
    "NOUN": ["peppa's", "george's", "let's", "pig's", "i'll", "rabbit's", "daddy's", "chloe's",
             "can't", "doesn't", "suzy's", "zebra's", "zoe's", "it's", "dog's", "dinosaur's", "they're", "grandpa's",
             "rebecca's", "we've", "there's", "you'll", "i'm", "we'll", "i've", "what's", "i'll", "that's", "you're",
             "we'd", "we're", "bit", "lot", "be",
             "dear", "love"],   # ("love" and "dear" are not used as nouns in the dataset)
    "ADJ": ["it's", "that's"],
}

TOKEN_MASK = "<MASK>"


def clean_lemma(lemma):
    lemma = lemma.lower()
    # Remove punctuation
    if lemma[-1] in [".", ",", "'", "?", "!"]:
        lemma = lemma[:-1]
    if lemma in SYNONYMS_REPLACE:
        lemma = SYNONYMS_REPLACE[lemma]
    return lemma


def load_realigned_data():
    nlp = spacy.load("en_core_web_sm")

    data_sentences = []
    data_tokens = []

    for root, dirs, files in os.walk(REALIGNED_DATA_DIR):
        for file in files:
            if file.endswith(".json"):
                path = os.path.join(root, file)
                item = json.load(open(path, "r"))
                fragment = "narration" if "narration" in root else "dialog"
                episode = int(path.split("/")[-3].split("_")[1])

                # Remove punctuation
                item["transcript"] = re.sub(r"\s*[\.!]+\s*$", "", item["transcript"])
                item["transcript"] = re.sub(r"\s*[-:\.♪]+\s*", " ", item["transcript"])

                # Remove whitespace
                item["transcript"] = re.sub(r"\s+$", "", item["transcript"])
                item["transcript"] = re.sub(r"^\s+", "", item["transcript"])
                item["transcript"] = re.sub(r"\s\s", " ", item["transcript"])

                tokenized = re.split(" ", item["transcript"])
                if len(tokenized) != len(item["words"]):
                    raise RuntimeError(
                        f"Not aligned: {tokenized} and {[w['word'] for w in item['words']]}"
                    )

                item["tokenized"] = [w.lower() for w in tokenized]

                doc = Doc(nlp.vocab, words=tokenized)
                for name, proc in nlp.pipeline:
                    doc = proc(doc)

                # Treat proper nouns the same way as nouns
                item["pos"] = [t.pos_ if t.pos_ != "PROPN" else "NOUN" for t in doc]

                item["lemmatized"] = [clean_lemma(t.lemma_) for t in doc]

                for i in range(len(item["words"])):
                    item["words"][i]["fragment"] = fragment
                    item["words"][i]["path"] = path
                    item["words"][i]["episode"] = episode
                    item["words"][i]["pos"] = item["pos"][i]
                    item["words"][i]["lemma"] = item["lemmatized"][i]

                data_tokens.extend(item["words"])

                item_sentence = item.copy()
                item_sentence["fragment"] = fragment
                item_sentence["episode"] = episode
                data_sentences.append(item_sentence)

    data_tokens = pd.DataFrame(data_tokens)
    data_sentences = pd.DataFrame(data_sentences)
    return data_sentences, data_tokens


def load_data():
    nlp = spacy.load("en_core_web_sm")

    data_sentences = []
    data_tokens = []

    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(".json"):
                path = os.path.join(root, file)
                data_file = json.load(open(path, "r"))
                fragment = "narration" if "narration" in root else "dialog"
                episode = int(path.split("/")[-2])

                for subtitle in data_file["subtitles"]:
                    item = {"transcript": subtitle["text"]}

                    # Remove punctuation
                    item["transcript"] = re.sub(
                        r"\s*[\.!]+\s*$", "", item["transcript"]
                    )
                    item["transcript"] = re.sub(
                        r"\s*[-:\.♪]+\s*", " ", item["transcript"]
                    )

                    # Remove whitespace
                    item["transcript"] = re.sub(r"\s+$", "", item["transcript"])
                    item["transcript"] = re.sub(r"^\s+", "", item["transcript"])
                    item["transcript"] = re.sub(r"\s\s", " ", item["transcript"])

                    tokenized = re.split(" ", item["transcript"])

                    item["tokenized"] = [w.lower() for w in tokenized]

                    doc = Doc(nlp.vocab, words=tokenized)
                    for name, proc in nlp.pipeline:
                        doc = proc(doc)

                    # Treat proper nouns the same way as nouns
                    item["pos"] = [t.pos_ if t.pos_ != "PROPN" else "NOUN" for t in doc]
                    item["lemmatized"] = [t.lemma_.lower() for t in doc]

                    item["words"] = [{"word": w} for w in item["tokenized"]]
                    for i in range(len(item["words"])):
                        item["words"][i]["fragment"] = fragment
                        item["words"][i]["path"] = path
                        item["words"][i]["episode"] = episode
                        item["words"][i]["pos"] = item["pos"][i]
                        item["words"][i]["lemma"] = item["lemmatized"][i]

                    data_tokens.extend(item["words"])

                    item_sentence = item.copy()
                    item_sentence["fragment"] = fragment
                    item_sentence["episode"] = episode
                    data_sentences.append(item_sentence)

    data_tokens = pd.DataFrame(data_tokens)
    data_sentences = pd.DataFrame(data_sentences)
    return data_sentences, data_tokens


def is_sublist(list_1, list_2):
    def get_all_in(one, another):
        for element in one:
            if element in another:
                yield element

    for x1, x2 in zip(get_all_in(list_1, list_2), get_all_in(list_2, list_1)):
        if x1 != x2:
            return False

    return True


def longest_intersection(tokens_1, tokens_2):
    longest_sublist = []
    mask_index = tokens_1.index(TOKEN_MASK)
    for i in range(len(tokens_1)):
        for j in range(i, len(tokens_1)):
            if i - 1 < mask_index < j + 1:
                sublist = tokens_1[i : j + 1]
                for k in range(len(tokens_2)):
                    for l in range(k, len(tokens_2)):
                        sublist_2 = tokens_2[k : l + 1]
                        if sublist == sublist_2:
                            if len(sublist) > len(longest_sublist):
                                longest_sublist = sublist

    return longest_sublist


def get_start_and_end_of_sublist(sentence, sublist):
    for i in range(len(sentence)):
        if sentence[i] == sublist[0]:
            start = i
            for j in range(len(sublist)):
                if sentence[i + j] != sublist[j]:
                    break
                if j == len(sublist) - 1:
                    end = i + j
                    return start, end

    raise RuntimeError(f"Could not find {sublist} in {sentence}")


def crop_and_create_example(example, start, end, target_word, distractor_word):
    example["tokenized"] = example["tokenized"][start : end + 1]

    example["words"] = example["words"][start : end + 1]

    example["start_token_idx"] = start
    example["end_token_idx"] = end

    # Update clip start and end time
    example["clipOffset"] = example["clipStart"]
    example["clipStart"] += example["words"][0]["start"]
    example["clipEnd"] = example["clipOffset"] + example["words"][-1]["end"]
    assert example["clipStart"] < example["clipEnd"]

    example["target_word"] = target_word
    example["distractor_word"] = distractor_word

    return example


def find_minimal_pairs(tuples, data, args):
    eval_set = []
    id = 0
    for lemma_1, lemma_2 in tqdm(tuples):
        used_counterexamples = []
        print(f"\nLooking for: {(lemma_1, lemma_2)}")
        for _, s1 in data.iterrows():
            best_example, best_counterexample, best_counterex_row = None, None, None
            len_longest_intersection = 0

            if lemma_1 in s1["lemmatized"]:
                example_candidate = s1.copy()
                s1_masked = [
                    w if lemma != lemma_1 else TOKEN_MASK
                    for w, lemma in zip(
                        example_candidate["tokenized"], example_candidate["lemmatized"]
                    )
                ]
                for row_counterexample, s2 in data.iterrows():
                    if row_counterexample in used_counterexamples:
                        continue

                    if lemma_2 not in s2["lemmatized"]:
                        continue

                    counterexample_candidate = s2.copy()
                    s2_masked = [
                        w if lemma != lemma_2 else TOKEN_MASK
                        for w, lemma in zip(
                            counterexample_candidate["tokenized"],
                            counterexample_candidate["lemmatized"],
                        )
                    ]

                    intersection = longest_intersection(s1_masked, s2_masked)
                    if not intersection:
                        continue

                    start, end = get_start_and_end_of_sublist(s1_masked, intersection)
                    first_word = example_candidate["words"][start]
                    last_word = example_candidate["words"][end]
                    if (
                        first_word["case"] != "success"
                        or last_word["case"] != "success"
                        or "end" not in last_word
                        or "start" not in first_word
                        or last_word["end"] - first_word["start"]
                        < args.min_phrase_duration
                    ):
                        continue

                    (counterex_start, counterex_end,) = get_start_and_end_of_sublist(
                        s2_masked, intersection
                    )
                    first_word = counterexample_candidate["words"][counterex_start]
                    last_word = counterexample_candidate["words"][counterex_end]
                    if (
                        first_word["case"] != "success"
                        or last_word["case"] != "success"
                        or "end" not in last_word
                        or "start" not in first_word
                        or last_word["end"] - first_word["start"]
                        < args.min_phrase_duration
                    ):
                        continue

                    if len(intersection) > len_longest_intersection:
                        example = crop_and_create_example(
                            example_candidate.copy(), start, end, lemma_1, lemma_2,
                        )

                        counterexample = crop_and_create_example(
                            counterexample_candidate.copy(),
                            counterex_start,
                            counterex_end,
                            lemma_2,
                            lemma_1,
                        )

                        len_longest_intersection = len(intersection)
                        best_example = example
                        best_counterexample = counterexample
                        best_counterex_row = row_counterexample

            if best_example is not None:
                best_example["id"] = id
                id += 1
                best_counterexample["id"] = id
                id += 1
                best_example["id_counterexample"] = best_counterexample["id"]
                best_counterexample["id_counterexample"] = best_example["id"]

                eval_set.append(best_example)
                eval_set.append(best_counterexample)
                print(best_example["tokenized"])
                print(best_counterexample["tokenized"], end="\n\n")

                used_counterexamples.append(best_counterex_row)

    eval_set = pd.DataFrame(eval_set)
    if len(eval_set) > 0:
        eval_set.set_index("id", inplace=True)

    return eval_set


def get_lemmatized_words(data_tokens, data_split, fragments=FRAGMENTS, pos=None):
    all_words = []
    for fragment in fragments:
        words = data_tokens[
            (data_tokens.fragment == fragment)
            & data_tokens.episode.isin(SPLIT_SPEC[fragment][data_split])
        ]
        if pos:
            words = words[words.pos == pos]
        lemmas = [w.lemma for _, w in words.iterrows()]
        all_words.extend(lemmas)

    return all_words


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min-occurrences",
        type=int,
        default=10,
        help="Minimum number of occurrences in val data of a word to be included",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=100,
        help="Minimum number of test samples for a word to be included",
    )
    parser.add_argument(
        "--min-phrase-duration",
        type=float,
        default=0.3,
        help="Minimum duration of a phrase (in seconds)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.makedirs(DATA_EVAL_DIR, exist_ok=True)

    data_sentences, data_tokens = load_realigned_data()

    for pos_name in POS_TAGS:
        print(f"Looking for {pos_name}s:")
        # Find most common words
        words = get_lemmatized_words(data_tokens, "val", fragments=FRAGMENTS, pos=pos_name)

        counter = Counter(words)

        words = [
            w
            for w, occ in counter.items()
            if occ > args.min_occurrences and w not in WORDS_IGNORE[pos_name]
        ]
        print("Considered words: ", words)
        tuples = list(itertools.combinations(words, 2))

        for fragment in FRAGMENTS:
            data_fragment = data_sentences[data_sentences.fragment == fragment]
            data_fragment_val = data_fragment[
                data_fragment.episode.isin(SPLIT_SPEC[fragment]["val"])
            ]

            eval_set = find_minimal_pairs(tuples, data_fragment_val, args)
            eval_set["fragment"] = fragment

            # Filter examples by min num samples
            counts = eval_set.target_word.value_counts()
            words_enough_samples = counts[counts > args.min_samples].keys().to_list()
            if len(words_enough_samples) == 0:
                print(f"No words with enough samples (>{args.min_samples}) found for POS {pos_name} and fragment {fragment}.")
            eval_set = eval_set[eval_set.target_word.isin(words_enough_samples) | eval_set.distractor_word.isin(words_enough_samples)]

            # Sort by duration
            eval_set["clipDuration"] = eval_set["clipEnd"] - eval_set["clipStart"]
            eval_set = eval_set.sort_values(by=['clipDuration'])

            file_name = f"eval_set_{fragment}_{pos_name}.csv"
            file_dir = os.path.join(DATA_EVAL_DIR, f"min_phrase_duration_{args.min_phrase_duration}")
            os.makedirs(file_dir, exist_ok=True)
            eval_set.to_csv(os.path.join(file_dir, file_name))
