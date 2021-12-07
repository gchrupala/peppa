import itertools
import json
import os
import re
from collections import Counter

import nltk as nltk
from nltk.stem import WordNetLemmatizer
import pandas as pd
from tqdm import tqdm

from pig.data import SPLIT_SPEC

DATA_DIR = "data/out/realign/"
DATA_EVAL_DIR = "data/eval/"

WORDS_IGNORE = {
    "VERB": ["be", "will", "can"],
    "NOUN": [
        "mr",
        "bit",
        "cannot",
        "time",
        "dear",
        "muddy",
        "it's",
        "oh",
        "look",
        "lot",
        "miss",
        "hello",
    ],
    "ADJ": [],
}
TUPLES_IGNORE = {"VERB": [("love", "like")], "NOUN": [], "ADJ": []}

POS_LEMMATIZER = {"VERB": "v", "NOUN": "n", "ADJ": "a"}

MIN_OCC = 10
MIN_PHRASE_LENGTH = 2

TOKEN_MASK = "<MASK>"

nltk.download("universal_tagset")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")

def load_data():
    data_sentences = []
    data_tokens = []

    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(".json"):
                path = os.path.join(root, file)
                item = json.load(open(path, "r"))
                type = "narration" if "narration" in root else "dialog"
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

                pos = nltk.pos_tag(tokenized, tagset="universal")
                pos = [p[1] for p in pos]
                item["pos"] = pos

                for i in range(len(item["words"])):
                    item["words"][i]["type"] = type
                    item["words"][i]["path"] = path
                    item["words"][i]["episode"] = episode
                    item["words"][i]["pos"] = pos[i]

                data_tokens.extend(item["words"])

                item_sentence = item.copy()
                item_sentence["type"] = type
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
        for j in range(i + MIN_PHRASE_LENGTH - 1, len(tokens_1)):
            if i - 1 < mask_index < j + 1:
                sublist = tokens_1[i : j + 1]
                for k in range(len(tokens_2)):
                    for l in range(k + MIN_PHRASE_LENGTH - 1, len(tokens_2)):
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


def find_minimal_pairs(tuples, data, lemmatizer):
    eval_set = []
    id = 0
    for lemma_1, lemma_2 in tqdm(tuples):
        used_counterexamples = []
        print(f"\nLooking for: {(lemma_1, lemma_2)}")
        for _, s1 in data.iterrows():
            best_example, best_counterexample, best_counterex_row = None, None, None
            len_longest_intersection = 0
            s1_lemmatized = [
                lemmatizer.lemmatize(w, POS_LEMMATIZER.get(pos, "n"))
                for w, pos in zip(s1["tokenized"], s1["pos"])
            ]
            if lemma_1 in s1_lemmatized:
                example_candidate = s1.copy()
                s1_masked = [
                    w
                    if lemmatizer.lemmatize(w, POS_LEMMATIZER.get(pos, "n")) != lemma_1
                    else TOKEN_MASK
                    for w, pos in zip(
                        example_candidate["tokenized"], example_candidate["pos"]
                    )
                ]
                for row_counterexample, s2 in data.iterrows():
                    if row_counterexample in used_counterexamples:
                        continue

                    s2_lemmatized = [
                        lemmatizer.lemmatize(w, POS_LEMMATIZER.get(pos, "n"))
                        for w, pos in zip(s2["tokenized"], s2["pos"])
                    ]
                    if lemma_2 not in s2_lemmatized:
                        continue

                    counterexample_candidate = s2.copy()
                    s2_masked = [
                        w
                        if lemmatizer.lemmatize(w, POS_LEMMATIZER.get(pos, "n"))
                        != lemma_2
                        else TOKEN_MASK
                        for w, pos in zip(
                            counterexample_candidate["tokenized"],
                            counterexample_candidate["pos"],
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
                    ):
                        continue

                    if len(intersection) > len_longest_intersection:
                        example = crop_and_create_example(
                            example_candidate.copy(), start, end, lemma_1, lemma_2,
                        )

                        example["id"] = id
                        id += 1

                        counterexample = crop_and_create_example(
                            counterexample_candidate.copy(),
                            counterex_start,
                            counterex_end,
                            lemma_2,
                            lemma_1,
                        )

                        counterexample["id"] = id
                        id += 1

                        example["id_counterexample"] = counterexample["id"]
                        counterexample["id_counterexample"] = example["id"]

                        len_longest_intersection = len(intersection)
                        best_example = example
                        best_counterexample = counterexample
                        best_counterex_row = row_counterexample

            if best_example is not None:
                eval_set.append(best_example)
                eval_set.append(best_counterexample)
                print(best_example["tokenized"])
                print(best_counterexample["tokenized"], end="\n\n")

                used_counterexamples.append(best_counterex_row)

    eval_set = pd.DataFrame(eval_set)
    if len(eval_set) > 0:
        eval_set.set_index("id", inplace=True)

    return eval_set


if __name__ == "__main__":
    os.makedirs(DATA_EVAL_DIR, exist_ok=True)

    lemmatizer = WordNetLemmatizer()

    data_sentences, data_tokens = load_data()

    for pos_name in ["NOUN", "VERB", "ADJ"]:
        print(f"Looking for {pos_name}s:")
        # Find most common words
        words_dialog = data_tokens[
            (data_tokens.type == "dialog")
            & data_tokens.episode.isin(SPLIT_SPEC["dialog"]["val"])
            & (data_tokens.pos == pos_name)
        ].word.values
        words_dialog = [
            lemmatizer.lemmatize(w.lower(), POS_LEMMATIZER[pos_name])
            for w in words_dialog
        ]
        words_narration = data_tokens[
            (data_tokens.type == "narration")
            & data_tokens.episode.isin(SPLIT_SPEC["narration"]["val"])
            & (data_tokens.pos == pos_name)
        ].word.values
        words_narration = [
            lemmatizer.lemmatize(w.lower(), POS_LEMMATIZER[pos_name])
            for w in words_narration
        ]

        counter = Counter(words_narration + words_dialog)

        words = [
            w
            for w, occ in counter.items()
            if occ > MIN_OCC and w not in WORDS_IGNORE[pos_name]
        ]
        print("Considered words: ", words)
        tuples = list(itertools.combinations(words, 2))

        tuples = [t for t in tuples if t not in TUPLES_IGNORE[pos_name]]

        eval_sets = []
        for fragment in ["narration", "dialog"]:
            data_fragment = data_sentences[data_sentences.type == fragment]
            data_fragment_val = data_fragment[
                data_fragment.episode.isin(SPLIT_SPEC[fragment]["val"])
            ]

            eval_set = find_minimal_pairs(tuples, data_fragment_val, lemmatizer)
            eval_set["fragment"] = fragment
            eval_sets.append(eval_set)

            file_name = f"eval_set_{fragment}_{pos_name}.csv"
            eval_set.to_csv(os.path.join(DATA_EVAL_DIR, file_name))
