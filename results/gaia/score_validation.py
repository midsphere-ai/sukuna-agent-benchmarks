"""Score validation/answers.jsonl against the GAIA validation answers.

The GAIA dataset is not in this repository. Download it from
https://huggingface.co/datasets/gaia-benchmark/GAIA and pass the path to
2023/validation/metadata.jsonl:

    python score_validation.py /path/to/2023/validation/metadata.jsonl

The comparison is the scorer published with the GAIA leaderboard
(https://huggingface.co/spaces/gaia-benchmark/leaderboard/blob/main/scorer.py).
"""

import json
import re
import string
import sys
from collections import Counter
from pathlib import Path


def normalize_number_str(number_str: str) -> float:
    for char in ["$", "%", ","]:
        number_str = number_str.replace(char, "")
    try:
        return float(number_str)
    except ValueError:
        return float("inf")


def normalize_str(input_str: str, remove_punct: bool = True) -> str:
    no_spaces = re.sub(r"\s", "", input_str)
    if remove_punct:
        return no_spaces.lower().translate(str.maketrans("", "", string.punctuation))
    return no_spaces.lower()


def is_float(element) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False


def question_scorer(model_answer: str, ground_truth: str) -> bool:
    if is_float(ground_truth):
        return normalize_number_str(model_answer) == float(ground_truth)

    if any(char in ground_truth for char in [",", ";"]):
        gt_elems = re.split("[,;]", ground_truth)
        ma_elems = re.split("[,;]", model_answer)
        if len(gt_elems) != len(ma_elems):
            return False
        return all(
            normalize_number_str(ma) == float(gt) if is_float(gt)
            else normalize_str(ma, remove_punct=False) == normalize_str(gt, remove_punct=False)
            for ma, gt in zip(ma_elems, gt_elems)
        )

    return normalize_str(model_answer) == normalize_str(ground_truth)


def main(metadata_path: str) -> None:
    gold, level = {}, {}
    for line in open(metadata_path, encoding="utf-8"):
        row = json.loads(line)
        gold[row["task_id"]] = row["Final answer"]
        level[row["task_id"]] = int(row["Level"])

    answers_path = Path(__file__).resolve().parent / "validation" / "answers.jsonl"
    total, correct = Counter(), Counter()
    for line in open(answers_path, encoding="utf-8"):
        row = json.loads(line)
        lv = level[row["task_id"]]
        total[lv] += 1
        correct[lv] += question_scorer(row["model_answer"], gold[row["task_id"]])

    for lv in sorted(total):
        print(f"level {lv}: {correct[lv]}/{total[lv]}")
    n, c = sum(total.values()), sum(correct.values())
    print(f"all: {c}/{n} = {100 * c / n:.2f}%")


def _self_check() -> None:
    assert question_scorer("1,000", "1000")
    assert question_scorer("b, a", "b,a")
    assert not question_scorer("a, b", "b, a")
    assert question_scorer("The Answer.", "the answer")
    assert not question_scorer("12", "13")


if __name__ == "__main__":
    _self_check()
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
