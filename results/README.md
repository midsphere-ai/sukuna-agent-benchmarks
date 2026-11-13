# Results

Every file here is output from our own runs, or public leaderboard data. GAIA questions, attached files and gold answers are not included. GAIA files hold `task_id` and `model_answer` only.

## gaia/test

The test split has 301 tasks. Its answers are private, so only the leaderboard can score a submission.

- `leaderboard_rows.json`: the ten rows the GAIA leaderboard holds for the organisation "Midsphere AI", copied from the public dataset `gaia-benchmark/results_public` on 2026-11-13. Scores are fractions of 301.

  | Entry | Date | Correct | Score |
  |---|---|---|---|
  | Sukuna V0.1 | 2025-10-01 | 156 | 51.83% |
  | Sukuna V0.1.1 | 2025-10-02 | 175 | 58.14% |
  | Sukuna V0.1.2 | 2025-10-03 | 180 | 59.80% |
  | Sukuna V0.1.3 | 2025-10-04 | 187 | 62.13% |
  | Sukuna V0.1.4 | 2025-10-05 | 195 | 64.78% |
  | Sukuna V0.1.5 | 2025-10-06 | 209 | 69.44% |
  | Sukuna V0.1.6 | 2025-10-08 | 208 | 69.10% |
  | Sukuna V0.1.7 | 2025-10-09 | 209 | 69.44% |
  | Sukuna V0.1.8 | 2025-10-10 | 213 | 70.76% |
  | Sukuna V0.1.9 | 2025-10-15 | 213 | 70.76% |

- `submissions/submission_01.jsonl` to `submission_10.jsonl`: ten merged answer files for the full test split, in the order they were made. There are ten files and ten leaderboard entries. The repository does not record which file was uploaded for which entry, and several files were changed in a later commit, so do not assume a one-to-one match.
- `model_runs/`: partial and full runs that fed the merged files, grouped by the folder name they were saved under (`gpt-5`, `claude-4`, `glm-4.6`, `gemini-2.5`, `perplexity`, and `merged` for files that mix models). The folder name is the label used at the time. It was not checked against the run logs, and at least one pair of files under different labels held identical answers. Version numbers in file names do not imply an order.

## gaia/validation

The validation split has 165 tasks. Its answers are public in the gated dataset.

- `answers.jsonl`: the final answer per task after five passes.
- `summary.json`: the score, by level, from `score_validation.py`.
- `exo_agent_single_task_2026-03.jsonl`: seven attempts by the 2026 Exo-based agent at one level 3 task. None matched. This is the only output kept from that agent.

To check the validation score, download GAIA and run:

```bash
python results/gaia/score_validation.py data/gaia/2023/validation/metadata.jsonl
```

Expected output: `all: 140/165 = 84.85%`.

## imo-2025

One run per problem on 2025-08-13 with gemini-2.5-pro as solver and grader.

- `solutions.json`: for each problem, the final solver output and the stored solver and grader rounds.
- `logs/solution_imoN.log.gz`: the full run log for each problem, including earlier attempts. Unpack with `gunzip`.

| Problem | Attempts in the log | Solver's final verdict | Final answer matches the official one |
|---|---|---|---|
| 1 | 7 | complete | yes |
| 2 | 2 | complete | proof, no numeric answer |
| 3 | 1 | complete | yes |
| 4 | 5 | complete | yes |
| 5 | 4 | complete | yes |
| 6 | 17 | partial, upper bound only | no |

No human graded these proofs.
