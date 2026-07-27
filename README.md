# QBugsSet: A Lifecycle-Based Dataset of Real-World Bugs in Quantum Software

This repository accompanies our paper **“How Quantum Bugs Live and Die: A Lifecycle-Based Empirical Study of Bugs in Quantum Software,”** accepted at the **IEEE/ACM International Conference on Automated Software Engineering (ASE 2026)**.

QBugsSet contains **4,222 file-level samples** collected from real-world Qiskit, Cirq, and PennyLane projects, including **2,429 bugs** and **1,793 non-buggy samples**. It can be used for quantum software bug analysis and LLM-based bug detection and repair.

## Dataset

The final, ready-to-use dataset is:

[`data/05_datasets/dataset_parent_all.json`](data/05_datasets/dataset_parent_all.json)

Each sample contains source code, its binary label (`1` for buggy and `0` for non-buggy), repository information, and available lifecycle and quantum-specificity annotations.

Bug repair uses the paired buggy and fixed files in:

[`data/03_extracted/method_level_single_merged.json`](data/03_extracted/method_level_single_merged.json)

## Installation

Python 3.10 or later is required.

```bash
git clone https://github.com/ArthurShy/QBugsSet.git
cd QBugsSet

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

List the configured models:

```bash
python llm/run.py --list-models
```

## LLM Bug Detection

Set the API key for the selected provider. For example:

```bash
export DEEPSEEK_API_KEY=your_api_key
```

Run bug detection:

```bash
python llm/run.py \
  --task detection \
  --model deepseek-chat \
  --dataset data/05_datasets/dataset_parent_all.json \
  --output-dir runs
```

For a quick test, add `--limit 10`. To run requests in parallel, add `--parallel 10`.

## LLM Bug Repair

The repair evaluator selects the buggy samples from the final dataset and joins them by sample ID with the included paired files. The prompt uses the full buggy file and asks the model to return directly applicable `SEARCH/REPLACE` edits.

Run bug repair:

```bash
python llm/run.py \
  --task repair \
  --model deepseek-chat \
  --dataset data/05_datasets/dataset_parent_all.json \
  --output-dir runs
```

Add `--limit 10` for a quick test or `--parallel 10` for parallel requests.

## Results

Results are saved automatically to:

```text
runs/detection/<api-or-local>/<model>/
runs/repair/<api-or-local>/<model>/
```

The detection report includes accuracy, precision, recall, F1, and per-sample predictions. The repair report includes edit validity, localization, patch similarity, and exact-match metrics.

Prompt definitions are available in [`PROMPTS.md`](PROMPTS.md).

## Build the Dataset (Optional)

> We recommend using the ready-to-use dataset at [`data/05_datasets/dataset_parent_all.json`](data/05_datasets/dataset_parent_all.json). Rebuilding it requires GitHub mining, LLM annotation, substantial storage, and API usage.
>
> The commands below write generated files under `data/`. Run them in a fresh clone or worktree if you want to preserve the published datasets unchanged.

Set the required credentials:

```bash
export GITHUB_TOKEN_1=your_github_token
export DEEPSEEK_API_KEY=your_deepseek_api_key
```

Collect and extract samples from the three supported frameworks:

```bash
for framework in qiskit cirq pennylane; do
  python data_preprocessing/get_repo.py --framework "$framework"
  python data_preprocessing/get_commit.py --framework "$framework"
  python data_preprocessing/get_bug.py --framework "$framework"
  python data_preprocessing/extract_method_level_single_func.py --framework "$framework"
done

python data_preprocessing/merge_framework_datasets.py
```

Run lifecycle annotation and construct the final dataset:

```bash
python data_preprocessing/deepseek_analyze/run.py
python data_preprocessing/deepseek_analyze/run.py --stage-three
python data_preprocessing/deepseek_analyze/run.py --stage-four
python data_preprocessing/deepseek_analyze/update_submodules.py

python data_postprocessing/code/download_parent_repos.py
python data_postprocessing/code/build_parent_all_func_negatives.py
```

The rebuilt dataset is saved to `data/05_datasets/dataset_parent_all.json`.

## Citation

```bibtex
@inproceedings{shi2026quantumbugs,
  author    = {Yasai Shi and Xiangxin Meng and Xiangjie Huang and
               Jian Zhang and Tianyu Wo and Xu Wang},
  title     = {How Quantum Bugs Live and Die: A Lifecycle-Based Empirical
               Study of Bugs in Quantum Software},
  booktitle = {IEEE/ACM International Conference on Automated Software Engineering (ASE)},
  year      = {2026}
}
```

## License

This project is released under the [MIT License](LICENSE).
