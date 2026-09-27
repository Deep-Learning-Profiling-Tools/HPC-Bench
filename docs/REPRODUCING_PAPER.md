# Reproducing the Paper

This guide regenerates the paper's figures from the released per-trial results, and describes how to re-run the full pipeline from generation to measurement.

## Figures from the released results

`final_data/` holds one row per (benchmark, setting, model, trial, input scale, thread count), including compile status, correctness, and kernel and total runtimes:

| File | Content |
|:---|:---|
| `final_data/unified_results.csv` | EX1 and EX2, 92 benchmarks, schema in `final_data/README.md` |
| `final_data/unified_results_ex3.csv` | EX3 |

Every figure below is computed from these two files only. Install the Python dependencies (`pip install -r requirements.txt`) and run the scripts from the repository root. Outputs are written to `analysis_summaries/`.

| Paper figure | Script | Output |
|:---|:---|:---|
| Fast@3 and Speedup@3 vs. p<sub>thr</sub>, EX1–EX3 (main results) | `scripts/analysis/plot_combined_main_figure.py` | `beautiful_figure/main_paper/figure_fast_speedup_at_3_combined.pdf` |
| Fast@k and Speedup@k vs. k, EX1 | `scripts/analysis/plot_k_sensitivity_new_style.py` | `beautiful_figure/main_paper/sensitivity_analysis/` |
| Per-input-scale Fast@3 and Speedup@3 | `scripts/analysis/plot_combined_main_figure.py` | `beautiful_figure/main_paper/figure_{fast,speedup}_at_3_per_dataset.pdf` |
| Best-of-N speedup distribution, EX1 vs. EX2 | `scripts/analysis/plot_best_of_n_distribution.py` | `beautiful_figure/main_paper/figure_best_of_n_distribution_ex1_vs_ex2.pdf` |
| Motif- and difficulty-stratified Fast@3 and Speedup@3 | `scripts/analysis/plot_appendix_stratified_stepfill.py` | `appendix_figure/fig_stratified_*` |
| Fast@k and Speedup@k vs. k, EX2 and EX3 | `scripts/analysis/plot_appendix_k_sensitivity_ex2_ex3.py` | `appendix_figure/*_vs_k_EX{2,3}_medium_*` |
| Fast@3 and Speedup@3 vs. thread count, EX2 | `scripts/analysis/plot_appendix_thread_sensitivity.py` | `appendix_figure/*_vs_threads_EX2_medium_*` |
| Total-runtime Fast@3 and Speedup@3 | `scripts/analysis/plot_total_time_figures.py` | `beautiful_figure/main_paper/figure_{fast,speedup}_at_3_total.pdf` |

```bash
python scripts/analysis/plot_combined_main_figure.py
python scripts/analysis/plot_k_sensitivity_new_style.py
python scripts/analysis/plot_best_of_n_distribution.py
python scripts/analysis/plot_appendix_stratified_stepfill.py
python scripts/analysis/plot_appendix_k_sensitivity_ex2_ex3.py
python scripts/analysis/plot_appendix_thread_sensitivity.py
python scripts/analysis/plot_total_time_figures.py
```

## Re-running the pipeline

Re-running measurements requires the input data, the toolchain listed in the README, and, for EX3, an NVIDIA GPU. Absolute runtimes depend on the machine; the paper used an AMD EPYC 7543 (32 cores) and an NVIDIA A100.

```bash
source tools/setup_env.sh
python tools/fetch_input_data.py

EX=EX1            # EX1, EX2, or EX3
BENCH=2mm         # or a comma-separated list, or all

# 1) Optional: generate new candidates (requires API keys). The paper's candidates are already included.
tools/run_auto_run.sh $EX $BENCH anthropic claude-sonnet-4-5-20250929 claude

# 2) Build the baseline and all candidates
python scripts/arg.py benchmark build --root . --ex-versions $EX --benchmarks $BENCH

# 3) Run the baseline and the candidates to capture outputs, then check correctness
python scripts/arg.py benchmark correctness-run --root . --ex-versions $EX --benchmarks $BENCH \
  --models baseline,claude,gpt5.1,qwen --datasets mini,small,medium,large,extra-large
tools/run_correctness.sh $EX $BENCH

# 4) Rebuild and measure runtime
python scripts/arg.py benchmark build --root . --ex-versions $EX --benchmarks $BENCH
tools/run_time_measurement.sh $EX $BENCH --models baseline,claude,gpt5.1,qwen \
  --datasets mini,small,medium,large,extra-large
```

For EX2, the thread count is set with `OMP_NUM_THREADS`; the paper reports 1, 2, 8, 16, and 32 threads and uses 16 threads for the main results. Warmup and repetition counts per benchmark and input scale are calibrated in `configs/runtime_hints.json` and `configs/runtime_strategy_<benchmark>.json`.

Correctness results are written to `results/correctness/<benchmark>/<EX>/correctness.jsonl` and timings to `results/time_measurements/<benchmark>/<EX>_time_measurements.jsonl`. `scripts/analysis/analysis_correctness.py` and `scripts/analysis/analysis_time_measurements.py` summarize these files into `analysis_summaries/`, and `scripts/analysis/build_unified_dataframe.py` joins the summaries into the `final_data/unified_results.csv` format.
