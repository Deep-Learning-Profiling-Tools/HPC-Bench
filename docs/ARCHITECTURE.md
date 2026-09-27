# Architecture

This document describes how HPC-Bench is organized and how to extend it.

## Directory layout

```
HPC-Bench/
├── EX1/                  # Serial CPU setting (92 benchmarks)
├── EX2/                  # OpenMP CPU setting (92 benchmarks)
├── EX3/                  # CUDA setting
├── common.mk             # Shared build rules
├── common_parboil/       # Parboil-suite Makefile infrastructure
├── common_rodinia/       # Rodinia-suite Makefile infrastructure
├── common_MachSuite/     # MachSuite Makefile infrastructure
├── driver_parboil/       # Parboil driver utilities
├── utilities/            # PolyBench utilities
├── scripts/
│   ├── arg.py            # Single CLI entry point
│   ├── benchmarks.py     # Core executor
│   ├── benchmark_catalog.py / benchmark_catalog_ex2.py
│   ├── benchmark_args.py # Per-benchmark runtime arguments
│   ├── ex_versions.py    # Experiment-version helpers
│   ├── llm/              # Provider-specific LLM clients
│   ├── checkers/         # Suite-specific correctness validators
│   └── analysis/         # Metrics, aggregation, and paper figures
├── configs/              # Per-benchmark runtime strategies
├── final_data/           # Per-trial results used in the paper
├── tools/                # fetch_input_data.py and pipeline wrappers
└── docs/                 # This documentation
```

Input data is not stored in Git. `tools/fetch_input_data.py` downloads it from the Hugging Face dataset `bcui2/HPC-Bench` into `EX<N>/<benchmark>/input_data/`.

## CLI entry point: `scripts/arg.py`

A single argparse-driven entry point with the following subcommands:

| Subcommand | Purpose |
|---|---|
| `benchmark list` | Enumerate available benchmarks |
| `benchmark auto-run` | Drive an LLM to generate optimized variants |
| `benchmark build` | Compile all variants of a benchmark |
| `benchmark check` | Run correctness validation against reference outputs |
| `benchmark correctness-run` | Capture reference outputs from baseline runs |
| `time_measurement` | Measure wall-clock time across variants |
| `benchmark cleanup` | Remove build artifacts |

All subcommands accept `--root .` (repo root), `--benchmarks` (comma list or `all`), and `--ex-version` / `--ex-versions`.

Each `tools/*.sh` script is a thin convenience wrapper around one of these subcommands; see the script source for the underlying call.

## Adding a new benchmark

1. Drop the benchmark sources into `EX<N>/<bench_name>/`. Provide `Makefile` and a top-level `Makefile.target`-style `<bench_name>.h` plus the existing kernel source.
2. Register metadata in `scripts/benchmark_catalog.py`:
   ```python
   "my_benchmark": {
       "suite": "polybench",
       "target_file": "EX1/my_benchmark/my_benchmark.c",
       "start_line": 42,
       "end_line": 88,
       "correctness_threshold": 1e-6,
       "datasets": ["mini", "small", "medium", "large"],
   },
   ```
3. Provide runtime arguments in `scripts/benchmark_args.py`.
4. Optional: write a per-suite checker in `scripts/checkers/` if the existing ones don't cover the new benchmark's output format.
5. Add a `runtime_strategy_<bench>.json` in `configs/` if the default warmup/measurement schedule is unsuitable.

## Adding a new LLM provider

LLM clients live in `scripts/llm/`. To add a new provider:

1. Subclass `LLMClient` from `scripts/llm/base.py` and implement `generate()`.
2. Register the new client in `scripts/llm/factory.py` keyed on `model-family`.
3. Read API keys from `os.environ` — never hard-code them. The setup helper `tools/setup_env.sh` documents the expected variables.

## Adding a new computational motif / suite

1. Place benchmarks under `EX<N>/<bench>/` and reuse the existing `common.mk` infrastructure where possible.
2. Provide a suite-specific `common_<suite>/` directory if the suite needs special build flags.
3. Add a checker module in `scripts/checkers/` and register it in `scripts/checkers/registry.py`.
4. Update `scripts/benchmark_catalog.py` with the new entries.

## Output layout

Pipeline outputs are written under `results/`, which is not tracked by Git:

```
results/
├── correctness/<bench>/<EX>/correctness.jsonl
├── make_results/<bench>/<EX>/build.log
├── missing_bins/<bench>/<EX>_missing_bins.json
└── time_measurements/<bench>/<EX>_time_measurements.jsonl
```

Program outputs captured by `benchmark correctness-run` are written to `EX<N>/<bench>/EX5_correctness/`. Analysis scripts write to `analysis_summaries/`, which is also not tracked.

## Build infrastructure

All Makefiles in `EX{1,2,3}/<bench>/` `include` a chain that ultimately resolves to the top-level `common.mk`. Path resolution uses `$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))` so the build is location-independent — moving the repo to a different parent path is fine.

The default optimization level is `-O3`. EX2 builds add `-fopenmp`; EX3 builds use `nvcc` targeting `sm_80` / `sm_90`.
