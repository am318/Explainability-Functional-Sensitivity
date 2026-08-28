# WikiText-2 AWD-LSTM sensitivity/pruning experiment (cluster run)

Self-contained copy of the `wikitext_lstm/` experiment (functional-sensitivity
tracking + top-k pruning-vs-epoch study on a word-level AWD-LSTM), sized to
roughly match Merity et al. 2017's own AWD-LSTM config for WikiText-2
(3-layer LSTM, 1150 hidden units, 400-dim tied embedding, 750 epochs).
Nothing here depends on anything outside this folder -- the shared
`common/` sensitivity/pruning/plotting/rank-stability code and the raw
WikiText-2 text files are bundled directly.

## Usage

```bash
bash run_full_pipeline.sh
```

This creates its own `venv/`, installs `requirements.txt` (plain `pip
install torch` pulls a CUDA-enabled build automatically on Linux -- no
manual CUDA wheel selection needed), runs a quick smoke test, then the full
pipeline: `train.py` -> `rank_stability.py` -> `analyze_distributions.py`
-> `pruning_experiment.py` -> `plot_pruning_story.py`. Everything writes
into `outputs/<experiment name>/`.

Every setting is an overridable environment variable, e.g.:

```bash
EPOCHS=200 CHECKPOINT_INTERVAL=10 EXPERIMENT_NAME=my_run bash run_full_pipeline.sh
```

See the top of `run_full_pipeline.sh` for the full list of defaults and
`train.py`'s `Config` dataclass for everything that can be overridden.

If your cluster uses a scheduler (SLURM/PBS/etc.), wrap this script in
whatever submission script your setup requires -- it makes no scheduler
assumptions itself, just needs a shell and (ideally) a visible CUDA GPU.

## Bringing results back

Copy the whole `outputs/<experiment name>/` directory back -- it contains
checkpoints, `history.json`, `results.json`, and every plot. Nothing else
in this folder is needed for analysis afterwards.

## Known deviations from the paper

Documented in `run_full_pipeline.sh`'s header comment: plain Adam instead
of NT-ASGD, fixed (not randomized) BPTT length, no AR/TAR activation
regularization. The paper's own ablation (Table 4) shows dropping NT-ASGD
alone costs ~5 perplexity points on WikiText-2 -- real, but not the
dominant factor versus model size / training length.
