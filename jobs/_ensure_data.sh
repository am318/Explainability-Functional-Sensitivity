#!/usr/bin/env bash
# Preflight: make sure every dataset the queued configs need is present on disk BEFORE the
# GPU workers fan out. Every jobs/*/cfg_*.json ships with "download": false on purpose --
# we do not want N parallel workers racing to unpack the same tarball. So this script does
# the one-time, serial fetch instead, and is idempotent (a dataset already on disk is left
# untouched). Sourced by jobs/run_cluster_sweep.sh and jobs/run_positive_evidence_sweep.sh.
#
# Usage:  ensure_data <data_dir> [<py>]        # defaults: ./data  ./venv/bin/python|python
#
# What it can fetch:
#   cifar10  / cifar100  -- via torchvision (download=True)
#   text corpora whose path is data/wikitext-2/{train,valid}.txt -- fetched as plain
#     text from the PyTorch examples repo (the old metamind S3 zip is dead).
# Any other "text" corpus path that is missing is a hard error -- we have no source for it.

ensure_data() {
  local data_dir="${1:-./data}"
  local py="${2:-}"
  if [ -z "$py" ]; then
    py="./venv/bin/python"; [ -x "$py" ] || py="python"
  fi

  # Which datasets do the queued configs actually reference? Scan every emitted config.
  local need_c10=0 need_c100=0 datasets=""
  local -a cfgs=() text_files=()
  while IFS= read -r f; do cfgs+=("$f"); done < <(ls jobs/*/cfg_*.json 2>/dev/null || true)
  if [ "${#cfgs[@]}" -gt 0 ]; then
    datasets=$(grep -h '"dataset"' "${cfgs[@]}" 2>/dev/null \
               | sed -E 's/.*"dataset"[: ]+"([^"]*)".*/\1/' | sort -u)
    printf '%s\n' "$datasets" | grep -qx 'cifar10'  && need_c10=1  || true
    printf '%s\n' "$datasets" | grep -qx 'cifar100' && need_c100=1 || true
    # collect distinct non-empty text_file values
    while IFS= read -r tf; do
      [ -n "$tf" ] && text_files+=("$tf")
    done < <(grep -h '"text_file"' "${cfgs[@]}" 2>/dev/null \
             | sed -E 's/.*"text_file"[: ]+"([^"]*)".*/\1/' | sort -u)
  else
    # no configs emitted yet -- be permissive and grab the vision sets
    need_c10=1; need_c100=1
  fi

  echo "[ensure_data] data_dir=$data_dir  cifar10=$need_c10 cifar100=$need_c100 text=${#text_files[@]}"
  mkdir -p "$data_dir"

  # ---- CIFAR-10 / CIFAR-100 via torchvision -----------------------------------------
  if [ "$need_c10" = 1 ] && [ ! -e "$data_dir/cifar-10-batches-py/data_batch_1" ]; then
    echo "[ensure_data] downloading CIFAR-10 -> $data_dir"
    "$py" -c "from torchvision import datasets; d='$data_dir'; \
datasets.CIFAR10(d, train=True, download=True); datasets.CIFAR10(d, train=False, download=True)"
  fi
  if [ "$need_c100" = 1 ] && [ ! -e "$data_dir/cifar-100-python/train" ]; then
    echo "[ensure_data] downloading CIFAR-100 -> $data_dir"
    "$py" -c "from torchvision import datasets; d='$data_dir'; \
datasets.CIFAR100(d, train=True, download=True); datasets.CIFAR100(d, train=False, download=True)"
  fi

  # ---- text corpora ----------------------------------------------------------------
  local tf
  for tf in "${text_files[@]}"; do
    if [ -s "$tf" ]; then
      continue
    fi
    local dir base
    dir=$(dirname "$tf"); base=$(basename "$dir")
    if [ "$base" = "wikitext-2" ]; then
      echo "[ensure_data] fetching WikiText-2 -> $dir"
      mkdir -p "$dir"
      # The old Salesforce/metamind S3 zip (research.metamind.io) is gone -- it now
      # answers with a tiny XML error body, which is why unzip died with BadZipFile.
      # WikiText-2 ships as plain text with the official PyTorch examples repo; pull
      # the splits straight from there (same source as tests/fetch_data.py).
      local wiki="https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2"
      local split
      for split in train valid test; do
        [ -s "$dir/$split.txt" ] && continue
        if command -v curl >/dev/null 2>&1; then
          curl -fL --retry 3 -A "Mozilla/5.0" -o "$dir/$split.txt" "$wiki/$split.txt"
        else
          wget --header="User-Agent: Mozilla/5.0" -O "$dir/$split.txt" "$wiki/$split.txt"
        fi
        [ -s "$dir/$split.txt" ] && echo "  wrote $dir/$split.txt" || rm -f "$dir/$split.txt"
      done
      if [ ! -s "$dir/train.txt" ]; then
        echo "[ensure_data] ERROR: could not fetch WikiText-2 train split from $wiki" >&2
        return 1
      fi
      if [ ! -s "$dir/valid.txt" ]; then
        cp "$dir/train.txt" "$dir/valid.txt"      # be defensive; keep a val split on disk
      fi
    else
      echo "[ensure_data] ERROR: text corpus '$tf' is missing and no download source is known." >&2
      echo "[ensure_data]        Put the corpus file there by hand, then re-run." >&2
      return 1
    fi
  done

  echo "[ensure_data] ok"
}
