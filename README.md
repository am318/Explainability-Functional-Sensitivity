# Repository Structure

```
.
├── Old Pruning Tests
│   ├── Pruning_Sensitivity_ViT_Tiny.py
│   └── sensitivity_pruning.py
├── Pruning_diagnostics
│   ├── Pruning_SNIP_ViT_Tiny.py
│   ├── Pruning_SYNFLOW_ViT_Tiny.py
│   ├── Pruning_Sensitivity_ViT_Tiny.py
│   ├── ViT_Model.py
│   ├── build_sparse_model.py
│   ├── dataset.py
│   ├── pruning_baselines.py
│   ├── sensitivity_metrics.py
│   ├── sensitivity_pruning.py
│   └── training_tools.py
└── requirements.txt
```

- `Pruning_diagnostics/` contains the main implementation of the zero-shot pruning methods and supporting modules.
- `Old Pruning Tests/` contains legacy implementations and experimental code retained for reference.
- `requirements.txt` lists the Python package dependencies.

# Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

# Running the Zero-Shot Pruning Algorithms

The main pruning scripts are located in the `Pruning_diagnostics` directory. Each script implements a different zero-shot pruning method for the ViT-Tiny model.

### SNIP

```bash
python Pruning_diagnostics/Pruning_SNIP_ViT_Tiny.py
```

### SynFlow

```bash
python Pruning_diagnostics/Pruning_SYNFLOW_ViT_Tiny.py
```

### Sensitivity

```bash
python Pruning_diagnostics/Pruning_Sensitivity_ViT_Tiny.py
```

Each script runs the corresponding zero-shot pruning algorithm (SNIP, SynFlow, or Sensitivity) using the shared utility modules in `Pruning_diagnostics`.
