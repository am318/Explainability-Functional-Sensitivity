"""Architecture-specific launcher for NanoGPT-style character-LM pruning-at-initialization sweeps."""
import os
os.environ.setdefault("ARCH", "nanogpt")
os.environ.setdefault("DATASET", "text")
from Transformer_Prune_At_Init_Benchmark import main

if __name__ == "__main__":
    main()
