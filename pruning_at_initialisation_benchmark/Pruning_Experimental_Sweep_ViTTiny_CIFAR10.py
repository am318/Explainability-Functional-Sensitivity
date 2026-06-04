"""Architecture-specific launcher for ViT-Tiny/CIFAR-10 pruning-at-initialization sweeps."""
import os
os.environ.setdefault("ARCH", "vit_tiny")
os.environ.setdefault("DATASET", "cifar10")
from Transformer_Prune_At_Init_Benchmark import main

if __name__ == "__main__":
    main()
