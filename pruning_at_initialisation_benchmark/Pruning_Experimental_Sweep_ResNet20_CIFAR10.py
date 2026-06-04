"""Architecture-specific launcher for ResNet-20/CIFAR-10 pruning-at-initialization sweeps."""
import os
os.environ.setdefault("ARCH", "resnet20")
os.environ.setdefault("DATASET", "cifar10")
from Transformer_Prune_At_Init_Benchmark import main

if __name__ == "__main__":
    main()
