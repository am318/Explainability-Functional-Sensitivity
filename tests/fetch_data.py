"""Fetch the datasets the experiments need.

  CIFAR-10    ~170 MB   https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz  (via torchvision)
  CIFAR-100   ~169 MB   https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz (via torchvision)
  WikiText-2   ~12 MB   https://raw.githubusercontent.com/pytorch/examples/main/
                        word_language_model/data/wikitext-2/{train,valid}.txt

The original Salesforce/metamind S3 bucket for WikiText-2 is gone (HTTP 301 with no
Location header). The corpus ships as plain text with the official PyTorch examples
repository, which suits the character-level model better than the HuggingFace parquet
release anyway -- no pyarrow or datasets dependency for a single file.
"""
import sys, urllib.request
from pathlib import Path
sys.path.insert(0, ".")
from torchvision import datasets

root = Path("data"); root.mkdir(exist_ok=True)

for cls in (datasets.CIFAR10, datasets.CIFAR100):
    for train in (True, False):
        cls(str(root), train=train, download=True)
    print(f"{cls.__name__} ok")

WIKI = ("https://raw.githubusercontent.com/pytorch/examples/main/"
        "word_language_model/data/wikitext-2")
dest = root / "wikitext-2"; dest.mkdir(exist_ok=True)
for name in ("train.txt", "valid.txt"):
    out = dest / name
    if out.exists() and out.stat().st_size > 0:
        print(f"wikitext-2/{name}: present ({out.stat().st_size/1e6:.1f} MB)")
        continue
    req = urllib.request.Request(f"{WIKI}/{name}", headers={"User-Agent": "Mozilla/5.0"})
    data = urllib.request.urlopen(req, timeout=180).read()
    out.write_bytes(data)
    print(f"wikitext-2/{name}: {len(data)/1e6:.1f} MB")

print("\nsizes:", {p.name: f"{sum(f.stat().st_size for f in p.rglob('*') if f.is_file())/1e6:.0f}MB"
                   for p in root.iterdir() if p.is_dir()})
