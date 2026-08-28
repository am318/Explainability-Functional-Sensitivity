"""E10 -- character-level LSTM, a fourth architecture family.

Neither convolutional nor attentional. If the granularity ladder --- layer ordering,
component ordering, parameter ordering --- behaves the same way in a recurrent model, the
result is about training dynamics rather than about one inductive bias. The LSTM is also
the setting the original stability observation came from, so measuring it under the same
protocol as the vision models puts the two on a common footing.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments._common import base, driver
from fsd import config as C


def configs():
    c = base("e10-lstm", "gpt", steps=20000, sens_samples=256)   # start from the text task
    c.model = C.ModelCfg(arch="lstm", width=256, depth=1, block_size=64)
    c.train.lr_schedule = "constant"
    c.train.batch_size = 32
    c.n_ckpts = 26
    c.track_criteria = False
    c.keep_scores = "none"
    return [c]


if __name__ == "__main__":
    raise SystemExit(driver("e10_lstm", configs, __doc__))
