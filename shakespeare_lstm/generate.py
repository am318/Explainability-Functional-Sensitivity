"""Autoregressive sampling from a trained CharLSTM."""

from typing import Dict, List

import torch

from model import CharLSTM


@torch.no_grad()
def generate_text(
    model: CharLSTM,
    start_string: str,
    char2idx: Dict[str, int],
    idx2char: List[str],
    device: torch.device,
    num_generate: int = 1000,
    temperature: float = 1.0,
) -> str:
    model.eval()
    input_indices = torch.tensor(
        [[char2idx[c] for c in start_string]], dtype=torch.long, device=device
    )

    hidden = model.init_hidden(1, device)
    generated: List[str] = []

    # Prime the hidden state on the start string; the logits for the last
    # position are the prediction for the first generated character.
    logits, hidden = model(input_indices, hidden)
    next_logits = logits[:, -1:, :]

    for _ in range(num_generate):
        probs = torch.softmax(next_logits[0, -1] / temperature, dim=-1)
        predicted_id = torch.multinomial(probs, num_samples=1).item()
        generated.append(idx2char[predicted_id])
        next_input = torch.tensor([[predicted_id]], dtype=torch.long, device=device)
        next_logits, hidden = model(next_input, hidden)

    return start_string + "".join(generated)
