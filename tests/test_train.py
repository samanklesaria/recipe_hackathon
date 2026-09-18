import numpy as np
import pytest

# unsloth only imports where there is a GPU stack; skip on the laptop.
train = pytest.importorskip("train")
import torch


def test_loss_is_bce_on_the_raw_dot_product():
    a = torch.tensor([[3.0, 0.0]])
    b = torch.tensor([[0.5, 9.9]])  # dot = 1.5, norms deliberately unequal
    loss = train.DotBCELoss(lambda f: {"sentence_embedding": f})
    got = loss([a, b], torch.tensor([0.95])).item()
    want = -(0.95 * np.log(1 / (1 + np.exp(-1.5)))
             + 0.05 * np.log(1 - 1 / (1 + np.exp(-1.5))))
    assert got == pytest.approx(want, rel=1e-5)


def test_label_words_map_to_their_probabilities():
    # Guards the argmax indexing in load_training_data against LABELS reorder.
    labels = np.array(["never", "always", "sometimes", "usually"])
    p = train.be.P[np.argmax(labels[:, None] == train.be.LABELS, axis=-1)]
    assert p.tolist() == [0.05, 0.95, 0.20, 0.80]
