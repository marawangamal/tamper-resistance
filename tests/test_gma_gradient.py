from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from trainers.tr_gma import GMABaseTrackTrainer


class ToyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 4, bias=False, dtype=torch.float64)

    def forward(self, input_ids, attention_mask=None, labels=None):
        logits = self.projection(input_ids)
        return SimpleNamespace(loss=F.cross_entropy(logits, labels))


def exact_average_pairwise_gradient_cosine(model, batch):
    gradients = []
    parameters = tuple(model.parameters())
    for index in range(len(batch["labels"])):
        output = model(
            input_ids=batch["input_ids"][index : index + 1],
            labels=batch["labels"][index : index + 1],
        )
        gradient = torch.autograd.grad(output.loss, parameters, create_graph=True)
        flattened = torch.cat([value.flatten() for value in gradient])
        gradients.append(flattened / flattened.norm())

    pairwise_cosines = [
        torch.dot(gradients[i], gradients[j])
        for i in range(len(gradients))
        for j in range(len(gradients))
        if i != j
    ]
    return torch.stack(pairwise_cosines).mean()


def test_finite_difference_gradient_matches_exact_gradient():
    torch.manual_seed(7)
    model = ToyClassifier()
    batch = {
        "input_ids": torch.randn(3, 3, dtype=torch.float64),
        "attention_mask": torch.ones(3, 1, dtype=torch.long),
        "labels": torch.tensor([0, 1, 3]),
    }

    exact_loss = exact_average_pairwise_gradient_cosine(model, batch)
    exact_gradient = torch.autograd.grad(exact_loss, tuple(model.parameters()))

    trainer = object.__new__(GMABaseTrackTrainer)
    trainer.eps_finite_diff = 1e-5
    trainer.eps_norm_div = 1e-12
    approximate_loss = trainer.compute_loss_avg_of_cosims(model, batch)
    approximate_gradient = torch.autograd.grad(
        approximate_loss, tuple(model.parameters())
    )

    torch.testing.assert_close(approximate_loss, exact_loss, rtol=1e-10, atol=1e-10)
    for approximate, exact in zip(approximate_gradient, exact_gradient):
        torch.testing.assert_close(approximate, exact, rtol=2e-4, atol=2e-5)
