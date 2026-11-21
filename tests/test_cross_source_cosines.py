import importlib
from pathlib import Path
import sys
import types

import pytest
import yaml


torch = pytest.importorskip("torch")
ROOT = Path(__file__).parents[1]


@pytest.fixture
def trainer_cls(monkeypatch):
    class BaseTrainer:
        def __init__(self, **kwargs):
            self.logged = []

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model(**inputs)
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        def log(self, values):
            self.logged.append(values)

    for name in ["axolotl", "axolotl.core", "axolotl.core.trainers", "axolotl.core.trainers.base"]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["axolotl.core.trainers.base"].AxolotlTrainer = BaseTrainer
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    names = ["trainers", "trainers.tr_seam", "trainers.utils"]
    previous = {name: sys.modules.pop(name, None) for name in names}
    try:
        yield importlib.import_module("trainers.tr_seam").SEAMAttackTrainer
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[0.2, -0.4], [0.3, 0.1]], dtype=torch.float64))
        self.bias = torch.nn.Parameter(torch.tensor([0.1, -0.2], dtype=torch.float64))
        self.frozen = torch.nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, input_ids, attention_mask, labels):
        logits = input_ids @ self.weight + self.bias
        return types.SimpleNamespace(loss=torch.nn.functional.cross_entropy(logits, labels))


def make_batch(sources):
    n = len(sources)
    return {
        "input_ids": torch.tensor([[1.0 + i / 4, (-1.0) ** i] for i in range(n)], dtype=torch.float64),
        "attention_mask": torch.ones(n, 2, dtype=torch.long),
        "labels": torch.tensor([i % 2 for i in range(n)]),
        "source_id": torch.tensor(sources),
    }


def explicit_cross_cosine(model, inputs, epsilon):
    params = tuple(p for p in model.parameters() if p.requires_grad)
    gradients = []
    for i in range(len(inputs["labels"])):
        example = {key: inputs[key][i:i + 1] for key in ["input_ids", "attention_mask", "labels"]}
        gradients_i = torch.autograd.grad(model(**example).loss, params)
        flat = torch.cat([g.reshape(-1) for g in gradients_i])
        gradients.append(flat / (flat.norm() + epsilon))
    pairs = [
        torch.dot(gradients[i], gradients[j])
        for i, source_a in enumerate(inputs["source_id"])
        for j, source_b in enumerate(inputs["source_id"])
        if source_a == 0 and source_b == 1
    ]
    return torch.stack(pairs).mean()


@pytest.mark.parametrize("sources", [
    [0, 1] * 4,
    [0, 1, 1, 1],
    [1, 0, 1, 0, 0],
    [0, 2, 1, 2, 1],
    [0, 1],
])
def test_matches_explicit_cross_pairs(trainer_cls, sources):
    trainer = trainer_cls(source_id_forget=0, source_id_retain=1)
    model = TinyModel()
    inputs = make_batch(sources)
    expected = explicit_cross_cosine(model, inputs, trainer.eps_norm_div)
    actual = trainer.compute_loss_avg_of_cosims(model, inputs, compute_loss_only=True)
    torch.testing.assert_close(actual, expected)
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("return_outputs", [False, True])
def test_tracking_preserves_harmful_training_loss_and_gradients(trainer_cls, return_outputs):
    trainer = trainer_cls(source_id_forget=0, source_id_retain=1)
    model = TinyModel()
    reference = TinyModel()
    inputs = make_batch([0, 1] * 4)
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.full_like(p, 0.7)
    harmful_mask = inputs["source_id"] == 0
    benign_mask = inputs["source_id"] == 1
    fields = ["input_ids", "attention_mask", "labels"]
    expected = reference(**{key: inputs[key][harmful_mask] for key in fields}).loss
    benign_loss = reference(**{key: inputs[key][benign_mask] for key in fields}).loss
    result = trainer.compute_loss(model, inputs, return_outputs=return_outputs)
    loss = result[0] if return_outputs else result
    torch.testing.assert_close(loss, expected)
    for p in model.parameters():
        if p.requires_grad:
            torch.testing.assert_close(p.grad, torch.full_like(p, 0.7))
    assert len(trainer.logged) == 1
    assert trainer.logged[0]["loss_forget"] == pytest.approx(expected.item())
    assert trainer.logged[0]["loss_retain"] == pytest.approx(benign_loss.item())
    assert trainer.logged[0]["avg_of_cosims"] == pytest.approx(
        explicit_cross_cosine(reference, inputs, trainer.eps_norm_div).item()
    )
    model.zero_grad()
    loss.backward()
    expected.backward()
    for actual, expected_param in zip(model.parameters(), reference.parameters()):
        if actual.requires_grad:
            torch.testing.assert_close(actual.grad, expected_param.grad)


def test_requires_both_groups(trainer_cls):
    trainer = trainer_cls(source_id_forget=0, source_id_retain=1)
    with pytest.raises(ValueError, match="both sources"):
        trainer.compute_loss_avg_of_cosims(TinyModel(), make_batch([0, 0]), compute_loss_only=True)


def test_rejects_overlapping_groups(trainer_cls):
    with pytest.raises(ValueError, match="distinct sources"):
        trainer_cls(source_id_forget=0, source_id_retain=0)


def test_zero_gradients_are_finite(trainer_cls):
    class ZeroModel(TinyModel):
        def forward(self, input_ids, attention_mask, labels):
            return types.SimpleNamespace(loss=(self.weight.sum() + self.bias.sum()) * 0)

    trainer = trainer_cls(source_id_forget=0, source_id_retain=1)
    result = trainer.compute_loss_avg_of_cosims(ZeroModel(), make_batch([0, 1]), compute_loss_only=True)
    assert result.item() == 0


def test_tracking_configs_select_cross_source_trainer():
    for fmt in ["sp", "nl"]:
        path = ROOT / f"configs/beavertails/stages/1_attack/llama2-7b--attack-fft-seam-{fmt}.yml"
        config = yaml.safe_load(path.read_text())
        assert config["trainer_cls"] == "trainers.tr_seam.SEAMAttackTrainer"
        assert config["custom_trainer_args"] == {"source_id_forget": 0, "source_id_retain": 1}
        assert config["wandb_project"] == "tamper-resistance"
        expected_separator = "space" if fmt == "sp" else "newline"
        assert config["datasets"][0]["message_property_mappings"]["response_separator"] == expected_separator
