"""Qi et al.'s epoch-wise learning-rate decay for the AdvBench attack proxy."""

from axolotl.core.trainers.base import AxolotlTrainer
from torch.optim.lr_scheduler import LambdaLR


class QiStepTrainer(AxolotlTrainer):
    def __init__(self, *args, steps_per_epoch: int, decay: float = 0.85, **kwargs):
        if steps_per_epoch < 1 or not 0 < decay <= 1:
            raise ValueError("steps_per_epoch must be positive and decay must be in (0, 1]")
        self.steps_per_epoch = steps_per_epoch
        self.decay = decay
        super().__init__(*args, **kwargs)

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.lr_scheduler is None:
            optimizer = optimizer or self.optimizer
            self.lr_scheduler = LambdaLR(
                optimizer,
                lr_lambda=lambda step: self.decay ** (step // self.steps_per_epoch),
            )
        return self.lr_scheduler
