from abc import ABC, abstractmethod

from experiment import Experiment

# NOTE: consider doing all config in init and just calling Trainer.train() in main,
# this would allow Hydra to fully switch in and out trainers


class Trainer(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def train(self):
        raise NotImplementedError()

    @abstractmethod
    def test(self):
        raise NotImplementedError()

    @abstractmethod
    def predict(self):
        raise NotImplementedError()


class StandardTrainer(Trainer):
    """Implementation of a standard training loop with logging and saving."""

    def __init__(
        self,
        experiment: Experiment,
        data,
        num_epochs: int,
        logger=None,
        log_every_n_steps: int = 100,
        ckpt_dir=None,
    ):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def predict(self):
        pass
