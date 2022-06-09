# Logger (abstract)
# TensorBoard logger
# Wandb logger

from abc import ABC


class Logger(ABC):
    pass


class TensorBoardLogger(Logger):
    pass


class WandbLogger(Logger):
    pass
