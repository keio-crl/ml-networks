import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import \
    RichProgressBarTheme
from schedulefree import RAdamScheduleFree, SGDScheduleFree
from pytorch_optimizer import ScheduleFreeSGD, ScheduleFreeRAdam, ScheduleFreeAdamW
from typing import Union, Iterable

ScheduleFreeOptimizers = Union[RAdamScheduleFree, SGDScheduleFree, 
                               ScheduleFreeSGD, ScheduleFreeRAdam, 
                               ScheduleFreeAdamW]

class ProgressBarCallback(RichProgressBar):
    """
    Make the progress bar richer.

    References
    ----------
    * https://qiita.com/akihironitta/items/edfd6b29dfb67b17fb00
    """

    def __init__(self) -> None:
        """Rich progress bar with custom theme."""
        theme = RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
        )
        super().__init__(theme=theme)

class SwitchOptimizer(pl.Callback):
    def __init__(self):
        super().__init__()

    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer,
        opt_idx: int = 0,
        ):

        optimizer = pl_module.optimizers()
        if isinstance(optimizer, ScheduleFreeOptimizers):
            optimizer.train()
        elif isinstance(optimizer, Iterable):
            for opt in optimizer:
                if isinstance(opt, ScheduleFreeOptimizers):
                    opt.train()

    def on_train_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        optimizer = pl_module.optimizers()
        if isinstance(optimizer, ScheduleFreeOptimizers):
            optimizer.train()
        elif isinstance(optimizer, Iterable):
            for opt in optimizer:
                if isinstance(opt, ScheduleFreeOptimizers):
                    opt.train()

    def on_validation_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        optimizer = pl_module.optimizers()
        if isinstance(optimizer, ScheduleFreeOptimizers):
            optimizer.eval()
        elif isinstance(optimizer, Iterable):
            for opt in optimizer:
                if isinstance(opt, ScheduleFreeOptimizers):
                    opt.eval()

    def on_validation_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule
    ) -> None:
        optimizer = pl_module.optimizers()
        if isinstance(optimizer, ScheduleFreeOptimizers):
            optimizer.eval()
        elif isinstance(optimizer, Iterable):
            for opt in optimizer:
                if isinstance(opt, ScheduleFreeOptimizers):
                    opt.eval()

      
