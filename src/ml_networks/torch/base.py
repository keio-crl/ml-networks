"""ベースモジュール."""

import pytorch_lightning as pl


class BaseModule(pl.LightningModule):
    """Base module for PyTorch Lightning."""

    def freeze_weights(self) -> None:
        """Freeze all weight parameters."""
        for name, param in self.named_parameters():
            if "weight" in name:
                param.requires_grad = False

    def freeze_biases(self) -> None:
        """Freeze all bias parameters."""
        for name, param in self.named_parameters():
            if "bias" in name:
                param.requires_grad = False

    def unfreeze_weights(self) -> None:
        """Unfreeze all weight parameters."""
        for name, param in self.named_parameters():
            if "weight" in name:
                param.requires_grad = True

    def unfreeze_biases(self) -> None:
        """Unfreeze all bias parameters."""
        for name, param in self.named_parameters():
            if "bias" in name:
                param.requires_grad = True
