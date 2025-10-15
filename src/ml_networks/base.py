import pytorch_lightning as pl

class BaseModule(pl.LightningModule):
    def freeze_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                param.requires_grad = False
    def freeze_biases(self):
        for name, param in self.named_parameters():
            if "bias" in name:
                param.requires_grad = False

    def unfreeze_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                param.requires_grad = True

    def unfreeze_biases(self):
        for name, param in self.named_parameters():
            if "bias" in name:
                param.requires_grad = True
