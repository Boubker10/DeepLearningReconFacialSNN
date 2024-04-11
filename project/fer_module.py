import torch
import pytorch_lightning as pl
from torch.nn import functional as F
import torchmetrics
from project.models.snn_models import SNNModule
from project.models.models import CNNModule

class FerModule(pl.LightningModule):
    def __init__(self, learning_rate: float, timesteps: int, n_classes: int, epochs: int, mode="snn", **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.epochs = epochs
        
        if "snn" in mode:
            self.model = SNNModule(
                2, timesteps=timesteps, n_classes=n_classes
            )
        else:
            self.model = CNNModule(2, timesteps, n_classes)

    def forward(self, x):
        # (T, B, C, H, W) --> (B, num_classes)
        x = self.model(x)

        return x
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        y_hat_prob = torch.softmax(y_hat, dim=1)
        acc = torchmetrics.functional.accuracy(y_hat_prob, y)

        self.log("train_loss", loss, on_epoch=True, prog_bar=False)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y.long())
        y_hat_prob = torch.softmax(y_hat, dim=1)
        acc = torchmetrics.functional.accuracy(y_hat_prob, y)

        # logs
        self.log("val_loss", loss, on_epoch=True, prog_bar=False)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        y_hat_prob = torch.softmax(y_hat, dim=1)
        acc = torchmetrics.functional.accuracy(y_hat_prob, y)

        self.log("test_loss", loss, on_epoch=True, prog_bar=False)
        self.log("test_acc", acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)

        return [optimizer], [scheduler]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)  # better perf
