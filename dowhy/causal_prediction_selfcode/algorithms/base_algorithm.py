import pytorch_lightning as pl
import torch
from torch.nn import functional as F


class PredictionAlgorithm(pl.LightningModule):

    def __init__(self, model, optimizer, lr, weight_decay, betas, momentum):

        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.momentum = momentum

        # Check if the optimizer is currently supported
        if self.optimizer not in ["Adam", "SGD"]:
            error_msg = self.optimizer + " is not implemented currently. Try Adam or SGD."
            raise Exception(error_msg)
        
    def training_step(self, train_batch, batch_idx):
        raise NotImplementedError
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch[0], list):
            x = torch.cat([x for x, y, _ in batch])
            y = torch.cat([y for x, y, _ in batch])
        else:
            x = batch[0]
            y = batch[1]
        
        out = self.model(x)
        loss = F.cross_entropy(out, y)
        acc = (torch.argmax(out, dim=1) == y).float().mean()

        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch[0], list):
            x = torch.cat([x for x, y, _ in batch])
            y = torch.cat([y for x, y, _ in batch])
        else:
            x = batch[0]
            y = batch[1]
        
        out = self.model(x)
        loss = F.cross_entropy(out, y)
        acc = (torch.argmax(out, dim=1) == y).float().mean()

        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        if self.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=self.betas
            )
        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum
            )
        
        return optimizer