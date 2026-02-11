import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics.aggregation as metrics
from torchmetrics import MetricCollection
from util import *


class LitCustomModule(L.LightningModule):
    def __init__(self, model, name):
        super().__init__()
        self.name = name
        self.model = model
        self.loss_function = LossFunction()
        self.train_metrics = MetricCollection(
            {
                "mse": metrics.MeanMetric(),
                "mre": metrics.MeanMetric(),
            },
            prefix=name + "train_",
        )

        self.val_metrics = MetricCollection(
            {
                "mse": metrics.MeanMetric(),
                "mre": metrics.MeanMetric(),
            },
            prefix=name + "val_",
        )

        self.test_metrics = MetricCollection(
            {
                "mse": metrics.MeanMetric(),
                "mre": metrics.MeanMetric(),
            },
            prefix=name + "test_",
        )

    def configure_optimizers(self):
        print("Custom Module Configuring optimizers...")
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        # scheduler_config = {
        #     "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
        #         optimizer, factor=0.5, patience=20
        #     ),
        #     "monitor": self.name + "val_mse",
        #     "name": self.name + "_lr",
        #     "interval": "epoch",
        #     "frequency": 1,
        # }

        return [optimizer]

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()


class LitVMM(LitCustomModule):
    def __init__(self, model, name, loss_type, lr=1e-3):
        super().__init__(model, name)
        self.loss_type = loss_type
        self.lr = lr

    def loss(self, y, y_hat=None, **kwargs):
        if self.loss_type == "mse":
            loss = F.mse_loss(y_hat, y, reduction="mean")
            return loss
        if self.loss_type == "mae":
            loss = F.l1_loss(y_hat, y, reduction="mean")
            return loss
        if self.loss_type == "adjoint":
            loss = self.model.adjoint_loss(y, **kwargs)
        elif self.loss_type == "weighted_mse":
            den = torch.square(y) + 1e-6
            num = F.mse_loss(y_hat, y, reduction="none")
            loss = torch.mean(num / den)
        return loss

    def configure_optimizers(self):
        print("VMM Module Configuring optimizers...")
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # scheduler_config = {
        #     "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
        #         optimizer, factor=0.5, patience=20
        #     ),
        #     "monitor": self.name + "val_mse",
        #     "name": self.name + "_lr",
        #     "interval": "epoch",
        #     "frequency": 1,
        # }

        return [optimizer]

    def training_step(self, batch):
        x, y = batch
        y_hat, xi = self.model(*x)
        if self.loss_type == "adjoint":
            kwargs = {e: x[0], E: x[1], Y: x[2], n: x[3], edot_0: x[4]}
            loss = self.loss(y, xi=xi, **kwargs)
        else:
            loss = self.loss(y_hat, y)
        rel_error = self.loss_function.L2RelativeError(y_hat, y, reduction=None)
        self.train_metrics["mse"].update(loss)
        self.train_metrics["mre"].update(rel_error)
        return loss

    def validation_step(self, batch):
        with torch.set_grad_enabled(True):
            x, y = batch
            y_hat, _ = self.model(*x)
            loss = self.loss(y_hat, y)
            rel_error = self.loss_function.L2RelativeError(y_hat, y, reduction=None)
            self.val_metrics["mse"].update(loss)
            self.val_metrics["mre"].update(rel_error)
        return loss

    def test_step(self, batch):
        with torch.set_grad_enabled(True):
            x, y = batch
            y_hat, _ = self.model(*x)
            loss = self.loss(y_hat, y)
            rel_error = self.loss_function.L2RelativeError(y_hat, y, reduction=None)
            self.test_metrics["mse"].update(loss)
            self.test_metrics["mre"].update(rel_error)
        return loss


class LitAutoEncoder(LitCustomModule):
    def __init__(self, model, name):
        super().__init__(model, name)
        self.name = name

    def training_step(self, batch):
        x, _ = batch
        if self.name == "E_":
            x = x[2]
        elif self.name == "nu_":
            x = x[3]
        x_recon = self.model(x)
        loss = F.mse_loss(x_recon, x, reduction="mean")
        rel_error = self.loss_function.L2RelativeError(
            x_recon.unsqueeze(-1), x.unsqueeze(-1), reduction=None
        )
        self.train_metrics["mse"].update(loss)
        self.train_metrics["mre"].update(rel_error)
        return loss

    def validation_step(self, batch):
        x, _ = batch
        if self.name == "E_":
            x = x[2]
        elif self.name == "nu_":
            x = x[3]
        x_recon = self.model(x)
        loss = F.mse_loss(x_recon, x, reduction="mean")
        rel_error = self.loss_function.L2RelativeError(
            x_recon.unsqueeze(-1), x.unsqueeze(-1), reduction=None
        )
        self.val_metrics["mse"].update(loss)
        self.val_metrics["mre"].update(rel_error)
        return loss

    def test_step(self, batch):
        x, _ = batch
        if self.name == "E_":
            x = x[2]
        elif self.name == "nu_":
            x = x[3]
        x_recon = self.model(x)
        loss = F.mse_loss(x_recon, x, reduction="mean")
        rel_error = self.loss_function.L2RelativeError(
            x_recon.unsqueeze(-1), x.unsqueeze(-1), reduction=None
        )
        self.test_metrics["mse"].update(loss)
        self.test_metrics["mre"].update(rel_error)
        return loss
