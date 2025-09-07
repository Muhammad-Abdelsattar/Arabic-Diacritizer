import torch
import lightning as L
from .optimizers import get_optimizer


class ModelingOrchestrator(L.LightningModule):
    """
    Lightning orchestrator for diacritization.
    Model- and loss- agnostic. Uses injected factories.
    """

    def __init__(
        self,
        model,
        loss_fn,
        optimizer_cfg: dict,
        metrics: dict = None,
        scheduler_cfg: dict = None,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.metrics = metrics or {}

        # Save hparams for reproducibility
        self.save_hyperparameters(ignore=["model", "loss_fn", "metrics"])

    def _forward_logits(self, inputs, lengths=None):
        return self.model(inputs, lengths)

    def _compute_loss(self, logits, labels):
        return self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

    def _compute_metrics(self, logits, labels):
        """Return dict of computed metrics."""
        out = {}
        for name, metric in self.metrics.items():
            out[name] = metric(logits, labels)
        return out

    def _log_metrics(self, metrics: dict, stage: str, loss=None):
        """Handle uniform logging for Lightning loggers (WandB, MLFlow, etc.)"""
        if loss is not None:
            self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
        for name, val in metrics.items():
            self.log(f"{stage}_{name}", val, prog_bar=True, on_epoch=True)

    def _step(self, batch, stage: str):
        inputs, labels, lengths = batch
        logits = self._forward_logits(inputs, lengths)
        loss = self._compute_loss(logits, labels)
        metrics = self._compute_metrics(logits, labels)
        self._log_metrics(metrics, stage, loss=loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def configure_optimizers(self):
        return get_optimizer(
            params=self.model.parameters(),
            config=self.optimizer_cfg,
            scheduler_cfg=self.scheduler_cfg,
        )
