import math
from typing import Optional, Dict, Any
import lightning as L
from .optimizers import get_optimizer
from ..metrics import MetricsManager


class ModelingOrchestrator(L.LightningModule):
    """
    Lightning orchestrator for diacritization.
    Uses simple MetricsManager for all metric operations.
    """

    def __init__(
        self,
        model,
        loss_fn,
        optimizer_cfg: dict,
        metrics: dict = None,
        scheduler_cfg: dict = None,
        reset_optimizer_and_scheduler: bool = False,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.reset_optimizer_and_scheduler = reset_optimizer_and_scheduler

        self.metrics = metrics

        self.train_metrics = (
            MetricsManager(self.metrics, "train", self.device) if self.metrics else None
        )
        self.val_metrics = (
            MetricsManager(self.metrics, "val", self.device) if self.metrics else None
        )
        self.test_metrics = (
            MetricsManager(self.metrics, "test", self.device) if self.metrics else None
        )

        # Save hparams for reproducibility
        self.save_hyperparameters(ignore=["model", "loss_fn"])

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Hook called when a checkpoint is loaded.
        Used to selectively reset the optimizer and scheduler states.
        """
        if self.reset_optimizer_and_scheduler:
            keys_to_remove = {"optimizer_states", "lr_schedulers"}
            for key in keys_to_remove:
                if key in checkpoint:
                    del checkpoint[key]

            print(
                "[INFO] Hook `on_load_checkpoint`: Removed optimizer and scheduler states from the checkpoint."
            )

    def _forward_logits(self, inputs, lengths=None):
        return self.model(inputs, lengths)

    def _compute_loss(self, logits, labels):
        return self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

    def _compute_metrics(self, logits, labels, metrics):
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        metrics.update(logits, labels)

    def training_step(self, batch, batch_idx):
        inputs, labels, lengths = batch
        logits = self._forward_logits(inputs, lengths)
        loss = self._compute_loss(logits, labels)

        # Update training metrics if they exist
        # if self.train_metrics:
        #     self.train_metrics.update(logits, labels)
        #     self._compute_metrics(logits, labels, self.train_metrics)

        # Log loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels, lengths = batch
        logits = self._forward_logits(inputs, lengths)
        loss = self._compute_loss(logits, labels)

        # Update validation metrics if they exist
        if self.val_metrics:
            # self.val_metrics.update(logits, labels)
            self._compute_metrics(logits, labels, self.val_metrics)

        # Log loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        inputs, labels, lengths = batch
        logits = self._forward_logits(inputs, lengths)
        loss = self._compute_loss(logits, labels)

        # Update test metrics if they exist
        if self.test_metrics:
            # self.test_metrics.update(logits, labels)
            self._compute_metrics(logits, labels, self.test_metrics)

        # Log loss
        self.log("test_loss", loss, on_step=False, on_epoch=True)

    def on_fit_start(self) -> None:
        if self.metrics:
            for metric in self.metrics.values():
                metric.to(self.device)
        return super().on_fit_start()

    def on_train_epoch_end(self):
        # Compute and log all training metrics
        if self.train_metrics:
            train_metrics = self.train_metrics.get_log_dict()
            self.log_dict(train_metrics, on_epoch=True)
            self.train_metrics.reset()

    def on_validation_epoch_end(self):
        # Compute and log all validation metrics
        if self.val_metrics:
            val_metrics = self.val_metrics.get_log_dict()
            self.log_dict(val_metrics, on_epoch=True)
            self.val_metrics.reset()

    def on_test_epoch_end(self):
        # Compute and log all test metrics
        if self.test_metrics:
            test_metrics = self.test_metrics.get_log_dict()
            self.log_dict(test_metrics, on_epoch=True)
            self.test_metrics.reset()

    def configure_optimizers(self):
        """
        Builds the optimizer and a dynamically configured scheduler.
        """
        scheduler_cfg = self.scheduler_cfg.copy() if self.scheduler_cfg else None

        # Dynamically calculate training steps if a warmup scheduler is used
        if scheduler_cfg and scheduler_cfg.get("name") == "linear_warmup":

            # It correctly handles multi-GPU, accelerators, and gradient accumulation.
            total_training_steps = self.trainer.estimated_stepping_batches

            if total_training_steps == float("inf"):
                raise ValueError(
                    "Cannot determine total training steps. Please set `max_epochs` or `max_steps` in the trainer config."
                )

            warmup_ratio = scheduler_cfg.pop("warmup_ratio", 0.1)
            num_warmup_steps = math.ceil(total_training_steps * warmup_ratio)

            # Add the dynamically calculated values to the scheduler config
            scheduler_cfg["num_training_steps"] = total_training_steps
            scheduler_cfg["num_warmup_steps"] = num_warmup_steps

            print("[INFO] Scheduler configured dynamically:")
            print(f"  - Total training steps: {total_training_steps}")
            print(f"  - Warmup steps ({warmup_ratio*100}%): {num_warmup_steps}")

        return get_optimizer(
            params=self.model.parameters(),
            config=self.optimizer_cfg,
            scheduler_cfg=scheduler_cfg,
        )
