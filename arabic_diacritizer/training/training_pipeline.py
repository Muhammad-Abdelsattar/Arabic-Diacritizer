import torch
import logging
from typing import Dict, Any, Optional

import lightning as L
from . import builder

_LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
)


class TrainingPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def run(self, ckpt_path: Optional[str] = None):
        """
        Executes the standard training workflow.
        - If ckpt_path is None, it starts a new run from scratch.
        - If ckpt_path is provided, it RESUMES the training state.
        """
        if ckpt_path:
            _LOGGER.info(f"Preparing to RESUME training from checkpoint: {ckpt_path}")
        else:
            _LOGGER.info("Preparing a NEW training run...")

        L.seed_everything(self.config.get("seed", 42), workers=True)
        datamodule = builder.build_data_manager(self.config["data"])

        lightning_module = builder.build_modeling_orchestrator(
            modeling_config=self.config["modeling_config"],
            datamodule=datamodule,
        )
        lightning_module.hparams["config"] = self.config

        trainer = builder.build_trainer(self.config["trainer"])

        # The trainer's `fit` method natively handles resuming when ckpt_path is provided.
        trainer.fit(model=lightning_module, datamodule=datamodule, ckpt_path=ckpt_path)

        _LOGGER.info("Model training finished.")
        trainer.test(datamodule=datamodule, ckpt_path="best")
        _LOGGER.info("Model testing finished.")

    def finetune(self, ckpt_path: str):
        """
        Executes a fine-tuning workflow.
        - Loads ONLY the model weights from the checkpoint.
        - Starts a fresh training run from epoch 0 with a new optimizer/scheduler.
        """
        _LOGGER.info(f"Preparing to FINETUNE from weights at: {ckpt_path}")

        L.seed_everything(self.config.get("seed", 42), workers=True)
        datamodule = builder.build_data_manager(self.config["data"])

        # This builder function is perfect for this: it loads the weights into the model structure.
        lightning_module = builder.load_lightning_module_from_checkpoint(
            ckpt_path=ckpt_path,
        )

        # Ensure the current config is attached for logging and saving.
        lightning_module.hparams["config"] = self.config

        trainer = builder.build_trainer(self.config["trainer"])

        # We call `fit` WITHOUT ckpt_path to start a fresh run.
        trainer.fit(model=lightning_module, datamodule=datamodule)

        _LOGGER.info("Model fine-tuning finished.")
        trainer.test(datamodule=datamodule, ckpt_path="best")
        _LOGGER.info("Model testing finished.")

    def evaluate(self, ckpt_path: str):
        """
        Executes an evaluation-only workflow on the test set.
        """
        _LOGGER.info(f"Starting the evaluation pipeline for checkpoint: {ckpt_path}...")

        L.seed_everything(self.config.get("seed", 42), workers=True)

        datamodule = builder.build_data_manager(self.config["data"])
        lightning_module = builder.build_modeling_orchestrator(
            modeling_config=self.config["modeling_config"], datamodule=datamodule
        )
        trainer = builder.build_trainer(self.config["trainer"])

        _LOGGER.info("Starting model evaluation (trainer.test)...")
        trainer.test(model=lightning_module, datamodule=datamodule, ckpt_path=ckpt_path)
        _LOGGER.info("Evaluation pipeline completed successfully.")
