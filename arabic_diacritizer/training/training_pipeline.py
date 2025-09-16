import torch
import logging
from typing import Dict, Any, Optional

import lightning as L
from . import builder

# Set up a logger for the pipeline
_LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
)


class TrainingPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def run(
        self,
        ckpt_path: Optional[str] = None,
        reset_optimizer_and_scheduler: bool = False,
    ):
        """
        Executes the training workflow. The config is assumed to be final.
        """
        if ckpt_path:
            _LOGGER.info(f"Resuming training from checkpoint: {ckpt_path}")
        else:
            _LOGGER.info("Starting a new training run...")

        L.seed_everything(self.config.get("seed", 42), workers=True)
        datamodule = builder.build_data_manager(self.config["data"])
        datamodule.setup(stage="fit")

        lightning_module = builder.build_modeling_orchestrator(
            modeling_config=self.config["modeling_config"],
            datamodule=datamodule,
            reset_optimizer_and_scheduler=reset_optimizer_and_scheduler,
        )

        # This ensures the checkpoint we save from this run can be resumed later.
        lightning_module.hparams["config"] = self.config

        trainer = builder.build_trainer(self.config["trainer"])

        trainer.fit(model=lightning_module, datamodule=datamodule, ckpt_path=ckpt_path)

        _LOGGER.info("Model training finished.")
        trainer.test(datamodule=datamodule, ckpt_path="best")
        _LOGGER.info("Model testing finished.")

    def evaluate(self, ckpt_path: str):
        """
        Executes an evaluation-only workflow on the test set.

        Args:
            ckpt_path: Path to the model checkpoint (.ckpt) to be evaluated.
        """
        _LOGGER.info(f"Starting the evaluation pipeline for checkpoint: {ckpt_path}...")

        L.seed_everything(self.config.get("seed", 42), workers=True)

        datamodule = builder.build_data_manager(self.config["data"])

        # Build the model structure to load the checkpoint weights into
        lightning_module = builder.build_modeling_orchestrator(
            modeling_config=self.config["modeling_config"], datamodule=datamodule
        )

        trainer = builder.build_trainer(self.config["trainer"])

        _LOGGER.info("Starting model evaluation (trainer.test)...")
        trainer.test(model=lightning_module, datamodule=datamodule, ckpt_path=ckpt_path)
        _LOGGER.info("Evaluation pipeline completed successfully.")
