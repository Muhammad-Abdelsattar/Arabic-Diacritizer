import logging
from typing import Dict, Any

import lightning as L
from . import builder

# Set up a logger for the pipeline
_LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')


class TrainingPipeline:
    """
    Orchestrates the entire training, validation, and testing process.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the pipeline with a configuration dictionary.
        """
        self.config = config

    def run(self):
        """
        Executes the main training and testing workflow.
        """
        _LOGGER.info("Starting the training pipeline...")

        # 1. Set the global seed for reproducibility
        L.seed_everything(self.config.get("seed", 42), workers=True)

        # 2. Build the DataManager
        datamodule = builder.build_data_manager(self.config["data"])

        # 3. Set up the DataManager to initialize the tokenizer
        _LOGGER.info("Setting up DataManager...")
        datamodule.setup(stage='fit')

        # 4. Build the ModelingOrchestrator
        lightning_module = builder.build_modeling_orchestrator(
            modeling_config=self.config["modeling_config"],
            datamodule=datamodule
        )

        # 5. Build the Trainer
        trainer = builder.build_trainer(self.config["trainer"])

        # 6. Start training
        _LOGGER.info("Starting model training (trainer.fit)...")
        trainer.fit(model=lightning_module, datamodule=datamodule)
        _LOGGER.info("Model training finished.")

        # 7. Run final test
        _LOGGER.info("Starting final model testing (trainer.test)...")
        trainer.test(datamodule=datamodule, ckpt_path="best")
        _LOGGER.info("Model testing finished.")
        _LOGGER.info("Training pipeline completed successfully.")

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
            modeling_config=self.config["modeling_config"],
            datamodule=datamodule
        )

        trainer = builder.build_trainer(self.config["trainer"])

        _LOGGER.info("Starting model evaluation (trainer.test)...")
        trainer.test(model=lightning_module, datamodule=datamodule, ckpt_path=ckpt_path)
        _LOGGER.info("Evaluation pipeline completed successfully.")

