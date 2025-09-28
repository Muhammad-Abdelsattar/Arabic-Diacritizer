from pathlib import Path
import numpy as np
import onnxruntime as ort
from .exceptions import ModelNotFound


class OnnxPredictor:
    def __init__(self, model_path: Path, use_gpu: bool = False):
        """
        Initializes the ONNX Runtime session.

        Args:
            model_path: Path to the .onnx model file.
            use_gpu: Whether to use the GPU for inference. Defaults to False.

        Raises:
            ModelNotFound: If the model file does not exist at the given path.
        """
        if not model_path.exists():
            raise ModelNotFound(f"ONNX model file not found at: {model_path}")

        providers = ["CPUExecutionProvider"]
        if use_gpu:
            # You can customize this list based on your target hardware
            providers.insert(0, "CUDAExecutionProvider")

        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.hints_name = self.session.get_inputs()[1].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, input_ids: np.ndarray, hints: np.ndarray) -> np.ndarray:
        """
        Runs inference on a batch of tokenized input IDs.

        Args:
            input_ids: A numpy array of shape (batch_size, sequence_length).
            hints: A numpy array of shape (batch_size, sequence_length).

        Returns:
            A numpy array of logits of shape (batch_size, sequence_length, num_classes).
        """
        ort_inputs = {self.input_name: input_ids, self.hints_name: hints}
        # The output is a list, we are interested in the first element
        logits = self.session.run([self.output_name], ort_inputs)[0]
        return logits
