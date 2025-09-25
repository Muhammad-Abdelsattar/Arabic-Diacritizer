class DiacritizerException(Exception):
    """Base exception for all errors raised by the diacritizer package."""
    pass

class ModelNotFound(DiacritizerException):
    """Raised when the model files (ONNX, vocab, etc.) cannot be found."""
    pass

class InvalidInputError(DiacritizerException):
    """Raised when the input text provided to the diacritizer is invalid."""
    pass
