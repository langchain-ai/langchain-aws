class InferenceEvaluatorError(Exception):
    """
    Exception raised if PII entities are detected.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)
