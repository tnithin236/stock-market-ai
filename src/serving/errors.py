"""Domain errors the API maps to HTTP status codes."""


class ModelNotTrainedError(Exception):
    """No saved model for this ticker (-> 404)."""


class DataUnavailableError(Exception):
    """Market data could not be fetched (-> 502)."""


class InsufficientHistoryError(Exception):
    """Not enough history to compute every indicator (-> 422)."""


class TrainingInProgressError(Exception):
    """A training job for this ticker is already queued/running (-> 409)."""
