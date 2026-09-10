from trace_models import FailureLayer


def set_failure_layer(error: Exception, layer: FailureLayer) -> None:
    try:
        error.failure_layer = layer
    except (AttributeError, TypeError):
        pass


__all__ = ["set_failure_layer"]
