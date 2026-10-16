"""Lightweight fatal-error contract shared by the MCP server and client."""

FATAL_ERROR_PREFIX = "[VIDEORAG_FATAL]"


class FatalVideoRAGError(RuntimeError):
    """The pipeline cannot continue safely; abort this evaluation, not a sample."""


def find_fatal_error(exc):
    """MCP/AnyIO context exits may wrap the error in nested ExceptionGroups."""
    if isinstance(exc, FatalVideoRAGError):
        return exc
    for nested in getattr(exc, "exceptions", ()):
        fatal = find_fatal_error(nested)
        if fatal is not None:
            return fatal
    return None
