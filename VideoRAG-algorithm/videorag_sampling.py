"""Qwen sampling defaults for the standalone VideoRAG MCP environment."""


def qwen35_sampling_params(model_name: str, *, enable_thinking: bool = True) -> dict:
    """Return Qwen3.5-4B's official general-task preset, or no override."""
    if "qwen3.5-4b" not in str(model_name).lower():
        return {}
    return {
        "temperature": 1.0 if enable_thinking else 0.7,
        "top_p": 0.95 if enable_thinking else 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    }
