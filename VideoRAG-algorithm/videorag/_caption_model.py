"""Local-only MiniCPM loading, including its otherwise lazy processor."""

import logging
import os
from pathlib import Path

_MODEL_ID = "openbmb/MiniCPM-V-2_6-int4"
_log = logging.getLogger(__name__)


def resolve_caption_snapshot(model_path=None):
    """Resolve existing files without an authenticated Hub metadata request."""
    model_path = model_path or os.environ.get("VIDEORAG_CAPTION_MODEL_PATH")
    if model_path:
        snapshot = Path(model_path).expanduser().absolute()
        if not snapshot.is_dir():
            raise FileNotFoundError(f"MiniCPM local model directory does not exist: {snapshot}")
    else:
        from huggingface_hub import snapshot_download

        try:
            snapshot = Path(snapshot_download(_MODEL_ID, local_files_only=True))
        except Exception as exc:
            raise RuntimeError(
                f"MiniCPM is not available in the local Hugging Face cache: {_MODEL_ID}. " "Populate that cache before evaluation, or set VIDEORAG_CAPTION_MODEL_PATH " "to an existing complete snapshot directory. No download was attempted."
            ) from exc
    return str(snapshot)


def load_caption_model(model_path=None):
    """Use one snapshot for weights, tokenizer, image processor and custom code."""
    from transformers import AutoModel, AutoProcessor

    snapshot = resolve_caption_snapshot(model_path)
    _log.info("Loading MiniCPM from local snapshot: %s", snapshot)
    # In this Transformers version the repo-ID/offline path does not tolerate
    # an absent optional processor_config.json. The actual directory does, and
    # falls back to this model's preprocessor_config.json. Validate on CPU first.
    processor = AutoProcessor.from_pretrained(snapshot, trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(snapshot, trust_remote_code=True, local_files_only=True)
    # MiniCPM.chat() would otherwise call AutoProcessor.from_pretrained again
    # without local_files_only, potentially contacting the Hub days into a run.
    model.processor = processor
    model.eval()
    return model, processor.tokenizer
