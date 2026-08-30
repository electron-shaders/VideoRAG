"""
mcp_server.py — VideoRAG MCP server (streamable-HTTP transport).

Runs in the 'videorag' conda environment. Exposes VideoRAG video QA tools
over the MCP protocol to lmms-eval.

Environment variables read at startup
--------------------------------------
    VIDEORAG_BASE_URL           VLM server, e.g. "http://127.0.0.1:9000/v1"
    VIDEORAG_EMBED_BASE_URL     Embedding server, e.g. "http://127.0.0.1:9001/v1"
    VIDEORAG_API_KEY            API key (default: "EMPTY")
    VIDEORAG_VLM_MODEL          VLM model name, e.g. "Qwen/Qwen3.5-4B"
    VIDEORAG_EMBED_MODEL        Embedding model name (default: "BAAI/bge-m3")
    VIDEORAG_EMBED_DIM          Embedding vector dimension (default: 1024)
    VIDEORAG_WORKING_DIR        Root dir for VideoRAG cache (default: ./.videorag_cache)
    VIDEORAG_SEGMENT_LENGTH     Video segment length in seconds (default: 30)
    VIDEORAG_RETRIEVAL_TOPK     Number of retrieved segments (default: 4)
    VIDEORAG_QUERY_MODE         Query mode: "videorag" or "videorag_multiple_choice"
    VIDEORAG_LOAD_CAPTION       Whether to load real caption model (default: "true")
    VIDEORAG_CUDA_DEVICE        CUDA visible device(s) for VideoRAG models
    VIDEORAG_HOST               Host to listen on (default: 127.0.0.1)
    VIDEORAG_PORT               Port to listen on (default: 9003)
    VIDEORAG_EMPTY_RESPONSE_MAX_ATTEMPTS
                                Attempts before an empty VLM response is fatal
                                (default: 3)
    VIDEORAG_STRUCTURED_ENABLE_THINKING
                                Enable Qwen thinking for structured RAG stages
                                such as extraction and summarization (default: false)
    VIDEORAG_FINAL_ENABLE_THINKING
                                Enable Qwen thinking for final answers (default: true)
    VIDEORAG_FINAL_THINKING_TOKEN_BUDGET
                                Maximum Qwen reasoning tokens for final answers
                                (default: 1024)
    VIDEORAG_MC_RESPONSE_MAX_ATTEMPTS
                                Attempts to obtain a parseable final MC answer
                                (default: 3)
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
import shutil
import sys
import threading

import numpy as np

# ---------------------------------------------------------------------------
# Configure CUDA device before PyTorch initialization
# ---------------------------------------------------------------------------
if os.environ.get("VIDEORAG_CUDA_DEVICE"):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(os.environ["VIDEORAG_CUDA_DEVICE"])

multiprocessing.set_start_method("spawn", force=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [VideoRAG MCP] %(message)s")
logger = logging.getLogger("videorag_mcp_server")

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
_base_url = os.environ.get("VIDEORAG_BASE_URL", "http://127.0.0.1:9000/v1")
_embed_base_url = os.environ.get("VIDEORAG_EMBED_BASE_URL", "") or _base_url
_api_key = os.environ.get("VIDEORAG_API_KEY", "EMPTY")
_vlm_model = os.environ.get("VIDEORAG_VLM_MODEL", "Qwen/Qwen3.5-4B")
_embed_model = os.environ.get("VIDEORAG_EMBED_MODEL", "BAAI/bge-m3")
_embed_dim = int(os.environ.get("VIDEORAG_EMBED_DIM", "1024"))
_working_dir = os.environ.get("VIDEORAG_WORKING_DIR", "./.videorag_cache")
_segment_length = int(os.environ.get("VIDEORAG_SEGMENT_LENGTH", "30"))
_retrieval_topk = int(os.environ.get("VIDEORAG_RETRIEVAL_TOPK", "4"))
_default_mode = os.environ.get("VIDEORAG_QUERY_MODE", "videorag_multiple_choice")
_load_caption = os.environ.get("VIDEORAG_LOAD_CAPTION", "true").lower() in ("true", "1", "yes")
_host = os.environ.get("VIDEORAG_HOST", "127.0.0.1")
_port = int(os.environ.get("VIDEORAG_PORT", "9003"))
_empty_response_max_attempts = max(
    1, int(os.environ.get("VIDEORAG_EMPTY_RESPONSE_MAX_ATTEMPTS", "3"))
)
_structured_enable_thinking = os.environ.get(
    "VIDEORAG_STRUCTURED_ENABLE_THINKING", "false"
).lower() in ("true", "1", "yes")
_final_enable_thinking = os.environ.get(
    "VIDEORAG_FINAL_ENABLE_THINKING", "true"
).lower() in ("true", "1", "yes")
_final_thinking_token_budget = int(
    os.environ.get("VIDEORAG_FINAL_THINKING_TOKEN_BUDGET", "1024")
)
if _final_thinking_token_budget < 0:
    raise ValueError("VIDEORAG_FINAL_THINKING_TOKEN_BUDGET must be non-negative")
_mc_response_max_attempts = max(
    1, int(os.environ.get("VIDEORAG_MC_RESPONSE_MAX_ATTEMPTS", "3"))
)

# Add VideoRAG repository root to sys.path
_repo_dir = os.path.dirname(os.path.abspath(__file__))
if _repo_dir not in sys.path:
    sys.path.insert(0, _repo_dir)

from mcp.server import MCPServer
from openai import AsyncOpenAI
from videorag._llm import LLMConfig
from videorag._utils import compute_args_hash
from videorag.base import BaseKVStorage, QueryParam
from videorag.videorag import VideoRAG

# ---------------------------------------------------------------------------
# OpenAI-compatible clients for vLLM
# ---------------------------------------------------------------------------
_vlm_client = AsyncOpenAI(api_key=_api_key, base_url=_base_url)
_embed_client = AsyncOpenAI(api_key=_api_key, base_url=_embed_base_url)
_pipeline_lock = threading.Lock()
_fatal_pipeline_error: str | None = None


async def _vllm_complete(
    model_name: str,
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list | None = None,
    **kwargs,
) -> str | None:
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    use_cache = kwargs.pop("use_cache", True)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None and use_cache:
        args_hash = compute_args_hash(model_name, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            cached_content = if_cache_return.get("return")
            if isinstance(cached_content, str) and cached_content.strip():
                return cached_content
            logger.warning("Ignoring an empty cached VLM response for model=%r", model_name)

    api_kwargs = {}
    for k in ("temperature", "max_tokens", "top_p", "stop"):
        if k in kwargs:
            api_kwargs[k] = kwargs[k]
    # VideoRAG's final answer calls have a system prompt; its structured helper
    # calls (extraction, rewriting, filtering, and summarization) do not. Keep
    # thinking out of the structured stages so their parsers receive an answer,
    # and put an explicit upper bound on final-answer reasoning.
    if "qwen3" in model_name.lower():
        is_final_answer = system_prompt is not None
        enable_thinking = (
            _final_enable_thinking
            if is_final_answer
            else _structured_enable_thinking
        )
        extra_body = {
            "chat_template_kwargs": {
                "enable_thinking": enable_thinking,
            }
        }
        if is_final_answer and enable_thinking:
            extra_body["thinking_token_budget"] = _final_thinking_token_budget
        api_kwargs["extra_body"] = extra_body

    content = None
    for attempt in range(1, _empty_response_max_attempts + 1):
        response = await _vlm_client.chat.completions.create(
            model=model_name, messages=messages, **api_kwargs
        )
        content = response.choices[0].message.content
        if isinstance(content, str) and content.strip():
            break

        finish_reason = getattr(response.choices[0], "finish_reason", None)
        logger.warning(
            "VLM returned empty content for model=%r (attempt %d/%d, finish_reason=%r)",
            model_name,
            attempt,
            _empty_response_max_attempts,
            finish_reason,
        )
        if attempt < _empty_response_max_attempts:
            await asyncio.sleep(min(0.5 * attempt, 2.0))
    else:
        logger.error(
            "VLM returned empty content after %d attempts for model=%r; "
            "returning no content to the caller's context-specific validator",
            _empty_response_max_attempts,
            model_name,
        )
        return None

    if hashing_kv is not None and use_cache:
        await hashing_kv.upsert(
            {args_hash: {"return": content, "model": model_name}}
        )
        await hashing_kv.index_done_callback()

    return content


async def _vllm_embedding(model_name: str, texts: list[str]) -> np.ndarray:
    response = await _embed_client.embeddings.create(
        model=model_name, input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])


# ---------------------------------------------------------------------------
# VideoRAG Singleton Management (Protected from spawn re-import)
# ---------------------------------------------------------------------------
_videorag: VideoRAG | None = None


def get_videorag() -> VideoRAG:
    global _videorag
    if _videorag is not None:
        return _videorag

    vllm_config = LLMConfig(
        embedding_func_raw=_vllm_embedding,
        embedding_model_name=_embed_model,
        embedding_dim=_embed_dim,
        embedding_max_token_size=8192,
        embedding_batch_num=32,
        embedding_func_max_async=16,
        query_better_than_threshold=0.2,
        best_model_func_raw=_vllm_complete,
        best_model_name=_vlm_model,
        best_model_max_token_size=32768,
        best_model_max_async=16,
        cheap_model_func_raw=_vllm_complete,
        cheap_model_name=_vlm_model,
        cheap_model_max_token_size=32768,
        cheap_model_max_async=16,
    )

    os.makedirs(_working_dir, exist_ok=True)
    _videorag = VideoRAG(
        llm=vllm_config,
        working_dir=_working_dir,
        video_segment_length=_segment_length,
        segment_retrieval_top_k=_retrieval_topk,
        multiple_choice_response_max_attempts=_mc_response_max_attempts,
    )

    if _load_caption:
        logger.info("Loading full MiniCPM caption model...")
        _videorag.load_caption_model(debug=False)
    else:
        _videorag.load_caption_model(debug=True)

    try:
        import decord
        decord.bridge.set_bridge("torch")
    except Exception:
        pass

    logger.info(
        f"VideoRAG initialized. working_dir='{_working_dir}', vlm='{_vlm_model}', embed='{_embed_model}'"
    )
    return _videorag


# ---------------------------------------------------------------------------
# FastMCP Server and Tool Definition
# ---------------------------------------------------------------------------
mcp = MCPServer("VideoRAG")


@mcp.tool()
async def run_videorag_query(
    video_path: str,
    question: str,
    mode: str = "videorag_multiple_choice",
) -> str:
    """
    Run VideoRAG retrieval-augmented video QA for a (video, question) pair.
    """
    query_mode = mode or _default_mode
    video_name = os.path.basename(video_path).split(".")[0]
    vr = get_videorag()

    def _configure_decord():
        try:
            import decord
            decord.bridge.set_bridge("torch")
        except Exception:
            pass

    def _insert_video_if_needed():
        if video_name not in vr.video_segments._data:
            cache_dir = os.path.join(vr.working_dir, "_cache", video_name)
            if os.path.exists(cache_dir):
                try:
                    shutil.rmtree(cache_dir)
                except Exception:
                    pass
            logger.info(f"Inserting video: {video_path}")
            vr.insert_video(video_path_list=[video_path])
        else:
            logger.info(f"Video already ingested: {video_name}")

    def _query_video():
        param = QueryParam(mode=query_mode)
        param.wo_reference = True

        response = vr.query(query=question, param=param)

        if isinstance(response, dict):
            answer = response.get("Answer")
        else:
            answer = response
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("VideoRAG returned an empty answer")
        return answer

    def _sync_process():
        global _fatal_pipeline_error
        # VideoRAG's in-memory graph and JSON/vector stores are shared mutable
        # state. Keep insertion and querying in one critical section so a query
        # cannot observe a partially inserted video.
        with _pipeline_lock:
            _configure_decord()
            if _fatal_pipeline_error is not None:
                raise RuntimeError(
                    "VideoRAG pipeline is halted after an earlier failure; restart "
                    f"the MCP server. Original failure: {_fatal_pipeline_error}"
                )
            try:
                _insert_video_if_needed()
            except Exception as exc:
                # Insertion mutates several shared stores, so an unexpected
                # insertion failure may leave in-memory state unsafe to reuse.
                _fatal_pipeline_error = f"{type(exc).__name__}: {exc}"
                raise
            # Query/answer-format failures do not mutate the index. Let the
            # evaluator retry or fail this sample without poisoning later ones.
            return _query_video()

    loop = asyncio.get_running_loop()
    try:
        answer = await loop.run_in_executor(None, _sync_process)
        return answer
    except Exception as exc:
        logger.exception("run_videorag_query failed")
        raise


if __name__ == "__main__":
    get_videorag()
    print(f"[videorag_mcp_server] Starting FastMCP SSE server on {_host}:{_port}", file=sys.stderr)
    mcp.run(transport="sse", host=_host, port=_port)
