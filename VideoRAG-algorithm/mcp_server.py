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
    VIDEORAG_CAPTION_MODEL_PATH Optional existing MiniCPM snapshot directory;
                                otherwise use the local Hugging Face cache only
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
                                outside raw-response evaluation mode (default: 3)
    VIDEORAG_TRUNCATION_MAX_TOKENS
                                Hard output-token cap for length retries (default:
                                16384). Budgets double on truncation; calls without
                                an explicit budget start at min(4096, cap).
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
_truncation_max_tokens = int(os.environ.get("VIDEORAG_TRUNCATION_MAX_TOKENS", "16384"))
if _truncation_max_tokens < 1:
    raise ValueError("VIDEORAG_TRUNCATION_MAX_TOKENS must be a positive integer")

# Add VideoRAG repository root to sys.path
_repo_dir = os.path.dirname(os.path.abspath(__file__))
if _repo_dir not in sys.path:
    sys.path.insert(0, _repo_dir)

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from openai import AsyncOpenAI, BadRequestError
from videorag_errors import FATAL_ERROR_PREFIX
from videorag_sampling import qwen35_sampling_params
from videorag._llm import LLMConfig
from videorag._utils import always_get_an_event_loop, compute_args_hash
from videorag._recovery import UnsafeVideoRAGState, drain_pipeline_tasks
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

    api_kwargs = {}
    for k in ("temperature", "max_tokens", "top_p", "stop"):
        if k in kwargs:
            api_kwargs[k] = kwargs[k]
    if "max_new_tokens" in kwargs:
        api_kwargs["max_tokens"] = int(kwargs["max_new_tokens"])
    initial_budget = int(api_kwargs.get("max_tokens", 4096))
    if initial_budget < 1:
        raise ValueError("max_new_tokens/max_tokens must be a positive integer")
    api_kwargs["max_tokens"] = min(initial_budget, _truncation_max_tokens)
    if initial_budget > _truncation_max_tokens:
        logger.warning(
            "Clamping VLM max_tokens=%d to VIDEORAG_TRUNCATION_MAX_TOKENS=%d",
            initial_budget, _truncation_max_tokens,
        )
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
        sampling = qwen35_sampling_params(model_name, enable_thinking=enable_thinking)
        extra_body = dict(kwargs.get("extra_body") or {}) if sampling else {}
        template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
        template_kwargs["enable_thinking"] = enable_thinking
        extra_body["chat_template_kwargs"] = template_kwargs
        # Task defaults (including VideoMME's greedy decoding) and extra_body
        # must not override the official preset at the actual API boundary.
        for name, value in sampling.items():
            extra_body.pop(name, None)
            if name in ("temperature", "top_p", "presence_penalty"):
                api_kwargs[name] = value
            else:
                extra_body[name] = value
        if is_final_answer and enable_thinking:
            extra_body["thinking_token_budget"] = kwargs.get("thinking_token_budget", _final_thinking_token_budget)
        api_kwargs["extra_body"] = extra_body

    if hashing_kv is not None and use_cache:
        # Qwen's old cache keys omitted sampling and thinking mode. Include the
        # effective request so a previous greedy answer cannot bypass this fix.
        args_hash = (
            compute_args_hash(model_name, messages, api_kwargs)
            if qwen35_sampling_params(model_name)
            else compute_args_hash(model_name, messages)
        )
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            cached_content = if_cache_return.get("return")
            if isinstance(cached_content, str) and cached_content.strip():
                return cached_content
            logger.warning("Ignoring an empty cached VLM response for model=%r", model_name)

    content = None
    empty_attempts = 0
    retrying_truncation = False
    while True:
        try:
            response = await _vlm_client.chat.completions.create(
                model=model_name, messages=messages, **api_kwargs
            )
        except BadRequestError as exc:
            # A larger output budget can exceed the model's remaining context
            # window. Keep the previous answer instead of losing it to a retry.
            context_error = any(
                marker in str(exc).lower()
                for marker in ("maximum context length", "max_model_len", "context window")
            )
            if not retrying_truncation or not context_error:
                raise
            logger.warning("Truncation retry exceeds model context; returning last response: %s", exc)
            return content
        content = response.choices[0].message.content
        finish_reason = getattr(response.choices[0], "finish_reason", None)
        if finish_reason == "length":
            budget = api_kwargs["max_tokens"]
            next_budget = min(budget * 2, _truncation_max_tokens)
            usage = getattr(response, "usage", None)
            logger.warning(
                "VLM response truncated (finish_reason=length, max_tokens=%d, completion_tokens=%s); %s",
                budget, getattr(usage, "completion_tokens", None),
                f"retrying with max_tokens={next_budget}"
                if next_budget > budget else "token cap reached; returning last response without caching",
            )
            if next_budget <= budget:
                return content
            api_kwargs["max_tokens"] = next_budget
            retrying_truncation = True
            continue

        retrying_truncation = False
        if isinstance(content, str) and content.strip():
            break

        empty_attempts += 1
        logger.warning(
            "VLM returned empty content for model=%r (attempt %d/%d, finish_reason=%r)",
            model_name,
            empty_attempts,
            _empty_response_max_attempts,
            finish_reason,
        )
        if empty_attempts >= _empty_response_max_attempts:
            logger.error(
                "VLM returned empty content after %d attempts for model=%r; "
                "returning no content to the caller's context-specific validator",
                _empty_response_max_attempts,
                model_name,
            )
            return None
        await asyncio.sleep(min(0.5 * empty_attempts, 2.0))

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
    # Publish only a completely initialized instance. A failed caption load
    # must not leave an apparently ready singleton for a subsequent request.
    _videorag = _create_videorag()
    return _videorag


def _create_videorag(caption_source=None) -> VideoRAG:

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
    vr = VideoRAG(
        llm=vllm_config,
        working_dir=_working_dir,
        video_segment_length=_segment_length,
        segment_retrieval_top_k=_retrieval_topk,
        multiple_choice_response_max_attempts=_mc_response_max_attempts,
    )

    if caption_source is not None:
        # Reload indexes and fresh LLM wrappers, not a second GPU copy of MiniCPM.
        vr.caption_model = caption_source.caption_model
        vr.caption_tokenizer = caption_source.caption_tokenizer
        vr.caption_model_path = caption_source.caption_model_path
    elif _load_caption:
        logger.info("Loading full MiniCPM caption model...")
        vr.load_caption_model(debug=False)
    else:
        vr.load_caption_model(debug=True)

    try:
        import decord
        decord.bridge.set_bridge("torch")
    except Exception:
        pass

    logger.info(
        f"VideoRAG initialized. working_dir='{_working_dir}', vlm='{_vlm_model}', embed='{_embed_model}'"
    )
    return vr


def _recover_failed_insertion(vr, original_error):
    """Called under _pipeline_lock; never clear the error and reuse dirty state."""
    global _videorag, _fatal_pipeline_error
    try:
        drain_pipeline_tasks(always_get_an_event_loop())
        if isinstance(original_error, UnsafeVideoRAGState):
            raise original_error
        # Default local stores persist index changes only at _insert_done().
        # That method flags partial commits as unsafe. Before commit, discard
        # all in-memory mutations and reopen the last successfully saved indexes.
        restored = _create_videorag(caption_source=vr)
    except Exception as recovery_error:
        _videorag = None
        _fatal_pipeline_error = (
            f"{type(original_error).__name__}: {original_error}; recovery unsafe/failed: "
            f"{type(recovery_error).__name__}: {recovery_error}"
        )
        logger.exception("Cannot safely recover VideoRAG; further index access is blocked")
    else:
        _videorag = restored
        _fatal_pipeline_error = None
        logger.warning("Discarded failed insertion and reloaded committed VideoRAG indexes; later requests may continue")


# ---------------------------------------------------------------------------
# FastMCP Server and Tool Definition
# ---------------------------------------------------------------------------
mcp = MCPServer("VideoRAG")


@mcp.tool()
async def run_videorag_query(
    video_path: str,
    question: str,
    mode: str = "videorag_multiple_choice",
    generation_kwargs: dict | None = None,
    return_raw_response: bool = False,
) -> str:
    """
    Run VideoRAG retrieval-augmented video QA for a (video, question) pair.
    """
    query_mode = mode or _default_mode
    video_name = os.path.basename(video_path).split(".")[0]

    def _configure_decord():
        try:
            import decord
            decord.bridge.set_bridge("torch")
        except Exception:
            pass

    def _insert_video_if_needed(vr):
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

    def _query_video(vr):
        param = QueryParam(
            mode=query_mode,
            generation_kwargs=dict(generation_kwargs or {}),
            return_raw_response=return_raw_response,
        )
        param.wo_reference = True

        response = vr.query(query=question, param=param)

        if return_raw_response:
            # No answer extraction or empty-answer rejection in evaluation mode.
            return response if isinstance(response, str) else ""
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
                raise UnsafeVideoRAGState(
                    "VideoRAG pipeline is halted because safe recovery failed; inspect "
                    "worker cleanup and persisted indexes before restarting the MCP server. "
                    f"Original failure: {_fatal_pipeline_error}"
                )
            # Resolve the singleton only after acquiring the lock: a queued
            # request must not retain the failed instance from before recovery.
            vr = get_videorag()
            try:
                _insert_video_if_needed(vr)
            except Exception as exc:
                _recover_failed_insertion(vr, exc)
                if _fatal_pipeline_error is not None:
                    raise UnsafeVideoRAGState(_fatal_pipeline_error) from exc
                raise
            # Query/answer-format failures do not mutate the index. Let the
            # evaluator retry or fail this sample without poisoning later ones.
            try:
                return _query_video(vr)
            except Exception:
                try:
                    drain_pipeline_tasks(always_get_an_event_loop())
                except Exception as cleanup_error:
                    _fatal_pipeline_error = f"Query cleanup failed: {cleanup_error}"
                    logger.exception("Cannot safely stop failed VideoRAG query tasks")
                    raise UnsafeVideoRAGState(_fatal_pipeline_error) from cleanup_error
                raise

    loop = asyncio.get_running_loop()
    try:
        answer = await loop.run_in_executor(None, _sync_process)
        return answer
    except UnsafeVideoRAGState as exc:
        logger.exception("Fatal VideoRAG pipeline failure; abort this evaluation")
        # MCP masks ordinary exceptions. ToolError deliberately preserves this
        # typed marker so the client can abort instead of grading 900 failures.
        raise ToolError(f"{FATAL_ERROR_PREFIX} {exc}") from exc
    except Exception as exc:
        logger.exception("run_videorag_query failed")
        raise


if __name__ == "__main__":
    get_videorag()
    print(f"[videorag_mcp_server] Starting FastMCP SSE server on {_host}:{_port}", file=sys.stderr)
    mcp.run(transport="sse", host=_host, port=_port)
