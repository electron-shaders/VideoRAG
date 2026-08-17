"""
videorag_adapter.py — MCP client adapter between lmms-eval and VideoRAG.

The VideoRAG pipeline runs as an MCP SSE server in the 'videorag' conda
environment. This adapter manages:
  - Launching the server subprocess once (init_videorag_instance).
  - Calling the VideoRAG tool via the official MCP ClientSession over SSE.

No VideoRAG or fastmcp dependencies are imported here; official `mcp` SDK is used.
"""

from __future__ import annotations

import atexit
import logging
import os
import subprocess
import threading
import time
from typing import Optional

import httpx
from mcp import ClientSession
from mcp.client.sse import sse_client

logger = logging.getLogger("videorag_adapter")

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
_init_lock = threading.Lock()
_initialized = False

_server_env: dict[str, str] = {}
_videorag_venv_python: str = "python"
_mcp_server_script: str = ""
_mcp_url: str = ""  # e.g. "http://127.0.0.1:9003"
_server_proc: subprocess.Popen | None = None


def _cleanup_server():
    global _server_proc
    if _server_proc is not None:
        try:
            _server_proc.terminate()
            _server_proc.wait(timeout=5)
        except Exception:
            try:
                _server_proc.kill()
            except Exception:
                pass
        _server_proc = None


atexit.register(_cleanup_server)


# ---------------------------------------------------------------------------
# Public initialization
# ---------------------------------------------------------------------------
def init_videorag_instance(
    videorag_path: str,
    base_url: str,
    api_key: str,
    vlm_model: str,
    embed_model: str = "BAAI/bge-m3",
    embed_dim: int = 1024,
    embed_base_url: str | None = None,
    working_dir: str = "./.videorag_cache",
    video_segment_length: int = 30,
    segment_retrieval_top_k: int = 4,
    videorag_query_mode: str = "videorag_multiple_choice",
    load_caption_model: bool = False,
    videorag_venv_python: str = "",
    videorag_host: str = "127.0.0.1",
    videorag_port: int = 9003,
    videorag_cuda_device: str = "",
    **kwargs,
) -> None:
    """
    Launch the VideoRAG MCP server as a background subprocess and wait
    for it to be ready. Must be called once before run_videorag_query().
    """
    global _initialized, _server_env, _videorag_venv_python, _mcp_server_script
    global _mcp_url, _server_proc

    if _initialized:
        return

    with _init_lock:
        if _initialized:
            return

        _videorag_venv_python = (
            videorag_venv_python
            or os.environ.get("VIDEORAG_VENV_PYTHON", "")
            or os.path.expanduser("~/miniconda/envs/videorag/bin/python")
        )
        _mcp_server_script = os.path.join(videorag_path, "mcp_server.py")
        _mcp_url = f"http://{videorag_host}:{videorag_port}"

        if not os.path.isfile(_mcp_server_script):
            raise FileNotFoundError(
                f"[videorag_adapter] mcp_server.py not found at '{_mcp_server_script}'."
            )

        if not os.path.isfile(_videorag_venv_python) or not os.access(_videorag_venv_python, os.X_OK):
            raise FileNotFoundError(
                f"[videorag_adapter] Python interpreter not executable at '{_videorag_venv_python}'."
            )

        cuda_dev = (
            videorag_cuda_device
            or os.environ.get("VIDEORAG_CUDA_DEVICE", "")
            or os.environ.get("VIDEORAG_GPUS", "")
        )

        _server_env = {
            **os.environ,
            "VIDEORAG_BASE_URL": base_url,
            "VIDEORAG_EMBED_BASE_URL": embed_base_url or base_url,
            "VIDEORAG_API_KEY": api_key or "EMPTY",
            "VIDEORAG_VLM_MODEL": vlm_model,
            "VIDEORAG_EMBED_MODEL": embed_model,
            "VIDEORAG_EMBED_DIM": str(embed_dim),
            "VIDEORAG_WORKING_DIR": working_dir,
            "VIDEORAG_SEGMENT_LENGTH": str(video_segment_length),
            "VIDEORAG_RETRIEVAL_TOPK": str(segment_retrieval_top_k),
            "VIDEORAG_QUERY_MODE": videorag_query_mode,
            "VIDEORAG_LOAD_CAPTION": "true" if load_caption_model else "false",
            "VIDEORAG_HOST": videorag_host,
            "VIDEORAG_PORT": str(videorag_port),
        }
        if cuda_dev:
            _server_env["VIDEORAG_CUDA_DEVICE"] = str(cuda_dev)
            _server_env["CUDA_VISIBLE_DEVICES"] = str(cuda_dev)

        # Check if an existing server is already running and ready on this port
        try:
            resp = httpx.get(f"{_mcp_url}/", timeout=1.0)
            if resp.status_code < 500:
                logger.info(f"[videorag_adapter] VideoRAG MCP server is already running at {_mcp_url}")
                _initialized = True
                return
        except Exception:
            pass

        # Use a cross-process lock to prevent duplicate subprocess spawns
        import fcntl
        lock_file_path = os.path.join(working_dir or "/tmp", "videorag_mcp_server.lock")
        os.makedirs(os.path.dirname(os.path.abspath(lock_file_path)), exist_ok=True)
        with open(lock_file_path, "w") as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                # Double-check inside lock
                try:
                    resp = httpx.get(f"{_mcp_url}/", timeout=1.0)
                    if resp.status_code < 500:
                        logger.info(f"[videorag_adapter] VideoRAG MCP server is already running at {_mcp_url}")
                        _initialized = True
                        return
                except Exception:
                    pass

                logger.info(
                    f"[videorag_adapter] Launching VideoRAG MCP server subprocess: "
                    f"{_videorag_venv_python} {_mcp_server_script} on {_mcp_url}"
                )

                _server_proc = subprocess.Popen(
                    [_videorag_venv_python, _mcp_server_script],
                    env=_server_env,
                    stdout=subprocess.DEVNULL,
                    stderr=None,  # inherit stderr so logs are visible
                )

                # Wait for the MCP server to accept connections
                elapsed = 0
                while True:
                    if _server_proc.poll() is not None:
                        raise RuntimeError(
                            f"[videorag_adapter] VideoRAG MCP server subprocess exited early "
                            f"(returncode={_server_proc.returncode})."
                        )
                    try:
                        resp = httpx.get(f"{_mcp_url}/", timeout=1.0)
                        if resp.status_code < 500:
                            logger.info(f"[videorag_adapter] VideoRAG MCP server is ready at {_mcp_url}")
                            break
                    except Exception:
                        pass

                    time.sleep(1.0)
                    elapsed += 1
                    if elapsed % 10 == 0:
                        logger.info(f"[videorag_adapter] Waiting for VideoRAG MCP server ({elapsed}s elapsed)...")

                _initialized = True
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Public query API
# ---------------------------------------------------------------------------
async def run_videorag_query(
    video_path: str,
    question: str,
    mode: str = "videorag_multiple_choice",
    **kwargs,
) -> str:
    """
    Run VideoRAG query via official MCP ClientSession over SSE.
    """
    if not _initialized:
        raise RuntimeError(
            "[videorag_adapter] init_videorag_instance() must be called before run_videorag_query()."
        )

    try:
        async with sse_client(f"{_mcp_url}/sse") as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(
                    "run_videorag_query",
                    arguments={
                        "video_path": video_path,
                        "question": question,
                        "mode": mode,
                    },
                )
                if result and result.content:
                    for item in result.content:
                        if hasattr(item, "text"):
                            return item.text
                        return str(item)
                return ""
    except Exception as exc:
        logger.exception("[videorag_adapter] MCP call_tool failed")
        return ""
