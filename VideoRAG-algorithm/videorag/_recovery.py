"""Safety checks before replacing a failed VideoRAG index instance."""

import asyncio


class UnsafeVideoRAGState(RuntimeError):
    """Recovery cannot prove that workers stopped or persisted indexes agree."""


def drain_pipeline_tasks(loop, timeout=10.0):
    """Cancel sibling tasks left behind by a failed gather on a worker loop.

    Call only while holding the pipeline lock, on the synchronous pipeline's
    worker thread, never on the MCP server's running event loop.
    """
    pending = asyncio.all_tasks(loop)
    if not pending:
        return
    for task in pending:
        task.cancel()

    async def drain():
        done, unfinished = await asyncio.wait(pending, timeout=timeout)
        for task in done:
            if not task.cancelled():
                task.exception()  # Retrieve secondary failures before discarding.
        if unfinished:
            raise UnsafeVideoRAGState("VideoRAG async workers did not stop; index reload is unsafe")

    loop.run_until_complete(drain())
