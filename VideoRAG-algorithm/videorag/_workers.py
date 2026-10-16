"""Lifecycle management for VideoRAG's segment-saving and caption workers."""

from queue import Empty

from ._recovery import UnsafeVideoRAGState

_SHUTDOWN_TIMEOUT = 5.0


def _cleanup_workers(workers):
    """Stop only our surviving children, reap them, and release their handles."""
    errors = []
    # Signal all survivors before waiting for any one of them.
    for worker in workers:
        try:
            if worker.pid is not None and worker.is_alive():
                worker.terminate()
        except Exception as exc:
            errors.append(exc)

    for worker in workers:
        try:
            if worker.pid is not None:
                worker.join(timeout=_SHUTDOWN_TIMEOUT)
                if worker.is_alive():
                    worker.kill()
                    worker.join(timeout=_SHUTDOWN_TIMEOUT)
                if worker.is_alive():
                    raise RuntimeError(f"VideoRAG worker {worker.name} did not exit after kill")
            # close() also accepts workers whose start() was never reached.
            worker.close()
        except Exception as exc:
            errors.append(exc)
    if errors:
        raise UnsafeVideoRAGState(f"VideoRAG worker cleanup failed: {errors}") from errors[0]


def run_video_workers(workers, error_queue):
    """Run a worker pair, fail promptly on either error, and always clean up.

    The caller must keep the manager alive until this function returns or
    raises: both the result proxies and error queue belong to that manager.
    """
    try:
        for worker in workers:
            worker.start()

        pending = list(workers)
        while pending:
            for worker in pending[:]:
                worker.join(timeout=0.1)
                if worker.exitcode is not None:
                    pending.remove(worker)

            errors = []
            while True:
                try:
                    errors.append(str(error_queue.get_nowait()))
                except Empty:
                    break

            # A killed/crashed child may never have written to error_queue.
            errors.extend(f"{worker.name} exited with code {worker.exitcode}" for worker in workers if worker.exitcode not in (None, 0))
            if errors:
                raise RuntimeError("VideoRAG segment processing failed:\n" + "\n".join(errors))
    except BaseException as original_error:
        try:
            _cleanup_workers(workers)
        except Exception as cleanup_error:
            # A surviving worker can still modify segment files. Do not reload
            # and start another insertion while that worker may be running.
            raise UnsafeVideoRAGState(
                f"VideoRAG worker cleanup failed after {type(original_error).__name__}: {original_error}"
            ) from cleanup_error
        raise
    else:
        _cleanup_workers(workers)
