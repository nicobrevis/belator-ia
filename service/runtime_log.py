from __future__ import annotations


def safe_console_log(message: object) -> None:
    """Best-effort worker logging that remains safe during process teardown."""

    try:
        print(message, flush=True)
    except (BrokenPipeError, OSError, ValueError):
        # Dedicated workers inherit a log stream from the supervisor.  During a
        # concurrent shutdown that stream can already be closed; logging must
        # never turn a recoverable publisher/source failure into a worker crash.
        return
