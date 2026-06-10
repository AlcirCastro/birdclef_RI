"""Process memory probes. Falls back to /proc on Linux if psutil isn't available."""

from __future__ import annotations

import os
import resource


def current_rss_bytes() -> int:
    try:
        import psutil
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except ImportError:
        try:
            with open(f"/proc/{os.getpid()}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) * 1024
        except OSError:
            pass
        return 0


def peak_rss_bytes() -> int:
    """Process peak RSS via getrusage. Linux reports kilobytes, macOS reports bytes."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if not usage:
        return 0
    # Heuristic: if value seems too small for a reasonable process, assume KB.
    return usage * 1024 if usage < 10**9 else usage
