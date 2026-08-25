import time
from collections import defaultdict, deque
from threading import Lock

from . import config

_lock = Lock()
_requests: dict[str, deque] = defaultdict(deque)


def is_rate_limited(identifier: str) -> bool:
    now = time.monotonic()
    window_start = now - config.RATE_LIMIT_WINDOW_SECONDS

    with _lock:
        bucket = _requests[identifier]
        while bucket and bucket[0] < window_start:
            bucket.popleft()

        if len(bucket) >= config.RATE_LIMIT_MAX_REQUESTS:
            return True

        bucket.append(now)
        return False
