from __future__ import annotations

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    from slowapi.util import get_remote_address

    RATE_LIMITING_AVAILABLE = True
    limiter = Limiter(key_func=get_remote_address)
    rate_limit_exceeded_handler = _rate_limit_exceeded_handler
except Exception:
    RATE_LIMITING_AVAILABLE = False
    RateLimitExceeded = Exception
    SlowAPIMiddleware = None

    class _NoopLimiter:
        def limit(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

    limiter = _NoopLimiter()

    def rate_limit_exceeded_handler(*args, **kwargs):
        return None
